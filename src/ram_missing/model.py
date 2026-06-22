"""Paper-faithful RAM-Missing fusion components.

The modality order in this module is always ``[WSI, CT, RNA]``. The retrieval
bank stores raw, pre-extracted features and is populated explicitly from the
current training fold. It is registered as buffers rather than trainable model
parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class RetrievalResult:
    """Outputs produced by a top-k memory-bank query."""

    proxy_ct: Tensor
    entropy: Tensor
    weights: Tensor
    indices: Tensor


@dataclass
class RAMOutput:
    """Model outputs required for optimization and routing audits."""

    logits: Tensor
    router_weights: Tensor
    proxy_ct: Tensor
    projected_ct: Tensor
    retrieval_entropy: Tensor
    retrieval_weights: Tensor
    retrieval_indices: Tensor
    retrieval_enabled: bool


class RetrievalMemoryBank(nn.Module):
    """Non-trainable, fold-local WSI/RNA keys and CT values."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("wsi", torch.empty(0), persistent=False)
        self.register_buffer("ct", torch.empty(0), persistent=False)
        self.register_buffer("rna", torch.empty(0), persistent=False)
        self.register_buffer("patient_codes", torch.empty(0, dtype=torch.long), persistent=False)

    @property
    def size(self) -> int:
        return 0 if self.ct.ndim == 1 else int(self.ct.shape[0])

    def set(
        self,
        *,
        wsi: Tensor,
        ct: Tensor,
        rna: Tensor,
        patient_codes: Tensor,
    ) -> None:
        """Replace the bank with complete cases from one training fold."""
        n = int(wsi.shape[0])
        if n == 0:
            raise ValueError("The retrieval bank cannot be empty")
        if ct.shape[0] != n or rna.shape[0] != n or patient_codes.shape[0] != n:
            raise ValueError("All memory-bank arrays must have the same first dimension")
        if patient_codes.ndim != 1:
            raise ValueError("patient_codes must be a one-dimensional integer tensor")
        self.wsi = wsi.detach().clone()
        self.ct = ct.detach().clone()
        self.rna = rna.detach().clone()
        self.patient_codes = patient_codes.detach().to(dtype=torch.long).clone()

    def retrieve(
        self,
        *,
        query_wsi: Tensor,
        query_rna: Tensor,
        wsi_encoder: nn.Module,
        ct_encoder: nn.Module,
        rna_encoder: nn.Module,
        top_k: int,
        temperature: float,
        query_patient_codes: Tensor | None = None,
    ) -> RetrievalResult:
        """Retrieve a softmax-weighted CT proxy using cosine similarity.

        When query patient codes are supplied, memory rows from the same patient
        are excluded. This prevents the proxy-consistency objective from learning
        an identity lookup on complete training cases.
        """
        if self.size == 0:
            raise RuntimeError("Retrieval memory has not been populated")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        mem_wsi = wsi_encoder(self.wsi)
        mem_rna = rna_encoder(self.rna)
        mem_ct = ct_encoder(self.ct)
        query = F.normalize(torch.cat([query_wsi, query_rna], dim=-1), dim=-1)
        keys = F.normalize(torch.cat([mem_wsi, mem_rna], dim=-1), dim=-1)
        similarities = query @ keys.transpose(0, 1)

        if query_patient_codes is not None:
            if query_patient_codes.ndim != 1 or query_patient_codes.shape[0] != query.shape[0]:
                raise ValueError("query_patient_codes must contain one code per query")
            same_patient = query_patient_codes[:, None].eq(self.patient_codes[None, :])
            available = (~same_patient).sum(dim=1)
            if torch.any(available == 0):
                raise ValueError("A query has no non-self patient available in the memory bank")
            similarities = similarities.masked_fill(same_patient, -torch.inf)
            k = min(top_k, int(available.min().item()))
        else:
            k = min(top_k, self.size)

        top_similarity, top_indices = similarities.topk(k, dim=1)
        weights = F.softmax(top_similarity / temperature, dim=1)
        values = mem_ct[top_indices]
        proxy = (weights.unsqueeze(-1) * values).sum(dim=1)
        entropy = -(weights * torch.log(weights.clamp_min(1e-8))).sum(dim=1, keepdim=True)
        return RetrievalResult(proxy, entropy, weights, top_indices)


class GatedSumExpert(nn.Module):
    """Stable mask-aware weighted sum of the three modality embeddings."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.gate = nn.Linear(hidden_dim * 3, 3)
        self.output = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, wsi: Tensor, ct: Tensor, rna: Tensor, mask: Tensor) -> Tensor:
        tokens = torch.stack([wsi, ct, rna], dim=1)
        gate_logits = self.gate(torch.cat([wsi, ct, rna], dim=1))
        gate_logits = gate_logits.masked_fill(mask.eq(0), -torch.inf)
        weights = F.softmax(gate_logits, dim=1)
        return self.output((tokens * weights.unsqueeze(-1)).sum(dim=1))


class CrossAttentionExpert(nn.Module):
    """Cross-modal attention from the observed-modality mean query."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.output = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, wsi: Tensor, ct: Tensor, rna: Tensor, mask: Tensor) -> Tensor:
        tokens = torch.stack([wsi, ct, rna], dim=1)
        denominator = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        query = (tokens * mask.unsqueeze(-1)).sum(dim=1) / denominator
        attended, _ = self.attention(
            query.unsqueeze(1),
            tokens,
            tokens,
            key_padding_mask=mask.eq(0),
            need_weights=False,
        )
        return self.output(attended.squeeze(1))


class MemoryAwareExpert(nn.Module):
    """Align retrieved CT evidence with the observed WSI/RNA context."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.context = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.ct_alignment = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.ct_gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
        self.output = nn.LayerNorm(hidden_dim)

    def forward(self, wsi: Tensor, ct: Tensor, rna: Tensor, mask: Tensor) -> Tensor:
        observed_context = self.context(torch.cat([wsi, rna], dim=1))
        ct_context = torch.cat([observed_context, ct], dim=1)
        aligned_ct = self.ct_alignment(ct_context)
        gate = self.ct_gate(ct_context)
        # The effective CT passed to this expert is real when present and the
        # retrieved proxy otherwise. The router receives the original mask.
        return self.output(observed_context + gate * aligned_ct)


class RAMMissing(nn.Module):
    """Retrieval-augmented missing-aware three-expert fusion model."""

    def __init__(
        self,
        *,
        wsi_dim: int,
        ct_dim: int,
        rna_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.3,
        top_k: int = 5,
        retrieval_temperature: float = 0.1,
        gate_temperature: float = 1.0,
        num_classes: int = 2,
        use_retrieval: bool = True,
        use_router: bool = True,
        use_uncertainty: bool = True,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if gate_temperature <= 0:
            raise ValueError("gate_temperature must be positive")
        self.wsi_encoder = nn.Linear(wsi_dim, hidden_dim)
        self.ct_encoder = nn.Linear(ct_dim, hidden_dim)
        self.rna_encoder = nn.Linear(rna_dim, hidden_dim)
        self.memory = RetrievalMemoryBank()
        self.top_k = top_k
        self.retrieval_temperature = retrieval_temperature
        self.gate_temperature = gate_temperature
        self.use_retrieval = use_retrieval
        self.use_router = use_router
        self.use_uncertainty = use_uncertainty

        self.experts = nn.ModuleList(
            [
                GatedSumExpert(hidden_dim),
                CrossAttentionExpert(hidden_dim, num_heads, dropout),
                MemoryAwareExpert(hidden_dim, dropout),
            ]
        )
        self.router = nn.Sequential(
            nn.Linear(hidden_dim * 3 + 3 + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(self.experts)),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def set_memory(
        self,
        *,
        wsi: Tensor,
        ct: Tensor,
        rna: Tensor,
        patient_codes: Tensor,
    ) -> None:
        self.memory.set(wsi=wsi, ct=ct, rna=rna, patient_codes=patient_codes)

    def forward(
        self,
        *,
        wsi: Tensor,
        ct: Tensor,
        rna: Tensor,
        mask: Tensor,
        patient_codes: Tensor | None = None,
        exclude_query_patient: bool = False,
    ) -> RAMOutput:
        if mask.ndim != 2 or mask.shape[1] != 3:
            raise ValueError("mask must have shape [batch, 3] in [WSI, CT, RNA] order")
        if torch.any(mask.sum(dim=1) == 0):
            raise ValueError("Every sample must have at least one observed modality")

        projected_wsi = self.wsi_encoder(wsi) * mask[:, 0:1]
        projected_ct = self.ct_encoder(ct)
        projected_rna = self.rna_encoder(rna) * mask[:, 2:3]
        if self.use_retrieval:
            retrieval = self.memory.retrieve(
                query_wsi=projected_wsi,
                query_rna=projected_rna,
                wsi_encoder=self.wsi_encoder,
                ct_encoder=self.ct_encoder,
                rna_encoder=self.rna_encoder,
                top_k=self.top_k,
                temperature=self.retrieval_temperature,
                query_patient_codes=patient_codes if exclude_query_patient else None,
            )
        else:
            batch_size = wsi.shape[0]
            retrieval = RetrievalResult(
                proxy_ct=torch.zeros_like(projected_ct),
                entropy=projected_ct.new_zeros((batch_size, 1)),
                weights=projected_ct.new_zeros((batch_size, 0)),
                indices=torch.empty((batch_size, 0), dtype=torch.long, device=wsi.device),
            )
        effective_ct = mask[:, 1:2] * projected_ct + (1.0 - mask[:, 1:2]) * retrieval.proxy_ct
        effective_mask = mask.clone()
        # The proxy makes a CT token available to experts, while the original
        # missingness bit remains part of the router condition.
        expert_mask = effective_mask.clone()
        if self.use_retrieval:
            expert_mask[:, 1] = 1.0

        router_entropy = (
            retrieval.entropy if self.use_uncertainty else torch.zeros_like(retrieval.entropy)
        )
        router_input = torch.cat(
            [projected_wsi, effective_ct, projected_rna, mask, router_entropy], dim=1
        )
        if self.use_router:
            router_logits = self.router(router_input) / self.gate_temperature
            router_weights = F.softmax(router_logits, dim=1)
        else:
            router_weights = projected_ct.new_full(
                (projected_ct.shape[0], len(self.experts)), 1.0 / len(self.experts)
            )
        expert_outputs = torch.stack(
            [
                expert(projected_wsi, effective_ct, projected_rna, expert_mask)
                for expert in self.experts
            ],
            dim=1,
        )
        fused = (router_weights.unsqueeze(-1) * expert_outputs).sum(dim=1)
        logits = self.classifier(fused)
        return RAMOutput(
            logits=logits,
            router_weights=router_weights,
            proxy_ct=retrieval.proxy_ct,
            projected_ct=projected_ct,
            retrieval_entropy=retrieval.entropy,
            retrieval_weights=retrieval.weights,
            retrieval_indices=retrieval.indices,
            retrieval_enabled=self.use_retrieval,
        )


def ram_missing_loss(
    output: RAMOutput,
    labels: Tensor,
    modality_mask: Tensor,
    *,
    proxy_weight: float = 0.5,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute ``L_cls + proxy_weight * L_proxy`` from the paper."""
    classification = F.cross_entropy(output.logits, labels)
    complete = modality_mask[:, 1].bool()
    if output.retrieval_enabled and complete.any():
        residual = output.proxy_ct[complete] - output.projected_ct[complete]
        proxy = residual.square().sum(dim=1).mean()
    else:
        proxy = output.logits.new_zeros(())
    total = classification + proxy_weight * proxy
    return total, {"classification": classification.detach(), "proxy": proxy.detach()}
