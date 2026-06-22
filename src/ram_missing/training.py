"""Leakage-aware training and evaluation utilities."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from .data import MultimodalArrays
from .model import RAMMissing, ram_missing_loss


@dataclass(frozen=True)
class FoldFeatures:
    wsi: np.ndarray
    ct: np.ndarray
    rna: np.ndarray


class FeatureStandardizer:
    """Fit each modality on the current training fold only."""

    def __init__(self) -> None:
        self.stats: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def fit(self, arrays: MultimodalArrays, train_indices: np.ndarray) -> "FeatureStandardizer":
        modality_columns = {"wsi": 0, "ct": 1, "rna": 2}
        for name, column in modality_columns.items():
            indices = train_indices[arrays.availability[train_indices, column] == 1]
            if len(indices) == 0:
                raise ValueError(f"No observed {name} samples in the training fold")
            values = getattr(arrays, name)[indices].astype(np.float64)
            mean = values.mean(axis=0)
            scale = values.std(axis=0)
            scale[scale < 1e-8] = 1.0
            self.stats[name] = (mean.astype(np.float32), scale.astype(np.float32))
        return self

    def transform(self, arrays: MultimodalArrays) -> FoldFeatures:
        transformed: dict[str, np.ndarray] = {}
        for column, name in enumerate(("wsi", "ct", "rna")):
            mean, scale = self.stats[name]
            values = (getattr(arrays, name).astype(np.float32) - mean) / scale
            values[arrays.availability[:, column] == 0] = 0.0
            transformed[name] = values.astype(np.float32)
        return FoldFeatures(**transformed)


class MultimodalDataset(Dataset[dict[str, Tensor]]):
    def __init__(
        self,
        arrays: MultimodalArrays,
        features: FoldFeatures,
        indices: np.ndarray,
        *,
        force_missing_ct: bool = False,
        forced_missing_ct_indices: set[int] | None = None,
    ) -> None:
        self.arrays = arrays
        self.features = features
        self.indices = np.asarray(indices, dtype=np.int64)
        self.force_missing_ct = force_missing_ct
        self.forced_missing_ct_indices = forced_missing_ct_indices or set()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, Tensor]:
        index = int(self.indices[item])
        mask = self.arrays.availability[index].copy()
        ct = self.features.ct[index].copy()
        if self.force_missing_ct or index in self.forced_missing_ct_indices:
            mask[1] = 0.0
            ct.fill(0.0)
        return {
            "wsi": torch.from_numpy(self.features.wsi[index]),
            "ct": torch.from_numpy(ct),
            "rna": torch.from_numpy(self.features.rna[index]),
            "mask": torch.from_numpy(mask),
            "label": torch.tensor(self.arrays.labels[index], dtype=torch.long),
            "patient_code": torch.tensor(
                self.arrays.patient_codes[index], dtype=torch.long
            ),
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Keep repeated runs stable on the same software/hardware stack.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def inner_train_validation_split(
    indices: np.ndarray, arrays: MultimodalArrays, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_relative, validation_relative = next(
        splitter.split(
            indices,
            arrays.labels[indices],
            groups=arrays.patient_ids[indices],
        )
    )
    return indices[train_relative], indices[validation_relative]


def _move(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def set_fold_memory(
    model: nn.Module,
    arrays: MultimodalArrays,
    features: FoldFeatures,
    indices: np.ndarray,
    device: torch.device,
) -> None:
    if not isinstance(model, RAMMissing) or not model.use_retrieval:
        return
    complete = indices[arrays.availability[indices, 1] == 1]
    if len(complete) < 2:
        raise ValueError("RAM-Missing requires at least two complete training patients")
    model.set_memory(
        wsi=torch.from_numpy(features.wsi[complete]).to(device),
        ct=torch.from_numpy(features.ct[complete]).to(device),
        rna=torch.from_numpy(features.rna[complete]).to(device),
        patient_codes=torch.from_numpy(arrays.patient_codes[complete]).to(device),
    )


def predict_probabilities(
    model: nn.Module,
    loader: DataLoader[dict[str, Tensor]],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            moved = _move(batch, device)
            if isinstance(model, RAMMissing):
                logits = model(
                    wsi=moved["wsi"],
                    ct=moved["ct"],
                    rna=moved["rna"],
                    mask=moved["mask"],
                ).logits
            else:
                logits = model(
                    wsi=moved["wsi"],
                    ct=moved["ct"],
                    rna=moved["rna"],
                    mask=moved["mask"],
                )
            probabilities.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            labels.append(moved["label"].cpu().numpy())
    return np.concatenate(probabilities), np.concatenate(labels)


def safe_auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    return float("nan") if len(np.unique(labels)) < 2 else float(roc_auc_score(labels, probabilities))


def train_with_early_stopping(
    model: nn.Module,
    *,
    arrays: MultimodalArrays,
    features: FoldFeatures,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    device: torch.device,
    epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    proxy_weight: float,
    num_workers: int = 0,
) -> tuple[nn.Module, int, float]:
    set_fold_memory(model, arrays, features, train_indices, device)
    train_loader = DataLoader(
        MultimodalDataset(arrays, features, train_indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    validation_loader = DataLoader(
        MultimodalDataset(arrays, features, validation_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    best_state: dict[str, Tensor] | None = None
    best_auc = -np.inf
    best_epoch = 0
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            moved = _move(batch, device)
            optimizer.zero_grad(set_to_none=True)
            if isinstance(model, RAMMissing):
                output = model(
                    wsi=moved["wsi"],
                    ct=moved["ct"],
                    rna=moved["rna"],
                    mask=moved["mask"],
                    patient_codes=moved["patient_code"],
                    exclude_query_patient=model.use_retrieval,
                )
                loss, _ = ram_missing_loss(
                    output,
                    moved["label"],
                    moved["mask"],
                    proxy_weight=proxy_weight if model.use_retrieval else 0.0,
                )
            else:
                logits = model(
                    wsi=moved["wsi"],
                    ct=moved["ct"],
                    rna=moved["rna"],
                    mask=moved["mask"],
                )
                loss = torch.nn.functional.cross_entropy(logits, moved["label"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        validation_probabilities, validation_labels = predict_probabilities(
            model, validation_loader, device
        )
        validation_auc = safe_auc(validation_labels, validation_probabilities)
        if np.isfinite(validation_auc) and validation_auc > best_auc:
            best_auc = validation_auc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("Training did not produce a valid validation AUROC")
    model.load_state_dict(best_state)
    return model, best_epoch, float(best_auc)


def evaluate_condition(
    model: nn.Module,
    *,
    arrays: MultimodalArrays,
    features: FoldFeatures,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int,
    condition: str,
) -> dict[str, float | int | str]:
    if condition == "Full":
        eval_indices = indices[arrays.availability[indices, 1] == 1]
        force_missing_ct = False
    elif condition == "Missing_CT":
        eval_indices = indices
        force_missing_ct = True
    elif condition == "Natural":
        eval_indices = indices
        force_missing_ct = False
    else:
        raise ValueError(f"Unknown condition: {condition}")
    if len(eval_indices) == 0:
        return {"condition": condition, "n": 0, "auroc": float("nan"), "accuracy": float("nan")}
    loader = DataLoader(
        MultimodalDataset(
            arrays, features, eval_indices, force_missing_ct=force_missing_ct
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    probabilities, labels = predict_probabilities(model, loader, device)
    predictions = (probabilities >= 0.5).astype(np.int64)
    return {
        "condition": condition,
        "n": int(len(labels)),
        "auroc": safe_auc(labels, probabilities),
        "accuracy": float(accuracy_score(labels, predictions)),
    }


def evaluate_missing_ratio(
    model: nn.Module,
    *,
    arrays: MultimodalArrays,
    features: FoldFeatures,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int,
    ratio: float,
    seed: int,
) -> dict[str, float | int]:
    """Mask a deterministic fraction of complete test patients.

    The evaluation population is restricted to patients with real CT so that
    ratio 0.0 and ratio 0.9 have an unambiguous meaning. This corrected protocol
    differs from the historical positional-slice experiment and is reported as
    such in the release audit.
    """
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("Missing ratio must be between 0 and 1")
    complete = indices[arrays.availability[indices, 1] == 1]
    if len(complete) == 0:
        return {"ratio": ratio, "n": 0, "n_masked": 0, "auroc": float("nan")}
    rng = np.random.default_rng(seed)
    n_masked = int(round(len(complete) * ratio))
    selected = (
        set(int(index) for index in rng.choice(complete, size=n_masked, replace=False))
        if n_masked
        else set()
    )
    loader = DataLoader(
        MultimodalDataset(
            arrays,
            features,
            complete,
            forced_missing_ct_indices=selected,
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    probabilities, labels = predict_probabilities(model, loader, device)
    return {
        "ratio": ratio,
        "n": int(len(complete)),
        "n_masked": n_masked,
        "auroc": safe_auc(labels, probabilities),
    }
