#!/usr/bin/env python3
"""Run patient-level cross-validation for paper baselines and RAM-Missing."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
import yaml

from ram_missing.data import load_aligned_arrays
from ram_missing.model import RAMMissing
from ram_missing.training import (
    FeatureStandardizer,
    evaluate_condition,
    evaluate_missing_ratio,
    inner_train_validation_split,
    set_fold_memory,
    set_seed,
    train_with_early_stopping,
)


DISPLAY_NAMES = {
    "ram_missing": "RAM-Missing",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-folds", type=int, default=5)
    parser.add_argument("--models", help="Comma-separated override")
    return parser.parse_args()


def build_model(name: str, arrays, config: dict) -> torch.nn.Module:
    model_config = config["model"]
    common = {
        "wsi_dim": arrays.wsi.shape[1],
        "ct_dim": arrays.ct.shape[1],
        "rna_dim": arrays.rna.shape[1],
        "hidden_dim": int(model_config["hidden_dim"]),
        "dropout": float(model_config["dropout"]),
    }
    if name != "ram_missing":
        raise ValueError(f"Unknown model: {name}")
    return RAMMissing(
        **common,
        num_heads=int(model_config["num_heads"]),
        top_k=int(model_config["top_k"]),
        retrieval_temperature=float(model_config["retrieval_temperature"]),
        gate_temperature=float(model_config["gate_temperature"]),
    )


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "n",
        "best_epoch",
        "best_val_auc",
        "auroc",
        "accuracy",
        "weighted_auc_paper",
        "weighted_auc_observed",
    ]
    summary = (
        metrics.groupby(["model", "model_key", "condition"], dropna=False)[numeric]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(part for part in column if part).rstrip("_")
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]
    return summary


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    models = (
        [item.strip() for item in args.models.split(",") if item.strip()]
        if args.models
        else list(config["models"])
    )
    unknown = set(models) - set(DISPLAY_NAMES)
    if unknown:
        raise ValueError(f"Unknown model keys: {sorted(unknown)}")

    set_seed(int(config["seed"]))
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested but is unavailable: {device}")
    arrays = load_aligned_arrays(args.manifest, args.feature_cache)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        **config,
        "manifest": str(args.manifest),
        "feature_cache": str(args.feature_cache),
        "inputs_sha256": {
            "config": sha256(args.config),
            "manifest": sha256(args.manifest),
            "feature_cache": sha256(args.feature_cache),
        },
        "models": models,
        "max_folds": args.max_folds,
        "device_resolved": str(device),
        "n_patients": int(len(arrays.labels)),
        "n_ct_present": int(arrays.availability[:, 1].sum()),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2) + "\n", encoding="utf-8"
    )

    rows: list[dict[str, float | int | str]] = []
    ratio_rows: list[dict[str, float | int | str]] = []
    folds = sorted(int(fold) for fold in np.unique(arrays.folds))[: args.max_folds]
    train_config = config["training"]
    prevalence = config["evaluation"]

    for fold in folds:
        outer_test = np.flatnonzero(arrays.folds == fold)
        outer_train = np.flatnonzero(arrays.folds != fold)
        for model_index, name in enumerate(models):
            set_seed(int(config["seed"]) + fold * 100 + model_index)
            candidate_train = outer_train
            candidate_test = outer_test
            if len(candidate_train) < 10 or len(candidate_test) == 0:
                continue

            train_indices, validation_indices = inner_train_validation_split(
                candidate_train, arrays, int(config["seed"]) + fold
            )
            standardizer = FeatureStandardizer().fit(arrays, train_indices)
            features = standardizer.transform(arrays)
            model = build_model(name, arrays, config).to(device)
            model, best_epoch, best_val_auc = train_with_early_stopping(
                model,
                arrays=arrays,
                features=features,
                train_indices=train_indices,
                validation_indices=validation_indices,
                device=device,
                epochs=int(train_config["epochs"]),
                patience=int(train_config["patience"]),
                batch_size=int(train_config["batch_size"]),
                learning_rate=float(train_config["learning_rate"]),
                weight_decay=float(train_config["weight_decay"]),
                proxy_weight=float(train_config["proxy_weight"]),
                num_workers=int(train_config["num_workers"]),
            )
            # Validation belongs to the outer training fold and can be added to
            # the retrieval bank only after early stopping is complete.
            set_fold_memory(model, arrays, features, candidate_train, device)

            conditions = ["Natural", "Full", "Missing_CT"]
            model_rows: list[dict[str, float | int | str]] = []
            for condition in conditions:
                result = evaluate_condition(
                    model,
                    arrays=arrays,
                    features=features,
                    indices=candidate_test,
                    device=device,
                    batch_size=int(train_config["batch_size"]),
                    condition=condition,
                )
                model_rows.append(
                    {
                        "model": DISPLAY_NAMES[name],
                        "model_key": name,
                        "fold": fold,
                        "best_epoch": best_epoch,
                        "best_val_auc": best_val_auc,
                        "weighted_auc_paper": float("nan"),
                        "weighted_auc_observed": float("nan"),
                        **result,
                    }
                )
            if len(model_rows) == 3:
                by_condition = {row["condition"]: row for row in model_rows}
                weighted_auc_paper = (
                    float(prevalence["paper_full_prevalence"])
                    * float(by_condition["Full"]["auroc"])
                    + float(prevalence["paper_missing_prevalence"])
                    * float(by_condition["Missing_CT"]["auroc"])
                )
                observed_full = float(arrays.availability[:, 1].mean())
                weighted_auc_observed = (
                    observed_full * float(by_condition["Full"]["auroc"])
                    + (1.0 - observed_full)
                    * float(by_condition["Missing_CT"]["auroc"])
                )
                for row in model_rows:
                    row["weighted_auc_paper"] = weighted_auc_paper
                    row["weighted_auc_observed"] = weighted_auc_observed
            rows.extend(model_rows)
            if name in set(prevalence.get("missing_ratio_models", [])):
                for ratio in prevalence.get("missing_ratios", []):
                    ratio_result = evaluate_missing_ratio(
                        model,
                        arrays=arrays,
                        features=features,
                        indices=candidate_test,
                        device=device,
                        batch_size=int(train_config["batch_size"]),
                        ratio=float(ratio),
                        seed=int(config["seed"]) + fold * 1000 + int(round(float(ratio) * 100)),
                    )
                    ratio_rows.append(
                        {
                            "model": DISPLAY_NAMES[name],
                            "model_key": name,
                            "fold": fold,
                            **ratio_result,
                        }
                    )
            pd.DataFrame(rows).to_csv(args.output_dir / "fold_metrics.partial.csv", index=False)
            if ratio_rows:
                pd.DataFrame(ratio_rows).to_csv(
                    args.output_dir / "missing_ratio_metrics.partial.csv", index=False
                )
            print(
                f"fold={fold} model={DISPLAY_NAMES[name]} "
                f"best_epoch={best_epoch} val_auc={best_val_auc:.4f}"
            )

    metrics = pd.DataFrame(rows)
    if metrics.empty:
        raise RuntimeError("No experiment rows were produced")
    metrics.to_csv(args.output_dir / "fold_metrics.csv", index=False)
    summarize(metrics).to_csv(args.output_dir / "summary.csv", index=False)
    if ratio_rows:
        ratio_metrics = pd.DataFrame(ratio_rows)
        ratio_metrics.to_csv(args.output_dir / "missing_ratio_metrics.csv", index=False)
        ratio_summary = (
            ratio_metrics.groupby(["model", "model_key", "ratio"])["auroc"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        ratio_summary.to_csv(args.output_dir / "missing_ratio_summary.csv", index=False)
    print(f"saved={args.output_dir}")


if __name__ == "__main__":
    main()
