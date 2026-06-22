"""Stable-ID manifest construction and feature loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


REQUIRED_MANIFEST_COLUMNS = {
    "patient_id",
    "label",
    "feature_row",
    "has_wsi",
    "has_ct",
    "has_rna",
    "fold",
}


@dataclass(frozen=True)
class MultimodalArrays:
    """Aligned arrays in the paper modality order WSI, CT, RNA."""

    patient_ids: np.ndarray
    patient_codes: np.ndarray
    labels: np.ndarray
    feature_rows: np.ndarray
    wsi: np.ndarray
    ct: np.ndarray
    rna: np.ndarray
    availability: np.ndarray
    folds: np.ndarray


def _nonempty_text(series: pd.Series) -> pd.Series:
    return series.notna() & series.astype(str).str.strip().ne("")


def build_aligned_patient_manifest(
    metadata: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    ct_features: np.ndarray,
    label_column: str = "task0_histology",
    n_splits: int = 5,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, int | float]]:
    """Map a one-row-per-patient label table onto a row-aligned feature table.

    Matching requires patient ID, exact WSI path, and exact RNA path. Multiple
    metadata records with the same inputs are resolved by the lowest source row
    index, making the result deterministic.
    """
    metadata_required = {"patient_id", "wsi_path", "rna_path", "ct_path"}
    label_required = {"patient_id", "wsi_path", "rna_absolute_path", label_column}
    missing_metadata = metadata_required - set(metadata.columns)
    missing_labels = label_required - set(labels.columns)
    if missing_metadata:
        raise ValueError(f"Metadata is missing columns: {sorted(missing_metadata)}")
    if missing_labels:
        raise ValueError(f"Label table is missing columns: {sorted(missing_labels)}")
    if len(metadata) != len(ct_features):
        raise ValueError("Metadata rows and feature-cache rows are not aligned")
    if labels["patient_id"].duplicated().any():
        raise ValueError("The patient label table must contain exactly one row per patient")

    source = metadata.reset_index(drop=True).reset_index(names="feature_row").copy()
    source["wsi_path"] = source["wsi_path"].fillna("").astype(str)
    source["rna_path"] = source["rna_path"].fillna("").astype(str)
    target = labels.copy()
    target["wsi_path"] = target["wsi_path"].fillna("").astype(str)
    target["rna_absolute_path"] = target["rna_absolute_path"].fillna("").astype(str)

    selected: list[dict[str, int | str]] = []
    ambiguous = 0
    for row in target.itertuples(index=False):
        candidates = source[
            source["patient_id"].eq(row.patient_id)
            & source["wsi_path"].eq(row.wsi_path)
            & source["rna_path"].eq(row.rna_absolute_path)
        ].sort_values("feature_row")
        if candidates.empty:
            raise ValueError(f"No stable-ID feature match for patient {row.patient_id}")
        ambiguous += int(len(candidates) > 1)
        match = candidates.iloc[0]
        feature_row = int(match["feature_row"])
        ct_nonzero = bool(np.any(np.abs(ct_features[feature_row]) > 1e-12))
        ct_metadata = bool(pd.notna(match["ct_path"]) and str(match["ct_path"]).strip())
        if ct_nonzero != ct_metadata:
            raise ValueError(f"CT metadata/cache disagreement for patient {row.patient_id}")
        selected.append(
            {
                "patient_id": str(row.patient_id),
                "label": int(getattr(row, label_column)),
                "feature_row": feature_row,
                "has_wsi": 1,
                "has_ct": int(ct_nonzero),
                "has_rna": 1,
            }
        )

    manifest = pd.DataFrame(selected)
    if manifest["feature_row"].duplicated().any():
        raise ValueError("Stable-ID mapping selected the same feature row for multiple patients")

    folds = np.full(len(manifest), -1, dtype=np.int64)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (_, test_index) in enumerate(
        splitter.split(manifest, manifest["label"], groups=manifest["patient_id"])
    ):
        folds[test_index] = fold
    if np.any(folds < 0):
        raise RuntimeError("Not every patient was assigned to a fold")
    manifest["fold"] = folds

    naive_rows = min(len(labels), len(ct_features))
    naive_ct = np.any(np.abs(ct_features[:naive_rows]) > 1e-12, axis=1)
    summary: dict[str, int | float] = {
        "n_patients": int(len(manifest)),
        "n_ct_present_stable_id": int(manifest["has_ct"].sum()),
        "ct_prevalence_stable_id": float(manifest["has_ct"].mean()),
        "n_ct_present_naive_positional_slice": int(naive_ct.sum()),
        "ct_prevalence_naive_positional_slice": float(naive_ct.mean()),
        "n_ambiguous_source_matches_resolved": int(ambiguous),
        "n_unique_feature_rows": int(manifest["feature_row"].nunique()),
    }
    return manifest, summary


def load_aligned_arrays(manifest_path: str | Path, feature_cache_path: str | Path) -> MultimodalArrays:
    """Load a validated manifest and its aligned NPZ feature cache."""
    manifest = pd.read_csv(manifest_path)
    missing = REQUIRED_MANIFEST_COLUMNS - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    if manifest["patient_id"].duplicated().any():
        raise ValueError("Canonical release manifests must contain one row per patient")
    if manifest["feature_row"].duplicated().any():
        raise ValueError("Canonical release manifests must contain unique feature rows")

    cache = np.load(feature_cache_path, allow_pickle=False)
    required_arrays = {"wsi", "ct", "rna"}
    missing_arrays = required_arrays - set(cache.files)
    if missing_arrays:
        raise ValueError(f"Feature cache is missing arrays: {sorted(missing_arrays)}")
    feature_rows = manifest["feature_row"].to_numpy(dtype=np.int64)
    if feature_rows.min() < 0 or feature_rows.max() >= len(cache["wsi"]):
        raise ValueError("Manifest feature_row is outside the cache")
    if not (len(cache["wsi"]) == len(cache["ct"]) == len(cache["rna"])):
        raise ValueError("Feature arrays have different row counts")

    patients = manifest["patient_id"].astype(str).to_numpy()
    patient_codes, _ = pd.factorize(patients, sort=True)
    availability = manifest[["has_wsi", "has_ct", "has_rna"]].to_numpy(dtype=np.float32)
    return MultimodalArrays(
        patient_ids=patients,
        patient_codes=patient_codes.astype(np.int64),
        labels=manifest["label"].to_numpy(dtype=np.int64),
        feature_rows=feature_rows,
        wsi=cache["wsi"][feature_rows].astype(np.float32),
        ct=cache["ct"][feature_rows].astype(np.float32),
        rna=cache["rna"][feature_rows].astype(np.float32),
        availability=availability,
        folds=manifest["fold"].to_numpy(dtype=np.int64),
    )
