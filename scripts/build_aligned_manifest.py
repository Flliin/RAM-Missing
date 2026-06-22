#!/usr/bin/env python3
"""Build the corrected one-row-per-patient manifest using stable IDs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ram_missing.data import build_aligned_patient_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--label-column", default="task0_histology")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = pd.read_csv(args.metadata_csv)
    labels = pd.read_csv(args.labels_csv)
    cache = np.load(args.feature_cache, allow_pickle=False)
    manifest, summary = build_aligned_patient_manifest(
        metadata,
        labels,
        ct_features=cache["ct"],
        label_column=args.label_column,
        seed=args.seed,
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output_csv, index=False)
    args.summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
