# RAM-Missing

Minimal PyTorch implementation of RAM-Missing for multimodal LUAD/LUSC
classification with missing CT data.

## Install

```bash
python -m pip install -e .
```

## Data

The NPZ feature cache must contain `wsi`, `ct`, and `rna` arrays. The CSV
manifest must contain:

```text
patient_id,label,feature_row,has_wsi,has_ct,has_rna,fold
```

## Train

```bash
python scripts/run_cv.py \
  --config configs/paper.yaml \
  --manifest /path/to/manifest.csv \
  --feature-cache /path/to/features.npz \
  --output-dir outputs/ram_missing
```

Data, checkpoints, and experiment results are not included.
