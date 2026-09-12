# Dataset Structure & Setup

This directory is intended to store the training and evaluation dataset for the waste classification pipeline.

## Expected Directory Hierarchy

```
data/
├── clean_dataset/
│   ├── train/
│   │   ├── 0_Recyclable/        # e.g., plastic bottles, glass jars, paper, metal cans
│   │   ├── 1_Electronic/        # e.g., mobile phones, keyboards, cables, printed circuit boards
│   │   └── 2_Organic/           # e.g., food leftovers, vegetable waste, fruits, leaves
│   └── test/
│       ├── 1.jpg
│       ├── 2.jpg
│       └── ...
└── sample/
    └── sample_submission.csv    # Benchmark template format (id, predicted)
```

## Dataset Specifications

- **Classes**: 3 Mutually Exclusive Categories:
  - `0_Recyclable` (Label Index: `0`)
  - `1_Electronic` (Label Index: `1`)
  - `2_Organic` (Label Index: `2`)
- **Total Training Images (Clean V3)**: 25,693 images
  - `0_Recyclable`: ~10,189 images
  - `1_Electronic`: ~3,944 images (minority class, balanced via class weighting / focal loss)
  - `2_Organic`: ~11,560 images
- **Test Set Size**: 1,458 unlabeled evaluation images

## Data Cleaning & Integrity Rules

1. **No Data Leakage**: In accordance with the competition rules (BDC Satria Data 2026), never mix test samples with training or validation splits.
2. **File Path Safety**: Ensure image paths do not exceed the Windows `MAX_PATH` 260-character limit.
3. **Deduplication**: Run `scripts/dedupe_exact_duplicates.py` before model training to eliminate byte-identical image clones across folds.
