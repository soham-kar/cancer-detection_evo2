# Evo2 Innovation: CRISPR Applications

This folder contains innovative Evo2 implementations for CRISPR applications where foundation models **actually add value**.

## Why This Folder Exists

On **binary off-target classification** (Kleinstiver), simple mismatch counting achieves **AUROC 0.92**. Evo2 adds marginal value there.

This folder focuses on **harder problems** where Evo2 excels:

| Task | Heuristic Performance | Evo2 Potential |
|------|----------------------|----------------|
| Binary classification | AUROC 0.92 | ❌ Overkill |
| **Quantitative regression** | Spearman ~0.3 | ✅ Substantial |
| **On-target efficiency** | Rule-based ~0.5 | ✅ Substantial |
| **Chromatin-aware scoring** | Impossible | ✅ Unique capability |

## File Structure

```
evo2_innovation/
├── README.md                    # This file
├── config.py                    # Configuration and hyperparameters
├── feature_extraction.py        # Mismatch-aware Evo2 feature extraction
├── quantitative_model.py        # Regression model for cleavage efficiency
├── data_preparation.py          # Download and prepare CHANGE-seq data
├── train_regression.py          # Training pipeline
├── evaluate.py                  # Evaluation metrics (Spearman, etc.)
└── modal_extract.py             # Modal job for GPU extraction
```

## Quick Start

```bash
# 1. Prepare CHANGE-seq data
python data_preparation.py

# 2. Extract Evo2 features (requires Modal)
modal run modal_extract.py

# 3. Train regression model
python train_regression.py

# 4. Evaluate
python evaluate.py
```

## Key Innovations

1. **Mismatch-aware pooling**: Extract embeddings AT mismatch positions
2. **Quantitative regression**: Predict continuous cleavage rates
3. **Multi-scale encoding**: Combine local (20bp) + global (8kb) features
4. **Zero-shot generalization**: Test on held-out genomic loci

## Success Metrics

- **Spearman > 0.6** on CHANGE-seq → Nature Biotechnology level
- **Spearman > 0.8** → Clinical utility for therapeutic dosing
