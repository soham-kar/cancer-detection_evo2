# Evo2 On-Target CRISPR Efficiency Prediction

Zero-shot prediction of CRISPR on-target efficiency using Evo2 genomic embeddings.

## Innovation

**Problem**: Existing models (DeepHF, CRISPRon) train on synthetic reporters but fail on endogenous genomic loci (r ~0.45).

**Solution**: Evo2's pre-trained chromatin knowledge enables zero-shot generalization to new cell types and endogenous sites.

| Method | Synthetic (training) | Endogenous (test) |
|--------|---------------------|-------------------|
| DeepHF | r = 0.87 | r = 0.45 |
| CRISPRon | r = 0.85 | r = 0.42 |
| **Evo2 (ours)** | - | **Target: r > 0.65** |

## Quick Start

```bash
# 1. Prepare data (stratified by cell type)
python 01_prepare_deephf_data.py

# 2. Extract Evo2 embeddings (~$24 for train, ~$0.60 for test)
modal run 02_modal_extract.py --input data/train_coords.csv --output features_train
modal run 02_modal_extract.py --input data/test_coords.csv --output features_test

# 3. Train predictor and evaluate zero-shot
python 03_train_efficiency_predictor.py
```

## Cost Estimate

| Phase | Samples | Cost |
|-------|---------|------|
| Train | 4,000 | $24 |
| Test (zero-shot) | 100 | $0.60 |
| **Total** | 4,100 | **$24.60** |

## Success Criteria

| Spearman r | Interpretation | Publication |
|------------|----------------|-------------|
| > 0.65 | Evo2 beats DeepHF | Nature Biotechnology |
| 0.50-0.65 | Marginal improvement | Bioinformatics |
| < 0.50 | No advantage | Negative result |

## Directory Structure

```
evo2_ontarget_efficiency/
├── 01_prepare_deephf_data.py    # Data download & stratification
├── 02_modal_extract.py          # Evo2 embedding extraction
├── 03_train_efficiency_predictor.py  # Train & evaluate
├── data/
│   ├── train_coords.csv         # Training coordinates
│   └── test_coords.csv          # Zero-shot test coordinates
├── features_train/              # Evo2 embeddings (train)
├── features_test/               # Evo2 embeddings (test)
└── results/                     # Output figures & metrics
```
