# Evo2 ATAC-seq Prediction

Predict chromatin accessibility (ATAC-seq signal) from genomic sequence using Evo2 embeddings.

## Why This Works

Evo2 was **pre-trained on ATAC-seq peak prediction** (Pearson r = 0.89 in paper). This pipeline leverages that pre-training for accessibility prediction in new cell types.

| Metric | Expected | Baseline |
|--------|----------|----------|
| Pearson r | 0.62+ | ~0.30 |
| Cost | $30 | - |
| Time | 3-4 days | - |

## Quick Start

```bash
# 1. Download ENCODE ATAC-seq data (or use mock for testing)
python 01_download_encode_atac.py --mock

# 2. Prepare 8kb coordinate windows
python 02_prepare_coordinates.py

# 3. Extract Evo2 embeddings (requires Modal + A100)
modal run 03_modal_extract_atac.py --n_samples 100 --test

# 4. Train predictor
python 04_train_atac_predictor.py --epochs 50

# 5. Evaluate and generate figures
python 05_evaluate.py
```

## Directory Structure

```
evo2_atac_prediction/
├── 01_download_encode_atac.py    # Download ENCODE data
├── 02_prepare_coordinates.py     # Create 8kb windows
├── 03_modal_extract_atac.py      # Evo2 embedding extraction
├── 04_train_atac_predictor.py    # Train MLP predictor
├── 05_evaluate.py                # Generate figures
├── data/
│   └── atac/                     # Peak files and coordinates
├── features_cache/               # Evo2 embeddings
├── results/                      # Training outputs
└── figures/                      # Publication figures
```

## Cost Estimate

| Step | Samples | Cost |
|------|---------|------|
| Test run | 100 | $0.60 |
| Full training | 5,000 | $30 |

## Expected Output

After running the full pipeline:
- `results/atac_predictor.pt` - Trained model
- `results/training_results.csv` - Metrics
- `figures/baseline_comparison.png` - Publication figure

## Success Criteria

| Result | Interpretation | Action |
|--------|----------------|--------|
| r ≥ 0.62 | Matches Evo2 paper | → Nature Methods |
| r ≥ 0.50 | Good performance | → Tune hyperparameters |
| r < 0.50 | Below target | → Check data quality |
