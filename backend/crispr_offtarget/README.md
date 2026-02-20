# CRISPR Off-Target Prediction with Evo2

**Gap 4: CRISPR Off-Target with Evolutionary Context**

## 🎯 Goal
Zero-shot CRISPR off-target prediction using Evo2's evolutionary context, without CRISPR-specific training data.

## 📊 Datasets

### Primary: GUIDE-seq (GSE232228)
- HEK293FT cells
- 6 gRNAs
- 1,115 on/off-targets
- Validation: `read_count > 10` = positive

### Secondary: CIRCLE-seq (GSE206347)
- 10 gRNAs
- 7,371 pairs
- More sensitive detection

### Benchmark: dagrate/public_data_crisprCas9
- Pre-curated datasets
- Direct download

## 🔧 Key Innovation

**Seed Weighting:**
- Positions 10-12 (seed): 3x weight
- Positions 18-20 (PAM-proximal): 2x weight
- Positions 1-7 (5' end): 0.5x weight (tolerated)

**Population-Aware:**
- gnomAD PAM site variation
- Population-specific penalties

## 📁 Structure

```
crispr_offtarget/
├── data/               # Datasets
├── production/         # Core scripts
│   ├── crispr_scorer.py
│   ├── crispr_utils.py
│   ├── analyze_offtarget.py
│   └── population_aware_crispr.py
└── results/
    └── figures/
```

## 🚀 Quick Start

```bash
# Day 1: Download data
python production/download_data.py

# Day 2-3: Score off-targets
python production/crispr_scorer.py --gRNA GAGTCCGAGCAGAAGAAGAA

# Day 5-6: Validate
python production/analyze_offtarget.py
```

## 📈 Expected Performance

| Method | AUROC |
|--------|-------|
| CFD Score | ~0.65 |
| **Evo2 (this work)** | **0.70-0.75** |
| DeepCRISPR | ~0.89 |

## 💰 Cost

- $0.012 per off-target site
- ~$12 for 100K candidates
