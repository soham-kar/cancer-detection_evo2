# Oral Cancer Interpretable Genomics

**Pathway-Attention Network for Interpretable Oral Cancer Risk Prediction**

## Overview

This module implements an intrinsically interpretable deep learning architecture for oral cancer genomics. Unlike traditional "black-box + SHAP" approaches, biological pathway structure is **baked into the model architecture**.

## Key Innovation

| Traditional Approach | Our Approach |
|---------------------|--------------|
| Black-box model → SHAP explanations | Pathway structure constrains attention |
| Post-hoc, potentially misleading | Intrinsic, biologically grounded |
| "Gene X has importance 0.85" | "TP53 pathway has attention 0.85" |

## Architecture

```
Gene Expression (20,000 genes)
        ↓
Pathway Grouping (50 hallmark pathways)
        ↓
Intra-Pathway Attention (genes within same pathway)
        ↓
Inter-Pathway Attention (pathway-to-pathway)
        ↓
Risk Score + Interpretable Attention Weights
```

## Data Source

- **TCGA HNSC** (Head-Neck Squamous Cell Carcinoma)
- Filtered for **oral cavity subsites** (C03-C06): buccal, gingiva, floor of mouth, palate
- ~300 patients with RNA-seq + clinical survival data

## Integration with biotech-evo2

This module connects to the existing Evo2 infrastructure:
- **Evo2 scoring**: Non-coding variants in pathway regulatory regions
- **Modal deployment**: Shared GPU infrastructure
- **Cost optimization**: Reuses batch processing patterns

## Quick Start

```bash
# 1. Download TCGA data
python -m oral_cancer_interpretable.src.data_engineering.tcga_loader

# 2. Create pathway masks
python -m oral_cancer_interpretable.src.data_engineering.pathway_mask

# 3. Train model
python -m oral_cancer_interpretable.src.models.pathway_attention --train

# 4. Analyze attention
python -m oral_cancer_interpretable.src.interpretability.attention_viz
```

## Folder Structure

```
oral_cancer_interpretable/
├── data/
│   ├── tcga_hnsc/           # Raw TCGA downloads
│   ├── processed/           # Gene expression matrices
│   └── pathways/            # MSigDB, KEGG gene sets
├── src/
│   ├── models/
│   │   ├── pathway_attention.py    # Core architecture
│   │   └── evo2_integration.py     # Bridge to Evo2
│   ├── data_engineering/
│   │   ├── tcga_loader.py
│   │   └── pathway_mask.py
│   └── interpretability/
│       ├── attention_viz.py
│       └── pathway_report.py
├── notebooks/
│   ├── 01_tcga_setup.ipynb
│   ├── 02_pathway_attention.ipynb
│   └── 03_evo2_bridge.ipynb
└── configs/
    └── oral_cancer.yaml
```

## References

- Biswas et al. "Distinct genomic pathology..." Nature Communications (2023)
- MSigDB Hallmark Gene Sets
- TCGA Head-Neck Squamous Cell Carcinoma (HNSC)
