# ClinVar VUS Classification with Evo2

Classifies ClinVar **Variants of Uncertain Significance (VUS)** as likely **Pathogenic** or likely **Benign** using the Evo2 genomic language model delta-score approach.

## How it Works

For each VUS variant:
1. Extract an **8kb genomic window** from the reference genome (UCSC API)
2. Create the **alternate sequence** (swap ref → alt at the variant position)
3. Score both sequences with **Evo2**: `delta = score(alt) - score(ref)`
4. A large **negative delta** → Likely Pathogenic (model is "surprised" by the mutation)
5. A near-zero delta → Likely Benign

This mirrors the approach validated on BRCA1 in `backend/main.py` (AUROC ~0.9).

## Pipeline Steps

| Script | What it does | Output |
|--------|-------------|--------|
| `01_download_vus.py` | Download ClinVar & filter for VUS | `data/clinvar_vus_raw.tsv` |
| `02_prepare_sequences.py` | Fetch 8kb windows via UCSC API | `data/sequences.json` |
| `03_modal_score.py` | Score ref/alt pairs with Evo2 on Modal GPU | `data/delta_scores.json` |
| `04_classify_and_plot.py` | Apply threshold & generate figures | `results/*.png` |

## Run in Order

```bash
python 01_download_vus.py
python 02_prepare_sequences.py
modal run 03_modal_score.py
python 04_classify_and_plot.py
```

## Threshold

Calibrated from BRCA1 analysis in `main.py`:
- `delta < -0.0009178519` → **Likely Pathogenic**
- `delta >= -0.0009178519` → **Likely Benign**

## Output

- `results/vus_classification_summary.png` — bar chart of Pathogenic vs Benign counts
- `results/delta_score_distribution.png` — histogram of raw scores
- `results/gene_breakdown.png` — per-gene classification breakdown
- `data/vus_classified.csv` — full results table
