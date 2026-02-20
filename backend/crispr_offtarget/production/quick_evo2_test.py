"""Quick test: Evo2 + seed weight on balanced CIRCLE-seq data"""
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

df = pd.read_csv('results/modal_evo2/circle_seq_evo2_scored.csv')
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

y = df['is_validated'].values
evo2 = df['evo2_risk_score'].values

# Check mismatch distribution
print(f"\nMismatch distribution: {df['mismatches'].value_counts().to_dict()}")
print(f"Seed penalty distribution: {df['seed_penalty'].value_counts().to_dict()}")

# AUROC for Evo2
evo2_auroc = roc_auc_score(y, evo2)
print(f"\nEvo2-only AUROC: {evo2_auroc:.4f}")

# Check if we can compute seed penalty from sequences
# We need grna_sequence which may not be in this file
# The file only has target_sequence, not grna

# For now, use mismatches as a proxy (if all 0, can't help)
if df['mismatches'].nunique() > 1:
    mm_auroc = roc_auc_score(y, -df['mismatches'])  # fewer mismatches = more cutting
    print(f"Mismatch-only AUROC: {mm_auroc:.4f}")
else:
    print("Mismatch column has no variance - can't use as feature")

# Try weighted_delta_ll which may have seed weighting built in
if 'weighted_delta_ll' in df.columns:
    wdll_auroc = roc_auc_score(y, df['weighted_delta_ll'])
    print(f"Weighted delta LL AUROC: {wdll_auroc:.4f}")
