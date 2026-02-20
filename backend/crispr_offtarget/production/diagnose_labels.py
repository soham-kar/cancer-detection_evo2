"""Diagnostic script to verify label direction"""
import pandas as pd
import numpy as np

df = pd.read_csv('data/evo2_features_balanced_mismatches.csv')
print('=== LABEL DIAGNOSTICS ===')
print(f"Label distribution: {df['is_validated'].value_counts().to_dict()}")

# Mismatch count by label
print('\nMismatch count by label:')
print(df.groupby('is_validated')['mismatch_count'].describe())

# Correlation
corr = df['mismatch_count'].corr(df['is_validated'])
print(f'\nCorrelation(mismatch_count, is_validated): {corr:.4f}')

# What this means:
# If POSITIVE: Higher mismatch → validated (label 1)   ← INVERTED (wrong)
# If NEGATIVE: Higher mismatch → not validated (label 0) ← CORRECT
print('\nInterpretation:')
if corr > 0:
    print('  ⚠️ INVERTED: Higher mismatch predicts cleavage (label 1)')
    print('  This is biologically backwards!')
else:
    print('  ✅ CORRECT: Higher mismatch predicts no cleavage (label 0)')

# Test inverting the heuristic score
from sklearn.metrics import roc_auc_score

SEED_WEIGHTS = np.ones(20)
SEED_WEIGHTS[0:7] = 0.5
SEED_WEIGHTS[17:20] = 3.0

# Compute mismatch positions
def get_mismatch_score(row):
    grna = str(row['grna_sequence'])
    target = str(row['target_sequence'])
    score = 0
    for i in range(min(len(grna), len(target), 20)):
        if grna[i] != target[i]:
            score += SEED_WEIGHTS[i]
    return score

df['seed_score'] = df.apply(get_mismatch_score, axis=1)
y = df['is_validated'].values

# Original direction
auroc_original = roc_auc_score(y, df['seed_score'])
# Inverted direction
auroc_inverted = roc_auc_score(y, -df['seed_score'])

print(f'\n=== HEURISTIC AUROC ===')
print(f'Original (higher score → more cleavage): {auroc_original:.4f}')
print(f'Inverted (lower score → more cleavage):  {auroc_inverted:.4f}')
