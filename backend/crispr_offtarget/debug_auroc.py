"""Debug score direction for AUROC"""
import pandas as pd
from sklearn.metrics import roc_auc_score

df = pd.read_csv('results/scored_sample.csv')

print('Score vs Validation Analysis:')
print('='*50)

validated = df[df['is_validated'] == True]['weighted_delta_ll']
not_validated = df[df['is_validated'] == False]['weighted_delta_ll']

print(f'Validated (True) mean: {validated.mean():.4f}')
print(f'Not validated (False) mean: {not_validated.mean():.4f}')

# Try both directions
auroc_neg = roc_auc_score(df['is_validated'], -df['weighted_delta_ll'])
auroc_pos = roc_auc_score(df['is_validated'], df['weighted_delta_ll'])

print(f'\nAUROC with -score: {auroc_neg:.4f}')
print(f'AUROC with +score: {auroc_pos:.4f}')

if auroc_pos > auroc_neg:
    print('\n✅ Use POSITIVE score direction')
    best_auroc = auroc_pos
else:
    print('\n✅ Use NEGATIVE score direction')
    best_auroc = auroc_neg

print(f'\nBest AUROC: {best_auroc:.4f}')
print(f'Target: 0.70')
print(f'Gap: {0.70 - best_auroc:.4f}')

# The issue: with 83% validation rate, the data is heavily pre-filtered
# In raw GUIDE-seq, validation rate is 5-15%
print('\n' + '='*50)
print('DATA QUALITY NOTE:')
print(f'  Validation rate: {df["is_validated"].mean()*100:.1f}%')
print('  Expected for raw GUIDE-seq: 5-15%')
print('  This sample appears PRE-FILTERED for high-confidence sites.')
print('  Mock scoring may not discriminate well on pre-filtered data.')
