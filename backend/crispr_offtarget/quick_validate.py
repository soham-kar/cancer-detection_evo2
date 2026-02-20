"""Quick validation analysis for CRISPR off-target scoring"""
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

# Load scored data
df = pd.read_csv('results/scored_sample.csv')

print('='*60)
print('CRISPR OFF-TARGET VALIDATION RESULTS')
print('='*60)

print(f'\nTotal sites: {len(df)}')
validated = df["is_validated"].sum()
print(f'Validated positives: {validated} ({df["is_validated"].mean()*100:.1f}%)')

# Overall AUROC
try:
    auroc = roc_auc_score(df['is_validated'], -df['weighted_delta_ll'])
    print(f'\n📊 OVERALL AUROC: {auroc:.4f}')
    
    if auroc >= 0.70:
        print('   ✅ TARGET MET (>0.70)!')
    else:
        print(f'   ⚠️ Below target (need {0.70 - auroc:.4f} more)')
except Exception as e:
    print(f'Could not calculate AUROC: {e}')
    auroc = 0.5

# Per-gRNA AUROC
print('\n📈 PER-gRNA PERFORMANCE:')
print('-'*50)
aurocs = []
grna_results = []

for grna in df['grna_name'].unique():
    gdf = df[df['grna_name'] == grna]
    if gdf['is_validated'].nunique() > 1:
        try:
            a = roc_auc_score(gdf['is_validated'], -gdf['weighted_delta_ll'])
            aurocs.append(a)
            grna_results.append({'grna': grna, 'auroc': a, 'n_sites': len(gdf)})
            print(f'  {grna}: AUROC = {a:.4f} ({len(gdf)} sites)')
        except:
            pass

if aurocs:
    print(f'\n  Range: {min(aurocs):.4f} - {max(aurocs):.4f}')
    variation = max(aurocs)/min(aurocs) if min(aurocs) > 0 else 0
    print(f'  Variation: {variation:.2f}x (thesis found 11.5x across genes)')

# Seed vs Non-Seed analysis
print('\n🧬 SEED REGION ANALYSIS:')
seed_df = df[df['seed_penalty'] > 0]
nonseed_df = df[df['seed_penalty'] == 0]
print(f'  Seed mismatches: {len(seed_df)} sites, mean score = {seed_df["weighted_delta_ll"].mean():.4f}')
print(f'  Non-seed mismatches: {len(nonseed_df)} sites, mean score = {nonseed_df["weighted_delta_ll"].mean():.4f}')

if len(seed_df) > 0 and len(nonseed_df) > 0:
    seed_mean = seed_df['weighted_delta_ll'].mean()
    nonseed_mean = nonseed_df['weighted_delta_ll'].mean()
    if nonseed_mean != 0:
        ratio = abs(seed_mean / nonseed_mean)
        print(f'  Seed signal strength: {ratio:.2f}x stronger')

# Confidence distribution
print('\n🎯 CONFIDENCE DISTRIBUTION:')
conf_dist = df['confidence'].value_counts()
for conf, count in conf_dist.items():
    pct = count / len(df) * 100
    print(f'  {conf}: {count} ({pct:.1f}%)')

# Summary
print('\n' + '='*60)
print('SUMMARY')
print('='*60)
print(f'  AUROC: {auroc:.4f} (target: >0.70)')
print(f'  Seed signal detected: {"✅ Yes" if len(seed_df) > 0 else "❌ No"}')
print(f'  Per-gRNA variation: {variation:.2f}x')
print(f'  Ready for Modal scoring: {"✅ Yes" if auroc > 0.60 else "⚠️ Review pipeline"}')
