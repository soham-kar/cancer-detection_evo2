"""
G2M Fold-Change Analysis
Validates if G2M pathway signal is real biology or noise
"""

import numpy as np
import pandas as pd

# Load aligned data
data = np.load('aligned_data.npz', allow_pickle=True)
X = data['X']  # (patients, genes)
y_time = data['y_time']
y_event = data['y_event']
gene_names = list(data['gene_names'])
pathway_mask = data['pathway_mask']
pathway_names = list(data['pathway_names'])

print("="*60)
print("G2M FOLD-CHANGE ANALYSIS")
print("="*60)
print(f"Patients: {X.shape[0]}")
print(f"Genes: {X.shape[1]}")
print(f"Events: {y_event.sum():.0f} ({y_event.mean()*100:.1f}%)")

# Core G2M genes to check
g2m_core_genes = ['CDK1', 'CCNB1', 'CCNB2', 'CDC20', 'TOP2A', 'AURKA', 'PLK1', 'BUB1']

# Find which of these are in our gene set
available_g2m = [g for g in g2m_core_genes if g in gene_names]
print(f"\nG2M genes available: {available_g2m}")

# Get indices
g2m_indices = [gene_names.index(g) for g in available_g2m]

# Define high-risk (short survival) vs low-risk (long survival)
# Use median survival time as cutoff
median_survival = np.median(y_time[y_event == 1])  # Median among those who died
print(f"Median survival (events only): {median_survival:.0f} days")

# High-risk: short survival (<median) and event occurred
# Low-risk: long survival (>median) OR censored with long followup
high_risk_mask = (y_time < median_survival) & (y_event == 1)
low_risk_mask = (y_time > median_survival * 1.5) | ((y_event == 0) & (y_time > median_survival))

print(f"\nHigh-risk patients: {high_risk_mask.sum()}")
print(f"Low-risk patients: {low_risk_mask.sum()}")

# Expression comparison
g2m_expr_high = X[high_risk_mask][:, g2m_indices].mean(axis=0)
g2m_expr_low = X[low_risk_mask][:, g2m_indices].mean(axis=0)

# Fold change (log2)
# Add small epsilon to avoid log(0)
epsilon = 1e-6
fold_change = np.log2((g2m_expr_high + epsilon) / (g2m_expr_low + epsilon))

print("\n" + "-"*60)
print("G2M GENE FOLD-CHANGES (High-Risk vs Low-Risk)")
print("-"*60)
for gene, fc in zip(available_g2m, fold_change):
    direction = "↑ HIGH" if fc > 0 else "↓ LOW"
    print(f"  {gene:10s}: {fc:+.2f} log2 {direction}")

mean_fc = fold_change.mean()
print(f"\n  MEAN:       {mean_fc:+.2f} log2")

print("\n" + "="*60)
print("INTERPRETATION")
print("="*60)
if mean_fc > 1.5:
    print("✅ STRONG SIGNAL: G2M expression is REAL biology")
    print("   High-risk patients have significantly higher G2M gene expression")
    print("   The model is learning a true biological signal")
elif mean_fc > 0.5:
    print("✅ MODERATE SIGNAL: G2M expression is biologically relevant")
    print("   Some signal, but pathway concentration may be due to regularization")
else:
    print("⚠️ WEAK SIGNAL: G2M pathway may be overfitting")
    print("   Low fold-change suggests model is fitting noise, not biology")

# Also check correlation with survival
from scipy.stats import pearsonr, spearmanr

g2m_mean = X[:, g2m_indices].mean(axis=1)
corr_spearman, p_value = spearmanr(g2m_mean, -y_time)  # Negative time = higher risk

print(f"\n📊 G2M Expression vs Survival Correlation:")
print(f"   Spearman r = {corr_spearman:.3f} (p = {p_value:.2e})")

if p_value < 0.05:
    print("   ✅ Statistically significant correlation")
else:
    print("   ⚠️ Not statistically significant")

# Check which pathways have strongest correlation with survival
print("\n" + "="*60)
print("ALL PATHWAY CORRELATIONS WITH SURVIVAL")
print("="*60)

pathway_correlations = []
for p_idx, p_name in enumerate(pathway_names):
    genes_in_p = pathway_mask[:, p_idx] > 0
    if genes_in_p.sum() > 0:
        pathway_expr = X[:, genes_in_p].mean(axis=1)
        corr, p_val = spearmanr(pathway_expr, -y_time)
        pathway_correlations.append({
            'pathway': p_name,
            'correlation': corr,
            'p_value': p_val,
            'n_genes': genes_in_p.sum()
        })

df_corr = pd.DataFrame(pathway_correlations).sort_values('correlation', ascending=False)

print("\nTop 10 Pathways by Survival Correlation:")
for _, row in df_corr.head(10).iterrows():
    sig = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
    print(f"  {row['pathway']:45s}: r={row['correlation']:+.3f} {sig}")

print("\nBottom 5 (Protective?):")
for _, row in df_corr.tail(5).iterrows():
    sig = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
    print(f"  {row['pathway']:45s}: r={row['correlation']:+.3f} {sig}")
