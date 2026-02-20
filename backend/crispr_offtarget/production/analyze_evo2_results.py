"""
Analyze Evo2 Modal scoring results
"""
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Load results
df = pd.read_csv('results/modal_evo2/kleinstiver_evo2_scored.csv')

print("=" * 60)
print("EVO2 MODAL SCORING RESULTS")
print("=" * 60)
print(f"Total samples: {len(df)}")

# Get scores and labels
scores = df['weighted_delta_ll'].values
labels = df['is_validated'].astype(int).values

print(f"Positives: {labels.sum()}")
print(f"Negatives: {len(labels) - labels.sum()}")

# Calculate AUROC - higher score = more likely positive
auroc = roc_auc_score(labels, scores)
auprc = average_precision_score(labels, scores)

print()
print("=" * 60)
print("RESULTS")
print("=" * 60)
print(f"AUROC: {auroc:.4f}")
print(f"AUPRC: {auprc:.4f}")
print()

# Score stats by class
pos_df = df[df['is_validated'] == True]
neg_df = df[df['is_validated'] == False]
print(f"Positives - mean: {pos_df['weighted_delta_ll'].mean():.4f}")
print(f"Negatives - mean: {neg_df['weighted_delta_ll'].mean():.4f}")

# Generate ROC curve
fpr, tpr, _ = roc_curve(labels, scores)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, label=f'Evo2 Modal (AUROC = {auroc:.3f})', linewidth=2.5, color='#e63946')
ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1, label='Random')
ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
ax.set_title('CRISPR Off-Target Prediction - Evo2 (Modal)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='lower right')
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.grid(True, alpha=0.3)
plt.tight_layout()

fig_dir = Path('results/modal_evo2/figures')
fig_dir.mkdir(exist_ok=True)
plt.savefig(fig_dir / 'roc_evo2_modal.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"\nSaved: {fig_dir / 'roc_evo2_modal.png'}")

# Save metrics
metrics = {
    'auroc': float(auroc),
    'auprc': float(auprc),
    'scorer': 'Evo2_7B_Modal',
    'n_samples': len(df),
    'n_positives': int(labels.sum())
}
with open('results/modal_evo2/evo2_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"Saved: results/modal_evo2/evo2_metrics.json")
