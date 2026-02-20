"""Create Figure 3: Bayes Factor Analysis"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load Bayesian results
df = pd.read_csv('results/bayesian_exploration/bayesian_tuned/brca1_bayesian_tuned.csv')
print(f'Loaded {len(df)} variants')
print(f'Mean BF: {df["bayes_factor"].mean():.2f}')

# Create Figure 3
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Panel A: BF Distribution (Histogram)
bf_log = np.log10(df['bayes_factor'].clip(0.01, 100))
ax1.hist(bf_log, bins=50, color='#e74c3c', alpha=0.7, edgecolor='black')
ax1.axvline(0, color='black', linestyle='--', linewidth=2, label='BF=1 (neutral)')
ax1.axvline(np.log10(1.61), color='green', linestyle='--', linewidth=2, label='Mean BF=1.61')
ax1.set_xlabel('log10(Bayes Factor)', fontweight='bold', fontsize=12)
ax1.set_ylabel('Count', fontweight='bold', fontsize=12)
ax1.set_title('A) Bayes Factor Distribution\n(Clustered around neutral)', fontweight='bold', fontsize=14)
ax1.legend(fontsize=10, loc='upper right') # Explicitly set legend location
ax1.grid(True, alpha=0.3, axis='y')

# Annotation - Moved to the top-left to avoid covering the legend
ax1.text(0.05, 0.95, 'Mean BF = 1.61\n(Weak signal)\n\nBF > 10: Strong\nBF ~ 1: Neutral\nBF < 0.1: Strong benign',
         transform=ax1.transAxes, ha='left', va='top',
         bbox=dict(boxstyle='round', facecolor='#f9f9f9', edgecolor='black', linewidth=2),
         fontweight='bold', fontsize=10)

# Panel B: BF vs Evo2 Score
ax2.scatter(df['evo2_score'], df['bayes_factor'], alpha=0.3, s=15, c='#3498db')
ax2.axhline(1, color='black', linestyle='--', linewidth=2, alpha=0.7, label='BF=1 (neutral)')
ax2.axhline(10, color='green', linestyle='--', linewidth=1, alpha=0.5, label='BF=10 (strong)')
ax2.axhline(0.1, color='red', linestyle='--', linewidth=1, alpha=0.5, label='BF=0.1 (benign)')
ax2.set_xlabel('Evo2 Score', fontweight='bold', fontsize=12)
ax2.set_ylabel('Bayes Factor', fontweight='bold', fontsize=12)
ax2.set_title('B) BF vs Evo2 Score\n(Weak correlation)', fontweight='bold', fontsize=14)
ax2.set_yscale('log')
ax2.set_ylim(0.01, 100)
ax2.legend(fontsize=9, loc='upper right')
ax2.grid(True, alpha=0.3)

# Calculate correlation
corr = df['evo2_score'].corr(df['bayes_factor'])
ax2.text(0.05, 0.95, f'r = {corr:.2f}\n(weak correlation)',
         transform=ax2.transAxes, ha='left', va='top',
         bbox=dict(boxstyle='round', facecolor='#fff3cd', edgecolor='black', linewidth=2),
         fontweight='bold', fontsize=11)

plt.tight_layout()
output_path = Path('results/population_thresholds/figure3_bayes_factor_analysis.png')
# Ensure the output directory exists
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f'Saved: {output_path}')
plt.close()