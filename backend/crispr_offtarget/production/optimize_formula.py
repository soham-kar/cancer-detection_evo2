"""
Optimize risk scoring formula to maximize AUROC.
Try different weighting schemes for mismatches and seed region.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score

RESULTS_DIR = Path(__file__).parent.parent / "results"
SCORED_FILE = RESULTS_DIR / "scored_modal_full.csv"

print("="*60)
print("OPTIMIZING RISK SCORING FORMULA")
print("="*60)

df = pd.read_csv(SCORED_FILE)
print(f"Loaded {len(df)} sites")

# Original AUROC
original_auroc = roc_auc_score(df['is_validated'], df['weighted_delta_ll'])
print(f"\nOriginal AUROC: {original_auroc:.4f}")

# Try different formulas
best_auroc = original_auroc
best_formula = "original"

# Formula 1: Exponential mismatch decay
df['score_exp'] = (
    df['weighted_delta_ll'] * 
    np.exp(-df['mismatches'] * 0.3)
)
auroc1 = roc_auc_score(df['is_validated'], df['score_exp'])
print(f"Exp decay (0.3): {auroc1:.4f} ({auroc1-original_auroc:+.4f})")
if auroc1 > best_auroc:
    best_auroc = auroc1
    best_formula = "exp_decay_0.3"

# Formula 2: Stronger exponential
df['score_exp2'] = (
    df['weighted_delta_ll'] * 
    np.exp(-df['mismatches'] * 0.5)
)
auroc2 = roc_auc_score(df['is_validated'], df['score_exp2'])
print(f"Exp decay (0.5): {auroc2:.4f} ({auroc2-original_auroc:+.4f})")
if auroc2 > best_auroc:
    best_auroc = auroc2
    best_formula = "exp_decay_0.5"

# Formula 3: Seed-focused
df['score_seed'] = (
    df['weighted_delta_ll'] * 
    (1 + df['seed_penalty'] * 2.0)  # Double seed weight
)
auroc3 = roc_auc_score(df['is_validated'], df['score_seed'])
print(f"2x seed weight: {auroc3:.4f} ({auroc3-original_auroc:+.4f})")
if auroc3 > best_auroc:
    best_auroc = auroc3
    best_formula = "seed_2x"

# Formula 4: Combined exp + seed
df['score_combined'] = (
    df['weighted_delta_ll'] * 
    np.exp(-df['mismatches'] * 0.2) *
    (1 + df['seed_penalty'])
)
auroc4 = roc_auc_score(df['is_validated'], df['score_combined'])
print(f"Combined:       {auroc4:.4f} ({auroc4-original_auroc:+.4f})")
if auroc4 > best_auroc:
    best_auroc = auroc4
    best_formula = "combined"

# Formula 5: Log transform
df['score_log'] = np.log1p(np.abs(df['weighted_delta_ll'])) * np.sign(df['weighted_delta_ll'])
auroc5 = roc_auc_score(df['is_validated'], df['score_log'])
print(f"Log transform:  {auroc5:.4f} ({auroc5-original_auroc:+.4f})")
if auroc5 > best_auroc:
    best_auroc = auroc5
    best_formula = "log"

# Formula 6: Quadratic
df['score_quad'] = df['weighted_delta_ll'] ** 2 * np.sign(df['weighted_delta_ll'])
auroc6 = roc_auc_score(df['is_validated'], df['score_quad'])
print(f"Quadratic:      {auroc6:.4f} ({auroc6-original_auroc:+.4f})")
if auroc6 > best_auroc:
    best_auroc = auroc6
    best_formula = "quadratic"

# Formula 7: Mismatch <= 2 focus
df['score_low_mm'] = df['weighted_delta_ll'] * (df['mismatches'] <= 2).astype(float) * 2 + df['weighted_delta_ll']
auroc7 = roc_auc_score(df['is_validated'], df['score_low_mm'])
print(f"Low MM focus:   {auroc7:.4f} ({auroc7-original_auroc:+.4f})")
if auroc7 > best_auroc:
    best_auroc = auroc7
    best_formula = "low_mm"

# Formula 8: Rank-based
df['score_rank'] = df['weighted_delta_ll'].rank() / len(df)
auroc8 = roc_auc_score(df['is_validated'], df['score_rank'])
print(f"Rank-based:     {auroc8:.4f} ({auroc8-original_auroc:+.4f})")
if auroc8 > best_auroc:
    best_auroc = auroc8
    best_formula = "rank"

print(f"\n{'='*60}")
print(f"BEST FORMULA: {best_formula}")
print(f"BEST AUROC: {best_auroc:.4f}")
print(f"Improvement: {best_auroc - original_auroc:+.4f}")
print(f"Target: 0.70, Gap: {0.70 - best_auroc:.4f}")
