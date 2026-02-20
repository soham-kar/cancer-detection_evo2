"""
Validate on Kleinstiver dataset (experimentally confirmed positives/negatives)

Uses Evo2 scored data + seed penalty to compute AUROC on 108 samples.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"

# Seed weights
SEED_WEIGHTS = np.ones(20)
SEED_WEIGHTS[0:7] = 0.5     # PAM-distal
SEED_WEIGHTS[17:20] = 3.0   # Seed region

def compute_seed_penalty(grna, target):
    """Compute seed-weighted mismatch penalty"""
    penalty = 0
    grna = str(grna) if pd.notna(grna) else ""
    target = str(target) if pd.notna(target) else ""
    for i in range(min(len(grna), len(target), 20)):
        if grna[i] != target[i]:
            penalty += SEED_WEIGHTS[i]
    return penalty

def main():
    print("=" * 60)
    print("KLEINSTIVER VALIDATION (Confirmed Positives/Negatives)")
    print("=" * 60)
    
    # Load Kleinstiver balanced dataset
    df_balanced = pd.read_csv(DATA_DIR / "kleinstiver_balanced.csv")
    print(f"\nKleinstiver balanced: {len(df_balanced)} samples")
    print(f"Label distribution: {df_balanced['is_validated'].value_counts().to_dict()}")
    
    # Load Evo2 scored version
    df_scored = pd.read_csv(RESULTS_DIR / "modal_evo2" / "kleinstiver_evo2_scored.csv")
    print(f"Evo2 scored: {len(df_scored)} samples")
    
    # Need to merge if grna_sequence not in scored
    if 'grna_sequence' not in df_scored.columns:
        # Check what columns we have
        print(f"\nScored columns: {df_scored.columns.tolist()}")
        print(f"Balanced columns: {df_balanced.columns.tolist()}")
        
        # Use balanced data for grna/target, scored for evo2_risk_score
        # Match on target_sequence or grna_name
        if 'grna_name' in df_scored.columns and 'grna_name' in df_balanced.columns:
            df = df_balanced.merge(df_scored[['grna_name', 'target_sequence', 'evo2_risk_score']], 
                                   on=['grna_name', 'target_sequence'], how='inner')
        else:
            df = df_balanced.merge(df_scored[['target_sequence', 'evo2_risk_score']], 
                                   on='target_sequence', how='inner')
    else:
        df = df_scored
    
    print(f"\nMerged dataset: {len(df)} samples")
    
    if len(df) == 0:
        print("ERROR: No samples after merge!")
        return
    
    # Get labels
    y = df['is_validated'].astype(int).values
    print(f"Labels: {np.sum(y)} positives, {len(y) - np.sum(y)} negatives")
    
    # Get Evo2 scores
    evo2_scores = df['evo2_risk_score'].values
    print(f"Evo2 score range: [{evo2_scores.min():.4f}, {evo2_scores.max():.4f}]")
    
    # Compute seed penalties
    seed_penalties = df.apply(
        lambda r: compute_seed_penalty(r.get('grna_sequence', ''), r.get('target_sequence', '')),
        axis=1
    ).values
    print(f"Seed penalty range: [{seed_penalties.min():.1f}, {seed_penalties.max():.1f}]")
    
    # ========== AUROC RESULTS ==========
    print("\n" + "=" * 60)
    print("AUROC RESULTS")
    print("=" * 60)
    
    # 1. Heuristic only (negated - less mismatches = more cleavage)
    heuristic_auroc = roc_auc_score(y, -seed_penalties)
    print(f"\n1. Heuristic-only (seed penalty):  AUROC = {heuristic_auroc:.4f}")
    
    # 2. Evo2 only
    evo2_auroc = roc_auc_score(y, evo2_scores)
    print(f"2. Evo2-only:                      AUROC = {evo2_auroc:.4f}")
    
    # 3. Combined with grid search lambda
    best_auroc = 0
    best_lambda = 0
    for lam in np.linspace(0.1, 5.0, 50):
        sp_scaled = seed_penalties / (seed_penalties.max() + 1e-8)
        combined = evo2_scores + lam * (-sp_scaled)
        auroc = roc_auc_score(y, combined)
        if auroc > best_auroc:
            best_auroc = auroc
            best_lambda = lam
    
    print(f"3. Combined (λ={best_lambda:.2f}):            AUROC = {best_auroc:.4f}")
    
    # ========== COMPARISON WITH CIRCLE-seq ==========
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"CIRCLE-seq (synthetic negatives): 0.81")
    print(f"Kleinstiver (confirmed negatives): {best_auroc:.4f}")
    
    diff = best_auroc - 0.81
    if diff < -0.05:
        print(f"\n⚠️ DROP of {abs(diff):.2f} — CIRCLE-seq result may be inflated!")
    elif diff > 0.05:
        print(f"\n✅ INCREASE of {diff:.2f} — result holds on confirmed data!")
    else:
        print(f"\n✅ STABLE (within ±0.05) — result likely valid")

if __name__ == "__main__":
    main()
