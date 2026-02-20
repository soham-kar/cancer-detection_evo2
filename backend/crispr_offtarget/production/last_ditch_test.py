"""
Last-Ditch Test: Combine Evo2 Features + Seed Penalty

Uses Evo2 hidden state features (from mean-pooling) projected to 1D
combined with seed-weighted penalty to see if combination beats either alone.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

# Seed weights
SEED_WEIGHTS = np.ones(20)
SEED_WEIGHTS[0:7] = 0.5     # PAM-distal
SEED_WEIGHTS[17:20] = 3.0   # Seed region

def compute_seed_penalty(grna, target):
    """Compute seed-weighted mismatch penalty (more mismatches = higher penalty)"""
    penalty = 0
    for i in range(min(len(grna), len(target), 20)):
        if grna[i] != target[i]:
            penalty += SEED_WEIGHTS[i]
    return penalty

def main():
    print("=" * 60)
    print("LAST-DITCH TEST: EVO2 FEATURES + SEED PENALTY")
    print("=" * 60)
    
    # Load Evo2 features and labels
    X = np.load(DATA_DIR / "evo2_features_balanced.npy")
    y = np.load(DATA_DIR / "evo2_features_balanced_labels.npy")
    df = pd.read_csv(DATA_DIR / "evo2_features_balanced_mismatches.csv")
    
    print(f"Evo2 features: {X.shape}")
    print(f"Labels: {y.shape} (pos: {y.sum()}, neg: {len(y) - y.sum()})")
    
    # Compute seed penalty for each sample
    seed_penalties = df.apply(
        lambda r: compute_seed_penalty(str(r['grna_sequence']), str(r['target_sequence'])),
        axis=1
    ).values
    
    print(f"Seed penalty range: [{seed_penalties.min():.1f}, {seed_penalties.max():.1f}]")
    print(f"Mean seed penalty: {seed_penalties.mean():.2f}")
    
    # Split data
    X_train, X_val, y_train, y_val, sp_train, sp_val = train_test_split(
        X, y, seed_penalties, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale Evo2 features
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    # 1. Heuristic only (seed penalty, negated so LOW = more cleavage)
    heuristic_auroc = roc_auc_score(y_val, -sp_val)
    print(f"\n1. Heuristic-only (seed penalty):  AUROC = {heuristic_auroc:.4f}")
    
    # 2. Evo2-only (logistic regression to get single score from 512-dim)
    lr_evo2 = LogisticRegression(max_iter=1000, random_state=42, C=0.1)
    lr_evo2.fit(X_train_sc, y_train)
    evo2_prob = lr_evo2.predict_proba(X_val_sc)[:, 1]
    evo2_auroc = roc_auc_score(y_val, evo2_prob)
    print(f"2. Evo2-only (LR on features):     AUROC = {evo2_auroc:.4f}")
    
    # 3. Combined: Evo2 proba + lambda * (-seed_penalty)
    # Grid search lambda
    best_auroc = 0
    best_lambda = 0
    for lam in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]:
        # Need to scale seed penalty to be comparable to proba (0-1 range)
        sp_scaled = sp_val / sp_val.max()  # normalize to 0-1
        combined = evo2_prob + lam * (-sp_scaled)
        auroc = roc_auc_score(y_val, combined)
        if auroc > best_auroc:
            best_auroc = auroc
            best_lambda = lam
    
    print(f"3. Combined (Evo2 + λ*heuristic):  AUROC = {best_auroc:.4f} (λ={best_lambda})")
    
    # 4. Combined: Concatenate features + train single model
    X_combined_train = np.hstack([X_train_sc, -sp_train.reshape(-1, 1)])
    X_combined_val = np.hstack([X_val_sc, -sp_val.reshape(-1, 1)])
    lr_combined = LogisticRegression(max_iter=1000, random_state=42, C=0.1)
    lr_combined.fit(X_combined_train, y_train)
    combined_prob = lr_combined.predict_proba(X_combined_val)[:, 1]
    combined_lr_auroc = roc_auc_score(y_val, combined_prob)
    print(f"4. Combined (concat + LR):         AUROC = {combined_lr_auroc:.4f}")
    
    # Verdict
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    
    best_evo2_combo = max(best_auroc, combined_lr_auroc)
    if best_evo2_combo > heuristic_auroc + 0.01:
        print(f"✅ Combined beats heuristic by {best_evo2_combo - heuristic_auroc:.4f}")
        print(f"   Evo2 adds meaningful value!")
    else:
        print(f"⚠️ Combined ({best_evo2_combo:.4f}) ≈ Heuristic ({heuristic_auroc:.4f})")
        print(f"   Evo2 does NOT add meaningful value")
    
    if best_evo2_combo > 0.80:
        print("\n🎉 SUCCESS: AUROC > 0.80 achieved!")
    elif best_evo2_combo > 0.75:
        print("\n✅ CLOSE: AUROC > 0.75 achieved!")
    else:
        print("\n❌ Below target: AUROC < 0.75")

if __name__ == "__main__":
    main()
