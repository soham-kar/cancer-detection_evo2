"""
Option A: Linear Probe on Evo2 Features

Train a simple logistic regression on Evo2 hidden state features
to predict CRISPR off-target cleavage.

Success criterion: AUROC > 0.75
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import matplotlib.pyplot as plt

# Reproducibility
np.random.seed(42)

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

def main():
    print("=" * 60)
    print("OPTION A: LINEAR PROBE ON EVO2 FEATURES")
    print("=" * 60)
    
    # Load features and labels
    features_path = DATA_DIR / "evo2_features_balanced.npy"
    labels_path = DATA_DIR / "evo2_features_balanced_labels.npy"
    
    print(f"\nLoading features from {features_path}")
    X = np.load(features_path)
    y = np.load(labels_path)
    
    print(f"Features shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Class distribution: Positive={y.sum():.0f}, Negative={(1-y).sum():.0f}")
    
    # ========== DIAGNOSTIC CHECKS ==========
    print("\n" + "-" * 40)
    print("DIAGNOSTIC: Feature Statistics (Pre-Scaling)")
    print("-" * 40)
    print(f"  Mean: {X.mean():.6f} (should be ~0 after scaling)")
    print(f"  Std:  {X.std():.6f}  (should be ~1 after scaling)")
    print(f"  Min:  {X.min():.6f}")
    print(f"  Max:  {X.max():.6f}")
    print(f"  NaN count: {np.isnan(X).sum()}")
    print(f"  Inf count: {np.isinf(X).sum()}")
    
    # Sanity check: correlation with labels
    from scipy.stats import pearsonr
    corr = pearsonr(X.mean(axis=1), y)[0]
    print(f"  Mean feature correlation with label: {corr:.4f} (should be non-zero)")
    
    if np.abs(corr) < 0.01:
        print("  ⚠️ WARNING: Very low correlation. Features may be wrong or uninformative!")
    
    # ========== FEATURE SCALING (CRITICAL) ==========
    print("\n" + "-" * 40)
    print("Applying StandardScaler (CRITICAL)")
    print("-" * 40)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save scaler for inference
    np.save(DATA_DIR / "feature_scaler_mean.npy", scaler.mean_)
    np.save(DATA_DIR / "feature_scaler_scale.npy", scaler.scale_)
    print(f"  Scaler saved to {DATA_DIR}")
    
    print(f"  Post-scaling mean: {X_scaled.mean():.6f}")
    print(f"  Post-scaling std:  {X_scaled.std():.6f}")
    
    # ========== TRAIN LINEAR PROBE ==========
    print("\n" + "-" * 40)
    print("Training Logistic Regression (5-fold CV)")
    print("-" * 40)
    
    clf = LogisticRegression(
        class_weight='balanced',
        max_iter=2000,  # Increased for convergence
        C=1.0,  # L2 regularization
        solver='lbfgs',
        random_state=42,
        verbose=0
    )
    
    # 5-fold stratified cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # AUROC scores
    auroc_scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='roc_auc')
    print(f"\n✅ AUROC per fold: {auroc_scores}")
    print(f"✅ Mean AUROC: {auroc_scores.mean():.4f} ± {auroc_scores.std():.4f}")
    
    # ========== FEATURE IMPORTANCE ANALYSIS ==========
    print("\n" + "-" * 40)
    print("Training on full dataset for analysis")
    print("-" * 40)
    
    clf.fit(X_scaled, y)
    
    # Check convergence
    if hasattr(clf, 'n_iter_'):
        print(f"  Convergence iterations: {clf.n_iter_}")
    
    # Compute AUPRC on full data (training set, for reference)
    y_pred_proba = clf.predict_proba(X_scaled)[:, 1]
    precision, recall, _ = precision_recall_curve(y, y_pred_proba)
    auprc = auc(recall, precision)
    print(f"  AUPRC (full data): {auprc:.4f}")
    
    # Feature importance
    feature_importance = np.abs(clf.coef_[0])
    top_k = 20
    top_indices = np.argsort(feature_importance)[-top_k:][::-1]
    
    print(f"\n  Top {top_k} important feature indices:")
    for i, idx in enumerate(top_indices[:10]):
        print(f"    {i+1}. Feature {idx}: weight={clf.coef_[0][idx]:.4f}")
    
    # ========== SAVE RESULTS ==========
    results = {
        'mean_auroc': auroc_scores.mean(),
        'std_auroc': auroc_scores.std(),
        'auprc_train': auprc,
        'fold_aurocs': auroc_scores.tolist(),
        'label_correlation': corr,
        'success': auroc_scores.mean() > 0.75
    }
    
    results_path = RESULTS_DIR / "linear_probe_results.csv"
    pd.DataFrame([results]).to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")
    
    # ========== FINAL VERDICT ==========
    print("\n" + "=" * 60)
    if results['success']:
        print("🎉 SUCCESS: AUROC > 0.75 achieved!")
        print("   Evo2 features contain CRISPR off-target signal!")
    else:
        print("⚠️ AUROC < 0.75: Evo2 features alone may be insufficient")
        print("   Proceed with Option B (Residual Learning)")
    print("=" * 60)
    
    return results

if __name__ == "__main__":
    main()
