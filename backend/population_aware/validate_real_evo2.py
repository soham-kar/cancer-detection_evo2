"""
Real Evo2 Validation Analysis

Analyzes performance of real Evo2-7B scores on BRCA1 variants
Output: results/evo2_results/
"""
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, roc_curve, f1_score, precision_recall_curve,
    confusion_matrix, precision_score, recall_score
)
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("="*70)
    print("REAL EVO2 VALIDATION ANALYSIS")
    print("="*70)
    
    # Load data
    print("\n📂 Loading Evo2 scores...")
    df = pd.read_csv("results/evo2_results/brca1_evo2_scores.csv")
    print(f"   Loaded {len(df)} variants")
    
    # Class distribution
    print("\n📊 Class distribution:")
    print(df['func_class'].value_counts())
    
    # Binary classification (FUNC vs LOF only)
    df['label'] = df['func_class'].map({'LOF': 1, 'FUNC': 0})
    df_binary = df[df['label'].notna()].copy()
    
    print(f"\n🔬 Binary classification dataset: {len(df_binary)} variants")
    print(f"  LOF (pathogenic): {(df_binary['label']==1).sum()}")
    print(f"  FUNC (benign): {(df_binary['label']==0).sum()}")
    
    # Correlation
    corr = df_binary['evo2_score'].corr(df_binary['func_score'])
    print(f"\n📈 Correlation (Evo2 vs Findlay func_score): {corr:.3f}")
    
    # Find optimal threshold
    fpr, tpr, thresholds = roc_curve(df_binary['label'], -df_binary['evo2_score'])
    youden_idx = np.argmax(tpr - fpr)
    optimal_threshold = -thresholds[youden_idx]
    print(f"\n🎯 Optimal threshold (Youden's Index): {optimal_threshold:.6f}")
    
    # Make predictions
    df_binary['prediction'] = (df_binary['evo2_score'] < optimal_threshold).astype(int)
    
    # Calculate metrics
    auroc = roc_auc_score(df_binary['label'], -df_binary['evo2_score'])
    f1 = f1_score(df_binary['label'], df_binary['prediction'])
    precision = precision_score(df_binary['label'], df_binary['prediction'])
    recall = recall_score(df_binary['label'], df_binary['prediction'])
    
    cm = confusion_matrix(df_binary['label'], df_binary['prediction'])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    
    print("\n" + "="*60)
    print("VALIDATION METRICS - REAL EVO2")
    print("="*60)
    print(f"AUROC: {auroc:.3f}")
    print(f"F1 Score: {f1:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall (Sensitivity): {recall:.3f}")
    print(f"Specificity: {specificity:.3f}")
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives:  {tn:4d} (correct benign)")
    print(f"  False Positives: {fp:4d} (benign called pathogenic)")
    print(f"  False Negatives: {fn:4d} (pathogenic called benign)")
    print(f"  True Positives:  {tp:4d} (correct pathogenic)")
    print("="*60)
    
    # Save metrics
    metrics = {
        "auroc": float(auroc),
        "f1_score": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "optimal_threshold": float(optimal_threshold),
        "correlation_with_findlay": float(corr),
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp),
            "fn": int(fn), "tp": int(tp)
        },
        "dataset_size": {
            "total": len(df_binary),
            "lof": int((df_binary['label']==1).sum()),
            "func": int((df_binary['label']==0).sum())
        }
    }
    
    import json
    with open("results/evo2_results/validation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("\n✅ Saved metrics to results/evo2_results/validation_metrics.json")
    
    # Plot ROC curve
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, linewidth=2, label=f'Real Evo2 (AUROC = {auroc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - Real Evo2 on BRCA1', fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/evo2_results/roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved ROC curve to results/evo2_results/roc_curve.png")
    
    # Plot Precision-Recall curve
    precision_vals, recall_vals, _ = precision_recall_curve(df_binary['label'], -df_binary['evo2_score'])
    plt.figure(figsize=(10, 8))
    plt.plot(recall_vals, precision_vals, linewidth=2)
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve - Real Evo2', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/evo2_results/precision_recall_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved P-R curve to results/evo2_results/precision_recall_curve.png")
    
    # Score statistics by class
    print("\n📊 Score Statistics by Functional Class:")
    print("="*60)
    stats = df.groupby('func_class')['evo2_score'].agg(['count', 'mean', 'std', 'min', 'max'])
    print(stats)
    print("="*60)
    
    print("\n🎉 Real Evo2 validation complete!")

if __name__ == "__main__":
    main()
