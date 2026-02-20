"""
Day 2: Validation Analysis
Compares Evo2 predictions against Findlay functional labels
"""
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, precision_recall_curve, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import json

def main():
    print("Loading Evo2 scores...")
    df = pd.read_csv("results/brca1_evo2_scores.csv")
    print(f"Loaded {len(df)} variants")
    
    # Map Findlay classes to binary labels
    # LOF = pathogenic (1), FUNC = benign (0), INT = uncertain (separate analysis)
    print("\nClass distribution:")
    print(df['func_class'].value_counts())
    
    # Binary classification (LOF vs FUNC only)
    df_binary = df[df['func_class'].isin(['LOF', 'FUNC'])].copy()
    df_binary['label'] = (df_binary['func_class'] == 'LOF').astype(int)
    
    print(f"\nBinary classification dataset: {len(df_binary)} variants")
    print(f"  LOF (pathogenic): {sum(df_binary['label'] == 1)}")
    print(f"  FUNC (benign): {sum(df_binary['label'] == 0)}")
    
    # Correlation with functional scores
    if 'func_score' in df.columns:
        correlation = df['evo2_score'].corr(df['func_score'])
        print(f"\nCorrelation (Evo2 vs Findlay func_score): {correlation:.3f}")
    
    # Find optimal threshold using Youden's Index
    fpr, tpr, thresholds = roc_curve(df_binary['label'], -df_binary['evo2_score'])
    youden_index = tpr - fpr
    optimal_idx = np.argmax(youden_index)
    optimal_threshold = -thresholds[optimal_idx]  # Convert back to original scale
    
    print(f"\nOptimal threshold (Youden's Index): {optimal_threshold:.6f}")
    
    # Classification with optimal threshold
    df_binary['pred'] = (df_binary['evo2_score'] < optimal_threshold).astype(int)
    
    # Metrics
    auc = roc_auc_score(df_binary['label'], -df_binary['evo2_score'])
    f1 = f1_score(df_binary['label'], df_binary['pred'])
    precision = sum((df_binary['pred'] == 1) & (df_binary['label'] == 1)) / sum(df_binary['pred'] == 1)
    recall = sum((df_binary['pred'] == 1) & (df_binary['label'] == 1)) / sum(df_binary['label'] == 1)
    cm = confusion_matrix(df_binary['label'], df_binary['pred'])
    
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    
    print("\n" + "="*60)
    print("VALIDATION METRICS")
    print("="*60)
    print(f"AUROC: {auc:.3f}")
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
        "total_variants": len(df),
        "binary_classification_variants": len(df_binary),
        "LOF_count": int(sum(df_binary['label'] == 1)),
        "FUNC_count": int(sum(df_binary['label'] == 0)),
        "auroc": float(auc),
        "f1_score": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "optimal_threshold": float(optimal_threshold),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        }
    }
    
    with open("results/validation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✅ Saved metrics to results/validation_metrics.json")
    
    # ROC Curve
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, linewidth=2, label=f'Evo2 (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve: Evo2 BRCA1 Prediction', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/roc_curve.png', dpi=300)
    print(f"✅ Saved ROC curve to results/roc_curve.png")
    plt.close()
    
    # Precision-Recall Curve
    precision_curve, recall_curve, _ = precision_recall_curve(df_binary['label'], -df_binary['evo2_score'])
    plt.figure(figsize=(8, 8))
    plt.plot(recall_curve, precision_curve, linewidth=2)
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve: Evo2 BRCA1 Prediction', fontsize=14)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/precision_recall_curve.png', dpi=300)
    print(f"✅ Saved P-R curve to results/precision_recall_curve.png")
    plt.close()

if __name__ == "__main__":
    main()
