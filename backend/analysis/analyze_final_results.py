import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

def analyze_and_tune():
    print("📊 Loading results...")
    df = pd.read_csv("../helixmind_benchmark_results.csv")
    
    # 1. Clean Data
    # Remove errors or empty predictions
    df = df[df["evo2_score"].notna()]
    # Filter for Ground Truth (Pathogenic vs Benign only for binary metrics)
    df = df[df["ground_truth"].isin(["Pathogenic", "Benign"])]
    
    # Map labels to 1 (Pathogenic) and 0 (Benign)
    # NOTE: Evo-2 scores are usually NEGATIVE for Pathogenic (Loss of Function)
    y_true = df["ground_truth"].apply(lambda x: 1 if x == "Pathogenic" else 0)
    y_scores = df["evo2_score"] # The raw delta score
    
    print(f"✅ Analyzing {len(df)} valid variants...")

    # 2. Calculate AUC (Area Under Curve)
    # We invert scores because lower score = more pathogenic
    auc = roc_auc_score(y_true, -y_scores) 
    print(f"\n🏆 AUROC Score: {auc:.4f} (Target: >0.90)")

    # 3. Find Optimal Threshold (Maximize F1 Score)
    best_f1 = 0
    best_thresh = 0
    best_acc = 0
    
    thresholds = np.linspace(df["evo2_score"].min(), df["evo2_score"].max(), 200)
    
    for t in thresholds:
        # If score < t, predict Pathogenic (1)
        y_pred = (y_scores < t).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        sensitivity = tp / (tp + fn) if (tp+fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn+fp) > 0 else 0
        precision = tp / (tp + fp) if (tp+fp) > 0 else 0
        accuracy = (tp + tn) / len(y_true)
        
        # F1 Score
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision+sensitivity) > 0 else 0
        
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
            best_acc = accuracy
            best_metrics = (sensitivity, specificity)

    print("\n🎯 --- OPTIMAL THRESHOLD FOUND ---")
    print(f"Threshold: {best_thresh:.4f}")
    print(f"Accuracy:  {best_acc*100:.2f}%")
    print(f"Sensitivity: {best_metrics[0]*100:.2f}% (Catching Bad Variants)")
    print(f"Specificity: {best_metrics[1]*100:.2f}% (Ignoring Safe Variants)")
    
    # 4. Save the ROC Curve for your Paper (Figure 1)
    fpr, tpr, _ = roc_curve(y_true, -y_scores)
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, label=f'HelixMind (AUC = {auc:.2f})', color='purple', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('Figure 1: HelixMind Classification Performance')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.savefig("figure1_roc_curve.png")
    print("\n🖼️  Saved 'figure1_roc_curve.png' - Use this in your paper!")

if __name__ == "__main__":
    analyze_and_tune()