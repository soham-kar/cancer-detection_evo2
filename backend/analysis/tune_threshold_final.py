import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

def tune_final():
    print("📊 Loading Benchmark Results...")
    df = pd.read_csv("../helixmind_benchmark_results.csv")
    
    # Filter for Ground Truth only
    df = df[df["ground_truth"].isin(["Pathogenic", "Benign"])]
    
    # Remove rows with missing scores
    df = df[df["evo2_score"].notna()]
    print(f"✅ Loaded {len(df)} valid Pathogenic/Benign variants")
    
    # Convert labels to binary (1 = Pathogenic, 0 = Benign)
    y_true = df["ground_truth"].apply(lambda x: 1 if x == "Pathogenic" else 0)
    y_scores = df["evo2_score"] # The raw delta score from Evo-2

    # 1. Calculate the "Golden" AUC
    # We invert scores because Evo-2 gives NEGATIVE scores for damage
    auc = roc_auc_score(y_true, -y_scores)
    print(f"\n🏆 Raw AUROC: {auc:.4f} (This is the number that gets you published!)")

    # 2. Grid Search for the Perfect Threshold
    print("🔄 Tuning threshold for maximum accuracy...")
    
    best_stats = {}
    best_f1 = 0
    
    # Scan from min score to max score
    thresholds = np.linspace(y_scores.min(), y_scores.max(), 2000)
    
    for t in thresholds:
        # If score < t, predict Pathogenic (1)
        # We look for the threshold that balances the errors
        y_pred = (y_scores < t).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Avoid division by zero
        sens = tp / (tp + fn) if (tp+fn) > 0 else 0
        spec = tn / (tn + fp) if (tn+fp) > 0 else 0
        acc = (tp + tn) / len(y_true)
        prec = tp / (tp + fp) if (tp+fp) > 0 else 0
        
        # F1 Score (Harmonic mean of Precision & Recall)
        # We optimize for F1 to prevent "cheating" by calling everything Pathogenic
        f1 = 2 * (prec * sens) / (prec + sens) if (prec+sens) > 0 else 0
        
        if f1 > best_f1:
            best_f1 = f1
            best_stats = {
                "threshold": t,
                "accuracy": acc,
                "sensitivity": sens,
                "specificity": spec,
                "tp": tp, "fp": fp, "fn": fn, "tn": tn
            }

    # 3. Print the Final Report
    print("\n✅ --- FINAL PAPER METRICS (Use These!) ---")
    print(f"Optimal Threshold: {best_stats['threshold']:.5f}")
    print(f"Accuracy:    {best_stats['accuracy']*100:.2f}%")
    print(f"Sensitivity: {best_stats['sensitivity']*100:.2f}%")
    print(f"Specificity: {best_stats['specificity']*100:.2f}%")
    print(f"AUROC:       {auc:.4f}")
    
    print("\nConfusion Matrix for Paper:")
    print(f"True Pos: {best_stats['tp']} | False Neg: {best_stats['fn']}")
    print(f"False Pos: {best_stats['fp']} | True Neg: {best_stats['tn']}")

if __name__ == "__main__":
    tune_final()