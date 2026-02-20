import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

def plot_paper_figure():
    # 1. Load the Good Data
    df = pd.read_csv("../helixmind_benchmark_results.csv")
    
    # Filter for valid rows
    df = df[df["ground_truth"].isin(["Pathogenic", "Benign"])]
    df = df[df["evo2_score"].notna()]  # Remove NaN scores
    
    y_true = df["ground_truth"].apply(lambda x: 1 if x == "Pathogenic" else 0)
    y_scores = df["evo2_score"] # Raw scores

    # 2. Calculate Curve
    # Invert scores because Evo-2 uses negative values for damage
    fpr, tpr, _ = roc_curve(y_true, -y_scores)
    auc = roc_auc_score(y_true, -y_scores)

    # 3. Create Professional Plot
    plt.figure(figsize=(8, 6), dpi=300) # High DPI for journal submission
    
    # The Main Line
    plt.plot(fpr, tpr, color='#800080', lw=2.5, label=f'HelixMind (AUC = {auc:.2f})')
    
    # The "Random Guess" Line
    plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label='Random Chance')

    # Styling
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title('Performance of HelixMind on ClinVar (N=899)', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.2)

    # 4. Save
    plt.savefig("Figure1_ROC.png")
    print(f"✅ Created Figure1_ROC.png (AUC = {auc:.4f})")
    print("Add this image to your manuscript immediately.")

if __name__ == "__main__":
    plot_paper_figure()