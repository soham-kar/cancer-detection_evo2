"""
Phase 1 / Week 2: Publication Figure Generation
=================================================
Generates publication-quality figures from benchmark results:
- Figure 1: ROC curves (Evo2 vs CADD vs AlphaMissense vs REVEL)
- Figure 2: Precision-Recall curves
- Figure 3: Calibration reliability diagram
- Figure 4: Confidence stratification bar chart
- Figure 5: Prediction distribution pie chart

Input: helixmind_benchmark_results.csv
Output: figures/ directory with PNG files (300 DPI)

Usage:
    python generate_figures.py
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import roc_curve, precision_recall_curve, auc

# =============================================================================
# CONFIGURATION
# =============================================================================

RESULTS_FILE = Path(__file__).parent.parent / "helixmind_benchmark_results.csv"
FIGURES_DIR = Path(__file__).parent / "figures"

# Publication style
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.figsize": (6, 5),
})

# Color scheme
COLORS = {
    "Evo2-7B": "#ef4444",
    "AlphaMissense": "#3b82f6",
    "CADD": "#f59e0b",
    "REVEL": "#8b5cf6",
    "pathogenic": "#ef4444",
    "benign": "#22c55e",
    "vus": "#f59e0b",
    "conflicting": "#8b5cf6",
}


def map_to_binary(label: str) -> int:
    label = str(label).lower()
    if "pathogenic" in label and "conflicting" not in label:
        return 1
    elif "benign" in label:
        return 0
    return -1


def generate_all_figures():
    """Generate all publication figures."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    if not RESULTS_FILE.exists():
        print(f"❌ Results file not found: {RESULTS_FILE}")
        return
    
    df = pd.read_csv(RESULTS_FILE)
    print(f"Loaded {len(df)} results")
    
    # Prepare data
    df["binary_label"] = df["clinical_significance"].apply(map_to_binary)
    binary_df = df[df["binary_label"] >= 0].copy()
    binary_df["score"] = -binary_df["delta_score"].astype(float)
    
    y_true = binary_df["binary_label"].values
    y_score = binary_df["score"].values
    valid = ~np.isnan(y_score)
    y_true = y_true[valid]
    y_score = y_score[valid]
    
    # ─── Figure 1: ROC Curve ───
    fig, ax = plt.subplots(figsize=(6, 5))
    
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    ax.plot(fpr, tpr, color=COLORS["Evo2-7B"], linewidth=2,
            label=f"Evo2-7B (AUROC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5, label="Random (0.500)")
    
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title("ROC Curve — Evo2-7B on ClinVar 4K Benchmark")
    ax.legend(loc="lower right", frameon=True, fancybox=True, shadow=True)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "figure1_roc_curve.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Figure 1: ROC Curve → figures/figure1_roc_curve.png (AUROC = {roc_auc:.3f})")
    
    # ─── Figure 2: Precision-Recall Curve ───
    fig, ax = plt.subplots(figsize=(6, 5))
    
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)
    baseline = y_true.mean()
    
    ax.plot(recall, precision, color=COLORS["Evo2-7B"], linewidth=2,
            label=f"Evo2-7B (AUPRC = {pr_auc:.3f})")
    ax.axhline(y=baseline, color="gray", linestyle="--", linewidth=0.8, alpha=0.5,
               label=f"Baseline ({baseline:.3f})")
    
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve — Evo2-7B on ClinVar 4K")
    ax.legend(loc="upper right", frameon=True, fancybox=True, shadow=True)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "figure2_pr_curve.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Figure 2: PR Curve → figures/figure2_pr_curve.png (AUPRC = {pr_auc:.3f})")
    
    # ─── Figure 3: Prediction Distribution ───
    fig, ax = plt.subplots(figsize=(6, 5))
    
    pred_counts = df["prediction"].value_counts()
    colors = [COLORS.get(p.lower().split()[0], "#94a3b8") for p in pred_counts.index]
    
    wedges, texts, autotexts = ax.pie(
        pred_counts.values,
        labels=pred_counts.index,
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
        pctdistance=0.85,
    )
    
    for autotext in autotexts:
        autotext.set_fontsize(8)
    
    ax.set_title("Evo2-7B Prediction Distribution\nClinVar 4K Benchmark")
    
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "figure3_prediction_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Figure 3: Prediction Distribution → figures/figure3_prediction_distribution.png")
    
    # ─── Figure 4: Delta Score Distribution ───
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    for ax_i, (label, color) in enumerate([
        ("Pathogenic", COLORS["pathogenic"]),
        ("Benign", COLORS["benign"]),
    ]):
        subset = df[df["clinical_significance"].str.lower().str.contains(label.lower(), na=False)]
        scores = -subset["delta_score"].dropna().astype(float)
        
        axes[0].hist(scores, bins=50, alpha=0.6, color=color, label=label, density=True)
        axes[1].hist(scores, bins=50, alpha=0.6, color=color, label=label, density=True, cumulative=True)
    
    axes[0].set_xlabel("Evo2 Score (higher = more pathogenic)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Score Distribution by Class")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel("Evo2 Score (higher = more pathogenic)")
    axes[1].set_ylabel("Cumulative Density")
    axes[1].set_title("Cumulative Score Distribution")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "figure4_score_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Figure 4: Score Distribution → figures/figure4_score_distribution.png")
    
    # ─── Figure 5: Confusion Matrix ───
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Build confusion data
    categories = ["Pathogenic", "Benign", "VUS", "Conflicting"]
    matrix = np.zeros((4, 4))
    
    pred_map = {
        "Likely Pathogenic": 0,
        "Likely Benign": 1,
        "Uncertain Significance": 2,
    }
    true_map = {
        "Pathogenic": 0,
        "Benign": 1,
        "VUS": 2,
        "Conflicting": 3,
    }
    
    for _, row in df.iterrows():
        true_label = str(row["clinical_significance"]).lower()
        pred_label = str(row["prediction"]).lower()
        
        true_idx = -1
        for key, idx in true_map.items():
            if key.lower() in true_label:
                true_idx = idx
                break
        
        pred_idx = -1
        for key, idx in pred_map.items():
            if key.lower() in pred_label:
                pred_idx = idx
                break
        
        if true_idx >= 0 and pred_idx >= 0:
            matrix[true_idx, pred_idx] += 1
    
    im = ax.imshow(matrix, cmap="Blues", aspect="auto")
    
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Pred Pathogenic", "Pred Benign", "Pred VUS"], rotation=45, ha="right")
    ax.set_yticks(range(4))
    ax.set_yticklabels(["True Pathogenic", "True Benign", "True VUS", "True Conflicting"])
    ax.set_title("Confusion Matrix — Evo2-7B on ClinVar 4K")
    
    # Add text annotations
    for i in range(4):
        for j in range(3):
            if matrix[i, j] > 0:
                text_color = "white" if matrix[i, j] > matrix.max() / 2 else "black"
                ax.text(j, i, f"{int(matrix[i, j])}", ha="center", va="center",
                        color=text_color, fontsize=8, fontweight="bold")
    
    plt.colorbar(im, ax=ax, shrink=0.8)
    
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "figure5_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Figure 5: Confusion Matrix → figures/figure5_confusion_matrix.png")
    
    print(f"\n✅ All figures saved to: {FIGURES_DIR}/")


if __name__ == "__main__":
    generate_all_figures()
