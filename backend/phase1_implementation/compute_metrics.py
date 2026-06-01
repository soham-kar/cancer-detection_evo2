"""
Benchmark Metrics & Figures — Phase 1
======================================
Computes AUROC, AUPRC, calibration curves for:
  - Evo2 (delta_score)
  - AlphaMissense (score)
  - Consensus (weighted ensemble)

Saves:
  - metrics.json (AUROC, AUPRC, per-class stats)
  - figures/*.png (ROC, PR, calibration, delta distribution)

Usage:
    python backend/phase1_implementation/compute_metrics.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
    confusion_matrix, classification_report
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURATION
# =============================================================================
PHASE1 = Path("backend/phase1_implementation")
INPUT_CSV = PHASE1 / "helixmind_benchmark_results_enriched.csv"
OUTPUT_DIR = PHASE1 / "figures"
METRICS_JSON = PHASE1 / "metrics.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# LABEL MAPPING
# =============================================================================

def map_label(label: str) -> int:
    """Map ClinVar label to binary: 1=Pathogenic, 0=Benign, -1=Exclude."""
    label = str(label).strip().lower()
    if "pathogenic" in label and "uncertain" not in label and "conflicting" not in label:
        return 1
    if "benign" in label and "uncertain" not in label and "conflicting" not in label:
        return 0
    return -1  # Uncertain / Conflicting — exclude from binary metrics

# =============================================================================
# SCORE PREPARATION
# =============================================================================

def prepare_scores(df: pd.DataFrame):
    """Extract binary labels and continuous scores for each model."""
    df["y_true"] = df["label"].apply(map_label)
    binary_df = df[df["y_true"] != -1].copy()

    n_total = len(df)
    n_binary = len(binary_df)
    print(f"Total variants: {n_total}")
    print(f"Binary-classifiable: {n_binary} ({n_binary/n_total:.1%})")
    print(f"  Pathogenic: {(binary_df['y_true']==1).sum()}")
    print(f"  Benign: {(binary_df['y_true']==0).sum()}")

    # Evo2: negative delta = pathogenic, so flip sign for ROC
    binary_df["evo2_score"] = -binary_df["delta_score"].astype(float)

    # AlphaMissense: score is already pathogenicity (0-1)
    binary_df["am_score"] = binary_df["alphamissense_score"].astype(float)

    # Consensus: use continuous weighted score from available predictors
    # When both Evo2 + AlphaMissense available: weighted average
    # When only Evo2: use Evo2 score directly
    # When neither: NaN (excluded from ROC)
    def consensus_continuous(row):
        evo2 = row["evo2_score"]
        am = row["am_score"]
        if pd.notna(evo2) and pd.notna(am):
            # Both available: weighted average (Evo2 0.6, AM 0.4)
            return 0.6 * evo2 + 0.4 * am
        elif pd.notna(evo2):
            # Only Evo2: use Evo2 score directly
            return evo2
        elif pd.notna(am):
            # Only AM (rare): use AM score
            return am
        return np.nan
    binary_df["consensus_score"] = binary_df.apply(consensus_continuous, axis=1)

    return binary_df

# =============================================================================
# METRICS COMPUTATION
# =============================================================================

def compute_metrics(df: pd.DataFrame) -> dict:
    """Compute AUROC, AUPRC, and calibration metrics."""
    y_true = df["y_true"].values

    results = {}
    for name, score_col in [
        ("Evo2", "evo2_score"),
        ("AlphaMissense", "am_score"),
        ("Consensus", "consensus_score"),
    ]:
        scores = df[score_col].values
        # Drop NaNs
        mask = ~np.isnan(scores)
        y = y_true[mask]
        s = scores[mask]

        if len(np.unique(y)) < 2:
            print(f"  ⚠️ {name}: only one class present, skipping AUROC")
            results[name] = {"auroc": None, "auprc": None, "n": len(y)}
            continue

        auroc = roc_auc_score(y, s)
        auprc = average_precision_score(y, s)

        # Optimal threshold (Youden's J)
        fpr, tpr, thresh = roc_curve(y, s)
        j_scores = tpr - fpr
        opt_idx = np.argmax(j_scores)
        opt_thresh = thresh[opt_idx]

        # Predictions at optimal threshold
        preds = (s >= opt_thresh).astype(int)
        cm = confusion_matrix(y, preds)
        tn, fp, fn, tp = cm.ravel()

        results[name] = {
            "auroc": round(auroc, 4),
            "auprc": round(auprc, 4),
            "optimal_threshold": round(float(opt_thresh), 6),
            "n": len(y),
            "confusion_matrix": {
                "tn": int(tn), "fp": int(fp),
                "fn": int(fn), "tp": int(tp),
            },
            "sensitivity": round(tp / (tp + fn), 4) if (tp + fn) > 0 else None,
            "specificity": round(tn / (tn + fp), 4) if (tn + fp) > 0 else None,
            "ppv": round(tp / (tp + fp), 4) if (tp + fp) > 0 else None,
            "npv": round(tn / (tn + fn), 4) if (tn + fn) > 0 else None,
        }
        print(f"  ✅ {name}: AUROC={auroc:.3f}, AUPRC={auprc:.3f}, n={len(y)}")

    return results

# =============================================================================
# FIGURE GENERATION
# =============================================================================

def plot_roc_curves(df: pd.DataFrame, out_path: Path):
    """Overlay ROC curves for all three models."""
    y_true = df["y_true"].values
    fig, ax = plt.subplots(figsize=(8, 8))

    colors = {"Evo2": "#e74c3c", "AlphaMissense": "#3498db", "Consensus": "#2ecc71"}

    for name, score_col in [
        ("Evo2", "evo2_score"),
        ("AlphaMissense", "am_score"),
        ("Consensus", "consensus_score"),
    ]:
        scores = df[score_col].values
        mask = ~np.isnan(scores)
        y = y_true[mask]
        s = scores[mask]
        if len(np.unique(y)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y, s)
        auc = roc_auc_score(y, s)
        ax.plot(fpr, tpr, color=colors[name], lw=2.5,
                label=f"{name} (AUROC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — Variant Pathogenicity Prediction", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊 Saved ROC: {out_path}")


def plot_pr_curves(df: pd.DataFrame, out_path: Path):
    """Overlay Precision-Recall curves."""
    y_true = df["y_true"].values
    fig, ax = plt.subplots(figsize=(8, 8))

    colors = {"Evo2": "#e74c3c", "AlphaMissense": "#3498db", "Consensus": "#2ecc71"}

    for name, score_col in [
        ("Evo2", "evo2_score"),
        ("AlphaMissense", "am_score"),
        ("Consensus", "consensus_score"),
    ]:
        scores = df[score_col].values
        mask = ~np.isnan(scores)
        y = y_true[mask]
        s = scores[mask]
        if len(np.unique(y)) < 2:
            continue
        precision, recall, _ = precision_recall_curve(y, s)
        auprc = average_precision_score(y, s)
        ax.plot(recall, precision, color=colors[name], lw=2.5,
                label=f"{name} (AUPRC={auprc:.3f})")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves", fontsize=14, fontweight="bold")
    ax.legend(loc="lower left", fontsize=11)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊 Saved PR: {out_path}")


def plot_delta_distribution(df: pd.DataFrame, out_path: Path):
    """Histogram of Evo2 delta scores by true label."""
    fig, ax = plt.subplots(figsize=(10, 6))

    pathogenic = df[df["y_true"] == 1]["delta_score"].dropna()
    benign = df[df["y_true"] == 0]["delta_score"].dropna()

    ax.hist(benign, bins=50, alpha=0.6, color="#3498db", label=f"Benign (n={len(benign)})")
    ax.hist(pathogenic, bins=50, alpha=0.6, color="#e74c3c", label=f"Pathogenic (n={len(pathogenic)})")

    ax.axvline(x=0, color="black", linestyle="--", lw=1, alpha=0.5)
    ax.set_xlabel("Evo2 Delta Score", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Evo2 Delta Score Distribution by True Label", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊 Saved distribution: {out_path}")


def plot_calibration(df: pd.DataFrame, out_path: Path):
    """Calibration plot: predicted probability vs observed frequency."""
    from sklearn.calibration import calibration_curve

    y_true = df["y_true"].values
    fig, ax = plt.subplots(figsize=(8, 8))

    colors = {"Evo2": "#e74c3c", "AlphaMissense": "#3498db", "Consensus": "#2ecc71"}

    for name, score_col in [
        ("Evo2", "evo2_score"),
        ("AlphaMissense", "am_score"),
        ("Consensus", "consensus_score"),
    ]:
        scores = df[score_col].values
        mask = ~np.isnan(scores)
        y = y_true[mask]
        s = scores[mask]
        if len(np.unique(y)) < 2 or len(s) < 100:
            continue
        # Normalize scores to [0,1] for calibration
        s_norm = (s - s.min()) / (s.max() - s.min() + 1e-9)
        prob_true, prob_pred = calibration_curve(y, s_norm, n_bins=10, strategy="uniform")
        ax.plot(prob_pred, prob_true, "s-", color=colors[name], lw=2,
                label=name, markersize=6)

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Fraction of Positives", fontsize=12)
    ax.set_title("Calibration Curves", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊 Saved calibration: {out_path}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("Benchmark Metrics & Figure Generation")
    print("=" * 60)

    df = pd.read_csv(INPUT_CSV)
    binary_df = prepare_scores(df)

    print("\nComputing metrics...")
    metrics = compute_metrics(binary_df)

    # Save metrics JSON
    with open(METRICS_JSON, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n💾 Saved metrics: {METRICS_JSON}")

    print("\nGenerating figures...")
    plot_roc_curves(binary_df, OUTPUT_DIR / "roc_curves.png")
    plot_pr_curves(binary_df, OUTPUT_DIR / "pr_curves.png")
    plot_delta_distribution(binary_df, OUTPUT_DIR / "delta_distribution.png")
    plot_calibration(binary_df, OUTPUT_DIR / "calibration.png")

    print(f"\n✅ All outputs in: {PHASE1}")
    print(f"   - metrics.json")
    print(f"   - figures/*.png")

if __name__ == "__main__":
    main()
