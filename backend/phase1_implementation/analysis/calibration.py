"""
Phase 1 / Week 2: Calibration Analysis
========================================
Computes reliability diagrams, Expected Calibration Error (ECE),
and Brier score for Evo2 confidence calibration.

Input: helixmind_benchmark_results.csv
Output: calibration_metrics.json

Paper: Angelopoulos & Bates (2021) — IEEE SPM
       "Conformal prediction for reliable machine learning"
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from typing import Dict

# =============================================================================
# CONFIGURATION
# =============================================================================

RESULTS_FILE = Path(__file__).parent.parent / "helixmind_benchmark_results.csv"
CALIBRATION_FILE = Path(__file__).parent / "calibration_metrics.json"

N_BINS = 10  # Number of bins for reliability diagram


def map_to_binary(label: str) -> int:
    label = str(label).lower()
    if "pathogenic" in label and "conflicting" not in label:
        return 1
    elif "benign" in label:
        return 0
    return -1


def compute_calibration(df: pd.DataFrame) -> Dict:
    """Compute calibration metrics."""
    
    df["binary_label"] = df["clinical_significance"].apply(map_to_binary)
    binary_df = df[df["binary_label"] >= 0].copy()
    
    # Use classification_confidence as predicted probability
    binary_df["confidence"] = binary_df["classification_confidence"].astype(float)
    
    # For pathogenic predictions, confidence is P(pathogenic)
    # For benign predictions, confidence is P(benign) → convert to P(pathogenic) = 1 - P(benign)
    binary_df["prob_pathogenic"] = binary_df.apply(
        lambda r: r["confidence"] if "pathogenic" in str(r["prediction"]).lower()
        else (1 - r["confidence"]) if "benign" in str(r["prediction"]).lower()
        else 0.5,
        axis=1
    )
    
    y_true = binary_df["binary_label"].values
    y_prob = binary_df["prob_pathogenic"].values
    
    valid = ~np.isnan(y_prob)
    y_true = y_true[valid]
    y_prob = y_prob[valid]
    
    n = len(y_true)
    
    print(f"Valid calibration samples: {n}")
    
    metrics = {"n_samples": n}
    
    if n < 10:
        print("⚠️  Not enough samples for calibration.")
        return metrics
    
    # ─── Reliability Diagram ───
    try:
        prob_true, prob_pred = calibration_curve(
            y_true, y_prob, n_bins=N_BINS, strategy="uniform"
        )
        metrics["reliability_diagram"] = {
            "bin_centers": [round(x, 4) for x in prob_pred.tolist()],
            "observed_frequency": [round(x, 4) for x in prob_true.tolist()],
            "n_bins": N_BINS,
        }
        
        # Print reliability table
        print(f"\n📊 Reliability Diagram ({N_BINS} bins):")
        print(f"   {'Bin Center':<12} {'Observed':<12} {'Gap':<12}")
        print(f"   {'-'*36}")
        for pb, pt in zip(prob_pred, prob_true):
            gap = pt - pb
            print(f"   {pb:<12.4f} {pt:<12.4f} {gap:<+12.4f}")
        
    except Exception as e:
        print(f"⚠️  Reliability diagram failed: {e}")
    
    # ─── Expected Calibration Error (ECE) ───
    try:
        prob_true, prob_pred = calibration_curve(
            y_true, y_prob, n_bins=N_BINS, strategy="uniform"
        )
        bin_counts = np.histogram(y_prob, bins=N_BINS, range=(0, 1))[0]
        ece = np.sum(bin_counts / n * np.abs(prob_true - prob_pred))
        metrics["ece"] = round(float(ece), 4)
        print(f"\nECE: {ece:.4f}")
    except Exception as e:
        print(f"⚠️  ECE failed: {e}")
    
    # ─── Brier Score ───
    try:
        brier = brier_score_loss(y_true, y_prob)
        metrics["brier_score"] = round(float(brier), 4)
        print(f"Brier Score: {brier:.4f}")
    except Exception as e:
        print(f"⚠️  Brier score failed: {e}")
    
    # ─── Confidence Stratification ───
    try:
        bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
        binary_df["conf_bin"] = pd.cut(binary_df["confidence"], bins=bins, labels=labels)
        
        stratification = {}
        for label in labels:
            subset = binary_df[binary_df["conf_bin"] == label]
            if len(subset) > 0:
                accuracy = subset["binary_label"].mean()
                stratification[label] = {
                    "count": len(subset),
                    "accuracy": round(float(accuracy), 4),
                }
        
        metrics["confidence_stratification"] = stratification
        
        print(f"\n📊 Confidence Stratification:")
        for label, stats in stratification.items():
            print(f"   {label}: n={stats['count']}, accuracy={stats['accuracy']:.4f}")
    
    except Exception as e:
        print(f"⚠️  Stratification failed: {e}")
    
    return metrics


def main():
    print("=" * 60)
    print("📊 Calibration Analysis")
    print("=" * 60)
    print()
    
    if not RESULTS_FILE.exists():
        print(f"❌ Results file not found: {RESULTS_FILE}")
        return
    
    df = pd.read_csv(RESULTS_FILE)
    print(f"Loaded {len(df)} results")
    print()
    
    metrics = compute_calibration(df)
    
    CALIBRATION_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CALIBRATION_FILE, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✅ Calibration metrics saved to: {CALIBRATION_FILE}")


if __name__ == "__main__":
    main()
