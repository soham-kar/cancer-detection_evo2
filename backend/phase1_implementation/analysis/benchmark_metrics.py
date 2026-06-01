"""
Phase 1 / Week 2: Benchmark Metrics Computation
================================================
Computes AUROC, AUPRC, sensitivity, specificity, and gene-stratified
performance from the benchmark results CSV.

Input: helixmind_benchmark_results.csv
Output: benchmark_metrics.json

Papers:
- Nguyen et al. (2024) — bioRxiv: Evo2
- Cheng et al. (2023) — Science: AlphaMissense
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
    brier_score_loss,
)
from typing import Dict, Tuple

# =============================================================================
# CONFIGURATION
# =============================================================================

RESULTS_FILE = Path(__file__).parent.parent / "helixmind_benchmark_results.csv"
METRICS_FILE = Path(__file__).parent / "benchmark_metrics.json"

# =============================================================================
# LABEL MAPPING
# =============================================================================

def map_to_binary(label: str) -> int:
    """Map ClinVar label to binary: 1 = pathogenic, 0 = benign."""
    label = str(label).lower()
    if "pathogenic" in label and "conflicting" not in label:
        return 1
    elif "benign" in label:
        return 0
    else:
        return -1  # VUS/Conflicting — excluded from binary metrics


def map_to_multiclass(label: str) -> str:
    """Map ClinVar label to standardized class."""
    label = str(label).lower()
    if "pathogenic" in label and "conflicting" not in label:
        return "Pathogenic"
    elif "benign" in label:
        return "Benign"
    elif "uncertain" in label:
        return "VUS"
    else:
        return "Conflicting"


# =============================================================================
# METRICS COMPUTATION
# =============================================================================

def compute_metrics(df: pd.DataFrame) -> Dict:
    """Compute all benchmark metrics."""
    
    # Filter to binary labels (exclude VUS and Conflicting)
    df["binary_label"] = df["clinical_significance"].apply(map_to_binary)
    binary_df = df[df["binary_label"] >= 0].copy()
    
    # Use delta_score as prediction score (negative delta = pathogenic)
    # Invert so higher score = more pathogenic
    binary_df["score"] = -binary_df["delta_score"].astype(float)
    
    y_true = binary_df["binary_label"].values
    y_score = binary_df["score"].values
    
    # Handle NaN scores
    valid = ~np.isnan(y_score)
    y_true = y_true[valid]
    y_score = y_score[valid]
    
    n_total = len(df)
    n_binary = len(binary_df)
    n_valid = len(y_true)
    n_pathogenic = int(y_true.sum())
    n_benign = n_valid - n_pathogenic
    
    print(f"Total variants: {n_total}")
    print(f"Binary (P/B): {n_binary} ({n_pathogenic} P, {n_benign} B)")
    print(f"Valid scores: {n_valid}")
    print()
    
    metrics = {
        "dataset": {
            "total_variants": n_total,
            "binary_variants": n_binary,
            "valid_scores": n_valid,
            "pathogenic": n_pathogenic,
            "benign": n_benign,
            "class_balance": round(n_pathogenic / n_valid, 3) if n_valid > 0 else 0,
        }
    }
    
    if n_valid < 10:
        print("⚠️  Not enough valid scores for metrics.")
        return metrics
    
    # ─── AUROC ───
    try:
        auroc = roc_auc_score(y_true, y_score)
        metrics["auroc"] = round(auroc, 4)
        print(f"AUROC: {auroc:.4f}")
    except Exception as e:
        print(f"⚠️  AUROC failed: {e}")
        metrics["auroc"] = None
    
    # ─── AUPRC ───
    try:
        auprc = average_precision_score(y_true, y_score)
        metrics["auprc"] = round(auprc, 4)
        print(f"AUPRC: {auprc:.4f}")
    except Exception as e:
        print(f"⚠️  AUPRC failed: {e}")
        metrics["auprc"] = None
    
    # ─── ROC Curve Data ───
    try:
        fpr, tpr, thresholds_roc = roc_curve(y_true, y_score)
        metrics["roc_curve"] = {
            "fpr": [round(x, 4) for x in fpr.tolist()],
            "tpr": [round(x, 4) for x in tpr.tolist()],
            "thresholds": [round(x, 6) for x in thresholds_roc.tolist()],
        }
    except Exception as e:
        print(f"⚠️  ROC curve failed: {e}")
    
    # ─── PR Curve Data ───
    try:
        precision, recall, thresholds_pr = precision_recall_curve(y_true, y_score)
        metrics["pr_curve"] = {
            "precision": [round(x, 4) for x in precision.tolist()],
            "recall": [round(x, 4) for x in recall.tolist()],
            "thresholds": [round(x, 6) for x in thresholds_pr.tolist()],
        }
    except Exception as e:
        print(f"⚠️  PR curve failed: {e}")
    
    # ─── Sensitivity at 90% Specificity ───
    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        target_fpr = 0.10  # 90% specificity = 10% FPR
        idx = np.argmin(np.abs(fpr - target_fpr))
        metrics["sensitivity_at_90_specificity"] = round(float(tpr[idx]), 4)
        print(f"Sensitivity @ 90% Specificity: {tpr[idx]:.4f}")
    except Exception as e:
        print(f"⚠️  Sensitivity computation failed: {e}")
    
    # ─── Optimal Threshold (Youden's J) ───
    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        metrics["optimal_threshold"] = {
            "threshold": round(float(thresholds[optimal_idx]), 6),
            "sensitivity": round(float(tpr[optimal_idx]), 4),
            "specificity": round(float(1 - fpr[optimal_idx]), 4),
            "youden_j": round(float(j_scores[optimal_idx]), 4),
        }
        print(f"Optimal threshold: {thresholds[optimal_idx]:.6f}")
        print(f"  Sensitivity: {tpr[optimal_idx]:.4f}")
        print(f"  Specificity: {1 - fpr[optimal_idx]:.4f}")
    except Exception as e:
        print(f"⚠️  Optimal threshold failed: {e}")
    
    # ─── Prediction Distribution ───
    try:
        pred_counts = df["prediction"].value_counts().to_dict()
        metrics["prediction_distribution"] = pred_counts
        print(f"\nPrediction distribution:")
        for pred, count in pred_counts.items():
            print(f"  {pred}: {count}")
    except:
        pass
    
    # ─── Gene-Stratified Performance ───
    # Extract gene from variant_id (chr-gene not available, skip for now)
    # This requires VEP annotation which isn't in the benchmark dataset
    
    return metrics


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("📊 HelixMind Benchmark Metrics")
    print("=" * 60)
    print()
    
    if not RESULTS_FILE.exists():
        print(f"❌ Results file not found: {RESULTS_FILE}")
        print("   Run benchmark_runner.py first.")
        return
    
    print(f"Loading results from: {RESULTS_FILE}")
    df = pd.read_csv(RESULTS_FILE)
    print(f"   {len(df)} rows loaded")
    print()
    
    metrics = compute_metrics(df)
    
    # Save metrics
    METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✅ Metrics saved to: {METRICS_FILE}")
    
    # Summary for paper
    print()
    print("=" * 60)
    print("📝 Paper-Ready Summary")
    print("=" * 60)
    if metrics.get("auroc"):
        print(f"AUROC: {metrics['auroc']:.4f}")
    if metrics.get("auprc"):
        print(f"AUPRC: {metrics['auprc']:.4f}")
    if metrics.get("sensitivity_at_90_specificity"):
        print(f"Sensitivity @ 90% Specificity: {metrics['sensitivity_at_90_specificity']:.4f}")
    if metrics.get("optimal_threshold"):
        opt = metrics["optimal_threshold"]
        print(f"Optimal Threshold: {opt['threshold']:.6f}")
        print(f"  Sensitivity: {opt['sensitivity']:.4f}")
        print(f"  Specificity: {opt['specificity']:.4f}")


if __name__ == "__main__":
    main()
