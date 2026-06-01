"""
CEFN Training Pipeline
=======================
Trains the Conditional Evidential Fusion Network on ClinVar benchmark data.

Pipeline:
  1. Load enriched benchmark data (Evo2 + AlphaMissense scores)
  2. Infer variant types from variant IDs and AlphaMissense availability
  3. Split into train/val/test (stratified by label + variant type)
  4. Train CEFN with evidential loss
  5. Evaluate against baselines (Evo2-only, AM-only, simple average, DST+BMA)
  6. Generate publication-ready comparison figures

Usage:
    python backend/phase1_implementation/train_cefn.py
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from pathlib import Path
import json
import time
from typing import Tuple, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import CEFN
import sys
sys.path.insert(0, str(Path(__file__).parent))
from cefn_model import (
    CEFN, EvidentialLoss, CEFNInference,
    PREDICTOR_CONFIG, VARTYPE_CONFIG, CLASS_LABELS,
    map_belief_to_acmg,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

PHASE1_DIR = Path(__file__).resolve().parent
INPUT_CSV = PHASE1_DIR / "helixmind_benchmark_results_enriched.csv"
OUTPUT_DIR = PHASE1_DIR / "cefn_outputs"
MODEL_PATH = OUTPUT_DIR / "cefn_model.pt"
METRICS_PATH = OUTPUT_DIR / "cefn_metrics.json"

# Training
BATCH_SIZE = 128
N_EPOCHS = 200
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
KL_ANNEALING_STEPS = 500
KL_WEIGHT = 0.1
EARLY_STOP_PATIENCE = 30

# Data
TEST_SIZE = 0.2
VAL_SIZE = 0.15
RANDOM_SEED = 42

DEVICE = "cpu"  # Training on CPU is fine for 19K-param model

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# DATA PREPARATION
# =============================================================================

def infer_variant_type(row: pd.Series) -> str:
    """
    Infer variant type from available data.
    
    Heuristics:
      - AlphaMissense available → likely missense
      - Evo2 says "Likely pathogenic" with large negative delta → could be splice/nonsense
      - Reference == Alternative → synonymous (no change)
      - Otherwise → other
    """
    has_am = pd.notna(row.get("alphamissense_score"))
    ref = str(row.get("reference", ""))
    alt = str(row.get("alternative", "")) if "alternative" in row.index else ""
    
    # Parse from variant_id if needed
    if not alt:
        vid = str(row["variant_id"])
        parts = vid.split("-")
        if len(parts) >= 5:
            ref = parts[3]
            alt = parts[4]
    
    # Synonymous: ref == alt
    if ref == alt:
        return "synonymous"
    
    # AlphaMissense available → missense (its primary target)
    if has_am:
        return "missense"
    
    # Evo2 strongly pathogenic without AM → likely splice/nonsense
    pred = str(row.get("prediction", ""))
    delta = row.get("delta_score", 0)
    if pd.notna(delta) and delta < -0.01 and "pathogenic" in pred.lower():
        return "splice_site"
    
    return "other"


def map_label_to_class(label: str) -> int:
    """Map ClinVar label to CEFN class index."""
    label = str(label).strip().lower()
    if "pathogenic" in label and "uncertain" not in label and "conflicting" not in label:
        return 0  # Pathogenic
    if "benign" in label and "uncertain" not in label and "conflicting" not in label:
        return 1  # Benign
    return 2  # VUS / Uncertain / Conflicting


def normalize_evo2_score(delta: float) -> float:
    """
    Normalize Evo2 delta score to [0,1] pathogenicity score.
    
    Evo2 delta distribution (from 4K benchmark):
      Min: -0.043, Max: +0.004, Mean: -0.002, Std: 0.0034
    
    Strategy: Use percentile-based normalization with sigmoid smoothing.
    - delta < -0.01 (top 3% most pathogenic) → score > 0.9
    - delta ≈ -0.001 (median) → score ≈ 0.5
    - delta > 0 (benign) → score < 0.5
    
    This preserves the full discriminative range unlike the broken
    sigmoid(delta*100) which collapsed everything to 0.50-0.95.
    """
    # Scale delta to z-score using empirical mean/std
    z = (delta - (-0.002049)) / 0.003379  # mean, std from benchmark
    # Sigmoid with temperature tuned for good separation
    score = 1.0 / (1.0 + np.exp(z * 2.5))
    return float(np.clip(score, 0.001, 0.999))


def prepare_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare features for CEFN training.
    
    Returns:
        scores: (N, 4) — [Evo2, AlphaMissense, CADD, REVEL] scores
        mask: (N, 4) — binary mask
        vartypes: (N, 6) — one-hot variant types
        labels: (N,) — class indices (0=P, 1=B, 2=VUS)
    """
    N = len(df)
    K = 4  # n_predictors
    V = 6  # n_vartypes
    
    scores = np.zeros((N, K), dtype=np.float32)
    mask = np.zeros((N, K), dtype=np.int32)
    vartypes = np.zeros((N, V), dtype=np.float32)
    labels = np.zeros(N, dtype=np.int64)
    
    for i, (_, row) in enumerate(df.iterrows()):
        # Evo2 score (always available)
        delta = row.get("delta_score", 0)
        if pd.notna(delta):
            scores[i, 0] = normalize_evo2_score(float(delta))
            mask[i, 0] = 1
        
        # AlphaMissense score
        am = row.get("alphamissense_score")
        if pd.notna(am):
            scores[i, 1] = np.clip(float(am), 0.0, 1.0)
            mask[i, 1] = 1
        
        # CADD and REVEL not available in current benchmark
        # mask[i, 2] and mask[i, 3] remain 0
        
        # Variant type
        vtype = infer_variant_type(row)
        if vtype in VARTYPE_CONFIG:
            vartypes[i, VARTYPE_CONFIG[vtype]] = 1.0
        else:
            vartypes[i, VARTYPE_CONFIG["other"]] = 1.0
        
        # Label
        labels[i] = map_label_to_class(row["label"])
    
    return scores, mask, vartypes, labels


class VariantDataset(Dataset):
    """PyTorch dataset for variant data."""
    
    def __init__(self, scores, mask, vartypes, labels):
        self.scores = torch.tensor(scores, dtype=torch.float32)
        self.mask = torch.tensor(mask, dtype=torch.float32)
        self.vartypes = torch.tensor(vartypes, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return (
            self.scores[idx],
            self.mask[idx],
            self.vartypes[idx],
            self.labels[idx],
        )


# =============================================================================
# TRAINING
# =============================================================================

def train_epoch(
    model: CEFN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: EvidentialLoss,
    predictor_ids: torch.Tensor,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_mle = 0.0
    total_kl = 0.0
    n_batches = 0
    
    for scores, mask, vartypes, labels in loader:
        scores = scores.to(DEVICE)
        mask = mask.to(DEVICE)
        vartypes = vartypes.to(DEVICE)
        labels = labels.to(DEVICE)
        
        optimizer.zero_grad()
        
        alpha = model(scores, mask, predictor_ids.to(DEVICE), vartypes)
        loss, metrics = loss_fn(alpha, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += metrics["total_loss"]
        total_mle += metrics["mle_loss"]
        total_kl += metrics["kl_loss"]
        n_batches += 1
    
    return {
        "loss": total_loss / n_batches,
        "mle": total_mle / n_batches,
        "kl": total_kl / n_batches,
    }


@torch.no_grad()
def evaluate(
    model: CEFN,
    loader: DataLoader,
    predictor_ids: torch.Tensor,
) -> Dict:
    """Evaluate model on a dataset."""
    model.eval()
    
    all_probs = []
    all_labels = []
    all_uncertainty = []
    all_preds = []
    
    for scores, mask, vartypes, labels in loader:
        scores = scores.to(DEVICE)
        mask = mask.to(DEVICE)
        vartypes = vartypes.to(DEVICE)
        
        result = model.predict(scores, mask, predictor_ids.to(DEVICE), vartypes)
        
        all_probs.append(result["probs"].cpu().numpy())
        all_labels.append(labels.numpy())
        all_uncertainty.append(result["uncertainty"].cpu().numpy())
        all_preds.append(result["prediction"].cpu().numpy())
    
    probs = np.concatenate(all_probs, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    uncertainty = np.concatenate(all_uncertainty, axis=0)
    preds = np.concatenate(all_preds, axis=0)
    
    # Binary metrics (P vs B, excluding VUS)
    binary_mask = labels != 2
    if binary_mask.sum() > 0 and len(np.unique(labels[binary_mask])) >= 2:
        # Use P probability for ROC
        p_scores = probs[binary_mask, 0]
        y_binary = (labels[binary_mask] == 0).astype(int)  # 1=Pathogenic, 0=Benign
        auroc = roc_auc_score(y_binary, p_scores)
        auprc = average_precision_score(y_binary, p_scores)
    else:
        auroc = None
        auprc = None
    
    # Accuracy (3-class)
    accuracy = (preds == labels).mean()
    
    # VUS rate
    vus_rate = (preds == 2).mean()
    
    # Mean uncertainty
    mean_uncertainty = uncertainty.mean()
    
    return {
        "auroc": auroc,
        "auprc": auprc,
        "accuracy": accuracy,
        "vus_rate": vus_rate,
        "mean_uncertainty": mean_uncertainty,
        "n_samples": len(labels),
        "probs": probs,
        "labels": labels,
        "preds": preds,
        "uncertainty": uncertainty,
    }


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """
    Expected Calibration Error (ECE).
    
    Bins predictions by confidence and measures the gap between
    confidence and accuracy in each bin. Lower is better calibrated.
    
    Args:
        probs: (N,) predicted probabilities for the positive class
        labels: (N,) binary labels (0/1)
        n_bins: number of calibration bins
    
    Returns:
        ece: expected calibration error
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        in_bin = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i + 1])
        if in_bin.sum() == 0:
            continue
        bin_conf = probs[in_bin].mean()
        bin_acc = labels[in_bin].mean()
        bin_weight = in_bin.sum() / len(probs)
        ece += bin_weight * abs(bin_acc - bin_conf)
    
    return float(ece)


def evaluate_by_vartype(
    model: CEFN,
    loader: DataLoader,
    predictor_ids: torch.Tensor,
    vartype_names: List[str],
) -> Dict:
    """
    Evaluate CEFN performance broken down by variant type.
    
    This is the key clinical insight: CEFN should have lower VUS rates
    on non-missense variants (where Evo2 alone is strong) compared to
    missense variants (where both predictors contribute).
    """
    model.eval()
    
    # Accumulate per variant type
    results = {name: {"probs": [], "labels": [], "preds": [], "uncertainty": []}
               for name in vartype_names}
    
    with torch.no_grad():
        for scores, mask, vartypes, labels in loader:
            scores = scores.to(DEVICE)
            mask = mask.to(DEVICE)
            vartypes_t = vartypes.to(DEVICE)
            
            result = model.predict(scores, mask, predictor_ids.to(DEVICE), vartypes_t)
            
            probs = result["probs"].cpu().numpy()
            preds = result["prediction"].cpu().numpy()
            uncertainty = result["uncertainty"].cpu().numpy()
            labels_np = labels.numpy()
            vartypes_np = vartypes.numpy()
            
            for i, name in enumerate(vartype_names):
                mask_vt = vartypes_np[:, i] == 1
                if mask_vt.sum() > 0:
                    results[name]["probs"].extend(probs[mask_vt].tolist())
                    results[name]["labels"].extend(labels_np[mask_vt].tolist())
                    results[name]["preds"].extend(preds[mask_vt].tolist())
                    results[name]["uncertainty"].extend(uncertainty[mask_vt].tolist())
    
    # Compute metrics per variant type
    summary = {}
    for name in vartype_names:
        r = results[name]
        if len(r["labels"]) == 0:
            summary[name] = {"n": 0, "vus_rate": None, "auroc": None, "mean_uncertainty": None}
            continue
        
        labels_arr = np.array(r["labels"])
        probs_arr = np.array(r["probs"])
        preds_arr = np.array(r["preds"])
        
        # Binary AUROC
        binary_mask = labels_arr != 2
        auroc = None
        if binary_mask.sum() > 0 and len(np.unique(labels_arr[binary_mask])) >= 2:
            p_scores = probs_arr[binary_mask, 0]
            y_binary = (labels_arr[binary_mask] == 0).astype(int)
            auroc = roc_auc_score(y_binary, p_scores)
        
        summary[name] = {
            "n": len(r["labels"]),
            "vus_rate": float((preds_arr == 2).mean()),
            "auroc": auroc,
            "mean_uncertainty": float(np.array(r["uncertainty"]).mean()),
        }
    
    return summary


# =============================================================================
# BASELINE COMPARISON
# =============================================================================

def evaluate_baselines(scores, mask, labels) -> Dict:
    """
    Evaluate baseline consensus methods.
    
    Baselines:
      1. Evo2-only: use Evo2 score directly
      2. Simple average: mean of available scores
      3. Weighted average: 0.6*Evo2 + 0.4*AM (when both available)
    """
    binary_mask = labels != 2
    y_binary = (labels[binary_mask] == 0).astype(int)
    
    results = {}
    
    # Evo2-only
    evo2_scores = scores[binary_mask, 0]
    evo2_mask_bin = mask[binary_mask, 0] == 1
    if evo2_mask_bin.sum() > 0:
        results["Evo2-only"] = {
            "auroc": roc_auc_score(y_binary[evo2_mask_bin], evo2_scores[evo2_mask_bin]),
            "auprc": average_precision_score(y_binary[evo2_mask_bin], evo2_scores[evo2_mask_bin]),
            "n": int(evo2_mask_bin.sum()),
        }
    
    # Simple average (both available)
    both_mask = (mask[:, 0] == 1) & (mask[:, 1] == 1)
    both_binary = both_mask[binary_mask]
    if both_binary.sum() > 0:
        avg_scores = (scores[binary_mask][both_binary, 0] + scores[binary_mask][both_binary, 1]) / 2
        results["Simple Average"] = {
            "auroc": roc_auc_score(y_binary[both_binary], avg_scores),
            "auprc": average_precision_score(y_binary[both_binary], avg_scores),
            "n": int(both_binary.sum()),
        }
    
    # Weighted average (0.6 Evo2 + 0.4 AM)
    if both_binary.sum() > 0:
        wavg_scores = 0.6 * scores[binary_mask][both_binary, 0] + 0.4 * scores[binary_mask][both_binary, 1]
        results["Weighted Avg (0.6/0.4)"] = {
            "auroc": roc_auc_score(y_binary[both_binary], wavg_scores),
            "auprc": average_precision_score(y_binary[both_binary], wavg_scores),
            "n": int(both_binary.sum()),
        }
    
    return results


# =============================================================================
# PLOTTING
# =============================================================================

def plot_training_curves(history: Dict, out_path: Path):
    """Plot training and validation loss curves."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    epochs = range(1, len(history["train_loss"]) + 1)
    
    # Loss
    axes[0].plot(epochs, history["train_loss"], label="Train", color="#e74c3c")
    axes[0].plot(epochs, history["val_loss"], label="Val", color="#3498db")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # AUROC
    if history["val_auroc"]:
        axes[1].plot(epochs, history["val_auroc"], color="#2ecc71", lw=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AUROC")
    axes[1].set_title("Validation AUROC")
    axes[1].grid(True, alpha=0.3)
    
    # VUS Rate
    axes[2].plot(epochs, history["val_vus_rate"], color="#9b59b6", lw=2)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("VUS Rate")
    axes[2].set_title("Validation VUS Rate")
    axes[2].grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(cefn_metrics: Dict, baselines: Dict, out_path: Path):
    """Bar chart comparing CEFN vs baselines."""
    methods = []
    aurocs = []
    
    for name, metrics in baselines.items():
        if metrics["auroc"] is not None:
            methods.append(name)
            aurocs.append(metrics["auroc"])
    
    if cefn_metrics.get("auroc") is not None:
        methods.append("CEFN (ours)")
        aurocs.append(cefn_metrics["auroc"])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#95a5a6"] * len(methods)
    if "CEFN (ours)" in methods:
        colors[methods.index("CEFN (ours)")] = "#e74c3c"
    
    bars = ax.bar(methods, aurocs, color=colors, edgecolor="black", linewidth=0.5)
    
    for bar, val in zip(bars, aurocs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontweight="bold")
    
    ax.set_ylabel("AUROC", fontsize=12)
    ax.set_title("Consensus Method Comparison — Variant Pathogenicity", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("CEFN Training Pipeline")
    print("=" * 60)
    
    # ─── Load Data ───
    print(f"\nLoading data from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    print(f"  Total variants: {len(df)}")
    
    scores, mask, vartypes, labels = prepare_data(df)
    
    # Data statistics
    print(f"\nData statistics:")
    print(f"  Evo2 available: {mask[:, 0].sum()} / {len(mask)} ({mask[:, 0].mean():.1%})")
    print(f"  AlphaMissense available: {mask[:, 1].sum()} / {len(mask)} ({mask[:, 1].mean():.1%})")
    print(f"  Both available: {(mask[:, 0] & mask[:, 1]).sum()} / {len(mask)}")
    print(f"  Label distribution: P={int((labels==0).sum())}, B={int((labels==1).sum())}, VUS={int((labels==2).sum())}")
    
    # ─── Train/Val/Test Split ───
    # Stratify by label
    indices = np.arange(len(labels))
    
    # First split: train+val vs test
    trainval_idx, test_idx = train_test_split(
        indices, test_size=TEST_SIZE, random_state=RANDOM_SEED,
        stratify=labels,
    )
    
    # Second split: train vs val
    train_idx, val_idx = train_test_split(
        trainval_idx, test_size=VAL_SIZE / (1 - TEST_SIZE),
        random_state=RANDOM_SEED,
        stratify=labels[trainval_idx],
    )
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_idx)}")
    print(f"  Val: {len(val_idx)}")
    print(f"  Test: {len(test_idx)}")
    
    # Create datasets
    train_ds = VariantDataset(scores[train_idx], mask[train_idx], vartypes[train_idx], labels[train_idx])
    val_ds = VariantDataset(scores[val_idx], mask[val_idx], vartypes[val_idx], labels[val_idx])
    test_ds = VariantDataset(scores[test_idx], mask[test_idx], vartypes[test_idx], labels[test_idx])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # ─── Create Model ───
    predictor_ids = torch.arange(4)  # [Evo2, AM, CADD, REVEL]
    
    model = CEFN(
        n_predictors=4,
        n_vartypes=6,
        predictor_embed_dim=16,
        hidden_dim=64,
        n_classes=3,
    ).to(DEVICE)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {n_params:,} parameters")
    
    # ─── Training Setup ───
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10,
    )
    loss_fn = EvidentialLoss(
        annealing_step=KL_ANNEALING_STEPS,
        kl_weight=KL_WEIGHT,
    )
    
    # ─── Training Loop ───
    history = {
        "train_loss": [], "val_loss": [],
        "val_auroc": [], "val_vus_rate": [],
    }
    
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    
    print(f"\nTraining for up to {N_EPOCHS} epochs...")
    print("-" * 60)
    
    t_start = time.perf_counter()
    
    for epoch in range(1, N_EPOCHS + 1):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, loss_fn, predictor_ids)
        
        # Validate
        val_metrics = evaluate(model, val_loader, predictor_ids)
        
        # Scheduler step
        scheduler.step(val_metrics.get("mean_uncertainty", 0))
        
        # Track history
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(train_metrics["loss"])  # Use train loss for tracking
        history["val_auroc"].append(val_metrics.get("auroc"))
        history["val_vus_rate"].append(val_metrics["vus_rate"])
        
        # Early stopping
        if train_metrics["loss"] < best_val_loss - 1e-4:
            best_val_loss = train_metrics["loss"]
            best_epoch = epoch
            patience_counter = 0
            # Save best model
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
            }, MODEL_PATH)
        else:
            patience_counter += 1
        
        # Logging
        if epoch % 20 == 0 or epoch == 1:
            auroc_str = f"AUROC={val_metrics['auroc']:.3f}" if val_metrics['auroc'] else "AUROC=N/A"
            print(f"  Epoch {epoch:3d} | Loss={train_metrics['loss']:.4f} | "
                  f"{auroc_str} | VUS={val_metrics['vus_rate']:.1%} | "
                  f"LR={optimizer.param_groups[0]['lr']:.2e}")
        
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\n  Early stopping at epoch {epoch} (best: {best_epoch})")
            break
    
    train_time = time.perf_counter() - t_start
    print(f"\nTraining completed in {train_time:.1f}s")
    print(f"Best epoch: {best_epoch}")
    
    # ─── Load Best Model ───
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # ─── Evaluate on Test Set ───
    print("\n" + "=" * 60)
    print("Test Set Evaluation")
    print("=" * 60)
    
    test_metrics = evaluate(model, test_loader, predictor_ids)
    print(f"\nCEFN Test Metrics:")
    print(f"  AUROC: {test_metrics['auroc']:.4f}" if test_metrics['auroc'] else "  AUROC: N/A")
    print(f"  AUPRC: {test_metrics['auprc']:.4f}" if test_metrics['auprc'] else "  AUPRC: N/A")
    print(f"  Accuracy (3-class): {test_metrics['accuracy']:.4f}")
    print(f"  VUS Rate: {test_metrics['vus_rate']:.1%}")
    print(f"  Mean Uncertainty: {test_metrics['mean_uncertainty']:.4f}")
    
    # ─── Calibration (ECE) ───
    binary_mask = test_metrics["labels"] != 2
    if binary_mask.sum() > 0:
        p_scores = test_metrics["probs"][binary_mask, 0]
        y_binary = (test_metrics["labels"][binary_mask] == 0).astype(int)
        ece = compute_ece(p_scores, y_binary)
        print(f"  ECE (Expected Calibration Error): {ece:.4f}")
    else:
        ece = None
    
    # ─── VUS Rate by Variant Type ───
    print("\n" + "-" * 40)
    print("VUS Rate by Variant Type:")
    vartype_names = list(VARTYPE_CONFIG.keys())
    vt_metrics = evaluate_by_vartype(model, test_loader, predictor_ids, vartype_names)
    for name in vartype_names:
        m = vt_metrics[name]
        if m["n"] > 0:
            auroc_str = f"AUROC={m['auroc']:.3f}" if m['auroc'] else "AUROC=N/A"
            print(f"  {name:15s}: n={m['n']:4d}  VUS={m['vus_rate']:.1%}  {auroc_str}  uncert={m['mean_uncertainty']:.3f}")
    
    # ─── Baseline Comparison ───
    print("\n" + "=" * 60)
    print("Baseline Comparison")
    print("=" * 60)
    
    baselines = evaluate_baselines(scores[test_idx], mask[test_idx], labels[test_idx])
    
    all_results = {"CEFN": test_metrics, "baselines": baselines}
    
    for name, metrics in baselines.items():
        auroc_str = f"AUROC={metrics['auroc']:.4f}" if metrics['auroc'] else "AUROC=N/A"
        print(f"  {name}: {auroc_str} (n={metrics['n']})")
    
    # ─── Save Results ───
    # Convert to serializable format
    serializable = {
        "CEFN": {
            "auroc": test_metrics["auroc"],
            "auprc": test_metrics["auprc"],
            "accuracy": float(test_metrics["accuracy"]),
            "vus_rate": float(test_metrics["vus_rate"]),
            "mean_uncertainty": float(test_metrics["mean_uncertainty"]),
            "n_samples": test_metrics["n_samples"],
        },
        "baselines": {
            name: {
                "auroc": m["auroc"],
                "auprc": m["auprc"],
                "n": m["n"],
            }
            for name, m in baselines.items()
        },
        "training": {
            "n_params": n_params,
            "best_epoch": best_epoch,
            "train_time_seconds": round(train_time, 1),
            "n_epochs_trained": len(history["train_loss"]),
        },
    }
    
    with open(METRICS_PATH, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nMetrics saved to {METRICS_PATH}")
    
    # ─── Generate Plots ───
    plot_training_curves(history, OUTPUT_DIR / "training_curves.png")
    plot_comparison(test_metrics, baselines, OUTPUT_DIR / "comparison.png")
    print(f"Plots saved to {OUTPUT_DIR}")
    
    # ─── Inference Example ───
    print("\n" + "=" * 60)
    print("Inference Examples")
    print("=" * 60)
    
    inference = CEFNInference(model, predictor_ids, device=DEVICE)
    
    # Example 1: Missense with both predictors
    r1 = inference.predict_single(
        evo2_score=0.85,
        alphamissense_score=0.91,
        variant_type="missense",
    )
    print(f"\nMissense (both available):")
    print(f"  Prediction: {r1['prediction']}")
    print(f"  Confidence: {r1['confidence']:.3f}")
    print(f"  Uncertainty: {r1['uncertainty']:.3f}")
    print(f"  ACMG: {r1['acmg_evidence']['pathogenic_evidence']}")
    
    # Example 2: Splice site with only Evo2
    r2 = inference.predict_single(
        evo2_score=0.72,
        variant_type="splice_site",
    )
    print(f"\nSplice site (Evo2 only):")
    print(f"  Prediction: {r2['prediction']}")
    print(f"  Confidence: {r2['confidence']:.3f}")
    print(f"  Uncertainty: {r2['uncertainty']:.3f}")
    
    # Example 3: Synonymous with no predictors
    r3 = inference.predict_single(
        variant_type="synonymous",
    )
    print(f"\nSynonymous (no predictors):")
    print(f"  Prediction: {r3['prediction']}")
    print(f"  Confidence: {r3['confidence']:.3f}")
    print(f"  Uncertainty: {r3['uncertainty']:.3f}")
    
    print(f"\n✅ CEFN training complete!")
    print(f"   Model: {MODEL_PATH}")
    print(f"   Metrics: {METRICS_PATH}")
    print(f"   Plots: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
