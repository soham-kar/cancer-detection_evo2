"""
CEFN v2 — Full Production Pipeline
====================================
Binary-only training + Platt Scaling + Predictor-specific missing tokens
+ Fixed α_VUS + Chromosome-wise split + DST+BMA baseline + ECE + VUS-by-vartype

Adapted for our actual data format (2 predictors: Evo2 + AlphaMissense).

Usage:
    python backend/phase1_implementation/train_cefn_v2.py
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from pathlib import Path
import json
import time
import joblib
from typing import Dict, Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import shared components
from cefn_v2_common import (
    CEFN_v2, DeepSetEncoder,
    map_label, infer_variant_type,
    compute_beliefs, predict_class,
    compute_ece,
    evidential_loss_binary,
    DST_BMA_Consensus,
    evaluate_full,
    PREDICTOR_COLS, N_PREDICTORS,
    VARTYPE_COLS, N_VARTYPES,
    DEVICE,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

PHASE1_DIR = Path(__file__).resolve().parent
INPUT_CSV = PHASE1_DIR / "helixmind_benchmark_results_enriched.csv"
OUTPUT_DIR = PHASE1_DIR / "cefn_v2_outputs"
MODEL_PATH = OUTPUT_DIR / "cefn_v2_model.pt"
METRICS_PATH = OUTPUT_DIR / "cefn_v2_metrics.json"
SCALER_PATH = OUTPUT_DIR / "platt_scalers.joblib"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Training
BATCH_SIZE = 128
N_EPOCHS = 100
LEARNING_RATE = 1e-3
LAMBDA_REG = 0.05
EARLY_STOP_PATIENCE = 20

# Predictor AUROCs (from our metrics.json)
PREDICTOR_AUROCS = {
    "evo2_score": 0.975,
    "alphamissense_score": 0.975,
}

# =============================================================================
# 1. DATA LOADING & ADAPTATION
# =============================================================================

def load_and_adapt_data(csv_path: str) -> pd.DataFrame:
    """
    Load our CSV and convert to the format CEFN v2 expects.
    
    Adds columns:
      - label_int: 0=P, 1=B, 2=VUS
      - evo2_score: raw delta (will be Platt-scaled later)
      - alphamissense_score: already 0-1
      - chromosome: parsed from variant_id
      - chr_int: integer chromosome for splitting
      - One-hot variant type columns
    """
    df = pd.read_csv(csv_path)
    
    # Map labels
    df["label_int"] = df["label"].apply(map_label)
    
    # Evo2: use raw delta (Platt scaling will calibrate it)
    df["evo2_score"] = df["delta_score"].copy()
    
    # AlphaMissense: already calibrated 0-1, just ensure float
    df["alphamissense_score"] = df["alphamissense_score"].astype(float)
    
    # Parse chromosome
    df["chromosome"] = df["variant_id"].apply(
        lambda x: x.split("-")[0].replace("chr", "")
    )
    chr_map = {str(i): i for i in range(1, 23)}
    chr_map.update({"X": 23, "Y": 24})
    df["chr_int"] = df["chromosome"].map(chr_map).fillna(0).astype(int)
    
    # Infer variant types
    for vt in VARTYPE_COLS:
        df[f"vt_{vt}"] = 0
    
    for i, (_, row) in enumerate(df.iterrows()):
        vt = infer_variant_type(row)
        col = f"vt_{vt}"
        if col in df.columns:
            df.at[i, col] = 1
    
    return df


def chromosome_wise_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split by chromosome: chr1-16 → train, chr17-18 → val, chr19-22 → test.
    """
    train_df = df[df["chr_int"].between(1, 16)]
    val_df = df[df["chr_int"].between(17, 18)]
    test_df = df[df["chr_int"].between(19, 22)]
    
    # Fallback: if any split is too small, use random split
    if len(train_df) < 100 or len(val_df) < 50 or len(test_df) < 50:
        print("⚠️  Chromosome-wise split produced tiny sets. Falling back to random split.")
        from sklearn.model_selection import train_test_split
        train_df, temp = train_test_split(df, test_size=0.3, random_state=42, stratify=df["label_int"])
        val_df, test_df = train_test_split(temp, test_size=0.5, random_state=42, stratify=temp["label_int"])
    
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    return train_df, val_df, test_df


# =============================================================================
# 2. PLATT SCALING
# =============================================================================

def fit_platt_scalers(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """
    Fit Platt scaling (logistic calibration) for Evo2 on binary training data.
    AlphaMissense is already calibrated 0-1, so we skip it.
    
    Returns transformed DataFrames and the fitted scalers dict.
    """
    binary_train = train_df[train_df["label_int"] != 2].copy()
    scalers = {}
    
    # Platt scale Evo2
    col = "evo2_score"
    mask = binary_train[col].notna()
    if mask.sum() >= 20:
        X = binary_train.loc[mask, col].values.reshape(-1, 1)
        y = (binary_train.loc[mask, "label_int"] == 0).astype(int)  # 1=pathogenic
        lr = LogisticRegression(penalty=None, max_iter=1000)
        lr.fit(X, y)
        scalers[col] = lr
        print(f"  Platt scaler for {col}: w={lr.coef_[0][0]:.4f}, b={lr.intercept_[0]:.4f}")
    
    # AlphaMissense: already 0-1, no scaling needed
    # But we still clip to [0,1] for safety
    
    def transform(df_in: pd.DataFrame) -> pd.DataFrame:
        df_out = df_in.copy()
        for col, lr in scalers.items():
            mask = df_out[col].notna()
            if mask.sum() == 0:
                continue
            X = df_out.loc[mask, col].values.reshape(-1, 1)
            df_out.loc[mask, col] = lr.predict_proba(X)[:, 1]
        # Clip AlphaMissense
        am_mask = df_out["alphamissense_score"].notna()
        df_out.loc[am_mask, "alphamissense_score"] = df_out.loc[am_mask, "alphamissense_score"].clip(0, 1)
        return df_out
    
    # Save scalers for external validation
    joblib.dump(scalers, SCALER_PATH)
    print(f"  Platt scalers saved to {SCALER_PATH}")

    return transform(train_df), transform(val_df), transform(test_df), scalers


# (Model, metrics, baselines imported from cefn_v2_common.py)

# =============================================================================
# 8. TRAINING
# =============================================================================

def train_cefn_v2(
    model: CEFN_v2,
    train_loader: DataLoader,
    val_loader: DataLoader,
    predictor_ids: torch.Tensor,
    epochs: int = 100,
    lr: float = 1e-3,
    lambda_reg: float = 0.05,
    patience: int = 20,
) -> Dict:
    """Train CEFN v2 with binary-only evidential loss."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10,
    )
    
    history = {"train_loss": [], "val_auroc": [], "val_vus_rate": [], "val_ece": []}
    best_val_loss = float("inf")
    best_val_ece = float("inf")
    best_epoch = 0
    best_ece_epoch = 0
    patience_counter = 0
    
    for epoch in range(epochs):
        # ─── Train ───
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            scores, mask, vartype, labels = batch
            scores = scores.to(DEVICE)
            mask = mask.to(DEVICE)
            vartype = vartype.to(DEVICE)
            labels = labels.to(DEVICE)
            
            aP, aB, aV = model(scores, mask, predictor_ids, vartype)
            annealing = min(1.0, epoch / (epochs * 0.5))
            loss = evidential_loss_binary(aP, aB, aV, labels, annealing, lambda_reg)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)
        
        # ─── Validate ───
        model.eval()
        val_loss = 0.0
        y_true_bin, y_prob_bin = [], []
        all_preds = []
        
        with torch.no_grad():
            for batch in val_loader:
                scores, mask, vartype, labels = batch
                scores = scores.to(DEVICE)
                mask = mask.to(DEVICE)
                vartype = vartype.to(DEVICE)
                labels = labels.to(DEVICE)
                
                aP, aB, aV = model(scores, mask, predictor_ids, vartype)
                loss = evidential_loss_binary(aP, aB, aV, labels, 1.0, lambda_reg)
                val_loss += loss.item()
                
                preds, _, _ = predict_class(aP, aB, aV, threshold=0.5)
                all_preds.extend(preds.cpu().numpy())
                
                bin_mask = labels < 2
                if bin_mask.any():
                    p_path = aP[bin_mask] / (aP[bin_mask] + aB[bin_mask] + aV[bin_mask])
                    y_true_bin.extend((labels[bin_mask] == 0).int().cpu().numpy())
                    y_prob_bin.extend(p_path.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        val_auroc = roc_auc_score(y_true_bin, y_prob_bin) if len(set(y_true_bin)) > 1 else 0.5
        val_vus = np.mean(np.array(all_preds) == 2)
        val_ece = compute_ece(np.array(y_true_bin), np.array(y_prob_bin)) if len(y_true_bin) > 0 else 1.0

        history["val_auroc"].append(val_auroc)
        history["val_vus_rate"].append(val_vus)
        history["val_ece"].append(val_ece)

        scheduler.step(avg_val_loss)

        if epoch % 10 == 0 or epoch < 5:
            print(f"  Epoch {epoch+1:3d} | Loss={avg_train_loss:.4f} | "
                  f"Val AUROC={val_auroc:.4f} | ECE={val_ece:.4f} | VUS={val_vus:.1%} | "
                  f"LR={optimizer.param_groups[0]['lr']:.2e}")

        # Save best-loss checkpoint (for training continuity)
        if avg_train_loss < best_val_loss - 1e-4:
            best_val_loss = avg_train_loss
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save({
                "epoch": best_epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, MODEL_PATH)

        # Save best-ECE checkpoint (for publication — best calibration)
        if val_ece < best_val_ece - 1e-5:
            best_val_ece = val_ece
            best_ece_epoch = epoch + 1
            torch.save({
                "epoch": best_ece_epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_ece": val_ece,
                "val_auroc": val_auroc,
            }, OUTPUT_DIR / "cefn_v2_model_best_ece.pt")
        if patience_counter >= patience:
            print(f"\n  Early stopping at epoch {epoch+1} (best: {best_epoch})")
            break
    
    return history


# (evaluate_full imported from cefn_v2_common.py)

# =============================================================================
# 10. PLOTTING
# =============================================================================

def plot_results(history: Dict, cefn_metrics: Dict, baselines: Dict, out_dir: Path):
    """Generate publication-ready plots."""
    
    # ─── Training curves ───
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs = range(1, len(history["train_loss"]) + 1)
    
    axes[0].plot(epochs, history["train_loss"], color="#e74c3c", lw=2)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss"); axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(epochs, history["val_auroc"], color="#2ecc71", lw=2)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("AUROC")
    axes[1].set_title("Validation AUROC"); axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(epochs, history["val_vus_rate"], color="#9b59b6", lw=2)
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("VUS Rate")
    axes[2].set_title("Validation VUS Rate"); axes[2].grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    # ─── Comparison bar chart ───
    methods = []
    aurocs = []
    for name, m in baselines.items():
        if m.get("auroc") is not None:
            methods.append(name)
            aurocs.append(m["auroc"])
    if cefn_metrics.get("auroc") is not None:
        methods.append("CEFN v2 (ours)")
        aurocs.append(cefn_metrics["auroc"])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#95a5a6"] * len(methods)
    if "CEFN v2 (ours)" in methods:
        colors[methods.index("CEFN v2 (ours)")] = "#e74c3c"
    
    bars = ax.bar(methods, aurocs, color=colors, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, aurocs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontweight="bold")
    ax.set_ylabel("AUROC"); ax.set_ylim(0, 1.05)
    ax.set_title("Consensus Method Comparison", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    # ─── VUS by variant type ───
    vt_data = cefn_metrics.get("vt_summary", {})
    vts = [vt for vt in VARTYPE_COLS if vt_data.get(vt, {}).get("n", 0) > 0]
    vus_rates = [vt_data[vt]["vus_rate"] for vt in vts]
    ns = [vt_data[vt]["n"] for vt in vts]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(vts, vus_rates, color="#3498db", edgecolor="black", linewidth=0.5)
    for bar, rate, n in zip(bars, vus_rates, ns):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{rate:.0%}\n(n={n})", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("VUS Rate"); ax.set_ylim(0, 1.1)
    ax.set_title("VUS Rate by Variant Type — CEFN v2", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "vus_by_vartype.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    print(f"Plots saved to {out_dir}")


# =============================================================================
# 11. MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("CEFN v2 — Full Production Pipeline")
    print("=" * 60)
    
    # ─── 1. Load & Adapt Data ───
    print("\n[1/7] Loading data...")
    df = load_and_adapt_data(str(INPUT_CSV))
    print(f"  Total variants: {len(df)}")
    print(f"  Labels: P={int((df['label_int']==0).sum())}, "
          f"B={int((df['label_int']==1).sum())}, "
          f"VUS={int((df['label_int']==2).sum())}")
    
    # ─── 2. Chromosome-wise Split ───
    print("\n[2/7] Chromosome-wise split...")
    train_df, val_df, test_df = chromosome_wise_split(df)
    
    # ─── 3. Platt Scaling ───
    print("\n[3/7] Platt scaling...")
    train_df, val_df, test_df, scalers = fit_platt_scalers(train_df, val_df, test_df)
    
    # ─── 4. Prepare Tensors ───
    print("\n[4/7] Preparing tensors...")
    predictor_ids = torch.arange(N_PREDICTORS)
    
    vt_cols = [f"vt_{vt}" for vt in VARTYPE_COLS]
    
    def df_to_tensors(df_in):
        scores = df_in[PREDICTOR_COLS].fillna(0).values.astype(np.float32)
        mask = df_in[PREDICTOR_COLS].notna().values.astype(np.float32)
        vartype = df_in[vt_cols].values.astype(np.float32)
        labels = df_in["label_int"].values.astype(np.int64)
        return (
            torch.tensor(scores),
            torch.tensor(mask),
            torch.tensor(vartype),
            torch.tensor(labels),
        )
    
    train_s, train_m, train_vt, train_l = df_to_tensors(train_df)
    val_s, val_m, val_vt, val_l = df_to_tensors(val_df)
    test_s, test_m, test_vt, test_l = df_to_tensors(test_df)
    
    train_ds = TensorDataset(train_s, train_m, train_vt, train_l)
    val_ds = TensorDataset(val_s, val_m, val_vt, val_l)
    test_ds = TensorDataset(test_s, test_m, test_vt, test_l)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"  Train batches: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")
    
    # ─── 5. Train CEFN v2 ───
    print("\n[5/7] Training CEFN v2...")
    model = CEFN_v2(
        n_predictors=N_PREDICTORS,
        n_vartypes=N_VARTYPES,
        embed_dim=16,
        hidden_dim=128,
    ).to(DEVICE)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    
    t_start = time.perf_counter()
    history = train_cefn_v2(
        model, train_loader, val_loader, predictor_ids,
        epochs=N_EPOCHS, lr=LEARNING_RATE,
        lambda_reg=LAMBDA_REG, patience=EARLY_STOP_PATIENCE,
    )
    train_time = time.perf_counter() - t_start
    
# Load best-ECE model for evaluation (publication-quality calibration)
    BEST_ECE_PATH = OUTPUT_DIR / "cefn_v2_model_best_ece.pt"
    if BEST_ECE_PATH.exists():
        checkpoint = torch.load(BEST_ECE_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"  Training: {train_time:.1f}s, best-ECE epoch: {checkpoint['epoch']} (val ECE={checkpoint.get('val_ece', 'N/A')})")
    else:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"  Training: {train_time:.1f}s, best-loss epoch: {checkpoint['epoch']}")
    
    # ─── 6. Evaluate ───
    print("\n[6/7] Evaluating...")
    cefn_metrics = evaluate_full(model, test_loader, predictor_ids)

    print(f"\n  CEFN v2 Results:")
    print(f"    AUROC: {cefn_metrics['auroc']:.4f}" if cefn_metrics['auroc'] else "    AUROC: N/A")
    print(f"    AUPRC: {cefn_metrics['auprc']:.4f}" if cefn_metrics['auprc'] else "    AUPRC: N/A")
    print(f"    ECE:   {cefn_metrics['ece']:.4f}")
    print(f"    VUS Rate: {cefn_metrics['vus_rate']:.1%}")
    print(f"    Mean Uncertainty: {cefn_metrics['mean_uncertainty']:.4f}")

    print(f"\n  VUS Rate by Variant Type:")
    for vt in VARTYPE_COLS:
        m = cefn_metrics["vt_summary"][vt]
        if m["n"] > 0:
            auroc_str = f"AUROC={m['auroc']:.3f}" if m['auroc'] else "AUROC=N/A"
            print(f"    {vt:15s}: n={m['n']:4d}  VUS={m['vus_rate']:.1%}  {auroc_str}  uncert={m['mean_uncertainty']:.3f}")

    # ─── 7. Baselines ───
    print("\n[7/7] Baselines...")
    
    # Evo2-only (on test set, binary only)
    test_binary = test_df[test_df["label_int"] != 2]
    evo2_mask = test_binary["evo2_score"].notna()
    evo2_auroc = None
    if evo2_mask.sum() > 0 and len(np.unique(test_binary.loc[evo2_mask, "label_int"])) > 1:
        evo2_auroc = roc_auc_score(
            (test_binary.loc[evo2_mask, "label_int"] == 0).astype(int),
            test_binary.loc[evo2_mask, "evo2_score"],
        )
    
    # Simple average (both available, binary only)
    both_mask = test_binary["evo2_score"].notna() & test_binary["alphamissense_score"].notna()
    avg_auroc = None
    if both_mask.sum() > 0 and len(np.unique(test_binary.loc[both_mask, "label_int"])) > 1:
        avg_scores = (test_binary.loc[both_mask, "evo2_score"] + test_binary.loc[both_mask, "alphamissense_score"]) / 2
        avg_auroc = roc_auc_score(
            (test_binary.loc[both_mask, "label_int"] == 0).astype(int), avg_scores,
        )
    
    # DST+BMA
    dst = DST_BMA_Consensus(PREDICTOR_AUROCS)
    dst_preds = []
    for _, row in test_df.iterrows():
        sc = {
            "evo2_score": row["evo2_score"] if pd.notna(row["evo2_score"]) else None,
            "alphamissense_score": row["alphamissense_score"] if pd.notna(row["alphamissense_score"]) else None,
        }
        dst_preds.append(dst.predict(sc))
    dst_preds = np.array(dst_preds)
    
    dst_bin_mask = (test_l.numpy() < 2) & (dst_preds < 2)
    dst_auroc = None
    if dst_bin_mask.sum() > 1 and len(np.unique(test_l.numpy()[dst_bin_mask])) > 1:
        dst_auroc = roc_auc_score(
            (test_l.numpy()[dst_bin_mask] == 0).astype(int),
            (dst_preds[dst_bin_mask] == 0).astype(int),
        )
    dst_vus = float((dst_preds == 2).mean())
    
    baselines = {
        "Evo2-only": {"auroc": evo2_auroc, "n": int(evo2_mask.sum())},
        "Simple Average": {"auroc": avg_auroc, "n": int(both_mask.sum())},
        "DST+BMA": {"auroc": dst_auroc, "vus_rate": dst_vus, "n": len(test_df)},
    }
    
    print(f"\n  Baseline Comparison:")
    for name, m in baselines.items():
        auroc_str = f"AUROC={m['auroc']:.4f}" if m['auroc'] else "AUROC=N/A"
        extra = f" VUS={m.get('vus_rate', 0):.1%}" if 'vus_rate' in m else ""
        print(f"    {name:20s}: {auroc_str}{extra} (n={m['n']})")
    
    # ─── Save Results ───
    results = {
        "CEFN_v2": {
            "auroc": cefn_metrics["auroc"],
            "auprc": cefn_metrics["auprc"],
            "ece": cefn_metrics["ece"],
            "vus_rate": cefn_metrics["vus_rate"],
            "mean_uncertainty": cefn_metrics["mean_uncertainty"],
            "n_samples": cefn_metrics["n_samples"],
            "vt_summary": {
                vt: {k: v for k, v in m.items()}
                for vt, m in cefn_metrics["vt_summary"].items()
            },
        },
        "baselines": {
            name: {k: v for k, v in m.items()}
            for name, m in baselines.items()
        },
        "training": {
            "n_params": n_params,
            "best_epoch": checkpoint["epoch"],
            "train_time_seconds": round(train_time, 1),
            "n_epochs_trained": len(history["train_loss"]),
            "platt_scalers": {
                col: {"coef": lr.coef_[0][0], "intercept": lr.intercept_[0]}
                for col, lr in scalers.items()
            } if scalers else {},
        },
    }
    
    with open(METRICS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n  Metrics saved to {METRICS_PATH}")
    
    # ─── Plots ───
    plot_results(history, cefn_metrics, baselines, OUTPUT_DIR)
    
    # ─── Inference Examples ───
    print("\n" + "=" * 60)
    print("Inference Examples")
    print("=" * 60)
    
    model.eval()
    pid_t = predictor_ids.to(DEVICE)
    
    # Example 1: Missense, both available, pathogenic
    s1 = torch.tensor([[0.85, 0.91]], dtype=torch.float32)
    m1 = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
    vt1 = torch.zeros(1, N_VARTYPES); vt1[0, 0] = 1  # missense
    aP, aB, aV = model(s1, m1, pid_t, vt1)
    preds, bel_P, bel_B = predict_class(aP, aB, aV)
    _, _, ign = compute_beliefs(aP, aB, aV)
    print(f"\n  Missense (both available, high scores):")
    print(f"    Prediction: {'Pathogenic' if preds[0]==0 else 'Benign' if preds[0]==1 else 'VUS'}")
    print(f"    bel_P={bel_P[0].item():.3f}, bel_B={bel_B[0].item():.3f}, ignorance={ign[0].item():.3f}")
    
    # Example 2: Non-missense, only Evo2
    s2 = torch.tensor([[0.72, 0.0]], dtype=torch.float32)
    m2 = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    vt2 = torch.zeros(1, N_VARTYPES); vt2[0, 2] = 1  # splice_site
    aP, aB, aV = model(s2, m2, pid_t, vt2)
    preds, bel_P, bel_B = predict_class(aP, aB, aV)
    _, _, ign = compute_beliefs(aP, aB, aV)
    print(f"\n  Splice site (Evo2 only):")
    print(f"    Prediction: {'Pathogenic' if preds[0]==0 else 'Benign' if preds[0]==1 else 'VUS'}")
    print(f"    bel_P={bel_P[0].item():.3f}, bel_B={bel_B[0].item():.3f}, ignorance={ign[0].item():.3f}")
    
    # Example 3: No predictors
    s3 = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    m3 = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    vt3 = torch.zeros(1, N_VARTYPES); vt3[0, 4] = 1  # synonymous
    aP, aB, aV = model(s3, m3, pid_t, vt3)
    preds, bel_P, bel_B = predict_class(aP, aB, aV)
    _, _, ign = compute_beliefs(aP, aB, aV)
    print(f"\n  Synonymous (no predictors):")
    print(f"    Prediction: {'Pathogenic' if preds[0]==0 else 'Benign' if preds[0]==1 else 'VUS'}")
    print(f"    bel_P={bel_P[0].item():.3f}, bel_B={bel_B[0].item():.3f}, ignorance={ign[0].item():.3f}")
    
    print(f"\n✅ CEFN v2 pipeline complete!")
    print(f"   Model: {MODEL_PATH}")
    print(f"   Metrics: {METRICS_PATH}")
    print(f"   Plots: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
