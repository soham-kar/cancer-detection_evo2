"""
paper_figures.py
================
Generates publication-ready figures for the CEFN v2 manuscript:
  - Figure 2: ROC curves (internal test + temporal external)
  - Figure 3: Calibration curves (reliability diagram)
  - Figure 5: Ablation study bar chart
  - Figure 6: Decision curve analysis

Usage:
    python backend/phase1_implementation/paper_figures.py
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
import joblib
import json
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Import CEFN v2 components
# =============================================================================
from cefn_v2_common import (
    CEFN_v2, DeepSetEncoder, predict_class, compute_beliefs, compute_ece,
    DST_BMA_Consensus, evidential_loss_binary,
    PREDICTOR_COLS, VARTYPE_COLS, N_PREDICTORS, N_VARTYPES, DEVICE,
)

# =============================================================================
# Configuration
# =============================================================================
PHASE1_DIR = Path(__file__).resolve().parent
MODEL_DIR = PHASE1_DIR / "cefn_v2_outputs"
MODEL_PATH = MODEL_DIR / "cefn_v2_model.pt"
SCALER_PATH = MODEL_DIR / "platt_scalers.joblib"
ABLATION_JSON = PHASE1_DIR / "ablation_outputs" / "ablation_results.json"
FIGURES_DIR = MODEL_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

INPUT_CSV = PHASE1_DIR / "helixmind_benchmark_results_enriched.csv"
EXTERNAL_BALANCED_CSV = PHASE1_DIR / "clinvar" / "external_balanced_am.csv"

PREDICTOR_AUROCS = {"evo2_score": 0.975, "alphamissense_score": 0.975}

# Color palette (consistent across all figures)
COLORS = {
    "cefn": "#2E86AB",
    "evo2": "#A23B72",
    "am": "#F18F01",
    "dst": "#C73E1D",
    "avg": "#6A4C93",
    "gray": "#8D99AE",
    "dark": "#2B2D42",
    "light_gray": "#EDF2F4",
}

# Set publication style
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
})


# =============================================================================
# Data Preprocessing (from train_cefn_v2.py)
# =============================================================================
def map_label(label: str) -> int:
    label = str(label).strip().lower()
    if "pathogenic" in label and "uncertain" not in label and "conflicting" not in label:
        return 0
    if "benign" in label and "uncertain" not in label and "conflicting" not in label:
        return 1
    return 2

def infer_variant_type(row: pd.Series) -> str:
    has_am = pd.notna(row.get("alphamissense_score"))
    ref = str(row.get("reference", ""))
    vid = str(row.get("variant_id", ""))
    parts = vid.split("-")
    alt = parts[3] if len(parts) >= 4 else ref
    if ref == alt:
        return "synonymous"
    if has_am:
        return "missense"
    delta = row.get("delta_score", 0)
    pred = str(row.get("prediction", ""))
    if pd.notna(delta) and delta < -0.01 and "pathogenic" in pred.lower():
        return "splice_site"
    return "other"

def load_and_adapt_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["label_int"] = df["label"].apply(map_label)
    df["evo2_score"] = df["delta_score"].copy()
    df["alphamissense_score"] = pd.to_numeric(df["alphamissense_score"], errors="coerce")
    df["chromosome"] = df["variant_id"].apply(lambda x: x.split("-")[0].replace("chr", ""))
    chr_map = {str(i): i for i in range(1, 23)}
    chr_map.update({"X": 23, "Y": 24})
    df["chr_int"] = df["chromosome"].map(chr_map).fillna(0).astype(int)
    for vt in VARTYPE_COLS:
        df[f"vt_{vt}"] = 0
    for i, (_, row) in enumerate(df.iterrows()):
        vt = infer_variant_type(row)
        col = f"vt_{vt}"
        if col in df.columns:
            df.at[i, col] = 1
    return df

def chromosome_wise_split(df: pd.DataFrame):
    train_df = df[df["chr_int"].between(1, 16)]
    val_df = df[df["chr_int"].between(17, 18)]
    test_df = df[df["chr_int"].between(19, 22)]
    if len(train_df) < 100 or len(val_df) < 50 or len(test_df) < 50:
        from sklearn.model_selection import train_test_split
        train_df, temp = train_test_split(df, test_size=0.3, random_state=42, stratify=df["label_int"])
        val_df, test_df = train_test_split(temp, test_size=0.5, random_state=42, stratify=temp["label_int"])
    return train_df, val_df, test_df

def fit_platt_scalers(train_df, predictor_cols):
    binary_train = train_df[train_df["label_int"] != 2]
    scalers = {}
    for col in predictor_cols:
        mask = binary_train[col].notna()
        if mask.sum() < 20:
            continue
        X = binary_train.loc[mask, col].values.reshape(-1, 1)
        y = (binary_train.loc[mask, "label_int"] == 0).astype(int)
        lr = LogisticRegression(penalty=None, max_iter=1000)
        lr.fit(X, y)
        scalers[col] = lr
    return scalers

def apply_platt_scaling(df_in, scalers):
    df_out = df_in.copy()
    for col, lr in scalers.items():
        mask = df_out[col].notna()
        if mask.sum() == 0:
            continue
        X = df_out.loc[mask, col].values.reshape(-1, 1)
        df_out.loc[mask, col] = lr.predict_proba(X)[:, 1]
    if "alphamissense_score" in df_out.columns:
        mask = df_out["alphamissense_score"].notna()
        df_out.loc[mask, "alphamissense_score"] = df_out.loc[mask, "alphamissense_score"].clip(0, 1)
    return df_out


def prepare_external_data(csv_path, scalers):
    """Load and preprocess external validation CSV (AM-only)."""
    df = pd.read_csv(csv_path)
    df["label_int"] = df["label"].apply(map_label)
    # External data has delta_score (Evo2, all NaN) and alphamissense_score
    if "delta_score" in df.columns:
        df["evo2_score"] = df["delta_score"].copy()
    else:
        df["evo2_score"] = np.nan
    df["alphamissense_score"] = pd.to_numeric(df["alphamissense_score"], errors="coerce")
    # Apply Platt scaling to evo2_score (will be all NaN, but keep for consistency)
    df = apply_platt_scaling(df, scalers)
    # Infer variant types
    for vt in VARTYPE_COLS:
        df[f"vt_{vt}"] = 0
    for i, (_, row) in enumerate(df.iterrows()):
        vt = infer_variant_type(row)
        col = f"vt_{vt}"
        if col in df.columns:
            df.at[i, col] = 1
    return df


# =============================================================================
# Model Loading & Prediction
# =============================================================================
def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model = CEFN_v2(N_PREDICTORS, N_VARTYPES, embed_dim=16, hidden_dim=128).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

def get_predictions(model, df, predictor_ids):
    """Run model on DataFrame, return binary labels, pathogenic probs, and preds."""
    vt_cols = [f"vt_{vt}" for vt in VARTYPE_COLS]
    scores = df[PREDICTOR_COLS].fillna(0).values.astype(np.float32)
    mask = df[PREDICTOR_COLS].notna().values.astype(np.float32)
    vartype = df[vt_cols].values.astype(np.float32)
    labels = df["label_int"].values.astype(np.int64)

    with torch.no_grad():
        s_t = torch.tensor(scores)
        m_t = torch.tensor(mask)
        vt_t = torch.tensor(vartype)
        aP, aB, aV = model(s_t, m_t, predictor_ids, vt_t)
        preds, _, _ = predict_class(aP, aB, aV, threshold=0.5)
        p_path = (aP / (aP + aB + aV)).numpy()

    bin_mask = labels < 2
    return labels[bin_mask], p_path[bin_mask], preds.numpy()


# =============================================================================
# Figure 2: ROC Curves
# =============================================================================
def figure_2_roc(model, test_df, ext_df):
    predictor_ids = torch.arange(N_PREDICTORS)

    # --- Internal Test ---
    int_labels, int_cefn_prob, _ = get_predictions(model, test_df, predictor_ids)

    # Evo2-only
    evo2_mask = test_df["evo2_score"].notna() & (test_df["label_int"] != 2)
    evo2_labels = (test_df.loc[evo2_mask, "label_int"] == 0).astype(int).values
    evo2_prob = test_df.loc[evo2_mask, "evo2_score"].values

    # AlphaMissense (missense only)
    am_mask = test_df["alphamissense_score"].notna() & (test_df["label_int"] != 2)
    am_labels = (test_df.loc[am_mask, "label_int"] == 0).astype(int).values
    am_prob = test_df.loc[am_mask, "alphamissense_score"].values

    # Simple Average (both available)
    both_mask = evo2_mask & am_mask
    avg_labels = (test_df.loc[both_mask, "label_int"] == 0).astype(int).values
    avg_prob = (test_df.loc[both_mask, "evo2_score"] + test_df.loc[both_mask, "alphamissense_score"]) / 2.0

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
    dst_bin_mask = (test_df["label_int"] < 2) & (dst_preds < 2)
    dst_labels = (test_df.loc[dst_bin_mask, "label_int"] == 0).astype(int).values
    # DST gives hard class; for ROC we use class as probability proxy
    dst_prob = (dst_preds[dst_bin_mask] == 0).astype(float)

    # sklearn expects positive=1; CEFN labels are Pathogenic=0, Benign=1 -> flip only CEFN
    int_labels_sk = 1 - int_labels
    # Baseline labels already converted: (label_int == 0).astype(int) gives Pathogenic=1

    fpr_cefn, tpr_cefn, _ = roc_curve(int_labels_sk, int_cefn_prob)
    fpr_evo2, tpr_evo2, _ = roc_curve(evo2_labels, evo2_prob)
    fpr_am, tpr_am, _ = roc_curve(am_labels, am_prob)
    fpr_avg, tpr_avg, _ = roc_curve(avg_labels, avg_prob)
    fpr_dst, tpr_dst, _ = roc_curve(dst_labels, dst_prob)

    auc_cefn = auc(fpr_cefn, tpr_cefn)
    auc_evo2 = auc(fpr_evo2, tpr_evo2)
    auc_am = auc(fpr_am, tpr_am)
    auc_avg = auc(fpr_avg, tpr_avg)
    auc_dst = auc(fpr_dst, tpr_dst)

    # --- Temporal External ---
    ext_labels, ext_cefn_prob, _ = get_predictions(model, ext_df, predictor_ids)
    ext_labels_sk = 1 - ext_labels
    fpr_ext, tpr_ext, _ = roc_curve(ext_labels_sk, ext_cefn_prob)
    auc_ext = auc(fpr_ext, tpr_ext)

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(180/25.4, 90/25.4), constrained_layout=True)

    # Panel A: Internal test
    ax1.plot(fpr_cefn, tpr_cefn, label=f'CEFN v2 (AUROC = {auc_cefn:.3f})', lw=1.5, color=COLORS["cefn"])
    ax1.plot(fpr_evo2, tpr_evo2, label=f'Evo2-only (AUROC = {auc_evo2:.3f})', lw=1.2, color=COLORS["evo2"], linestyle='--')
    ax1.plot(fpr_am, tpr_am, label=f'AlphaMissense (AUROC = {auc_am:.3f})', lw=1.2, color=COLORS["am"], linestyle=':')
    ax1.plot(fpr_avg, tpr_avg, label=f'Simple Average (AUROC = {auc_avg:.3f})', lw=1.2, color=COLORS["avg"], linestyle='-.')
    ax1.plot(fpr_dst, tpr_dst, label=f'DST+BMA (AUROC = {auc_dst:.3f})', lw=1.2, color=COLORS["dst"], linestyle=(0, (3, 1, 1, 1)))
    ax1.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.4, label='Random')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('A  Internal test (chr19–22)')
    ax1.legend(fontsize=7, loc='lower right', frameon=False)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.2, linestyle='--')

    # Panel B: Temporal external
    ax2.plot(fpr_ext, tpr_ext, label=f'CEFN v2 (AUROC = {auc_ext:.3f})', lw=1.5, color=COLORS["cefn"])
    ax2.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.4, label='Random')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('B  Temporal external (ClinVar 2026–05–23)')
    ax2.legend(fontsize=8, loc='lower right', frameon=False)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.2, linestyle='--')

    fig.savefig(FIGURES_DIR / "figure2_roc.pdf", format='pdf')
    fig.savefig(FIGURES_DIR / "figure2_roc.png", format='png')
    plt.close(fig)
    print("Figure 2 saved: ROC curves")


# =============================================================================
# Figure 3: Calibration Curves
# =============================================================================
def figure_3_calibration(model, test_df):
    predictor_ids = torch.arange(N_PREDICTORS)
    int_labels, int_prob, _ = get_predictions(model, test_df, predictor_ids)

    # Evo2-only calibration
    evo2_mask = test_df["evo2_score"].notna() & (test_df["label_int"] != 2)
    evo2_labels = (test_df.loc[evo2_mask, "label_int"] == 0).astype(int).values
    evo2_prob = test_df.loc[evo2_mask, "evo2_score"].values

    # sklearn expects positive=1; CEFN labels are Pathogenic=0, Benign=1 -> flip only CEFN
    int_labels_sk = 1 - int_labels
    # evo2_labels already has Pathogenic=1 from (label_int == 0).astype(int)

    fraction_positive_cefn, mean_predicted_cefn = calibration_curve(int_labels_sk, int_prob, n_bins=10, strategy='uniform')
    fraction_positive_evo2, mean_predicted_evo2 = calibration_curve(evo2_labels, evo2_prob, n_bins=10, strategy='uniform')

    ece_cefn = compute_ece(int_labels_sk, int_prob)
    ece_evo2 = compute_ece(evo2_labels, evo2_prob)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(180/25.4, 80/25.4), constrained_layout=True)

    # Panel A: Reliability diagram
    ax1.plot(mean_predicted_cefn, fraction_positive_cefn, 's-', label=f'CEFN v2 (ECE = {ece_cefn:.3f})',
             color=COLORS["cefn"], markersize=6, lw=1.5)
    ax1.plot(mean_predicted_evo2, fraction_positive_evo2, 'o-', label=f'Evo2-only (ECE = {ece_evo2:.3f})',
             color=COLORS["evo2"], markersize=6, lw=1.5)
    ax1.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.4, label='Perfect calibration')
    ax1.set_xlabel('Mean predicted probability')
    ax1.set_ylabel('Observed fraction of positives')
    ax1.set_title('A  Reliability diagram')
    ax1.legend(fontsize=8, loc='upper left', frameon=False)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.2, linestyle='--')

    # Panel B: Probability histogram
    ax2.hist(int_prob, bins=20, alpha=0.6, color=COLORS["cefn"], label='CEFN v2', edgecolor='white')
    ax2.hist(evo2_prob, bins=20, alpha=0.6, color=COLORS["evo2"], label='Evo2-only', edgecolor='white')
    ax2.axvline(x=0.5, color=COLORS["dark"], linestyle='--', lw=0.8, alpha=0.5, label='Threshold = 0.5')
    ax2.set_xlabel('Predicted probability of pathogenicity')
    ax2.set_ylabel('Count')
    ax2.set_title('B  Probability distribution')
    ax2.legend(fontsize=8, loc='upper right', frameon=False)
    ax2.grid(True, alpha=0.2, linestyle='--', axis='y')

    fig.savefig(FIGURES_DIR / "figure3_calibration.pdf", format='pdf')
    fig.savefig(FIGURES_DIR / "figure3_calibration.png", format='png')
    plt.close(fig)
    print("Figure 3 saved: Calibration curves")


# =============================================================================
# Figure 5: Ablation Study Bar Chart
# =============================================================================
def figure_5_ablation():
    with open(ABLATION_JSON, 'r') as f:
        results = json.load(f)

    # Order: Full first, then by AUROC descending
    models = list(results.keys())
    full_idx = models.index("Full CEFN v2")
    other = [m for i, m in enumerate(models) if i != full_idx]
    other_sorted = sorted(other, key=lambda m: results[m]["AUROC"], reverse=True)
    ordered_models = ["Full CEFN v2"] + other_sorted

    aurocs = [results[m]["AUROC"] for m in ordered_models]
    eces = [results[m]["ECE"] for m in ordered_models]
    vus_rates = [results[m]["VUS Rate"] for m in ordered_models]

    # Clean journal-style labels (no minus signs, component names only)
    short_names = {
        "Full CEFN v2": "Full",
        "No Prior Network": "Prior",
        "No Deep Sets": "Deep Sets",
        "Shared Missing Token": "Missing emb.",
        "Learnable α_VUS": "Fixed α_VUS",
        "No Platt Scaling": "Platt",
    }
    display_names = [short_names.get(m, m) for m in ordered_models]

    fig, ax = plt.subplots(figsize=(160/25.4, 100/25.4))

    x = np.arange(len(ordered_models))
    width = 0.25

    # AUROC bars
    bars1 = ax.bar(x - width, aurocs, width, label='AUROC', color=COLORS["cefn"], edgecolor='white', linewidth=0.5)
    # ECE bars
    bars2 = ax.bar(x, eces, width, label='ECE', color=COLORS["am"], edgecolor='white', linewidth=0.5)
    # VUS rate bars
    bars3 = ax.bar(x + width, vus_rates, width, label='VUS rate', color=COLORS["dst"], edgecolor='white', linewidth=0.5)

    # Highlight full model
    bars1[0].set_edgecolor(COLORS["dark"])
    bars1[0].set_linewidth(1.5)
    bars2[0].set_edgecolor(COLORS["dark"])
    bars2[0].set_linewidth(1.5)
    bars3[0].set_edgecolor(COLORS["dark"])
    bars3[0].set_linewidth(1.5)

    # Reference lines
    ax.axhline(y=0.96, color=COLORS["gray"], linestyle='--', lw=0.8, alpha=0.5)
    ax.axhline(y=0.05, color=COLORS["gray"], linestyle='--', lw=0.8, alpha=0.5)

    ax.set_ylabel('Metric value')
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, rotation=0, ha='center', fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, loc='upper center', frameon=False, ncol=3,
              bbox_to_anchor=(0.5, 1.06), columnspacing=1.5)
    ax.set_title('Ablation study: component contribution', pad=25)
    ax.grid(True, alpha=0.2, linestyle='--', axis='y')

    # Value labels: ECE only (small, informative); VUS only if >5%
    for i, (a, e, v) in enumerate(zip(aurocs, eces, vus_rates)):
        # ECE label (skip if ≥0.99 since bar at top is obvious)
        if e < 0.99:
            ax.text(i, e + 0.02, f'{e:.3f}', ha='center', va='bottom', fontsize=7, color=COLORS["am"])
        # VUS label only if significant
        if v > 0.05:
            ax.text(i + width, v + 0.02, f'{v:.1%}', ha='center', va='bottom', fontsize=7, color=COLORS["dst"])

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "figure5_ablation.png", format='png')
    try:
        fig.savefig(FIGURES_DIR / "figure5_ablation.pdf", format='pdf')
    except PermissionError:
        print("  Warning: PDF locked by viewer, PNG saved successfully")
    plt.close(fig)
    print("Figure 5 saved: Ablation study")


# =============================================================================
# Figure 6: Decision Curve Analysis
# =============================================================================
def figure_6_dca(model, test_df):
    predictor_ids = torch.arange(N_PREDICTORS)
    int_labels, int_prob, _ = get_predictions(model, test_df, predictor_ids)

    # Evo2-only
    evo2_mask = test_df["evo2_score"].notna() & (test_df["label_int"] != 2)
    evo2_labels = (test_df.loc[evo2_mask, "label_int"] == 0).astype(int).values
    evo2_prob = test_df.loc[evo2_mask, "evo2_score"].values

    # sklearn convention: positive=1; CEFN labels are Pathogenic=0 -> flip
    int_labels_sk = 1 - int_labels
    # evo2_labels already has Pathogenic=1
    prevalence = np.mean(int_labels_sk == 1)

    thresholds = np.linspace(0.01, 0.99, 100)

    def net_benefit(probs, labels, pt):
        pred = (probs >= pt).astype(int)
        tp = np.sum((pred == 1) & (labels == 1))
        fp = np.sum((pred == 1) & (labels == 0))
        n = len(labels)
        return (tp / n) - (fp / n) * (pt / (1 - pt))

    nb_cefn = [net_benefit(int_prob, int_labels_sk, pt) for pt in thresholds]
    nb_evo2 = [net_benefit(evo2_prob, evo2_labels, pt) for pt in thresholds]
    # Clip "treat all" to avoid singularity at pt->1
    nb_all = []
    for pt in thresholds:
        if pt >= 0.99:
            nb_all.append(-10)
        else:
            nb_all.append(prevalence - (1 - prevalence) * (pt / (1 - pt)))
    nb_all = np.clip(nb_all, -0.5, 0.5)
    nb_none = [0] * len(thresholds)

    fig, ax = plt.subplots(figsize=(120/25.4, 100/25.4), constrained_layout=True)

    ax.plot(thresholds, nb_cefn, label='CEFN v2', color=COLORS["cefn"], lw=2)
    ax.plot(thresholds, nb_evo2, label='Evo2-only', color=COLORS["evo2"], lw=1.5, linestyle='--')
    ax.plot(thresholds, nb_all, label='Treat all', color=COLORS["gray"], lw=1, linestyle=':')
    ax.plot(thresholds, nb_none, label='Treat none', color=COLORS["dark"], lw=1, linestyle='-.')

    # Clinical threshold markers
    for pt, label in [(0.05, 'Screening'), (0.20, 'Diagnostic'), (0.50, 'Treatment')]:
        ax.axvline(x=pt, color=COLORS["gray"], linestyle='--', lw=0.5, alpha=0.4)
        ax.text(pt + 0.01, 0.40, label, rotation=90, fontsize=7,
                color=COLORS["gray"], ha='left', va='top')

    # Shade where CEFN > Evo2
    ax.fill_between(thresholds, nb_evo2, nb_cefn, where=np.array(nb_cefn) > np.array(nb_evo2),
                    alpha=0.15, color=COLORS["cefn"], label='CEFN advantage')

    ax.set_xlabel('Threshold probability')
    ax.set_ylabel('Net benefit')
    ax.set_title('Decision curve analysis')
    ax.legend(fontsize=8, loc='upper right', frameon=False)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.1, 0.5)
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.axhline(y=0, color='k', lw=0.5, alpha=0.3)

    fig.savefig(FIGURES_DIR / "figure6_dca.pdf", format='pdf')
    fig.savefig(FIGURES_DIR / "figure6_dca.png", format='png')
    plt.close(fig)
    print("Figure 6 saved: Decision curve analysis")


# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 60)
    print("CEFN v2 — Publication Figure Generation")
    print("=" * 60)

    print("\n[1/4] Loading model and data...")
    model = load_model()
    scalers = joblib.load(SCALER_PATH)

    # Load and preprocess internal test data
    df = load_and_adapt_data(str(INPUT_CSV))
    train_df, val_df, test_df = chromosome_wise_split(df)
    test_df = apply_platt_scaling(test_df, scalers)

    # Load external data
    ext_df = prepare_external_data(str(EXTERNAL_BALANCED_CSV), scalers)

    print(f"  Internal test: {len(test_df)} variants")
    print(f"  Temporal external: {len(ext_df)} variants")

    print("\n[2/4] Generating Figure 2: ROC curves...")
    figure_2_roc(model, test_df, ext_df)

    print("\n[3/4] Generating Figure 3: Calibration curves...")
    figure_3_calibration(model, test_df)

    print("\n[4/4] Generating Figure 5: Ablation study...")
    figure_5_ablation()

    print("\n[5/4] Generating Figure 6: Decision curve analysis...")
    figure_6_dca(model, test_df)

    print("\n" + "=" * 60)
    print("All figures saved to:", FIGURES_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
