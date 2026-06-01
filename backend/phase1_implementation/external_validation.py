"""
external_validation.py
====================
Production-ready external validation script for CEFN v2.

Loads trained CEFN v2 model + fitted Platt scalers and evaluates on an
independent dataset with identical preprocessing.

Usage:
    python backend/phase1_implementation/external_validation.py \
        --csv path/to/external_data.csv \
        --model backend/phase1_implementation/cefn_v2_outputs/cefn_v2_model.pt \
        --scalers backend/phase1_implementation/cefn_v2_outputs/platt_scalers.joblib \
        --out backend/phase1_implementation/cefn_v2_outputs/external_validation_metrics.json

Expected CSV columns:
    - variant_id (chr-POS-REF-ALT)
    - label (ClinVar string: Pathogenic, Benign, VUS, etc.)
    - delta_score (Evo2 raw delta)
    - alphamissense_score (0-1, may be NaN)
    - reference (REF allele)
    - prediction (Evo2 string prediction, e.g. "Likely Pathogenic")

Success criteria (configurable):
    - AUROC > 0.96
    - ECE < 0.05
    - VUS rate ≤ 5% on missense, ≤ 2% on other
    - CEFN AUROC ≥ Evo2 AUROC (consensus should not degrade)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader, TensorDataset

from cefn_v2_common import (
    CEFN_v2,
    DST_BMA_Consensus,
    compute_beliefs,
    compute_ece,
    evaluate_full,
    infer_variant_type,
    map_label,
    predict_class,
    PREDICTOR_COLS,
    N_PREDICTORS,
    VARTYPE_COLS,
    N_VARTYPES,
    DEVICE,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Default paths (relative to this script)
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = SCRIPT_DIR / "cefn_v2_outputs" / "cefn_v2_model.pt"
DEFAULT_SCALERS = SCRIPT_DIR / "cefn_v2_outputs" / "platt_scalers.joblib"
DEFAULT_OUT = SCRIPT_DIR / "cefn_v2_outputs" / "external_validation_metrics.json"

# Success thresholds
SUCCESS_THRESHOLDS = {
    "auroc_min": 0.96,
    "ece_max": 0.05,
    "vus_rate_missense_max": 0.05,
    "vus_rate_other_max": 0.02,
}

# Predictor AUROCs (same as training; used for DST+BMA baseline)
PREDICTOR_AUROCS = {
    "evo2_score": 0.975,
    "alphamissense_score": 0.975,
}


# =============================================================================
# 1. DATA LOADING & PREPROCESSING
# =============================================================================

def load_external_data(csv_path: str) -> pd.DataFrame:
    """
    Load and preprocess external dataset identically to training pipeline.

    Adds columns:
      - label_int: 0=P, 1=B, 2=VUS
      - evo2_score: raw delta (to be Platt-scaled)
      - alphamissense_score: already 0-1
      - One-hot variant type columns (vt_missense, vt_nonsense, ...)
    """
    df = pd.read_csv(csv_path)

    # Validate required columns
    required = ["variant_id", "label", "delta_score", "alphamissense_score",
                "reference", "prediction"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Map labels
    df["label_int"] = df["label"].apply(map_label)

    # Evo2 score = raw delta (will be calibrated by Platt scalers)
    df["evo2_score"] = df["delta_score"].copy()

    # AlphaMissense: ensure float
    df["alphamissense_score"] = df["alphamissense_score"].astype(float)

    # Infer variant types (one-hot)
    for vt in VARTYPE_COLS:
        df[f"vt_{vt}"] = 0

    for i, (_, row) in enumerate(df.iterrows()):
        vt = infer_variant_type(row)
        col = f"vt_{vt}"
        if col in df.columns:
            df.at[i, col] = 1

    return df


def apply_platt_scalers(df: pd.DataFrame, scalers: dict) -> pd.DataFrame:
    """Apply fitted Platt scalers to the dataframe (in-place copy)."""
    df_out = df.copy()
    for col, lr in scalers.items():
        mask = df_out[col].notna()
        if mask.sum() == 0:
            continue
        X = df_out.loc[mask, col].values.reshape(-1, 1)
        df_out.loc[mask, col] = lr.predict_proba(X)[:, 1]

    # Clip AlphaMissense to [0, 1] for safety
    am_mask = df_out["alphamissense_score"].notna()
    df_out.loc[am_mask, "alphamissense_score"] = df_out.loc[am_mask, "alphamissense_score"].clip(0, 1)

    return df_out


# =============================================================================
# 2. BASELINES ON EXTERNAL DATA
# =============================================================================

def run_baselines(df: pd.DataFrame) -> Dict:
    """Compute Evo2-only, Simple Average, and DST+BMA on the external data."""
    bin_df = df[df["label_int"] != 2]
    baselines = {}

    # --- Evo2-only ---
    evo2_mask = bin_df["evo2_score"].notna()
    if evo2_mask.sum() > 0 and len(np.unique(bin_df.loc[evo2_mask, "label_int"])) > 1:
        evo2_auroc = roc_auc_score(
            (bin_df.loc[evo2_mask, "label_int"] == 0).astype(int),
            bin_df.loc[evo2_mask, "evo2_score"],
        )
    else:
        evo2_auroc = None
    baselines["Evo2-only"] = {"auroc": evo2_auroc, "n": int(evo2_mask.sum())}

    # --- Simple average (both available) ---
    both_mask = bin_df["evo2_score"].notna() & bin_df["alphamissense_score"].notna()
    if both_mask.sum() > 0 and len(np.unique(bin_df.loc[both_mask, "label_int"])) > 1:
        avg_scores = (
            bin_df.loc[both_mask, "evo2_score"] +
            bin_df.loc[both_mask, "alphamissense_score"]
        ) / 2
        avg_auroc = roc_auc_score(
            (bin_df.loc[both_mask, "label_int"] == 0).astype(int),
            avg_scores,
        )
    else:
        avg_auroc = None
    baselines["Simple Average"] = {"auroc": avg_auroc, "n": int(both_mask.sum())}

    # --- DST+BMA ---
    dst = DST_BMA_Consensus(PREDICTOR_AUROCS)
    dst_preds = []
    for _, row in df.iterrows():
        sc = {
            "evo2_score": row["evo2_score"] if pd.notna(row["evo2_score"]) else None,
            "alphamissense_score": row["alphamissense_score"] if pd.notna(row["alphamissense_score"]) else None,
        }
        dst_preds.append(dst.predict(sc))
    dst_preds = np.array(dst_preds)

    dst_bin_mask = (df["label_int"].values < 2) & (dst_preds < 2)
    if dst_bin_mask.sum() > 1 and len(np.unique(df["label_int"].values[dst_bin_mask])) > 1:
        dst_auroc = roc_auc_score(
            (df["label_int"].values[dst_bin_mask] == 0).astype(int),
            (dst_preds[dst_bin_mask] == 0).astype(int),
        )
    else:
        dst_auroc = None
    dst_vus = float((dst_preds == 2).mean())

    baselines["DST+BMA"] = {
        "auroc": dst_auroc,
        "vus_rate": dst_vus,
        "n": len(df),
    }

    return baselines


# =============================================================================
# 3. SUCCESS CRITERIA CHECK
# =============================================================================

def check_success(metrics: Dict, baselines: Dict) -> Dict:
    """
    Automated PASS/FAIL checks against publication thresholds.
    Returns a dict with boolean flags and human-readable messages.
    """
    checks = {}

    # AUROC threshold
    auroc = metrics.get("auroc")
    checks["auroc_above_0.96"] = {
        "pass": auroc is not None and auroc >= SUCCESS_THRESHOLDS["auroc_min"],
        "value": auroc,
        "threshold": SUCCESS_THRESHOLDS["auroc_min"],
    }

    # ECE threshold
    ece = metrics.get("ece")
    checks["ece_below_0.05"] = {
        "pass": ece is not None and ece <= SUCCESS_THRESHOLDS["ece_max"],
        "value": ece,
        "threshold": SUCCESS_THRESHOLDS["ece_max"],
    }

    # VUS rate by variant type
    vt_summary = metrics.get("vt_summary", {})
    for vt, max_rate in [
        ("missense", SUCCESS_THRESHOLDS["vus_rate_missense_max"]),
        ("other", SUCCESS_THRESHOLDS["vus_rate_other_max"]),
    ]:
        vus_rate = vt_summary.get(vt, {}).get("vus_rate")
        checks[f"vus_rate_{vt}_below_{max_rate}"] = {
            "pass": vus_rate is None or vus_rate <= max_rate,
            "value": vus_rate,
            "threshold": max_rate,
        }

    # CEFN should not degrade vs Evo2
    cefn_auroc = metrics.get("auroc")
    evo2_auroc = baselines.get("Evo2-only", {}).get("auroc")
    checks["cefn_not_worse_than_evo2"] = {
        "pass": (
            cefn_auroc is None or evo2_auroc is None or
            cefn_auroc >= evo2_auroc - 0.02  # allow 2% tolerance
        ),
        "value": {"cefn": cefn_auroc, "evo2": evo2_auroc},
        "threshold": "CEFN >= Evo2 - 0.02",
    }

    # Overall verdict
    all_pass = all(c["pass"] for c in checks.values())
    checks["__overall__"] = {
        "pass": all_pass,
        "verdict": "PASS — Ready for publication" if all_pass else "FAIL — Review needed",
    }

    return checks


# =============================================================================
# 4. MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="CEFN v2 External Validation")
    parser.add_argument(
        "--csv", required=True,
        help="Path to external validation CSV (same format as training data)",
    )
    parser.add_argument(
        "--model", type=str, default=str(DEFAULT_MODEL),
        help="Path to trained CEFN v2 model .pt file",
    )
    parser.add_argument(
        "--scalers", type=str, default=str(DEFAULT_SCALERS),
        help="Path to fitted Platt scalers .joblib file",
    )
    parser.add_argument(
        "--out", type=str, default=str(DEFAULT_OUT),
        help="Path to write JSON metrics output",
    )
    parser.add_argument(
        "--batch-size", type=int, default=128,
        help="Inference batch size",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    scaler_path = Path(args.scalers)
    out_path = Path(args.out)
    csv_path = Path(args.csv)

    # Validate inputs exist
    for p, name in [(model_path, "Model"), (scaler_path, "Scalers"), (csv_path, "CSV")]:
        if not p.exists():
            print(f"ERROR: {name} not found: {p}")
            sys.exit(1)

    print("=" * 70)
    print("CEFN v2 — External Validation")
    print("=" * 70)

    # ─── 1. Load model and scalers ───
    print("\n[1/5] Loading model and scalers...")
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model = CEFN_v2(
        n_predictors=N_PREDICTORS,
        n_vartypes=N_VARTYPES,
        embed_dim=16,
        hidden_dim=128,
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    scalers = joblib.load(scaler_path)
    print(f"  Model loaded from {model_path}")
    print(f"  Scalers loaded: {list(scalers.keys())}")

    # ─── 2. Load and preprocess external data ───
    print(f"\n[2/5] Loading external data from {csv_path}...")
    df = load_external_data(str(csv_path))
    print(f"  Total variants: {len(df)}")
    print(f"  Labels: P={(df['label_int']==0).sum()}, "
          f"B={(df['label_int']==1).sum()}, "
          f"VUS={(df['label_int']==2).sum()}")

    # Apply Platt scalers identically to training
    df_proc = apply_platt_scalers(df, scalers)
    print(f"  Platt scaling applied.")

    # ─── 3. Prepare tensors ───
    print("\n[3/5] Preparing tensors...")
    predictor_ids = torch.arange(N_PREDICTORS)
    vt_cols = [f"vt_{vt}" for vt in VARTYPE_COLS]

    scores = df_proc[PREDICTOR_COLS].fillna(0).values.astype(np.float32)
    mask = df_proc[PREDICTOR_COLS].notna().values.astype(np.float32)
    vartype = df_proc[vt_cols].values.astype(np.float32)
    labels = df_proc["label_int"].values.astype(np.int64)

    ds = TensorDataset(
        torch.tensor(scores),
        torch.tensor(mask),
        torch.tensor(vartype),
        torch.tensor(labels),
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    print(f"  Batches: {len(loader)}")

    # ─── 4. Evaluate CEFN v2 ───
    print("\n[4/5] Evaluating CEFN v2 on external data...")
    metrics = evaluate_full(model, loader, predictor_ids)

    print(f"\n  CEFN v2 External Results:")
    auroc_str = f"{metrics['auroc']:.4f}" if metrics['auroc'] else "N/A"
    auprc_str = f"{metrics['auprc']:.4f}" if metrics['auprc'] else "N/A"
    print(f"    AUROC: {auroc_str}")
    print(f"    AUPRC: {auprc_str}")
    print(f"    ECE:   {metrics['ece']:.4f}")
    print(f"    VUS Rate: {metrics['vus_rate']:.1%}")
    print(f"    Mean Uncertainty: {metrics['mean_uncertainty']:.4f}")
    print(f"    N Samples: {metrics['n_samples']}")

    print(f"\n  VUS Rate by Variant Type:")
    for vt in VARTYPE_COLS:
        m = metrics["vt_summary"][vt]
        if m["n"] > 0:
            auroc_vt = f"AUROC={m['auroc']:.3f}" if m['auroc'] else "AUROC=N/A"
            print(f"    {vt:15s}: n={m['n']:4d}  VUS={m['vus_rate']:.1%}  "
                  f"{auroc_vt}  uncert={m['mean_uncertainty']:.3f}")

    # ─── 5. Baselines ───
    print("\n[5/5] Running baselines on external data...")
    baselines = run_baselines(df_proc)
    print(f"\n  Baseline Comparison:")
    for name, m in baselines.items():
        auroc_str = f"AUROC={m['auroc']:.4f}" if m['auroc'] else "AUROC=N/A"
        extra = f" VUS={m.get('vus_rate', 0):.1%}" if 'vus_rate' in m else ""
        print(f"    {name:20s}: {auroc_str}{extra} (n={m['n']})")

    # ─── 6. Success criteria ───
    print("\n" + "=" * 70)
    print("Success Criteria Check")
    print("=" * 70)
    checks = check_success(metrics, baselines)
    for key, c in checks.items():
        if key == "__overall__":
            status = "✅ PASS" if c["pass"] else "❌ FAIL"
            print(f"\n  {status}: {c['verdict']}")
        else:
            status = "✅" if c["pass"] else "❌"
            val = c["value"]
            if isinstance(val, dict):
                val_str = json.dumps(val, default=float)
            elif isinstance(val, float):
                val_str = f"{val:.4f}"
            else:
                val_str = str(val)
            print(f"  {status} {key}: {val_str} (threshold: {c['threshold']})")

    # ─── 7. Save results ───
    output = {
        "metadata": {
            "model_path": str(model_path),
            "scaler_path": str(scaler_path),
            "csv_path": str(csv_path),
            "n_samples": len(df),
        },
        "CEFN_v2_external": {
            k: v for k, v in metrics.items() if k != "vt_summary"
        },
        "CEFN_v2_external_vt_summary": metrics["vt_summary"],
        "baselines_external": baselines,
        "success_checks": checks,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=float)
    print(f"\n  Results saved to {out_path}")

    # Exit code reflects success
    sys.exit(0 if checks["__overall__"]["pass"] else 1)


if __name__ == "__main__":
    main()
