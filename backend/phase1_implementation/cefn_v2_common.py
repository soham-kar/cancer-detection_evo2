"""
cefn_v2_common.py
=================
Shared components for CEFN v2 training and external validation.

Contains:
  - Constants (predictor columns, variant types, device)
  - Data helpers (label mapping, variant type inference)
  - Model architecture (DeepSetEncoder, CEFN_v2)
  - Belief / prediction helpers
  - Metrics (ECE)
  - Loss function (binary evidential loss)
  - Baseline (DST+BMA consensus)
  - Full evaluation routine
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from typing import Dict, Tuple, List, Optional
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

PREDICTOR_COLS = ["evo2_score", "alphamissense_score"]
N_PREDICTORS = len(PREDICTOR_COLS)

VARTYPE_COLS = ["missense", "nonsense", "splice_site", "frameshift", "synonymous", "other"]
N_VARTYPES = len(VARTYPE_COLS)

DEVICE = "cpu"


# =============================================================================
# 1. DATA HELPERS
# =============================================================================

def map_label(label: str) -> int:
    """Map ClinVar string label to int: 0=P, 1=B, 2=VUS."""
    label = str(label).strip().lower()
    if "pathogenic" in label and "uncertain" not in label and "conflicting" not in label:
        return 0
    if "benign" in label and "uncertain" not in label and "conflicting" not in label:
        return 1
    return 2


def infer_variant_type(row: pd.Series) -> str:
    """Infer variant type from available data."""
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


# =============================================================================
# 2. MODEL ARCHITECTURE
# =============================================================================

class DeepSetEncoder(nn.Module):
    """Permutation-invariant set encoder."""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        emb = self.phi(x)        # (B, K, hidden)
        pooled = emb.mean(dim=1)  # (B, hidden) — permutation-invariant mean pooling
        return self.rho(pooled)


class CEFN_v2(nn.Module):
    """
    CEFN v2 — Evidential fusion with:
      - Predictor-specific missing embeddings
      - Fixed α_VUS = 1 (VUS emerges from low evidence)
      - Only two output logits (Pathogenic, Benign)
    """
    def __init__(self, n_predictors: int, n_vartypes: int,
                 embed_dim: int = 16, hidden_dim: int = 128):
        super().__init__()
        self.n_predictors = n_predictors

        # Predictor ID embedding (for present predictors)
        self.pred_embed = nn.Embedding(n_predictors, embed_dim)
        # Predictor-specific missing embeddings (one per predictor)
        self.missing_embeds = nn.Parameter(torch.randn(n_predictors, embed_dim) * 0.1)

        # Input per predictor: score (1), presence mask (1), embedding (embed_dim)
        input_dim = 1 + 1 + embed_dim
        self.encoder = DeepSetEncoder(input_dim, hidden_dim, hidden_dim)

        # Prior network (variant type → logits for P, B)
        self.prior_net = nn.Sequential(
            nn.Linear(n_vartypes, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),  # only P and B
        )

        # Evidence network
        self.evidence_net = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, scores, mask, predictor_ids, vartype):
        """
        Args:
            scores: (B, K) — calibrated scores in [0,1]
            mask: (B, K) — binary mask
            predictor_ids: (K,) — predictor indices
            vartype: (B, V) — one-hot variant types

        Returns:
            alpha_P, alpha_B, alpha_VUS (alpha_VUS = 1.0 fixed)
        """
        B, K = scores.shape
        pid = predictor_ids.unsqueeze(0).expand(B, -1)  # (B, K)

        # Embedding: use pred_embed if present, else predictor-specific missing
        present_emb = self.pred_embed(pid)              # (B, K, embed)
        missing_emb = self.missing_embeds[pid]           # (B, K, embed)
        emb = torch.where(mask.unsqueeze(-1).bool(), present_emb, missing_emb)

        # Concatenate features: [score, mask, embedding]
        score_ch = scores.unsqueeze(-1)       # (B, K, 1)
        mask_ch = mask.float().unsqueeze(-1)  # (B, K, 1)
        x = torch.cat([score_ch, mask_ch, emb], dim=-1)

        # Deep Sets encoding
        z_pred = self.encoder(x)  # (B, hidden)

        # Prior (P, B logits)
        prior_logit = self.prior_net(vartype)  # (B, 2)
        alpha_prior = F.softplus(prior_logit) + 1.0  # > 1

        # Evidence
        h = torch.cat([z_pred, alpha_prior], dim=-1)
        evidence = F.softplus(self.evidence_net(h))  # > 0

        # Final Dirichlet parameters
        alpha_P = alpha_prior[:, 0] + evidence[:, 0]
        alpha_B = alpha_prior[:, 1] + evidence[:, 1]
        alpha_VUS = torch.ones_like(alpha_P)  # fixed to 1

        return alpha_P, alpha_B, alpha_VUS


# =============================================================================
# 3. BELIEF & PREDICTION HELPERS
# =============================================================================

def compute_beliefs(alpha_P, alpha_B, alpha_VUS):
    """Belief masses and ignorance from Dirichlet parameters."""
    total = alpha_P + alpha_B + alpha_VUS
    bel_P = (alpha_P - 1) / total
    bel_B = (alpha_B - 1) / total
    ignorance = 3.0 / total
    return bel_P, bel_B, ignorance


def predict_class(alpha_P, alpha_B, alpha_VUS, threshold=0.5):
    """Classify based on belief masses. 0=P, 1=B, 2=VUS."""
    bel_P, bel_B, _ = compute_beliefs(alpha_P, alpha_B, alpha_VUS)
    pred = torch.where(bel_P > threshold, 0,
            torch.where(bel_B > threshold, 1, 2))
    return pred, bel_P, bel_B


# =============================================================================
# 4. EVALUATION METRICS
# =============================================================================

def compute_ece(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (y_pred >= bin_boundaries[i]) & (y_pred < bin_boundaries[i + 1])
        if in_bin.sum() == 0:
            continue
        bin_conf = y_pred[in_bin].mean()
        bin_acc = y_true[in_bin].mean()
        bin_weight = in_bin.sum() / len(y_pred)
        ece += bin_weight * abs(bin_acc - bin_conf)
    return float(ece)


# =============================================================================
# 5. BINARY EVIDENTIAL LOSS
# =============================================================================

def evidential_loss_binary(
    alpha_P, alpha_B, alpha_VUS, labels,
    annealing_coef=1.0, lambda_reg=0.1,
):
    """
    Binary evidential loss. Ignores VUS samples (label=2).
    labels: 0=pathogenic, 1=benign, 2=VUS (ignored).
    """
    binary_mask = labels < 2
    if binary_mask.sum() == 0:
        return torch.tensor(0.0, device=alpha_P.device, requires_grad=True)

    a_P = alpha_P[binary_mask]
    a_B = alpha_B[binary_mask]
    a_V = alpha_VUS[binary_mask]
    y = (labels[binary_mask] == 0).float()  # 1=pathogenic, 0=benign

    total = a_P + a_B + a_V

    # Expected log likelihood
    digamma_total = torch.digamma(total)
    expected_log_P = torch.digamma(a_P) - digamma_total
    expected_log_B = torch.digamma(a_B) - digamma_total

    ll = y * expected_log_P + (1 - y) * expected_log_B
    L_data = -ll.mean()

    # KL regulariser
    alpha_tilde_P = y * a_P + (1 - y) * torch.ones_like(a_P)
    alpha_tilde_B = (1 - y) * a_B + y * torch.ones_like(a_B)
    alpha_tilde = torch.stack([alpha_tilde_P, alpha_tilde_B, a_V], dim=1)
    target_uniform = torch.ones_like(alpha_tilde)

    kl = (
        torch.lgamma(alpha_tilde.sum(1)) - torch.lgamma(target_uniform.sum(1))
        - torch.lgamma(alpha_tilde).sum(1) + torch.lgamma(target_uniform).sum(1)
        + ((alpha_tilde - target_uniform)
           * (torch.digamma(alpha_tilde)
              - torch.digamma(alpha_tilde.sum(1, keepdim=True)))).sum(1)
    )
    L_reg = kl.mean()

    return L_data + lambda_reg * annealing_coef * L_reg


# =============================================================================
# 6. DST+BMA BASELINE
# =============================================================================

class DST_BMA_Consensus:
    """
    Dempster-Shafer + Bayesian Model Averaging consensus.
    Uses AUROC-based weights and Dempster's rule for evidence combination.
    """
    def __init__(self, predictor_aurocs: Dict[str, float],
                 temperature: float = 10.0, belief_threshold: float = 0.5):
        self.predictors = list(predictor_aurocs.keys())
        auroc_values = np.array([predictor_aurocs[p] for p in self.predictors])
        weights = F.softmax(torch.tensor(temperature * auroc_values), dim=0).numpy()
        self.bma_weights = dict(zip(self.predictors, weights))
        self.frame = frozenset({"P", "B"})
        self.threshold = belief_threshold

    def _predictor_mass(self, score: float, pred_name: str) -> Dict:
        """Convert a predictor score to belief mass assignment."""
        p = np.clip(score, 0, 1)
        w = self.bma_weights[pred_name]
        return {
            frozenset({"P"}): (1 - w) * p,
            frozenset({"B"}): (1 - w) * (1 - p),
            self.frame: w,  # ignorance mass
        }

    def predict(self, scores_dict: Dict[str, Optional[float]]) -> int:
        """
        Combine available predictor scores using Dempster's rule.
        Returns: 0=P, 1=B, 2=VUS.
        """
        masses = []
        for p in self.predictors:
            s = scores_dict.get(p)
            if s is not None and not np.isnan(s):
                masses.append(self._predictor_mass(s, p))
            else:
                # Predictor missing → total ignorance
                masses.append({self.frame: 1.0})

        # Dempster's combination
        combined = masses[0]
        for m2 in masses[1:]:
            new = {}
            for A, mA in combined.items():
                for B, mB in m2.items():
                    inter = A & B
                    if inter:
                        new[inter] = new.get(inter, 0) + mA * mB
            total = sum(new.values())
            if total > 0:
                for k in new:
                    new[k] /= total
            combined = new

        # Belief masses
        bel_P = sum(m for A, m in combined.items() if A.issubset({"P"}))
        bel_B = sum(m for A, m in combined.items() if A.issubset({"B"}))

        if bel_P > self.threshold and bel_P > bel_B:
            return 0
        elif bel_B > self.threshold and bel_B > bel_P:
            return 1
        else:
            return 2


# =============================================================================
# 7. FULL EVALUATION
# =============================================================================

@torch.no_grad()
def evaluate_full(
    model: CEFN_v2,
    loader: DataLoader,
    predictor_ids: torch.Tensor,
) -> Dict:
    """Full evaluation: AUROC, AUPRC, ECE, VUS-by-vartype."""
    model.eval()

    y_true_bin, y_prob_bin = [], []
    all_preds = []
    all_uncertainty = []

    # Per variant-type accumulators
    vt_data = {vt: {"preds": [], "labels": [], "probs": [], "uncertainty": []}
               for vt in VARTYPE_COLS}

    for batch_idx, (scores, mask, vartype, labels) in enumerate(loader):
        scores = scores.to(DEVICE)
        mask = mask.to(DEVICE)
        vartype = vartype.to(DEVICE)
        labels = labels.to(DEVICE)

        aP, aB, aV = model(scores, mask, predictor_ids, vartype)
        preds, bel_P, bel_B = predict_class(aP, aB, aV, threshold=0.5)
        _, _, ignorance = compute_beliefs(aP, aB, aV)

        all_preds.extend(preds.cpu().numpy())
        all_uncertainty.extend(ignorance.cpu().numpy())

        # Binary metrics
        bin_mask = labels < 2
        if bin_mask.any():
            p_path = aP[bin_mask] / (aP[bin_mask] + aB[bin_mask] + aV[bin_mask])
            y_true_bin.extend((labels[bin_mask] == 0).int().cpu().numpy())
            y_prob_bin.extend(p_path.cpu().numpy())

        # Per variant-type
        vt_np = vartype.cpu().numpy()
        preds_np = preds.cpu().numpy()
        labels_np = labels.cpu().numpy()
        probs_np = (aP / (aP + aB + aV)).cpu().numpy()
        ign_np = ignorance.cpu().numpy()

        for vi, vt in enumerate(VARTYPE_COLS):
            mask_vt = vt_np[:, vi] > 0.5
            if mask_vt.sum() > 0:
                vt_data[vt]["preds"].extend(preds_np[mask_vt].tolist())
                vt_data[vt]["labels"].extend(labels_np[mask_vt].tolist())
                vt_data[vt]["probs"].extend(probs_np[mask_vt].tolist())
                vt_data[vt]["uncertainty"].extend(ign_np[mask_vt].tolist())

    # ─── Aggregate metrics ───
    y_true_bin = np.array(y_true_bin)
    y_prob_bin = np.array(y_prob_bin)
    all_preds = np.array(all_preds)

    auroc = roc_auc_score(y_true_bin, y_prob_bin) if len(set(y_true_bin)) > 1 else None
    auprc = average_precision_score(y_true_bin, y_prob_bin) if len(set(y_true_bin)) > 1 else None
    ece = compute_ece(y_true_bin, y_prob_bin)
    vus_rate = float((all_preds == 2).mean())
    mean_uncertainty = float(np.mean(all_uncertainty))

    # Per variant-type
    vt_summary = {}
    for vt in VARTYPE_COLS:
        d = vt_data[vt]
        n = len(d["labels"])
        if n == 0:
            vt_summary[vt] = {"n": 0, "vus_rate": None, "auroc": None, "mean_uncertainty": None}
            continue

        labels_arr = np.array(d["labels"])
        probs_arr = np.array(d["probs"])
        preds_arr = np.array(d["preds"])

        bin_mask = labels_arr != 2
        vt_auroc = None
        if bin_mask.sum() > 0 and len(np.unique(labels_arr[bin_mask])) >= 2:
            p_scores = probs_arr[bin_mask]
            y_bin = (labels_arr[bin_mask] == 0).astype(int)
            vt_auroc = roc_auc_score(y_bin, p_scores)

        vt_summary[vt] = {
            "n": n,
            "vus_rate": float((preds_arr == 2).mean()),
            "auroc": vt_auroc,
            "mean_uncertainty": float(np.mean(d["uncertainty"])),
        }

    return {
        "auroc": auroc,
        "auprc": auprc,
        "ece": ece,
        "vus_rate": vus_rate,
        "mean_uncertainty": mean_uncertainty,
        "n_samples": len(all_preds),
        "vt_summary": vt_summary,
    }
