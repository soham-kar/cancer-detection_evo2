"""
Ablation Study — CEFN v2
=========================
Removes each architectural component one at a time and evaluates on the
internal test set (chr19-22). All models are trained with the same hyperparameters.

Components ablated:
1. No prior network (variant-type conditioning removed)
2. Shared missing token (instead of predictor‑specific)
3. Learnable α_VUS (3‑class output, trained with VUS labels)
4. No Deep Sets (simple MLP on concatenated features)
5. No Platt scaling (raw delta sigmoid*100)

Usage:
    python backend/phase1_implementation/ablation_study.py
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
import joblib
import json
import time
from pathlib import Path
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================
PHASE1_DIR = Path(__file__).resolve().parent
INPUT_CSV = PHASE1_DIR / "helixmind_benchmark_results_enriched.csv"
OUTPUT_DIR = PHASE1_DIR / "ablation_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

PREDICTOR_COLS = ["evo2_score", "alphamissense_score"]
N_PREDICTORS = len(PREDICTOR_COLS)
VARTYPE_COLS = ["missense", "nonsense", "splice_site", "frameshift", "synonymous", "other"]
N_VARTYPES = len(VARTYPE_COLS)

BATCH_SIZE = 128
N_EPOCHS = 100
LEARNING_RATE = 1e-3
LAMBDA_REG = 0.05
EARLY_STOP_PATIENCE = 20
DEVICE = "cpu"

# Predictor AUROCs for DST+BMA (not used in ablation but kept for reference)
PREDICTOR_AUROCS = {"evo2_score": 0.975, "alphamissense_score": 0.975}

# =============================================================================
# Reuse data loading & preprocessing from train_cefn_v2.py
# (copy-paste necessary functions to avoid circular imports)
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
    df["alphamissense_score"] = df["alphamissense_score"].astype(float)
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
    val_df   = df[df["chr_int"].between(17, 18)]
    test_df  = df[df["chr_int"].between(19, 22)]
    if len(train_df) < 100 or len(val_df) < 50 or len(test_df) < 50:
        from sklearn.model_selection import train_test_split
        train_df, temp = train_test_split(df, test_size=0.3, random_state=42, stratify=df["label_int"])
        val_df, test_df = train_test_split(temp, test_size=0.5, random_state=42, stratify=temp["label_int"])
    return train_df, val_df, test_df

def fit_platt_scalers(train_df, val_df, test_df, predictor_cols):
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
    def transform(df_in):
        df_out = df_in.copy()
        for col, lr in scalers.items():
            mask = df_out[col].notna()
            if mask.sum() == 0: continue
            X = df_out.loc[mask, col].values.reshape(-1, 1)
            df_out.loc[mask, col] = lr.predict_proba(X)[:, 1]
        if "alphamissense_score" in df_out.columns:
            mask = df_out["alphamissense_score"].notna()
            df_out.loc[mask, "alphamissense_score"] = df_out.loc[mask, "alphamissense_score"].clip(0, 1)
        return df_out
    return transform(train_df), transform(val_df), transform(test_df), scalers

# =============================================================================
# Base Architecture Components (copied from cefn_v2_common.py)
# =============================================================================
class DeepSetEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
        )
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, output_dim), nn.LayerNorm(output_dim), nn.ReLU(),
        )
    def forward(self, x):
        emb = self.phi(x)          # (B, K, H)
        pooled = emb.mean(dim=1)   # mean pooling
        return self.rho(pooled)

# Full CEFN v2 (reference)
class CEFN_v2_Full(nn.Module):
    def __init__(self, n_predictors, n_vartypes, embed_dim=16, hidden_dim=128):
        super().__init__()
        self.n_predictors = n_predictors
        self.pred_embed = nn.Embedding(n_predictors, embed_dim)
        self.missing_embeds = nn.Parameter(torch.randn(n_predictors, embed_dim) * 0.1)
        input_dim = 1 + 1 + embed_dim
        self.encoder = DeepSetEncoder(input_dim, hidden_dim, hidden_dim)
        self.prior_net = nn.Sequential(
            nn.Linear(n_vartypes, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(),
            nn.Linear(hidden_dim//2, 2),
        )
        self.evidence_net = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(),
            nn.Linear(hidden_dim//2, 2),
        )
    def forward(self, scores, mask, predictor_ids, vartype):
        B, K = scores.shape
        pid = predictor_ids.unsqueeze(0).expand(B, -1)
        present_emb = self.pred_embed(pid)
        missing_emb = self.missing_embeds[pid]
        emb = torch.where(mask.unsqueeze(-1).bool(), present_emb, missing_emb)
        x = torch.cat([scores.unsqueeze(-1), mask.float().unsqueeze(-1), emb], dim=-1)
        z_pred = self.encoder(x)
        prior_logit = self.prior_net(vartype)
        alpha_prior = F.softplus(prior_logit) + 1.0
        h = torch.cat([z_pred, alpha_prior], dim=-1)
        evidence = F.softplus(self.evidence_net(h))
        alpha_P = alpha_prior[:,0] + evidence[:,0]
        alpha_B = alpha_prior[:,1] + evidence[:,1]
        alpha_VUS = torch.ones_like(alpha_P)
        return alpha_P, alpha_B, alpha_VUS

# =============================================================================
# Ablated Model Variants
# =============================================================================

# 1. No Prior Network (replace prior with constant α=1)
class CEFN_NoPrior(nn.Module):
    def __init__(self, n_predictors, n_vartypes, embed_dim=16, hidden_dim=128):
        super().__init__()
        self.n_predictors = n_predictors
        self.pred_embed = nn.Embedding(n_predictors, embed_dim)
        self.missing_embeds = nn.Parameter(torch.randn(n_predictors, embed_dim) * 0.1)
        input_dim = 1 + 1 + embed_dim
        self.encoder = DeepSetEncoder(input_dim, hidden_dim, hidden_dim)
        # No prior network, instead use constant ones for alpha_prior
        self.evidence_net = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(),
            nn.Linear(hidden_dim//2, 2),
        )
    def forward(self, scores, mask, predictor_ids, vartype):
        B, K = scores.shape
        pid = predictor_ids.unsqueeze(0).expand(B, -1)
        present_emb = self.pred_embed(pid)
        missing_emb = self.missing_embeds[pid]
        emb = torch.where(mask.unsqueeze(-1).bool(), present_emb, missing_emb)
        x = torch.cat([scores.unsqueeze(-1), mask.float().unsqueeze(-1), emb], dim=-1)
        z_pred = self.encoder(x)
        alpha_prior = torch.ones(B, 2, device=scores.device) * 1.0   # uniform prior
        h = torch.cat([z_pred, alpha_prior], dim=-1)
        evidence = F.softplus(self.evidence_net(h))
        alpha_P = alpha_prior[:,0] + evidence[:,0]
        alpha_B = alpha_prior[:,1] + evidence[:,1]
        alpha_VUS = torch.ones_like(alpha_P)
        return alpha_P, alpha_B, alpha_VUS

# 2. Shared missing token (single embedding for all missing predictors)
class CEFN_SharedMissing(nn.Module):
    def __init__(self, n_predictors, n_vartypes, embed_dim=16, hidden_dim=128):
        super().__init__()
        self.n_predictors = n_predictors
        self.pred_embed = nn.Embedding(n_predictors, embed_dim)
        self.shared_missing = nn.Parameter(torch.randn(1, embed_dim) * 0.1)
        input_dim = 1 + 1 + embed_dim
        self.encoder = DeepSetEncoder(input_dim, hidden_dim, hidden_dim)
        self.prior_net = nn.Sequential(
            nn.Linear(n_vartypes, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(),
            nn.Linear(hidden_dim//2, 2),
        )
        self.evidence_net = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(),
            nn.Linear(hidden_dim//2, 2),
        )
    def forward(self, scores, mask, predictor_ids, vartype):
        B, K = scores.shape
        pid = predictor_ids.unsqueeze(0).expand(B, -1)
        present_emb = self.pred_embed(pid)
        # Shared missing token expanded
        missing_emb = self.shared_missing.expand(B, K, -1)
        emb = torch.where(mask.unsqueeze(-1).bool(), present_emb, missing_emb)
        x = torch.cat([scores.unsqueeze(-1), mask.float().unsqueeze(-1), emb], dim=-1)
        z_pred = self.encoder(x)
        prior_logit = self.prior_net(vartype)
        alpha_prior = F.softplus(prior_logit) + 1.0
        h = torch.cat([z_pred, alpha_prior], dim=-1)
        evidence = F.softplus(self.evidence_net(h))
        alpha_P = alpha_prior[:,0] + evidence[:,0]
        alpha_B = alpha_prior[:,1] + evidence[:,1]
        alpha_VUS = torch.ones_like(alpha_P)
        return alpha_P, alpha_B, alpha_VUS

# 3. Learnable α_VUS (3-class output, trained with VUS labels)
class CEFN_LearnableVUS(nn.Module):
    def __init__(self, n_predictors, n_vartypes, embed_dim=16, hidden_dim=128):
        super().__init__()
        self.n_predictors = n_predictors
        self.pred_embed = nn.Embedding(n_predictors, embed_dim)
        self.missing_embeds = nn.Parameter(torch.randn(n_predictors, embed_dim) * 0.1)
        input_dim = 1 + 1 + embed_dim
        self.encoder = DeepSetEncoder(input_dim, hidden_dim, hidden_dim)
        self.prior_net = nn.Sequential(
            nn.Linear(n_vartypes, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(),
            nn.Linear(hidden_dim//2, 3),  # now 3 classes
        )
        self.evidence_net = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(),
            nn.Linear(hidden_dim//2, 3),
        )
    def forward(self, scores, mask, predictor_ids, vartype):
        B, K = scores.shape
        pid = predictor_ids.unsqueeze(0).expand(B, -1)
        present_emb = self.pred_embed(pid)
        missing_emb = self.missing_embeds[pid]
        emb = torch.where(mask.unsqueeze(-1).bool(), present_emb, missing_emb)
        x = torch.cat([scores.unsqueeze(-1), mask.float().unsqueeze(-1), emb], dim=-1)
        z_pred = self.encoder(x)
        prior_logit = self.prior_net(vartype)
        alpha_prior = F.softplus(prior_logit) + 1.0
        h = torch.cat([z_pred, alpha_prior], dim=-1)
        evidence = F.softplus(self.evidence_net(h))
        return alpha_prior + evidence   # (B, 3) Dirichlet parameters

# 4. No Deep Sets (simple MLP on concatenated features)
class CEFN_NoDeepSets(nn.Module):
    def __init__(self, n_predictors, n_vartypes, embed_dim=16, hidden_dim=128):
        super().__init__()
        self.n_predictors = n_predictors
        self.pred_embed = nn.Embedding(n_predictors, embed_dim)
        self.missing_embeds = nn.Parameter(torch.randn(n_predictors, embed_dim) * 0.1)
        # Concatenate all predictor features into one vector: K * (1+1+embed_dim)
        concat_dim = n_predictors * (1 + 1 + embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.prior_net = nn.Sequential(
            nn.Linear(n_vartypes, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(),
            nn.Linear(hidden_dim//2, 2),
        )
        self.evidence_net = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(),
            nn.Linear(hidden_dim//2, 2),
        )
    def forward(self, scores, mask, predictor_ids, vartype):
        B, K = scores.shape
        pid = predictor_ids.unsqueeze(0).expand(B, -1)
        present_emb = self.pred_embed(pid)
        missing_emb = self.missing_embeds[pid]
        emb = torch.where(mask.unsqueeze(-1).bool(), present_emb, missing_emb)
        # Concatenate along feature dimension: (B, K*(1+1+embed))
        flat = torch.cat([scores.unsqueeze(-1), mask.float().unsqueeze(-1), emb], dim=-1)
        flat = flat.reshape(B, -1)
        z_pred = self.mlp(flat)
        prior_logit = self.prior_net(vartype)
        alpha_prior = F.softplus(prior_logit) + 1.0
        h = torch.cat([z_pred, alpha_prior], dim=-1)
        evidence = F.softplus(self.evidence_net(h))
        alpha_P = alpha_prior[:,0] + evidence[:,0]
        alpha_B = alpha_prior[:,1] + evidence[:,1]
        alpha_VUS = torch.ones_like(alpha_P)
        return alpha_P, alpha_B, alpha_VUS

# 5. No Platt scaling (raw delta sigmoid*100)
class CEFN_NoPlatt(nn.Module):
    # Architecture identical to Full, but input scores are raw delta normalized by sigmoid(100*delta)
    def __init__(self, n_predictors, n_vartypes, embed_dim=16, hidden_dim=128):
        super().__init__()
        self.n_predictors = n_predictors
        self.pred_embed = nn.Embedding(n_predictors, embed_dim)
        self.missing_embeds = nn.Parameter(torch.randn(n_predictors, embed_dim) * 0.1)
        input_dim = 1 + 1 + embed_dim
        self.encoder = DeepSetEncoder(input_dim, hidden_dim, hidden_dim)
        self.prior_net = nn.Sequential(
            nn.Linear(n_vartypes, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(),
            nn.Linear(hidden_dim//2, 2),
        )
        self.evidence_net = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(),
            nn.Linear(hidden_dim//2, 2),
        )
    def forward(self, scores, mask, predictor_ids, vartype):
        # scores expected to be raw delta (already normalized by sigmoid(100*delta) externally)
        B, K = scores.shape
        pid = predictor_ids.unsqueeze(0).expand(B, -1)
        present_emb = self.pred_embed(pid)
        missing_emb = self.missing_embeds[pid]
        emb = torch.where(mask.unsqueeze(-1).bool(), present_emb, missing_emb)
        x = torch.cat([scores.unsqueeze(-1), mask.float().unsqueeze(-1), emb], dim=-1)
        z_pred = self.encoder(x)
        prior_logit = self.prior_net(vartype)
        alpha_prior = F.softplus(prior_logit) + 1.0
        h = torch.cat([z_pred, alpha_prior], dim=-1)
        evidence = F.softplus(self.evidence_net(h))
        alpha_P = alpha_prior[:,0] + evidence[:,0]
        alpha_B = alpha_prior[:,1] + evidence[:,1]
        alpha_VUS = torch.ones_like(alpha_P)
        return alpha_P, alpha_B, alpha_VUS

# =============================================================================
# Training and Evaluation Helpers
# =============================================================================
def compute_beliefs(alpha_P, alpha_B, alpha_VUS):
    total = alpha_P + alpha_B + alpha_VUS
    bel_P = (alpha_P - 1) / total
    bel_B = (alpha_B - 1) / total
    ignorance = 3.0 / total
    return bel_P, bel_B, ignorance

def predict_class(alpha_P, alpha_B, alpha_VUS, threshold=0.5):
    bel_P, bel_B, _ = compute_beliefs(alpha_P, alpha_B, alpha_VUS)
    pred = torch.where(bel_P > threshold, 0, torch.where(bel_B > threshold, 1, 2))
    return pred

def evidential_loss_binary(alpha_P, alpha_B, alpha_VUS, labels, annealing_coef=1.0, lambda_reg=0.1):
    binary_mask = labels < 2
    if binary_mask.sum() == 0:
        return torch.tensor(0.0, device=alpha_P.device, requires_grad=True)
    a_P = alpha_P[binary_mask]; a_B = alpha_B[binary_mask]; a_V = alpha_VUS[binary_mask]
    y = (labels[binary_mask] == 0).float()
    total = a_P + a_B + a_V
    digamma_total = torch.digamma(total)
    expected_log_P = torch.digamma(a_P) - digamma_total
    expected_log_B = torch.digamma(a_B) - digamma_total
    ll = y * expected_log_P + (1 - y) * expected_log_B
    L_data = -ll.mean()
    alpha_tilde_P = y * a_P + (1 - y) * torch.ones_like(a_P)
    alpha_tilde_B = (1 - y) * a_B + y * torch.ones_like(a_B)
    alpha_tilde = torch.stack([alpha_tilde_P, alpha_tilde_B, a_V], dim=1)
    target_uniform = torch.ones_like(alpha_tilde)
    kl = (torch.lgamma(alpha_tilde.sum(1)) - torch.lgamma(target_uniform.sum(1))
          - torch.lgamma(alpha_tilde).sum(1) + torch.lgamma(target_uniform).sum(1)
          + ((alpha_tilde - target_uniform) * (torch.digamma(alpha_tilde) - torch.digamma(alpha_tilde.sum(1, keepdim=True)))).sum(1))
    L_reg = kl.mean()
    return L_data + lambda_reg * annealing_coef * L_reg

def compute_ece(y_true, y_pred, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins+1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (y_pred >= bin_boundaries[i]) & (y_pred < bin_boundaries[i+1])
        if in_bin.sum() == 0: continue
        bin_conf = y_pred[in_bin].mean()
        bin_acc = y_true[in_bin].mean()
        ece += (in_bin.sum() / len(y_pred)) * abs(bin_acc - bin_conf)
    return float(ece)

def train_model(model, train_loader, val_loader, predictor_ids, epochs=100, lr=1e-3, lambda_reg=0.05, patience=20):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            scores, mask, vartype, labels = batch
            scores, mask, vartype, labels = scores.to(DEVICE), mask.to(DEVICE), vartype.to(DEVICE), labels.to(DEVICE)
            if isinstance(model, CEFN_LearnableVUS):
                alpha = model(scores, mask, predictor_ids, vartype)
                loss = evidential_loss_3class(alpha, labels, annealing_coef=min(1.0, epoch/(epochs*0.5)), lambda_reg=lambda_reg)
            else:
                aP, aB, aV = model(scores, mask, predictor_ids, vartype)
                loss = evidential_loss_binary(aP, aB, aV, labels, annealing_coef=min(1.0, epoch/(epochs*0.5)), lambda_reg=lambda_reg)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                scores, mask, vartype, labels = batch
                scores, mask, vartype, labels = scores.to(DEVICE), mask.to(DEVICE), vartype.to(DEVICE), labels.to(DEVICE)
                if isinstance(model, CEFN_LearnableVUS):
                    alpha = model(scores, mask, predictor_ids, vartype)
                    loss = evidential_loss_3class(alpha, labels, annealing_coef=1.0, lambda_reg=lambda_reg)
                else:
                    aP, aB, aV = model(scores, mask, predictor_ids, vartype)
                    loss = evidential_loss_binary(aP, aB, aV, labels, annealing_coef=1.0, lambda_reg=lambda_reg)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        if avg_val_loss < best_val_loss - 1e-4:
            best_val_loss = avg_val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def evidential_loss_3class(alpha, labels, annealing_coef=1.0, lambda_reg=0.1):
    # 3-class loss, all samples used, labels: 0=P,1=B,2=VUS
    alpha0 = alpha.sum(dim=-1)
    p = alpha / alpha0.unsqueeze(-1)
    # Expected log likelihood
    digamma_alpha = torch.digamma(alpha)
    digamma_alpha0 = torch.digamma(alpha0)
    expected_log = digamma_alpha - digamma_alpha0.unsqueeze(-1)
    L_data = -expected_log[range(len(labels)), labels].mean()
    # KL regulariser
    y_onehot = F.one_hot(labels, num_classes=3).float()
    alpha_tilde = y_onehot + (1 - y_onehot) * alpha
    target_uniform = torch.ones_like(alpha)
    kl = (torch.lgamma(alpha_tilde.sum(1)) - torch.lgamma(target_uniform.sum(1))
          - torch.lgamma(alpha_tilde).sum(1) + torch.lgamma(target_uniform).sum(1)
          + ((alpha_tilde - target_uniform) * (torch.digamma(alpha_tilde) - torch.digamma(alpha_tilde.sum(1, keepdim=True)))).sum(1))
    L_reg = kl.mean()
    return L_data + lambda_reg * annealing_coef * L_reg

def evaluate_model(model, loader, predictor_ids):
    model.eval()
    y_true_bin, y_prob_bin = [], []
    all_preds = []
    with torch.no_grad():
        for scores, mask, vartype, labels in loader:
            scores, mask, vartype, labels = scores.to(DEVICE), mask.to(DEVICE), vartype.to(DEVICE), labels.to(DEVICE)
            if isinstance(model, CEFN_LearnableVUS):
                alpha = model(scores, mask, predictor_ids, vartype)
                aP, aB, aV = alpha[:,0], alpha[:,1], alpha[:,2]
            else:
                aP, aB, aV = model(scores, mask, predictor_ids, vartype)
            preds = predict_class(aP, aB, aV, threshold=0.5)
            all_preds.extend(preds.cpu().numpy())
            bin_mask = labels < 2
            if bin_mask.any():
                p_path = aP[bin_mask] / (aP[bin_mask] + aB[bin_mask] + aV[bin_mask])
                y_true_bin.extend((labels[bin_mask]==0).int().cpu().numpy())
                y_prob_bin.extend(p_path.cpu().numpy())
    auroc = roc_auc_score(y_true_bin, y_prob_bin) if len(set(y_true_bin))>1 and not np.isnan(y_prob_bin).any() else 0.5
    ece = compute_ece(np.array(y_true_bin), np.array(y_prob_bin)) if not np.isnan(y_prob_bin).any() else 1.0
    vus_rate = np.mean(np.array(all_preds)==2)
    return auroc, ece, vus_rate

# =============================================================================
# Main Ablation Execution
# =============================================================================
def main():
    print("="*60)
    print("CEFN v2 Ablation Study")
    print("="*60)
    # Load and split data
    df = load_and_adapt_data(str(INPUT_CSV))
    train_df, val_df, test_df = chromosome_wise_split(df)
    # Platt scaling for baseline and full model; for NoPlatt we'll use raw sigmoid
    train_platt, val_platt, test_platt, scalers = fit_platt_scalers(train_df, val_df, test_df, PREDICTOR_COLS)
    # Prepare tensors for Platt-scaled data
    vt_cols = [f"vt_{vt}" for vt in VARTYPE_COLS]
    def df_to_tensors(df_in, use_platt=True):
        if use_platt:
            scores = df_in[PREDICTOR_COLS].fillna(0).values.astype(np.float32)
        else:
            # Use raw delta normalized by sigmoid(100*delta)
            scores = df_in[PREDICTOR_COLS].copy()
            scores['evo2_score'] = 1.0 / (1.0 + np.exp(-100 * scores['evo2_score'].values))
            scores['alphamissense_score'] = df_in['alphamissense_score'].fillna(0).astype(np.float32)
            scores = scores.values.astype(np.float32)
        mask = df_in[PREDICTOR_COLS].notna().values.astype(np.float32)
        vartype = df_in[vt_cols].values.astype(np.float32)
        labels = df_in["label_int"].values.astype(np.int64)
        return (torch.tensor(scores), torch.tensor(mask), torch.tensor(vartype), torch.tensor(labels))

    train_s, train_m, train_vt, train_l = df_to_tensors(train_platt)
    val_s, val_m, val_vt, val_l = df_to_tensors(val_platt)
    test_s, test_m, test_vt, test_l = df_to_tensors(test_platt)
    train_ds = TensorDataset(train_s, train_m, train_vt, train_l)
    val_ds = TensorDataset(val_s, val_m, val_vt, val_l)
    test_ds = TensorDataset(test_s, test_m, test_vt, test_l)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    predictor_ids = torch.arange(N_PREDICTORS)

    # For NoPlatt, prepare separate loaders with raw sigmoid scores
    train_noplatt_s, train_noplatt_m, train_noplatt_vt, train_noplatt_l = df_to_tensors(train_df, use_platt=False)
    val_noplatt_s, val_noplatt_m, val_noplatt_vt, val_noplatt_l = df_to_tensors(val_df, use_platt=False)
    test_noplatt_s, test_noplatt_m, test_noplatt_vt, test_noplatt_l = df_to_tensors(test_df, use_platt=False)
    train_noplatt_ds = TensorDataset(train_noplatt_s, train_noplatt_m, train_noplatt_vt, train_noplatt_l)
    val_noplatt_ds = TensorDataset(val_noplatt_s, val_noplatt_m, val_noplatt_vt, val_noplatt_l)
    test_noplatt_ds = TensorDataset(test_noplatt_s, test_noplatt_m, test_noplatt_vt, test_noplatt_l)
    train_noplatt_loader = DataLoader(train_noplatt_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_noplatt_loader = DataLoader(val_noplatt_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_noplatt_loader = DataLoader(test_noplatt_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Define models
    models = {
        "Full CEFN v2": CEFN_v2_Full(N_PREDICTORS, N_VARTYPES, embed_dim=16, hidden_dim=128),
        "No Prior Network": CEFN_NoPrior(N_PREDICTORS, N_VARTYPES, embed_dim=16, hidden_dim=128),
        "Shared Missing Token": CEFN_SharedMissing(N_PREDICTORS, N_VARTYPES, embed_dim=16, hidden_dim=128),
        "Learnable α_VUS": CEFN_LearnableVUS(N_PREDICTORS, N_VARTYPES, embed_dim=16, hidden_dim=128),
        "No Deep Sets": CEFN_NoDeepSets(N_PREDICTORS, N_VARTYPES, embed_dim=16, hidden_dim=128),
        "No Platt Scaling": CEFN_NoPlatt(N_PREDICTORS, N_VARTYPES, embed_dim=16, hidden_dim=128),
    }

    results = {}
    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        t0 = time.perf_counter()
        # Use appropriate data loaders
        if name == "No Platt Scaling":
            tr_loader, vl_loader, ts_loader = train_noplatt_loader, val_noplatt_loader, test_noplatt_loader
        else:
            tr_loader, vl_loader, ts_loader = train_loader, val_loader, test_loader
        model = train_model(model, tr_loader, vl_loader, predictor_ids, epochs=N_EPOCHS, lr=LEARNING_RATE, lambda_reg=LAMBDA_REG, patience=EARLY_STOP_PATIENCE)
        elapsed = time.perf_counter() - t0
        auroc, ece, vus = evaluate_model(model, ts_loader, predictor_ids)
        results[name] = {"AUROC": auroc, "ECE": ece, "VUS Rate": vus, "Train Time (s)": elapsed}
        print(f"{name}: AUROC={auroc:.4f}, ECE={ece:.4f}, VUS={vus:.3f}, Time={elapsed:.1f}s")

    # Save results
    with open(OUTPUT_DIR / "ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    # Print table
    print("\n" + "="*70)
    print("Ablation Study Summary")
    print("="*70)
    print(f"{'Model':30s} {'AUROC':>8s} {'ECE':>8s} {'VUS Rate':>10s}")
    print("-"*70)
    for name, metrics in results.items():
        print(f"{name:30s} {metrics['AUROC']:8.4f} {metrics['ECE']:8.4f} {metrics['VUS Rate']:10.1%}")
    print("-"*70)

if __name__ == "__main__":
    main()
