"""
Conditional Evidential Fusion Network (CEFN)
=============================================
Novel missing-data-aware consensus engine for variant pathogenicity prediction.

Architecture:
  1. Deep Sets encoder — permutation-invariant fusion of any predictor subset
  2. Prior network — variant-type metadata informs baseline pathogenicity rates
  3. Evidential output — Dirichlet distribution over {Pathogenic, Benign, VUS}

Key innovations:
  - Structured missingness: model knows WHY a predictor is missing (variant type)
  - No imputation: missing predictors get learned "missing token" embeddings
  - Calibrated uncertainty: Dirichlet evidence maps to ACMG/AMP strength levels
  - End-to-end learned: no ad-hoc weights; all combination logic is data-driven

Reference:
  Zaheer et al. (2017) NeurIPS — "Deep Sets"
  Sensoy et al. (2018) NeurIPS — "Evidential Deep Learning"
  Malinin & Gales (2018) NeurIPS — "Prior Networks"

Author: HelixMind Phase 1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

class DeepSetEncoder(nn.Module):
    """
    Permutation-invariant set encoder (Zaheer et al., 2017).
    
    Takes a variable-size set of predictor embeddings and produces
    a fixed-dimensional representation via sum-pooling.
    
    φ: individual element embedding
    ρ: pooled representation projection
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
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
    
    def forward(self, X: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            X: (B, K, input_dim) — predictor features
            mask: (B, K) — binary mask (1=present, 0=missing)
        Returns:
            (B, output_dim) — pooled representation
        """
        embeddings = self.phi(X)  # (B, K, H)
        
        if mask is not None:
            # Zero out missing predictor embeddings before pooling
            mask_expanded = mask.unsqueeze(-1).float()  # (B, K, 1)
            embeddings = embeddings * mask_expanded
        
        # Sum-pooling (permutation invariant)
        pooled = embeddings.sum(dim=1)  # (B, H)
        
        # Normalize by number of present predictors to avoid scale issues
        if mask is not None:
            n_present = mask.sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1)
            pooled = pooled / n_present
        
        return self.rho(pooled)


class CEFN(nn.Module):
    """
    Conditional Evidential Fusion Network.
    
    Combines any subset of variant effect predictors with variant-type
    metadata to produce a Dirichlet distribution over clinical classes.
    
    Args:
        n_predictors: Number of predictor types (Evo2, AlphaMissense, CADD, REVEL)
        n_vartypes: Number of variant type categories (missense, nonsense, splice, etc.)
        predictor_embed_dim: Dimension of predictor identity embeddings
        hidden_dim: Hidden dimension for all MLPs
        n_classes: Number of output classes (default: 3 = P, B, VUS)
    """
    
    def __init__(
        self,
        n_predictors: int = 4,
        n_vartypes: int = 6,
        predictor_embed_dim: int = 16,
        hidden_dim: int = 64,
        n_classes: int = 3,
    ):
        super().__init__()
        
        self.n_predictors = n_predictors
        self.n_classes = n_classes
        
        # ─── Predictor Identity Embeddings ───
        # Each predictor type gets a learned embedding
        self.predictor_embed = nn.Embedding(n_predictors, predictor_embed_dim)
        # Special "missing" token for absent predictors
        self.missing_embed = nn.Parameter(torch.randn(predictor_embed_dim) * 0.1)
        
        # ─── Deep Sets Encoder ───
        # Input per predictor: [score, presence_mask, predictor_embed]
        input_dim = 1 + 1 + predictor_embed_dim
        self.encoder = DeepSetEncoder(input_dim, hidden_dim, hidden_dim)
        
        # ─── Prior Network (Variant-Type Context) ───
        # Encodes variant metadata into a prior Dirichlet
        self.prior_net = nn.Sequential(
            nn.Linear(n_vartypes, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_classes),
        )
        
        # ─── Evidence Network ───
        # Combines set encoding + prior to produce evidence
        self.evidence_net = nn.Sequential(
            nn.Linear(hidden_dim + n_classes, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_classes),
        )
        
        # ─── Uncertainty Calibration ───
        # Learnable temperature for Dirichlet concentration
        self.log_temperature = nn.Parameter(torch.zeros(1))
    
    def forward(
        self,
        scores: torch.Tensor,
        mask: torch.Tensor,
        predictor_ids: torch.Tensor,
        vartype: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass producing Dirichlet parameters.
        
        Args:
            scores: (B, K) — predictor scores in [0,1], NaN→0 (masked out)
            mask: (B, K) — binary mask (1=present)
            predictor_ids: (K,) — integer IDs for each predictor column
            vartype: (B, V) — one-hot variant type metadata
        
        Returns:
            alpha: (B, C) — Dirichlet concentration parameters (>1)
        """
        B, K = scores.shape
        
        # ─── Build Per-Predictor Features ───
        scores_in = scores.unsqueeze(-1)  # (B, K, 1)
        mask_in = mask.unsqueeze(-1).float()  # (B, K, 1)
        
        # Predictor identity embeddings
        pid = predictor_ids.unsqueeze(0).expand(B, -1)  # (B, K)
        emb = self.predictor_embed(pid)  # (B, K, D)
        # Replace missing predictor embeddings with learned missing token
        missing_emb = self.missing_embed.unsqueeze(0).unsqueeze(0).expand(B, K, -1)
        emb = torch.where(mask.unsqueeze(-1).bool(), emb, missing_emb)
        
        # Concatenate: [score, presence, predictor_identity]
        X = torch.cat([scores_in, mask_in, emb], dim=-1)  # (B, K, input_dim)
        
        # ─── Encode Predictor Set ───
        z_pred = self.encoder(X, mask)  # (B, hidden_dim)
        
        # ─── Prior from Variant Type ───
        # softplus ensures alpha_prior >= 1 (valid Dirichlet)
        alpha_prior = F.softplus(self.prior_net(vartype)) + 1.0  # (B, C)
        
        # ─── Evidence from Combined Representation ───
        h = torch.cat([z_pred, alpha_prior], dim=-1)  # (B, hidden_dim + C)
        evidence = F.softplus(self.evidence_net(h))  # (B, C), positive
        
        # ─── Posterior Dirichlet ───
        # Prior + learned evidence (Malinin & Gales, 2018)
        temperature = torch.exp(self.log_temperature)
        alpha_posterior = alpha_prior + evidence * temperature
        
        return alpha_posterior
    
    def predict(
        self,
        scores: torch.Tensor,
        mask: torch.Tensor,
        predictor_ids: torch.Tensor,
        vartype: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Produce clinical predictions from Dirichlet parameters.
        
        Returns:
            dict with:
                - alpha: Dirichlet parameters (B, C)
                - probs: expected class probabilities (B, C)
                - uncertainty: total evidential uncertainty (B,)
                - belief: belief masses for each class (B, C)
                - prediction: hard class prediction (B,)
                - confidence: prediction confidence (B,)
        """
        alpha = self.forward(scores, mask, predictor_ids, vartype)
        
        # Total evidence strength
        S = alpha.sum(dim=-1, keepdim=True)  # (B, 1)
        
        # Expected probabilities
        probs = alpha / S  # (B, C)
        
        # Evidential uncertainty (ignorance mass)
        uncertainty = self.n_classes / S.squeeze(-1)  # (B,)
        
        # Belief masses (Dempster-Shafer interpretation)
        belief = (alpha - 1) / S  # (B, C)
        
        # Hard prediction
        prediction = torch.argmax(probs, dim=-1)  # (B,)
        
        # Confidence = max probability
        confidence = probs.max(dim=-1).values  # (B,)
        
        return {
            "alpha": alpha,
            "probs": probs,
            "uncertainty": uncertainty,
            "belief": belief,
            "prediction": prediction,
            "confidence": confidence,
        }


# =============================================================================
# LOSS FUNCTION
# =============================================================================

class EvidentialLoss(nn.Module):
    """
    Evidential loss for Dirichlet networks (Sensoy et al., 2018).
    
    Combines:
      1. Maximum likelihood under Dirichlet (fits data)
      2. KL regularization (prevents overconfident misclassifications)
    
    Args:
        annealing_step: Number of steps for KL weight annealing
        kl_weight: Final KL regularization weight
    """
    
    def __init__(self, annealing_step: int = 1000, kl_weight: float = 0.1):
        super().__init__()
        self.annealing_step = annealing_step
        self.kl_weight = kl_weight
        self.register_buffer("step", torch.zeros(1))
    
    def forward(
        self,
        alpha: torch.Tensor,
        y_true: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            alpha: (B, C) — Dirichlet parameters from CEFN
            y_true: (B,) — integer class labels (0=P, 1=B, 2=VUS)
        
        Returns:
            loss: scalar loss
            metrics: dict with loss components
        """
        B, C = alpha.shape
        y_onehot = F.one_hot(y_true, num_classes=C).float()  # (B, C)
        
        # ─── 1. Maximum Likelihood Loss ───
        S = alpha.sum(dim=-1, keepdim=True)  # (B, 1)
        
        # log(Dirichlet) = sum((alpha-1)*log(p)) - log(B(alpha))
        # Expected log-likelihood: E_q[log p(y|x)]
        # = ψ(α_y) - ψ(α_0)  where ψ is digamma
        digamma_S = torch.digamma(S).squeeze(-1)  # (B,)
        digamma_alpha = torch.digamma(alpha)  # (B, C)
        
        # Expected log probability for true class
        ll = (digamma_alpha - digamma_S.unsqueeze(-1)) * y_onehot  # (B, C)
        ll = ll.sum(dim=-1)  # (B,)
        
        mle_loss = -ll.mean()
        
        # ─── 2. KL Regularization ───
        # Penalize high evidence for wrong predictions
        # α̃ = y + (1-y)⊙α  (remove evidence for incorrect classes)
        alpha_tilde = y_onehot + (1 - y_onehot) * alpha  # (B, C)
        S_tilde = alpha_tilde.sum(dim=-1, keepdim=True)  # (B, 1)
        
        # KL(Dir(α̃) || Dir(1))
        # = log(Γ(S̃)/Γ(C)) - Σ log(Γ(α̃_i)/Γ(1)) + Σ (α̃_i-1)(ψ(α̃_i)-ψ(S̃))
        digamma_alpha_tilde = torch.digamma(alpha_tilde)  # (B, C)
        digamma_S_tilde = torch.digamma(S_tilde)  # (B, 1)
        
        kl = (
            torch.lgamma(S_tilde).squeeze(-1)  # (B,)
            - torch.lgamma(torch.tensor(C, dtype=alpha.dtype, device=alpha.device))
            - (torch.lgamma(alpha_tilde) - torch.lgamma(torch.ones_like(alpha_tilde))).sum(dim=-1)
            + ((alpha_tilde - 1) * (digamma_alpha_tilde - digamma_S_tilde)).sum(dim=-1)
        )
        
        kl_loss = kl.mean()
        
        # ─── Annealing ───
        annealing_coef = torch.min(
            torch.ones(1, device=alpha.device),
            self.step / self.annealing_step,
        )
        
        # ─── Total Loss ───
        loss = mle_loss + annealing_coef * self.kl_weight * kl_loss
        
        # Update step counter
        self.step += 1
        
        metrics = {
            "mle_loss": mle_loss.item(),
            "kl_loss": kl_loss.item(),
            "annealing_coef": annealing_coef.item(),
            "total_loss": loss.item(),
        }
        
        return loss, metrics


# =============================================================================
# PREDICTOR CONFIGURATION
# =============================================================================

# Predictor IDs (must match order in data preparation)
PREDICTOR_CONFIG = {
    "Evo2": 0,
    "AlphaMissense": 1,
    "CADD": 2,
    "REVEL": 3,
}

# Variant type categories (one-hot encoding)
VARTYPE_CONFIG = {
    "missense": 0,
    "nonsense": 1,
    "splice_site": 2,
    "frameshift": 3,
    "synonymous": 4,
    "other": 5,
}

# Class labels
CLASS_LABELS = {
    0: "Likely Pathogenic",
    1: "Likely Benign",
    2: "Uncertain Significance",
}

# =============================================================================
# ACMG EVIDENCE MAPPING
# =============================================================================

def map_belief_to_acmg(belief: torch.Tensor, uncertainty: torch.Tensor) -> Dict:
    """
    Map Dirichlet belief masses to ACMG/AMP evidence strength categories.
    
    ACMG evidence levels:
      - Supporting: belief > 0.5
      - Moderate: belief > 0.7
      - Strong: belief > 0.9
      - Very Strong: belief > 0.95
    
    Args:
        belief: (C,) — belief masses for each class
        uncertainty: scalar — total evidential uncertainty
    
    Returns:
        dict with ACMG evidence codes and strengths
    """
    belief_np = belief.detach().cpu().numpy()
    uncertainty_np = uncertainty.detach().cpu().item()
    
    p_belief = belief_np[0]  # Pathogenic belief
    b_belief = belief_np[1]  # Benign belief
    
    result = {
        "pathogenic_evidence": None,
        "benign_evidence": None,
        "uncertainty_level": None,
    }
    
    # Pathogenic evidence
    if p_belief > 0.95:
        result["pathogenic_evidence"] = {"code": "PVS1/PS1", "strength": "Very Strong"}
    elif p_belief > 0.90:
        result["pathogenic_evidence"] = {"code": "PS3", "strength": "Strong"}
    elif p_belief > 0.70:
        result["pathogenic_evidence"] = {"code": "PM1/PM5", "strength": "Moderate"}
    elif p_belief > 0.50:
        result["pathogenic_evidence"] = {"code": "PP3", "strength": "Supporting"}
    
    # Benign evidence
    if b_belief > 0.95:
        result["benign_evidence"] = {"code": "BA1", "strength": "Standalone"}
    elif b_belief > 0.90:
        result["benign_evidence"] = {"code": "BS1/BS2", "strength": "Strong"}
    elif b_belief > 0.70:
        result["benign_evidence"] = {"code": "BP1/BP2", "strength": "Moderate"}
    elif b_belief > 0.50:
        result["benign_evidence"] = {"code": "BP4", "strength": "Supporting"}
    
    # Uncertainty
    if uncertainty_np > 0.5:
        result["uncertainty_level"] = "High"
    elif uncertainty_np > 0.3:
        result["uncertainty_level"] = "Moderate"
    else:
        result["uncertainty_level"] = "Low"
    
    return result


# =============================================================================
# INFERENCE WRAPPER
# =============================================================================

class CEFNInference:
    """
    Production-ready inference wrapper for CEFN.
    
    Handles:
      - Score normalization
      - Missing data encoding
      - Variant type inference from metadata
      - ACMG evidence mapping
    """
    
    def __init__(
        self,
        model: CEFN,
        predictor_ids: torch.Tensor,
        device: str = "cpu",
    ):
        self.model = model
        self.model.eval()
        self.predictor_ids = predictor_ids
        self.device = device
    
    @torch.no_grad()
    def predict_single(
        self,
        evo2_score: Optional[float] = None,
        alphamissense_score: Optional[float] = None,
        cadd_score: Optional[float] = None,
        revel_score: Optional[float] = None,
        variant_type: str = "other",
    ) -> Dict:
        """
        Predict for a single variant.
        
        Args:
            evo2_score: Evo2 pathogenicity score (0-1, higher=pathogenic)
            alphamissense_score: AlphaMissense score (0-1)
            cadd_score: CADD PHRED normalized (0-1)
            revel_score: REVEL score (0-1)
            variant_type: One of 'missense', 'nonsense', 'splice_site', 
                         'frameshift', 'synonymous', 'other'
        
        Returns:
            dict with prediction, confidence, uncertainty, ACMG evidence
        """
        K = len(self.predictor_ids)
        
        # Build scores and mask
        raw_scores = [evo2_score, alphamissense_score, cadd_score, revel_score]
        scores = torch.zeros(1, K, device=self.device)
        mask = torch.zeros(1, K, device=self.device)
        
        for i, s in enumerate(raw_scores):
            if s is not None and not np.isnan(s):
                scores[0, i] = float(s)
                mask[0, i] = 1.0
        
        # Build variant type one-hot
        vartype = torch.zeros(1, len(VARTYPE_CONFIG), device=self.device)
        if variant_type in VARTYPE_CONFIG:
            vartype[0, VARTYPE_CONFIG[variant_type]] = 1.0
        else:
            vartype[0, VARTYPE_CONFIG["other"]] = 1.0
        
        # Predict
        result = self.model.predict(
            scores, mask,
            self.predictor_ids.to(self.device),
            vartype,
        )
        
        # Extract
        pred_idx = result["prediction"][0].item()
        confidence = result["confidence"][0].item()
        uncertainty = result["uncertainty"][0].item()
        probs = result["probs"][0].cpu().numpy()
        belief = result["belief"][0]
        
        # ACMG mapping
        acmg = map_belief_to_acmg(belief, result["uncertainty"][0])
        
        return {
            "prediction": CLASS_LABELS.get(pred_idx, "Unknown"),
            "confidence": round(confidence, 4),
            "uncertainty": round(uncertainty, 4),
            "probabilities": {
                "pathogenic": round(float(probs[0]), 4),
                "benign": round(float(probs[1]), 4),
                "vus": round(float(probs[2]), 4),
            },
            "acmg_evidence": acmg,
            "predictors_available": int(mask.sum().item()),
            "predictors_total": K,
        }
    
    @torch.no_grad()
    def predict_batch(
        self,
        scores: np.ndarray,
        mask: np.ndarray,
        vartypes: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Predict for a batch of variants.
        
        Args:
            scores: (B, K) — predictor scores
            mask: (B, K) — binary mask
            vartypes: (B, V) — one-hot variant types
        
        Returns:
            dict with numpy arrays for all outputs
        """
        scores_t = torch.tensor(scores, dtype=torch.float32, device=self.device)
        mask_t = torch.tensor(mask, dtype=torch.float32, device=self.device)
        vartypes_t = torch.tensor(vartypes, dtype=torch.float32, device=self.device)
        
        result = self.model.predict(
            scores_t, mask_t,
            self.predictor_ids.to(self.device),
            vartypes_t,
        )
        
        return {
            "alpha": result["alpha"].cpu().numpy(),
            "probs": result["probs"].cpu().numpy(),
            "uncertainty": result["uncertainty"].cpu().numpy(),
            "belief": result["belief"].cpu().numpy(),
            "prediction": result["prediction"].cpu().numpy(),
            "confidence": result["confidence"].cpu().numpy(),
        }


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("CEFN — Conditional Evidential Fusion Network")
    print("=" * 60)
    
    # Create model
    model = CEFN(
        n_predictors=4,
        n_vartypes=6,
        predictor_embed_dim=16,
        hidden_dim=64,
        n_classes=3,
    )
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")
    
    # Test forward pass
    B, K = 8, 4
    scores = torch.rand(B, K)
    mask = torch.ones(B, K)
    mask[:, 1] = 0  # AlphaMissense missing for some
    mask[:, 2] = 0  # CADD missing for some
    predictor_ids = torch.arange(K)
    vartype = torch.zeros(B, 6)
    vartype[:, 0] = 1  # missense
    
    alpha = model(scores, mask, predictor_ids, vartype)
    print(f"Alpha shape: {alpha.shape}")
    print(f"Alpha range: [{alpha.min().item():.3f}, {alpha.max().item():.3f}]")
    
    # Test prediction
    result = model.predict(scores, mask, predictor_ids, vartype)
    print(f"\nPredictions:")
    for i in range(min(3, B)):
        print(f"  Variant {i}: pred={CLASS_LABELS[result['prediction'][i].item()]}, "
              f"conf={result['confidence'][i].item():.3f}, "
              f"uncertainty={result['uncertainty'][i].item():.3f}")
    
    # Test loss
    loss_fn = EvidentialLoss()
    y_true = torch.randint(0, 3, (B,))
    loss, metrics = loss_fn(alpha, y_true)
    print(f"\nLoss: {loss.item():.4f}")
    print(f"Metrics: {metrics}")
    
    # Test inference wrapper
    print("\n" + "=" * 60)
    print("Inference Wrapper Test")
    print("=" * 60)
    
    inference = CEFNInference(model, predictor_ids)
    
    # Test 1: Both predictors available (missense)
    result1 = inference.predict_single(
        evo2_score=0.85,
        alphamissense_score=0.91,
        variant_type="missense",
    )
    print(f"\nMissense (both available):")
    print(f"  Prediction: {result1['prediction']}")
    print(f"  Confidence: {result1['confidence']}")
    print(f"  Uncertainty: {result1['uncertainty']}")
    print(f"  ACMG: {result1['acmg_evidence']}")
    
    # Test 2: Only Evo2 available (splice site)
    result2 = inference.predict_single(
        evo2_score=0.72,
        variant_type="splice_site",
    )
    print(f"\nSplice site (Evo2 only):")
    print(f"  Prediction: {result2['prediction']}")
    print(f"  Confidence: {result2['confidence']}")
    print(f"  Uncertainty: {result2['uncertainty']}")
    print(f"  ACMG: {result2['acmg_evidence']}")
    
    # Test 3: No predictors (should default to prior)
    result3 = inference.predict_single(
        variant_type="synonymous",
    )
    print(f"\nSynonymous (no predictors):")
    print(f"  Prediction: {result3['prediction']}")
    print(f"  Confidence: {result3['confidence']}")
    print(f"  Uncertainty: {result3['uncertainty']}")
    print(f"  ACMG: {result3['acmg_evidence']}")
    
    print("\n✅ CEFN model ready for training!")
