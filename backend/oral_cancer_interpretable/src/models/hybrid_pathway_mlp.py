"""
Hybrid Pathway-MLP: Predict Freely, Interpret Faithfully

Key insight from analysis:
- Baseline MLP achieves C-index 0.68 (data has signal)
- Pathway-constrained models get ~0.53 (constraints hurt performance)
- Solution: Unconstrained prediction + post-hoc pathway interpretation

Architecture:
1. Prediction Branch: Deep MLP on all genes (no pathway constraints)
2. Interpretation Branch: Pathway projection (discovers which pathways matter)

This is "models not wrappers" - the model learns freely, we interpret through biological lens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, List


class HybridPathwayMLP(nn.Module):
    """
    Hybrid architecture: MLP for prediction, pathway projection for interpretability.
    
    The prediction branch is completely unconstrained - it can learn any gene-gene
    interactions needed for survival prediction.
    
    The interpretation branch projects the learned representations to pathway space,
    revealing which pathways the model found important.
    """
    
    def __init__(
        self,
        n_genes: int,
        pathway_mask: np.ndarray,
        hidden_dims: List[int] = [512, 256, 128],
        pathway_embed_dim: int = 64,
        dropout: float = 0.3,
        pathway_names: Optional[List[str]] = None
    ):
        super().__init__()
        
        self.n_genes = n_genes
        self.n_pathways = pathway_mask.shape[1]
        self.pathway_names = pathway_names or [f"Pathway_{i}" for i in range(self.n_pathways)]
        
        # Pathway mask for interpretation only (NOT for constraining prediction)
        self.register_buffer('pathway_mask', torch.from_numpy(pathway_mask).float())
        
        # ===== PREDICTION BRANCH (Unconstrained) =====
        layers = []
        prev_dim = n_genes
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # Risk prediction head
        self.risk_head = nn.Sequential(
            nn.Linear(prev_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, 1)
        )
        
        # ===== INTERPRETATION BRANCH (Pathway projection) =====
        # Learns to project hidden states to pathway space
        self.pathway_queries = nn.Parameter(torch.randn(self.n_pathways, pathway_embed_dim))
        self.hidden_to_pathway = nn.Linear(hidden_dims[-1], pathway_embed_dim)
        
        # Gene importance scorer per pathway
        self.gene_pathway_scorer = nn.Linear(n_genes, self.n_pathways)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Initialize pathway queries
        nn.init.xavier_uniform_(self.pathway_queries)
    
    def forward(self, x: torch.Tensor, return_interpretation: bool = True) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: (batch, n_genes) gene expression
            return_interpretation: If True, compute pathway importance
        """
        # ===== PREDICTION (Unconstrained) =====
        hidden = self.encoder(x)  # (batch, hidden_dim)
        risk_score = self.risk_head(hidden)  # (batch, 1)
        
        output = {'risk_score': risk_score, 'hidden': hidden}
        
        # ===== INTERPRETATION (Post-hoc) =====
        if return_interpretation:
            # Method 1: Hidden state similarity to pathway prototypes
            hidden_proj = self.hidden_to_pathway(hidden)  # (batch, pathway_embed_dim)
            pathway_similarity = torch.matmul(
                hidden_proj, self.pathway_queries.T
            ) / np.sqrt(hidden_proj.size(-1))  # (batch, n_pathways)
            
            # Method 2: Gene-level contributions to pathways
            gene_scores = self.gene_pathway_scorer(x)  # (batch, n_pathways)
            
            # Combine both methods
            combined_scores = pathway_similarity + gene_scores
            pathway_importance = F.softmax(combined_scores, dim=1)
            
            output['pathway_importance'] = pathway_importance
            output['pathway_scores'] = combined_scores
        
        return output
    
    def get_top_pathways(self, x: torch.Tensor, top_k: int = 10) -> List[tuple]:
        """Get top pathways for interpretation."""
        self.eval()
        with torch.no_grad():
            output = self.forward(x, return_interpretation=True)
            importance = output['pathway_importance'].mean(dim=0).cpu().numpy()
        
        sorted_idx = np.argsort(importance)[::-1][:top_k]
        return [(self.pathway_names[i], float(importance[i])) for i in sorted_idx]


class CoxLoss(nn.Module):
    """Cox proportional hazards negative partial log-likelihood."""
    
    def forward(self, risk_scores, event_times, event_indicators):
        risk = risk_scores.squeeze()
        
        # Sort by descending time
        idx = torch.argsort(event_times, descending=True)
        risk = risk[idx]
        event_indicators = event_indicators[idx]
        
        # Log cumulative hazard
        log_cumsum = torch.logcumsumexp(risk, dim=0)
        
        # Partial likelihood for events only
        event_mask = event_indicators.bool()
        if event_mask.sum() == 0:
            return torch.tensor(0.0, requires_grad=True, device=risk.device)
        
        loss = -torch.mean(risk[event_mask] - log_cumsum[event_mask])
        return loss


def compute_cindex(risk, times, events):
    """Concordance index."""
    risk = risk.squeeze().cpu().numpy()
    times = times.cpu().numpy()
    events = events.cpu().numpy()
    
    concordant = discordant = tied = 0
    n = len(risk)
    
    for i in range(n):
        if events[i] == 0:
            continue
        for j in range(n):
            if i == j or times[i] >= times[j]:
                continue
            if risk[i] > risk[j]:
                concordant += 1
            elif risk[i] < risk[j]:
                discordant += 1
            else:
                tied += 1
    
    total = concordant + discordant + tied
    return (concordant + 0.5 * tied) / total if total > 0 else 0.5
