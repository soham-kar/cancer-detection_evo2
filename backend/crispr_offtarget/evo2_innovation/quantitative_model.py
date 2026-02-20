"""
Quantitative Cleavage Regression Model

The key insight: Predict CONTINUOUS cleavage rates, not binary labels.
This is where Evo2 adds real value over simple counting.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple

from config import HIDDEN_DIM, BATCH_SIZE, LEARNING_RATE, SEED
from feature_extraction import TOTAL_FEATURE_DIM


class Evo2CleavageRegressor(nn.Module):
    """
    Regression model for predicting quantitative cleavage efficiency.
    
    Architecture:
    - Input: Concatenated Evo2 features (mismatch + context + PAM + global + biophysics)
    - Hidden: 2-layer MLP with dropout
    - Output: Single value (log-transformed cleavage rate)
    """
    
    def __init__(
        self,
        input_dim: int = TOTAL_FEATURE_DIM,
        hidden_dim: int = 256,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features [batch, input_dim]
        
        Returns:
            Predicted log(cleavage_rate + 1) [batch, 1]
        """
        return self.network(x)


class Evo2EnsembleModel(nn.Module):
    """
    Ensemble model combining:
    1. Evo2-based regression
    2. Biophysical heuristic
    3. Learned position weights
    
    Uses residual learning: predict correction to heuristic baseline.
    """
    
    def __init__(
        self,
        evo2_dim: int = HIDDEN_DIM * 4,  # 4 types of Evo2 embeddings
        biophys_dim: int = 7,             # Biophysical features
        hidden_dim: int = 128,
        dropout: float = 0.2
    ):
        super().__init__()
        
        # Evo2 branch (deep)
        self.evo2_branch = nn.Sequential(
            nn.Linear(evo2_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Biophysics branch (shallow)
        self.biophys_branch = nn.Sequential(
            nn.Linear(biophys_dim, 32),
            nn.ReLU()
        )
        
        # Fusion head
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim // 2 + 32, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
        # Learned position weights (20 positions)
        self.position_weights = nn.Parameter(torch.ones(20))
    
    def forward(
        self,
        evo2_features: torch.Tensor,
        biophys_features: torch.Tensor,
        mismatch_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with ensemble fusion.
        
        Args:
            evo2_features: Evo2 embeddings [batch, evo2_dim]
            biophys_features: Biophysical features [batch, biophys_dim]
            mismatch_mask: Optional [batch, 20] binary mask of mismatch positions
        
        Returns:
            Predicted cleavage rate [batch, 1]
        """
        # Process branches
        evo2_out = self.evo2_branch(evo2_features)
        biophys_out = self.biophys_branch(biophys_features)
        
        # Fusion
        combined = torch.cat([evo2_out, biophys_out], dim=1)
        output = self.fusion(combined)
        
        # Optional: Add learned position penalty (residual)
        if mismatch_mask is not None:
            pos_weights = torch.sigmoid(self.position_weights)  # 0-1
            penalty = (mismatch_mask * pos_weights).sum(dim=1, keepdim=True)
            output = output - 0.1 * penalty  # Residual correction
        
        return output


class HeuristicBaseline(nn.Module):
    """
    Simple heuristic baseline for comparison.
    Uses only mismatch count and position weights.
    """
    
    def __init__(self, learn_weights: bool = True):
        super().__init__()
        
        if learn_weights:
            # Learnable position weights
            self.position_weights = nn.Parameter(torch.ones(20))
        else:
            # Fixed seed weights
            fixed_weights = torch.tensor([0.5]*7 + [1.0]*10 + [3.0]*3)
            self.register_buffer('position_weights', fixed_weights)
        
        self.learn_weights = learn_weights
        
        # Simple linear scaling
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(0.0))
    
    def forward(self, mismatch_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mismatch_mask: [batch, 20] binary mask
        
        Returns:
            Predicted score [batch, 1]
        """
        if self.learn_weights:
            weights = torch.softplus(self.position_weights)  # Keep positive
        else:
            weights = self.position_weights
        
        # Weighted sum (more mismatches = lower cleavage)
        penalty = (mismatch_mask * weights).sum(dim=1, keepdim=True)
        
        # Invert: fewer mismatches = higher cleavage
        return -self.scale * penalty + self.bias


def create_mismatch_mask(positions: list, max_len: int = 20) -> torch.Tensor:
    """Convert mismatch positions to binary mask."""
    mask = torch.zeros(max_len)
    for pos in positions:
        if 0 <= pos < max_len:
            mask[pos] = 1.0
    return mask


def spearman_correlation(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute Spearman rank correlation."""
    from scipy.stats import spearmanr
    
    pred_np = pred.detach().cpu().numpy().flatten()
    target_np = target.detach().cpu().numpy().flatten()
    
    corr, _ = spearmanr(pred_np, target_np)
    return corr if not np.isnan(corr) else 0.0


def pearson_correlation(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute Pearson correlation."""
    pred_np = pred.detach().cpu().numpy().flatten()
    target_np = target.detach().cpu().numpy().flatten()
    
    corr = np.corrcoef(pred_np, target_np)[0, 1]
    return corr if not np.isnan(corr) else 0.0


if __name__ == "__main__":
    # Test model dimensions
    batch_size = 8
    
    # Test single regressor
    model = Evo2CleavageRegressor()
    x = torch.randn(batch_size, TOTAL_FEATURE_DIM)
    out = model(x)
    print(f"Regressor output shape: {out.shape}")  # [8, 1]
    
    # Test ensemble
    ensemble = Evo2EnsembleModel()
    evo2_feat = torch.randn(batch_size, HIDDEN_DIM * 4)
    biophys_feat = torch.randn(batch_size, 7)
    mm_mask = torch.randint(0, 2, (batch_size, 20)).float()
    
    out = ensemble(evo2_feat, biophys_feat, mm_mask)
    print(f"Ensemble output shape: {out.shape}")  # [8, 1]
    
    # Test heuristic
    heuristic = HeuristicBaseline(learn_weights=True)
    out = heuristic(mm_mask)
    print(f"Heuristic output shape: {out.shape}")  # [8, 1]
    print(f"Learned position weights: {heuristic.position_weights.data}")
