
import torch
import torch.nn as nn
import torch.nn.functional as F

class ChromatinAwareCRISPR(nn.Module):
    def __init__(
        self,
        evo2_dim: int = 512,
        chromatin_bins: int = 100,
        hidden_dim: int = 256,
        dropout: float = 0.1
    ):
        """
        Evo2 Chromatin-Aware CRISPR Predictor.
        
        Args:
            evo2_dim: Dimension of Evo2 embeddings (512 for 7B)
            chromatin_bins: Number of bins for ATAC-seq profile (e.g., 100 bins for 1kb)
            hidden_dim: Hidden dimension for lightweight heads
            dropout: Dropout rate
        """
        super().__init__()
        
        # --- Chromatin Head ---
        # Predicts ATAC-seq profile from sequence embeddings using local context
        self.chromatin_head = nn.Sequential(
            nn.Linear(evo2_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, chromatin_bins)
        )
        
        # --- Cleavage Head ---
        # Predicts cleavage efficiency using:
        # 1. Global sequence embedding (Cell state / large context)
        # 2. Predicted chromatin accessibility (Mechanism)
        self.cleavage_head = nn.Sequential(
            # Input: Global (512) + Predicted Chromatin (100)
            nn.Linear(evo2_dim + chromatin_bins, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # Log cleavage rate
        )

    def forward(self, global_emb, center_emb):
        """
        Forward pass.
        
        Args:
            global_emb: (B, evo2_dim) - Global sequence embedding
            center_emb: (B, evo2_dim) - Local embedding at cleavage site
            
        Returns:
            Dictionary with:
            - 'log_cleavage': Predicted log reads
            - 'atac_pred': Predicted chromatin profile
        """
        # 1. Predict Chromatin from Local Context
        # We use the center embedding because chromatin state
        # is a local property of the binding site
        atac_pred = self.chromatin_head(center_emb)
        
        # 2. Predict Cleavage
        # Combine global context with predicted chromatin mechanism.
        # This forces the model to use the "chromatin bottleneck".
        combined = torch.cat([global_emb, atac_pred], dim=1)
        log_cleavage = self.cleavage_head(combined)
        
        return {
            'log_cleavage': log_cleavage,
            'atac_pred': atac_pred
        }

    def training_step(self, batch, lambda_chromatin=0.5):
        """
        Compute loss for a batch.
        
        Args:
            batch: Dict with inputs and targets
            lambda_chromatin: Weight for chromatin loss
            
        Returns:
            total_loss, metrics_dict
        """
        # Inputs
        global_emb = batch['global_emb']
        center_emb = batch['center_emb']
        
        # Targets
        true_cleavage = batch['log_cleavage']
        true_atac = batch['atac_signal']
        
        # Forward
        preds = self(global_emb, center_emb)
        
        # Losses
        loss_cleavage = F.mse_loss(preds['log_cleavage'].squeeze(), true_cleavage)
        loss_atac = F.mse_loss(preds['atac_pred'], true_atac)
        
        total_loss = loss_cleavage + lambda_chromatin * loss_atac
        
        return total_loss, {
            'loss': total_loss.item(),
            'mse_cleavage': loss_cleavage.item(),
            'mse_atac': loss_atac.item()
        }
