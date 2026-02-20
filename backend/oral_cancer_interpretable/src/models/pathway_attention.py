"""
Pathway-Attention Network for Interpretable Cancer Genomics

Core Innovation: Biological pathway structure is baked INTO the attention mechanism.
Genes can only attend to other genes in the same pathway (hard masking).

This provides intrinsic interpretability - the attention weights directly 
reflect biological pathway relationships, not a post-hoc approximation.

Key Features:
1. Hard pathway masking via attention_mask = (pathway @ pathway.T) > 0
2. Cox proportional hazards loss for survival prediction
3. Dual interpretability: gene-level and pathway-level attention

Usage:
    python -m oral_cancer_interpretable.src.models.pathway_attention --test
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from torch.utils.data import DataLoader, TensorDataset

# Paths
MODULE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = MODULE_DIR / "data"
RESULTS_DIR = MODULE_DIR / "results"


class PathwayAttention(nn.Module):
    """
    Attention mechanism constrained by biological pathways.
    
    Key innovation: Attention scores are masked so genes can only attend 
    to other genes in the same pathway. This is "interpretable by construction"
    rather than post-hoc explanation.
    
    Args:
        pathway_mask: (n_genes, n_pathways) binary matrix
        embed_dim: Dimension of gene embeddings
        num_heads: Number of attention heads (typically 1 for interpretability)
        dropout: Dropout rate
    """
    
    def __init__(
        self, 
        pathway_mask: np.ndarray,
        embed_dim: int = 64,
        num_heads: int = 1,
        dropout: float = 0.1,
        pathway_names: Optional[List[str]] = None
    ):
        super().__init__()
        
        self.n_genes, self.n_pathways = pathway_mask.shape
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.pathway_names = pathway_names or [f"Pathway_{i}" for i in range(self.n_pathways)]
        
        # Convert pathway mask to torch tensor (genes x pathways)
        # This is the CRITICAL constraint - hardcoded biology, not learned
        self.register_buffer('pathway_mask', torch.from_numpy(pathway_mask).float())
        
        # Create attention mask: gene i can attend to gene j if they share any pathway
        # attention_mask[i, j] = 1 if genes i and j are in the same pathway
        attention_mask_np = (pathway_mask @ pathway_mask.T) > 0  # (n_genes, n_genes)
        attention_mask_np = attention_mask_np.astype(np.float32)
        
        # Convert to additive mask for MultiheadAttention: 0 = allow, -inf = block
        # We'll handle this in forward pass
        self.register_buffer('attention_mask_binary', torch.from_numpy(attention_mask_np))
        
        # Gene embedding layer
        self.gene_embedding = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Self-attention for genes within pathways
        self.gene_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Pathway aggregation: pool gene embeddings within each pathway
        self.pathway_pool = nn.Linear(embed_dim, embed_dim)
        
        # Pathway-level attention (unconstrained - pathways can interact)
        self.pathway_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Output layers for pathway processing
        self.fc_pathway = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Final risk prediction head (Cox proportional hazards style)
        self.pathway_to_risk = nn.Linear(embed_dim, 1)  # Per-pathway risk
        self.global_risk = nn.Linear(self.n_pathways, 1)  # Aggregate risk
        
        # Initialize
        self._init_weights()
        
    def _init_weights(self):
        """Xavier initialization for stability."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self, 
        x: torch.Tensor,
        return_attention: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with pathway-constrained attention.
        
        Args:
            x: (batch_size, n_genes) expression values
            return_attention: If True, return attention weights for visualization
            
        Returns:
            Dictionary with:
            - risk_score: (batch_size, 1) log hazard ratio
            - gene_embeddings: (batch_size, n_genes, embed_dim)
            - gene_attention: (batch_size, n_genes, n_genes) gene-level attention
            - pathway_scores: (batch_size, n_pathways) pathway importance
            - pathway_importance: (batch_size, n_pathways) softmax pathway weights
        """
        batch_size = x.size(0)
        
        # ===== GENE EMBEDDING =====
        # Embed each gene independently: (batch, n_genes) -> (batch, n_genes, embed_dim)
        x_embed = self.gene_embedding(x.unsqueeze(-1))  # Add feature dim
        
        # ===== PATHWAY-CONSTRAINED GENE ATTENTION =====
        # Create additive mask: where binary mask is 0 (blocked), use -inf
        # Shape: (n_genes, n_genes)
        attn_mask = torch.where(
            self.attention_mask_binary.bool(),
            torch.zeros_like(self.attention_mask_binary),
            torch.full_like(self.attention_mask_binary, float('-inf'))
        )
        
        # Apply self-attention with pathway constraint
        gene_attn_out, gene_attn_weights = self.gene_attention(
            x_embed, x_embed, x_embed,
            attn_mask=attn_mask,
            need_weights=True,
            average_attn_weights=True  # Average over heads
        )
        # gene_attn_weights: (batch, n_genes, n_genes)
        
        # ===== PATHWAY AGGREGATION =====
        # Pool gene embeddings to pathway representations
        # pathway_embedding[p] = mean(gene_embeddings[genes_in_p])
        pathway_embeds_list = []
        pathway_gene_counts = []
        
        for p_idx in range(self.n_pathways):
            genes_in_pathway = self.pathway_mask[:, p_idx].bool()  # (n_genes,)
            n_genes_in_pathway = genes_in_pathway.sum().item()
            pathway_gene_counts.append(n_genes_in_pathway)
            
            if n_genes_in_pathway == 0:
                # No genes in this pathway - use zeros
                pathway_embeds_list.append(
                    torch.zeros(batch_size, self.embed_dim, device=x.device)
                )
            else:
                # Average gene embeddings in this pathway
                p_embed = gene_attn_out[:, genes_in_pathway, :].mean(dim=1)  # (batch, embed_dim)
                pathway_embeds_list.append(p_embed)
        
        pathway_embeds = torch.stack(pathway_embeds_list, dim=1)  # (batch, n_pathways, embed_dim)
        pathway_embeds = self.pathway_pool(pathway_embeds)
        
        # ===== PATHWAY-LEVEL ATTENTION =====
        # Pathways can interact freely (no constraint at this level)
        pathway_attn_out, pathway_attn_weights = self.pathway_attention(
            pathway_embeds, pathway_embeds, pathway_embeds,
            need_weights=True,
            average_attn_weights=True
        )
        # pathway_attn_out: (batch, n_pathways, embed_dim)
        # pathway_attn_weights: (batch, n_pathways, n_pathways)
        
        # Process pathway representations
        pathway_processed = self.fc_pathway(pathway_attn_out)
        
        # ===== RISK PREDICTION =====
        # Per-pathway risk contribution
        pathway_risks = self.pathway_to_risk(pathway_processed).squeeze(-1)  # (batch, n_pathways)
        
        # Pathway importance via softmax
        pathway_importance = F.softmax(pathway_risks, dim=-1)  # (batch, n_pathways)
        
        # Final risk score: weighted combination of pathway risks
        risk_score = self.global_risk(pathway_risks)  # (batch, 1)
        
        output = {
            'risk_score': risk_score,
            'gene_embeddings': x_embed,
            'gene_attention': gene_attn_weights if return_attention else None,
            'pathway_embeddings': pathway_embeds,
            'pathway_attention': pathway_attn_weights if return_attention else None,
            'pathway_risks': pathway_risks,
            'pathway_importance': pathway_importance,
        }
        
        return output
    
    def get_pathway_importance_df(self, x: torch.Tensor) -> pd.DataFrame:
        """
        Get interpretable pathway importance for a batch of patients.
        
        Returns DataFrame with pathway names and importance scores.
        """
        self.eval()
        with torch.no_grad():
            out = self.forward(x, return_attention=False)
        
        scores = out['pathway_importance'].cpu().numpy()
        
        df = pd.DataFrame(scores, columns=self.pathway_names)
        return df
    
    def get_top_pathways(self, x: torch.Tensor, top_k: int = 5) -> Dict[int, List[Tuple[str, float]]]:
        """
        Get top-k most important pathways for each patient.
        
        Returns:
            Dict mapping patient index to list of (pathway_name, importance) tuples
        """
        importance_df = self.get_pathway_importance_df(x)
        
        results = {}
        for i, row in importance_df.iterrows():
            sorted_pathways = row.sort_values(ascending=False)
            results[i] = [(name, score) for name, score in sorted_pathways.head(top_k).items()]
        
        return results


class CoxLoss(nn.Module):
    """
    Cox proportional hazards partial likelihood loss.
    
    This is the standard loss for survival analysis with right-censored data 
    (some patients still alive at last follow-up).
    
    For each patient who had an event, we compute:
    log(hazard_i) - log(sum_{j: time_j >= time_i} hazard_j)
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self, 
        risk_scores: torch.Tensor,  # (batch, 1) - log hazard ratios
        event_times: torch.Tensor,  # (batch,) - time to event or censoring
        event_indicators: torch.Tensor  # (batch,) - 1 if event occurred, 0 if censored
    ) -> torch.Tensor:
        """
        Compute negative log partial likelihood.
        """
        # Remove any patients with invalid times
        valid_mask = event_times > 0
        if not valid_mask.all():
            risk_scores = risk_scores[valid_mask]
            event_times = event_times[valid_mask]
            event_indicators = event_indicators[valid_mask]
        
        if len(risk_scores) == 0:
            return torch.tensor(0.0, requires_grad=True, device=risk_scores.device)
        
        # Flatten risk scores
        risk = risk_scores.squeeze(-1)  # (batch,)
        
        # Sort by event time (descending for cumsum logic)
        idx = torch.argsort(event_times, descending=True)
        risk = risk[idx]
        event_indicators = event_indicators[idx]
        event_times = event_times[idx]
        
        # Compute log cumulative hazard
        # At position i (sorted descending by time), all patients with index >= i
        # are "at risk" (their event time >= current time)
        log_cum_hazard = torch.logcumsumexp(risk, dim=0)
        
        # Only compute loss for patients who had events
        event_mask = event_indicators.bool()
        
        if event_mask.sum() == 0:
            return torch.tensor(0.0, requires_grad=True, device=risk.device)
        
        # Negative log partial likelihood
        # For each event: risk_score - log(sum of hazards at risk)
        loss = -torch.sum(risk[event_mask] - log_cum_hazard[event_mask])
        
        # Normalize by number of events
        loss = loss / event_mask.sum()
        
        return loss


class ConcordanceIndex:
    """
    Compute concordance index (C-index) for survival prediction.
    
    C-index measures how well the model ranks patients by risk:
    - 0.5 = random
    - 1.0 = perfect (higher risk = shorter survival)
    """
    
    @staticmethod
    def compute(
        risk_scores: torch.Tensor, 
        event_times: torch.Tensor, 
        event_indicators: torch.Tensor
    ) -> float:
        """
        Compute C-index.
        
        For each pair of patients (i, j) where i had an event before j,
        check if model correctly ranked i's risk higher than j's.
        """
        risk = risk_scores.squeeze().cpu().numpy()
        times = event_times.cpu().numpy()
        events = event_indicators.cpu().numpy()
        
        concordant = 0
        discordant = 0
        tied = 0
        
        n = len(risk)
        for i in range(n):
            if events[i] == 0:
                continue  # Skip censored patients as the "event" patient
            
            for j in range(n):
                if i == j:
                    continue
                if times[i] >= times[j]:
                    continue  # i's event must be before j's time
                
                # i had event before j's last observation
                # Check if risk[i] > risk[j] (correct ranking)
                if risk[i] > risk[j]:
                    concordant += 1
                elif risk[i] < risk[j]:
                    discordant += 1
                else:
                    tied += 1
        
        total = concordant + discordant + tied
        if total == 0:
            return 0.5
        
        return (concordant + 0.5 * tied) / total


class PathwayAttentionTrainer:
    """
    Training loop with validation, early stopping, and logging.
    """
    
    def __init__(
        self,
        model: PathwayAttention,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str = None
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        self.criterion = CoxLoss()
        self.history = {'train_loss': [], 'val_loss': [], 'val_cindex': []}
        
    def _create_batch(self, X, y_time, y_event, indices):
        """Create batch tensors."""
        return {
            'X': torch.from_numpy(X[indices]).float().to(self.device),
            'y_time': torch.from_numpy(y_time[indices]).float().to(self.device),
            'y_event': torch.from_numpy(y_event[indices]).float().to(self.device)
        }
        
    def train_epoch(self, X, y_time, y_event, batch_size=32) -> float:
        """Train for one epoch."""
        self.model.train()
        
        n_samples = len(X)
        indices = np.random.permutation(n_samples)
        total_loss = 0
        n_batches = 0
        
        for i in range(0, n_samples, batch_size):
            batch_idx = indices[i:i+batch_size]
            batch = self._create_batch(X, y_time, y_event, batch_idx)
            
            self.optimizer.zero_grad()
            
            output = self.model(batch['X'], return_attention=False)
            loss = self.criterion(output['risk_score'], batch['y_time'], batch['y_event'])
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / max(n_batches, 1)
    
    def validate(self, X, y_time, y_event) -> Tuple[float, float]:
        """Validate and compute C-index."""
        self.model.eval()
        
        with torch.no_grad():
            X_tensor = torch.from_numpy(X).float().to(self.device)
            y_time_tensor = torch.from_numpy(y_time).float().to(self.device)
            y_event_tensor = torch.from_numpy(y_event).float().to(self.device)
            
            output = self.model(X_tensor, return_attention=False)
            loss = self.criterion(output['risk_score'], y_time_tensor, y_event_tensor)
            
            cindex = ConcordanceIndex.compute(
                output['risk_score'], y_time_tensor, y_event_tensor
            )
        
        return loss.item(), cindex
    
    def fit(
        self, 
        X_train, y_time_train, y_event_train,
        X_val, y_time_val, y_event_val,
        epochs: int = 100,
        batch_size: int = 32,
        patience: int = 15,
        save_path: Optional[str] = None
    ):
        """Train with early stopping."""
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"Training on {len(X_train)} samples, validating on {len(X_val)} samples")
        print(f"Device: {self.device}")
        print("-" * 60)
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(X_train, y_time_train, y_event_train, batch_size)
            val_loss, val_cindex = self.validate(X_val, y_time_val, y_event_val)
            
            self.scheduler.step(val_loss)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_cindex'].append(val_cindex)
            
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                  f"C-index: {val_cindex:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                if save_path:
                    torch.save(self.model.state_dict(), save_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        if save_path and Path(save_path).exists():
            self.model.load_state_dict(torch.load(save_path))
            print(f"\nLoaded best model from {save_path}")
        
        return self.history


def test_model():
    """Test the model architecture with synthetic data."""
    print("="*60)
    print("PATHWAY ATTENTION MODEL TEST")
    print("="*60)
    
    # Create synthetic data
    batch_size = 8
    n_genes = 100
    n_pathways = 10
    
    # Random pathway mask (10% of genes in each pathway)
    np.random.seed(42)
    pathway_mask = (np.random.rand(n_genes, n_pathways) < 0.1).astype(np.float32)
    
    # Ensure each gene is in at least one pathway
    for i in range(n_genes):
        if pathway_mask[i].sum() == 0:
            pathway_mask[i, np.random.randint(n_pathways)] = 1
    
    print(f"\nPathway mask shape: {pathway_mask.shape}")
    print(f"Genes per pathway: {pathway_mask.sum(axis=0)}")
    
    # Create model
    model = PathwayAttention(
        pathway_mask=pathway_mask,
        embed_dim=32,
        num_heads=1,
        dropout=0.1
    )
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    print("\n[1] Testing forward pass...")
    X = torch.randn(batch_size, n_genes)
    output = model(X)
    
    print(f"   Input shape: {X.shape}")
    print(f"   Risk scores: {output['risk_score'].squeeze().tolist()}")
    print(f"   Pathway importance shape: {output['pathway_importance'].shape}")
    
    # Test interpretability
    print("\n[2] Testing interpretability...")
    top_pathways = model.get_top_pathways(X, top_k=3)
    print(f"   Top 3 pathways for patient 0: {top_pathways[0]}")
    
    # Test Cox loss
    print("\n[3] Testing Cox loss...")
    y_time = torch.tensor([100, 200, 150, 300, 50, 250, 180, 120], dtype=torch.float)
    y_event = torch.tensor([1, 0, 1, 1, 1, 0, 1, 0], dtype=torch.float)
    
    criterion = CoxLoss()
    loss = criterion(output['risk_score'], y_time, y_event)
    print(f"   Cox loss: {loss.item():.4f}")
    
    # Test backward pass
    print("\n[4] Testing backward pass...")
    loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    print(f"   Gradient norm: {grad_norm:.4f}")
    
    # Test C-index
    print("\n[5] Testing C-index...")
    with torch.no_grad():
        output = model(X)
    cindex = ConcordanceIndex.compute(output['risk_score'], y_time, y_event)
    print(f"   C-index: {cindex:.4f}")
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED")
    print("="*60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Test model architecture")
    parser.add_argument("--train", action="store_true", help="Train on TCGA data")
    args = parser.parse_args()
    
    if args.test:
        test_model()
    elif args.train:
        print("Training mode - use train.py instead")
    else:
        test_model()


if __name__ == "__main__":
    main()
