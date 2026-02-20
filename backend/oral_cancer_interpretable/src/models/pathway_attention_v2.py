"""
Pathway Attention V2 - Soft Regularization

Key Insight: Hard masking blocks gradient flow between isolated genes.
Solution: Allow full attention but penalize cross-pathway communication.

This gives the model flexibility to learn while encouraging biological structure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, List


class PathwayAttentionV2(nn.Module):
    """
    Soft pathway attention with regularization.
    
    Unlike V1 (hard masking), this allows full gene-gene attention 
    but adds a regularization loss that penalizes attention between
    genes that don't share pathways.
    
    This preserves gradient flow while encouraging biological structure.
    """
    
    def __init__(
        self, 
        pathway_mask: np.ndarray,
        embed_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.2,
        pathway_names: Optional[List[str]] = None
    ):
        super().__init__()
        
        self.n_genes, self.n_pathways = pathway_mask.shape
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.pathway_names = pathway_names or [f"P{i}" for i in range(self.n_pathways)]
        
        # Store pathway mask for regularization (NOT hard masking)
        self.register_buffer('pathway_mask', torch.from_numpy(pathway_mask).float())
        
        # Pathway co-membership matrix: how many pathways do genes i and j share?
        pathway_cooccurrence = (pathway_mask @ pathway_mask.T).astype(np.float32)
        self.register_buffer('pathway_cooccurrence', torch.from_numpy(pathway_cooccurrence))
        
        # Normalized co-occurrence for regularization [0, 1]
        max_cooc = pathway_cooccurrence.max()
        if max_cooc > 0:
            cooc_norm = pathway_cooccurrence / max_cooc
        else:
            cooc_norm = pathway_cooccurrence
        self.register_buffer('pathway_cooccurrence_norm', torch.from_numpy(cooc_norm))
        
        # Gene embedding
        self.gene_embed = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # FULL attention (no masking) - allows gradient flow
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Pathway pooling layer
        self.pathway_fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Risk prediction from pooled gene representation
        self.risk_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass with full attention.
        
        Args:
            x: (batch, n_genes) gene expression
            
        Returns:
            Dict with risk_score, pathway_importance, attention_weights
        """
        batch_size = x.size(0)
        
        # Embed genes: (batch, n_genes, embed_dim)
        x_embed = self.gene_embed(x.unsqueeze(-1))
        
        # Full self-attention (no masking!)
        attn_out, attn_weights = self.attention(
            x_embed, x_embed, x_embed,
            need_weights=True,
            average_attn_weights=True  # Average over heads
        )
        # attn_out: (batch, n_genes, embed_dim)
        # attn_weights: (batch, n_genes, n_genes)
        
        # ===== PATHWAY IMPORTANCE (Interpretability) =====
        # Measure how much attention each pathway's genes receive
        pathway_scores = []
        for p_idx in range(self.n_pathways):
            genes_in_p = self.pathway_mask[:, p_idx].bool()
            n_genes_p = genes_in_p.sum().item()
            
            if n_genes_p == 0:
                pathway_scores.append(torch.zeros(batch_size, device=x.device))
            else:
                # Sum of attention TO genes in this pathway
                p_attention = attn_weights[:, :, genes_in_p].sum(dim=(1, 2)) / n_genes_p
                pathway_scores.append(p_attention)
        
        pathway_scores = torch.stack(pathway_scores, dim=1)  # (batch, n_pathways)
        pathway_importance = F.softmax(pathway_scores, dim=1)
        
        # ===== RISK PREDICTION =====
        # Global mean pooling
        global_repr = attn_out.mean(dim=1)  # (batch, embed_dim)
        risk_score = self.risk_head(global_repr)  # (batch, 1)
        
        return {
            'risk_score': risk_score,
            'pathway_importance': pathway_importance,
            'attention_weights': attn_weights,
            'gene_embeddings': attn_out
        }
    
    def pathway_regularization_loss(self, attn_weights: torch.Tensor) -> torch.Tensor:
        """
        Soft regularization: penalize attention between genes that don't share pathways.
        
        This encourages the model to respect biological structure without
        completely blocking gradient flow.
        
        Args:
            attn_weights: (batch, n_genes, n_genes) attention weights
            
        Returns:
            Scalar regularization loss
        """
        # Penalty matrix: 1 - normalized_cooccurrence
        # High penalty for gene pairs that share no pathways
        penalty = 1.0 - self.pathway_cooccurrence_norm  # (n_genes, n_genes)
        
        # Weight attention by penalty
        # attention * penalty = high when attending to unrelated genes (bad)
        reg_loss = (attn_weights * penalty.unsqueeze(0)).mean()
        
        return reg_loss


class CoxLossV2(nn.Module):
    """
    Improved Cox loss with numerical stability.
    """
    
    def __init__(self, smoothing: float = 1e-7):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(
        self, 
        risk_scores: torch.Tensor,
        event_times: torch.Tensor,
        event_indicators: torch.Tensor
    ) -> torch.Tensor:
        """
        Negative log partial likelihood.
        """
        # Ensure valid inputs
        valid = (event_times > 0) & torch.isfinite(risk_scores.squeeze())
        if not valid.any():
            return torch.tensor(0.0, requires_grad=True, device=risk_scores.device)
        
        risk = risk_scores.squeeze()[valid]
        times = event_times[valid]
        events = event_indicators[valid]
        
        # Sort by time descending (for cumulative sum)
        idx = torch.argsort(times, descending=True)
        risk = risk[idx]
        events = events[idx]
        
        # Log cumulative sum of exp(risk) - numerically stable
        log_cum_hazard = torch.logcumsumexp(risk, dim=0)
        
        # Only compute loss for events
        event_mask = events.bool()
        if not event_mask.any():
            return torch.tensor(0.0, requires_grad=True, device=risk.device)
        
        # Negative partial log-likelihood
        loss = -torch.mean(risk[event_mask] - log_cum_hazard[event_mask])
        
        return loss


def compute_cindex(risk_scores, event_times, event_indicators):
    """
    Concordance index - probability that model correctly ranks pairs.
    
    C-index = 0.5: random
    C-index = 1.0: perfect
    """
    risk = risk_scores.squeeze().detach().cpu().numpy()
    times = event_times.detach().cpu().numpy()
    events = event_indicators.detach().cpu().numpy()
    
    concordant = 0
    discordant = 0
    tied = 0
    n = len(risk)
    
    for i in range(n):
        if events[i] == 0:
            continue  # Can only compare when i had event
        for j in range(n):
            if i == j:
                continue
            if times[i] >= times[j]:
                continue  # i's event must be before j's last observation
            
            # i had event first - should have higher risk
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


class PathwayTrainerV2:
    """
    Trainer with pathway regularization and C-index tracking.
    """
    
    def __init__(
        self,
        model: PathwayAttentionV2,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        pathway_reg_weight: float = 0.1,
        device: str = None
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.pathway_reg_weight = pathway_reg_weight
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=10
        )
        self.criterion = CoxLossV2()
        
        self.history = {
            'train_cox': [], 'train_reg': [], 'train_total': [],
            'val_loss': [], 'val_cindex': []
        }
    
    def train_epoch(self, X, y_time, y_event, batch_size=32):
        self.model.train()
        
        n = len(X)
        indices = np.random.permutation(n)
        total_cox = 0
        total_reg = 0
        n_batches = 0
        
        for i in range(0, n, batch_size):
            batch_idx = indices[i:i+batch_size]
            x = torch.from_numpy(X[batch_idx]).float().to(self.device)
            t = torch.from_numpy(y_time[batch_idx]).float().to(self.device)
            e = torch.from_numpy(y_event[batch_idx]).float().to(self.device)
            
            self.optimizer.zero_grad()
            
            output = self.model(x)
            
            # Cox loss (survival prediction)
            cox_loss = self.criterion(output['risk_score'], t, e)
            
            # Pathway regularization (soft structure)
            reg_loss = self.model.pathway_regularization_loss(output['attention_weights'])
            
            # Combined loss
            loss = cox_loss + self.pathway_reg_weight * reg_loss
            
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                total_cox += cox_loss.item()
                total_reg += reg_loss.item()
                n_batches += 1
        
        return total_cox / max(n_batches, 1), total_reg / max(n_batches, 1)
    
    def validate(self, X, y_time, y_event):
        self.model.eval()
        
        with torch.no_grad():
            x = torch.from_numpy(X).float().to(self.device)
            t = torch.from_numpy(y_time).float().to(self.device)
            e = torch.from_numpy(y_event).float().to(self.device)
            
            output = self.model(x)
            loss = self.criterion(output['risk_score'], t, e).item()
            cindex = compute_cindex(output['risk_score'], t, e)
        
        return loss, cindex
    
    def fit(
        self,
        X_train, y_time_train, y_event_train,
        X_val, y_time_val, y_event_val,
        epochs: int = 200,
        batch_size: int = 32,
        patience: int = 30,
        save_path: str = None
    ):
        best_cindex = 0.5
        patience_counter = 0
        best_state = None
        
        print(f"Training on {len(X_train)} samples, validating on {len(X_val)}")
        print(f"Device: {self.device}, Pathway reg weight: {self.pathway_reg_weight}")
        print("-" * 60)
        
        for epoch in range(epochs):
            cox_loss, reg_loss = self.train_epoch(
                X_train, y_time_train, y_event_train, batch_size
            )
            val_loss, val_cindex = self.validate(X_val, y_time_val, y_event_val)
            
            self.scheduler.step(val_cindex)
            
            self.history['train_cox'].append(cox_loss)
            self.history['train_reg'].append(reg_loss)
            self.history['train_total'].append(cox_loss + self.pathway_reg_weight * reg_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_cindex'].append(val_cindex)
            
            if epoch % 10 == 0 or val_cindex > best_cindex:
                print(f"Epoch {epoch+1:3d} | Cox: {cox_loss:.4f} | Reg: {reg_loss:.4f} | "
                      f"Val: {val_loss:.4f} | C-index: {val_cindex:.4f}")
            
            # Early stopping on C-index
            if val_cindex > best_cindex:
                best_cindex = val_cindex
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                if save_path:
                    torch.save(best_state, save_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        if best_state:
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})
        
        print(f"\nBest Val C-index: {best_cindex:.4f}")
        return self.history
    
    def get_pathway_importance(self, X):
        """Get average pathway importance across samples."""
        self.model.eval()
        
        with torch.no_grad():
            x = torch.from_numpy(X).float().to(self.device)
            output = self.model(x)
            importance = output['pathway_importance'].mean(dim=0).cpu().numpy()
        
        # Sort by importance
        sorted_idx = np.argsort(importance)[::-1]
        
        results = []
        for idx in sorted_idx:
            name = self.model.pathway_names[idx]
            score = importance[idx]
            results.append((name, score))
        
        return results
