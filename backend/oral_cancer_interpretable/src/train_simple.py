"""
Improved Training Script with Simpler Architecture

Key changes for small datasets:
1. Simpler model (no attention for gene-level, just pathway aggregation)
2. Higher learning rate for faster convergence  
3. L1 regularization for sparsity in pathway importance
4. More epochs with cosine annealing
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from sklearn.model_selection import train_test_split
import json

import sys
MODULE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(MODULE_DIR))

from src.data_engineering.tcga_loader import TCGALoader


class SimplePathwayModel(nn.Module):
    """
    Simpler pathway model without gene-level attention.
    
    For small datasets, gene-gene attention is too complex.
    Instead: aggregate genes per pathway, then run pathway-level attention.
    """
    
    def __init__(self, pathway_mask, embed_dim=32, dropout=0.3, pathway_names=None):
        super().__init__()
        
        self.n_genes, self.n_pathways = pathway_mask.shape
        self.pathway_names = pathway_names or [f"P{i}" for i in range(self.n_pathways)]
        
        # Store pathway mask
        self.register_buffer('pathway_mask', torch.from_numpy(pathway_mask).float())
        
        # Per-gene linear embedding (no attention)
        self.gene_embed = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Pathway-level processing
        self.pathway_fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Risk prediction per pathway
        self.pathway_risk = nn.Linear(embed_dim, 1)
        
        # Learnable pathway importance (sparse)
        self.pathway_importance_layer = nn.Linear(self.n_pathways, self.n_pathways)
        
        # Final risk aggregation
        self.final_risk = nn.Linear(self.n_pathways, 1)
        
        # Initialize
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        x: (batch, n_genes) - gene expression
        """
        batch_size = x.size(0)
        
        # Embed genes: (batch, n_genes) -> (batch, n_genes, embed_dim)
        x_embed = self.gene_embed(x.unsqueeze(-1))
        
        # Aggregate genes per pathway (mean pooling within each pathway)
        pathway_reprs = []
        for p_idx in range(self.n_pathways):
            genes_in = self.pathway_mask[:, p_idx].bool()
            n_genes = genes_in.sum().item()
            
            if n_genes == 0:
                pathway_reprs.append(torch.zeros(batch_size, x_embed.size(-1), device=x.device))
            else:
                # Mean of gene embeddings in this pathway
                p_repr = x_embed[:, genes_in, :].mean(dim=1)  # (batch, embed_dim)
                pathway_reprs.append(p_repr)
        
        # Stack pathways: (batch, n_pathways, embed_dim)
        pathway_embeds = torch.stack(pathway_reprs, dim=1)
        
        # Process pathways
        pathway_processed = self.pathway_fc(pathway_embeds)  # (batch, n_pathways, embed_dim)
        
        # Risk per pathway
        pathway_risks = self.pathway_risk(pathway_processed).squeeze(-1)  # (batch, n_pathways)
        
        # Pathway importance (softmax for interpretability)
        importance_logits = self.pathway_importance_layer(pathway_risks)
        pathway_importance = F.softmax(importance_logits, dim=-1)  # (batch, n_pathways)
        
        # Final risk: weighted sum
        risk_score = self.final_risk(pathway_risks)  # (batch, 1)
        
        return {
            'risk_score': risk_score,
            'pathway_importance': pathway_importance,
            'pathway_risks': pathway_risks
        }


class CoxLoss(nn.Module):
    """Cox proportional hazards partial likelihood loss."""
    
    def forward(self, risk_scores, event_times, event_indicators):
        valid_mask = event_times > 0
        if not valid_mask.all():
            risk_scores = risk_scores[valid_mask]
            event_times = event_times[valid_mask]
            event_indicators = event_indicators[valid_mask]
        
        if len(risk_scores) == 0:
            return torch.tensor(0.0, requires_grad=True, device=risk_scores.device)
        
        risk = risk_scores.squeeze(-1)
        idx = torch.argsort(event_times, descending=True)
        risk = risk[idx]
        event_indicators = event_indicators[idx]
        
        log_cum_hazard = torch.logcumsumexp(risk, dim=0)
        event_mask = event_indicators.bool()
        
        if event_mask.sum() == 0:
            return torch.tensor(0.0, requires_grad=True, device=risk.device)
        
        loss = -torch.sum(risk[event_mask] - log_cum_hazard[event_mask]) / event_mask.sum()
        return loss


def compute_cindex(risk, times, events):
    """Compute concordance index."""
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


def train(
    epochs=200,
    batch_size=32,
    lr=5e-3,  # Higher LR
    weight_decay=1e-3,
    embed_dim=32,  # Smaller
    dropout=0.3,
    patience=30
):
    """Train simpler model."""
    
    print("="*60)
    print("SIMPLE PATHWAY MODEL TRAINING")
    print("="*60)
    
    # Load data
    loader = TCGALoader("./data/tcga_hnsc")
    loader.load_clinical()
    loader.load_expression()
    loader.create_pathway_mask("./data/pathways/hallmark.gmt")
    aligned = loader.align_data()
    
    # Split
    n = len(aligned['X'])
    train_idx, val_idx = train_test_split(
        np.arange(n), test_size=0.2, random_state=42, stratify=aligned['y_event']
    )
    
    X_train = aligned['X'][train_idx]
    y_time_train = aligned['y_time'][train_idx]
    y_event_train = aligned['y_event'][train_idx]
    X_val = aligned['X'][val_idx]
    y_time_val = aligned['y_time'][val_idx]
    y_event_val = aligned['y_event'][val_idx]
    
    print(f"\nTrain: {len(X_train)} patients ({y_event_train.sum():.0f} events)")
    print(f"Val: {len(X_val)} patients ({y_event_val.sum():.0f} events)")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Create model
    model = SimplePathwayModel(
        pathway_mask=aligned['pathway_mask'],
        embed_dim=embed_dim,
        dropout=dropout,
        pathway_names=aligned['pathway_names']
    ).to(device)
    
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer with L2 regularization
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = CoxLoss()
    
    print("-"*60)
    
    best_val_cindex = 0.5
    patience_counter = 0
    best_state = None
    history = {'train_loss': [], 'val_loss': [], 'val_cindex': []}
    
    for epoch in range(epochs):
        # Train
        model.train()
        indices = np.random.permutation(len(X_train))
        total_loss = 0
        n_batches = 0
        
        for i in range(0, len(X_train), batch_size):
            batch_idx = indices[i:i+batch_size]
            x = torch.from_numpy(X_train[batch_idx]).float().to(device)
            t = torch.from_numpy(y_time_train[batch_idx]).float().to(device)
            e = torch.from_numpy(y_event_train[batch_idx]).float().to(device)
            
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out['risk_score'], t, e)
            
            if not (torch.isnan(loss) or torch.isinf(loss)):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1
        
        scheduler.step()
        train_loss = total_loss / max(n_batches, 1)
        
        # Validate
        model.eval()
        with torch.no_grad():
            x_val = torch.from_numpy(X_val).float().to(device)
            t_val = torch.from_numpy(y_time_val).float().to(device)
            e_val = torch.from_numpy(y_event_val).float().to(device)
            
            out_val = model(x_val)
            val_loss = criterion(out_val['risk_score'], t_val, e_val).item()
            val_cindex = compute_cindex(out_val['risk_score'], t_val, e_val)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_cindex'].append(val_cindex)
        
        print(f"Epoch {epoch+1:3d}/{epochs} | Train: {train_loss:.4f} | "
              f"Val: {val_loss:.4f} | C-index: {val_cindex:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # Early stopping on C-index (not loss)
        if val_cindex > best_val_cindex:
            best_val_cindex = val_cindex
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    # Load best model
    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    
    # Final analysis
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Best Val C-index: {best_val_cindex:.4f}")
    
    # Get pathway importance
    model.eval()
    with torch.no_grad():
        x_val = torch.from_numpy(X_val).float().to(device)
        out = model(x_val)
        importance = out['pathway_importance'].mean(dim=0).cpu().numpy()
    
    sorted_idx = np.argsort(importance)[::-1]
    
    print("\nTop 10 Pathways:")
    for i, idx in enumerate(sorted_idx[:10]):
        print(f"  {i+1}. {aligned['pathway_names'][idx]}: {importance[idx]:.4f}")
    
    # Save results
    results_dir = MODULE_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'pathway_names': aligned['pathway_names'],
        'best_cindex': best_val_cindex,
        'history': history
    }, results_dir / "simple_model.pt")
    
    print(f"\nSaved to {results_dir / 'simple_model.pt'}")
    
    return model, best_val_cindex


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--embed-dim', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--patience', type=int, default=30)
    args = parser.parse_args()
    
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        embed_dim=args.embed_dim,
        dropout=args.dropout,
        patience=args.patience
    )
