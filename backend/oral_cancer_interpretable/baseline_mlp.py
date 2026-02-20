"""
Baseline MLP to verify data has learnable signal.

If this gets C-index > 0.60, the data is fine and attention model needs fixing.
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import train_test_split
import sys

MODULE_DIR = Path(__file__).parent
sys.path.insert(0, str(MODULE_DIR / "src"))

from data_engineering.tcga_loader import TCGALoader


class BaselineMLP(nn.Module):
    """Simple MLP for survival prediction - no pathway structure."""
    
    def __init__(self, n_genes, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_genes, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x):
        return {'risk_score': self.net(x)}


class CoxLoss(nn.Module):
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


def train_baseline(epochs=200, hidden_dim=256, lr=1e-3, weight_decay=1e-4):
    print("="*60)
    print("BASELINE MLP TRAINING")
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
    print(f"Genes: {X_train.shape[1]}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Create model
    model = BaselineMLP(n_genes=X_train.shape[1], hidden_dim=hidden_dim).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = CoxLoss()
    
    print("-"*60)
    
    best_val_cindex = 0.5
    patience = 30
    patience_counter = 0
    batch_size = 64
    
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
        
        if epoch % 10 == 0 or val_cindex > best_val_cindex:
            print(f"Epoch {epoch+1:3d}/{epochs} | Train: {train_loss:.4f} | "
                  f"Val: {val_loss:.4f} | C-index: {val_cindex:.4f}")
        
        if val_cindex > best_val_cindex:
            best_val_cindex = val_cindex
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    print("\n" + "="*60)
    print("BASELINE RESULTS")
    print("="*60)
    print(f"Best Val C-index: {best_val_cindex:.4f}")
    
    if best_val_cindex > 0.60:
        print("✅ Data has learnable signal! C-index > 0.60")
        print("   Problem is in the attention architecture, not the data.")
    elif best_val_cindex > 0.55:
        print("⚠️  Weak signal (C-index 0.55-0.60)")
        print("   May need feature selection or different model.")
    else:
        print("❌ No signal (C-index ≤ 0.55)")
        print("   Data may not have survival signal, or severe class imbalance.")
    
    return best_val_cindex


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    
    train_baseline(epochs=args.epochs, hidden_dim=args.hidden_dim, lr=args.lr)
