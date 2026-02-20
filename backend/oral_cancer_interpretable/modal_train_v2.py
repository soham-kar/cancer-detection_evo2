"""
Modal Training Script for PathwayAttentionV2

Runs on Modal T4 GPU with soft pathway regularization.
"""

import modal

app = modal.App("oral-cancer-pathway-v2")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch",
        "numpy", 
        "pandas",
        "scikit-learn",
    ])
)


@app.function(
    image=image,
    gpu="T4",
    timeout=3600,
)
def train_pathway_v2(
    X_train, y_time_train, y_event_train,
    X_val, y_time_val, y_event_val,
    pathway_mask, pathway_names,
    epochs: int = 150,
    batch_size: int = 32,
    lr: float = 5e-4,
    reg_weight: float = 0.1,
    patience: int = 30
):
    """Train PathwayAttentionV2 on Modal GPU."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    
    print("="*60)
    print("PATHWAY ATTENTION V2 - MODAL GPU")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Train: {len(X_train)} patients ({y_event_train.sum():.0f} events)")
    print(f"Val: {len(X_val)} patients ({y_event_val.sum():.0f} events)")
    print(f"Genes: {X_train.shape[1]}, Pathways: {pathway_mask.shape[1]}")
    print(f"Reg weight: {reg_weight}")
    
    # ===== MODEL DEFINITION (inline) =====
    class PathwayAttentionV2(nn.Module):
        def __init__(self, pathway_mask, embed_dim=128, num_heads=4, dropout=0.2):
            super().__init__()
            n_genes, n_pathways = pathway_mask.shape
            self.n_genes = n_genes
            self.n_pathways = n_pathways
            
            self.register_buffer('pathway_mask', torch.from_numpy(pathway_mask).float())
            cooc = (pathway_mask @ pathway_mask.T).astype(np.float32)
            cooc_norm = cooc / cooc.max() if cooc.max() > 0 else cooc
            self.register_buffer('pathway_cooccurrence_norm', torch.from_numpy(cooc_norm))
            
            self.gene_embed = nn.Sequential(
                nn.Linear(1, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            
            self.attention = nn.MultiheadAttention(
                embed_dim, num_heads, dropout=dropout, batch_first=True
            )
            
            self.risk_head = nn.Sequential(
                nn.Linear(embed_dim, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1)
            )
            
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        
        def forward(self, x):
            batch_size = x.size(0)
            x_embed = self.gene_embed(x.unsqueeze(-1))
            attn_out, attn_weights = self.attention(
                x_embed, x_embed, x_embed,
                need_weights=True, average_attn_weights=True
            )
            
            # Pathway importance
            pathway_scores = []
            for p_idx in range(self.n_pathways):
                genes_in = self.pathway_mask[:, p_idx].bool()
                n_g = genes_in.sum().item()
                if n_g == 0:
                    pathway_scores.append(torch.zeros(batch_size, device=x.device))
                else:
                    p_attn = attn_weights[:, :, genes_in].sum(dim=(1, 2)) / n_g
                    pathway_scores.append(p_attn)
            
            pathway_scores = torch.stack(pathway_scores, dim=1)
            pathway_importance = F.softmax(pathway_scores, dim=1)
            
            global_repr = attn_out.mean(dim=1)
            risk_score = self.risk_head(global_repr)
            
            return {
                'risk_score': risk_score,
                'pathway_importance': pathway_importance,
                'attention_weights': attn_weights
            }
        
        def pathway_reg_loss(self, attn_weights):
            penalty = 1.0 - self.pathway_cooccurrence_norm
            return (attn_weights * penalty.unsqueeze(0)).mean()
    
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
    
    # ===== TRAINING =====
    model = PathwayAttentionV2(pathway_mask, embed_dim=128, num_heads=4, dropout=0.2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
    
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("-"*60)
    
    history = {'train_cox': [], 'train_reg': [], 'val_cindex': []}
    best_cindex = 0.5
    patience_counter = 0
    best_state = None
    
    for epoch in range(epochs):
        # Train
        model.train()
        indices = np.random.permutation(len(X_train))
        total_cox = total_reg = 0
        n_batches = 0
        
        for i in range(0, len(X_train), batch_size):
            batch_idx = indices[i:i+batch_size]
            x = torch.from_numpy(X_train[batch_idx]).float().to(device)
            t = torch.from_numpy(y_time_train[batch_idx]).float().to(device)
            e = torch.from_numpy(y_event_train[batch_idx]).float().to(device)
            
            optimizer.zero_grad()
            out = model(x)
            
            # Cox loss
            risk = out['risk_score'].squeeze()
            idx = torch.argsort(t, descending=True)
            risk_sorted = risk[idx]
            e_sorted = e[idx]
            log_cum = torch.logcumsumexp(risk_sorted, dim=0)
            event_mask = e_sorted.bool()
            if event_mask.sum() > 0:
                cox_loss = -torch.mean(risk_sorted[event_mask] - log_cum[event_mask])
            else:
                cox_loss = torch.tensor(0.0, device=device)
            
            # Reg loss
            reg_loss = model.pathway_reg_loss(out['attention_weights'])
            
            loss = cox_loss + reg_weight * reg_loss
            
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_cox += cox_loss.item()
                total_reg += reg_loss.item()
                n_batches += 1
        
        avg_cox = total_cox / max(n_batches, 1)
        avg_reg = total_reg / max(n_batches, 1)
        
        # Validate
        model.eval()
        with torch.no_grad():
            x_val = torch.from_numpy(X_val).float().to(device)
            t_val = torch.from_numpy(y_time_val).float().to(device)
            e_val = torch.from_numpy(y_event_val).float().to(device)
            out_val = model(x_val)
            val_cindex = compute_cindex(out_val['risk_score'], t_val, e_val)
        
        scheduler.step(val_cindex)
        
        history['train_cox'].append(avg_cox)
        history['train_reg'].append(avg_reg)
        history['val_cindex'].append(val_cindex)
        
        if epoch % 10 == 0 or val_cindex > best_cindex:
            print(f"Epoch {epoch+1:3d} | Cox: {avg_cox:.4f} | Reg: {avg_reg:.4f} | C-index: {val_cindex:.4f}")
        
        if val_cindex > best_cindex:
            best_cindex = val_cindex
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
    
    # Get pathway importance
    model.eval()
    with torch.no_grad():
        x_val = torch.from_numpy(X_val).float().to(device)
        out = model(x_val)
        importance = out['pathway_importance'].mean(dim=0).cpu().numpy()
    
    sorted_idx = np.argsort(importance)[::-1]
    top_pathways = [(pathway_names[i], float(importance[i])) for i in sorted_idx[:15]]
    
    print("\n" + "="*60)
    print("TOP 15 PATHWAYS")
    print("="*60)
    for i, (name, score) in enumerate(top_pathways):
        print(f"  {i+1}. {name}: {score:.4f}")
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Best Val C-index: {best_cindex:.4f}")
    
    # Check biological relevance
    cancer_pathways = ['MYC', 'E2F', 'G2M', 'DNA_REPAIR', 'APOPTOSIS', 'HYPOXIA', 'P53', 'INFLAMMATORY']
    top_5_names = [name for name, _ in top_pathways[:5]]
    relevant = sum(1 for name in top_5_names if any(cp in name for cp in cancer_pathways))
    print(f"Cancer-relevant pathways in top 5: {relevant}/5")
    
    return {
        'best_cindex': best_cindex,
        'epochs_trained': len(history['train_cox']),
        'top_pathways': top_pathways,
        'history': history
    }


@app.local_entrypoint()
def main(epochs: int = 150, lr: float = 5e-4, reg_weight: float = 0.1, batch_size: int = 32):
    """Load data locally and train on Modal GPU."""
    import sys
    from pathlib import Path
    import numpy as np
    from sklearn.model_selection import train_test_split
    
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    from data_engineering.tcga_loader import TCGALoader
    
    print("🚀 PathwayAttentionV2 on Modal GPU")
    print("="*60)
    
    # Load data
    print("[1/2] Loading data...")
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
    
    print(f"   Train: {len(train_idx)}, Val: {len(val_idx)}")
    
    # Train on Modal
    print("[2/2] Training on Modal T4 GPU...")
    results = train_pathway_v2.remote(
        X_train=aligned['X'][train_idx],
        y_time_train=aligned['y_time'][train_idx],
        y_event_train=aligned['y_event'][train_idx],
        X_val=aligned['X'][val_idx],
        y_time_val=aligned['y_time'][val_idx],
        y_event_val=aligned['y_event'][val_idx],
        pathway_mask=aligned['pathway_mask'],
        pathway_names=aligned['pathway_names'],
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        reg_weight=reg_weight
    )
    
    print("\n" + "="*60)
    print("📊 TRAINING COMPLETE")
    print("="*60)
    print(f"Best C-index: {results['best_cindex']:.4f}")
    print(f"Epochs trained: {results['epochs_trained']}")
    
    print("\nTop 5 Pathways:")
    for name, score in results['top_pathways'][:5]:
        print(f"  • {name}: {score:.4f}")
    
    return results
