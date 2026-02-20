"""
Modal Training Script for PathwayAttentionV3 - Memory Efficient

Uses pathway-level attention (50×50) instead of gene-level (4193×4193).
This fits in T4 GPU memory.
"""

import modal

app = modal.App("oral-cancer-pathway-v3")

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
def train_pathway_v3(
    X_train, y_time_train, y_event_train,
    X_val, y_time_val, y_event_val,
    pathway_mask, pathway_names,
    epochs: int = 150,
    batch_size: int = 32,
    lr: float = 5e-4,
    patience: int = 30
):
    """
    Train memory-efficient PathwayAttentionV3 on Modal GPU.
    
    Key insight: Aggregate genes to pathway-level FIRST, then run attention
    over pathways (50×50) instead of genes (4193×4193).
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    
    print("="*60)
    print("PATHWAY ATTENTION V3 - MEMORY EFFICIENT")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Train: {len(X_train)} patients ({y_event_train.sum():.0f} events)")
    print(f"Val: {len(X_val)} patients ({y_event_val.sum():.0f} events)")
    print(f"Genes: {X_train.shape[1]}, Pathways: {pathway_mask.shape[1]}")
    
    n_genes, n_pathways = pathway_mask.shape
    
    # ===== MEMORY-EFFICIENT MODEL =====
    class PathwayAttentionV3(nn.Module):
        """
        Pathway-first attention: aggregate genes to pathways, then attention over pathways.
        
        This is O(n_pathways^2) instead of O(n_genes^2) - massive memory savings.
        """
        
        def __init__(self, pathway_mask, embed_dim=64, num_heads=4, dropout=0.2):
            super().__init__()
            n_genes, n_pathways = pathway_mask.shape
            self.n_genes = n_genes
            self.n_pathways = n_pathways
            self.embed_dim = embed_dim
            
            self.register_buffer('pathway_mask', torch.from_numpy(pathway_mask).float())
            
            # Gene embedding (lightweight)
            self.gene_embed = nn.Linear(1, embed_dim)
            
            # Pathway attention over aggregated gene embeddings
            self.pathway_attention = nn.MultiheadAttention(
                embed_dim, num_heads, dropout=dropout, batch_first=True
            )
            
            # Pathway MLP
            self.pathway_mlp = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 2, embed_dim)
            )
            
            # Risk prediction
            self.risk_head = nn.Sequential(
                nn.Linear(embed_dim, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, 1)
            )
            
            # Pathway importance head
            self.importance_head = nn.Linear(embed_dim, 1)
            
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        
        def forward(self, x):
            """
            x: (batch, n_genes) gene expression
            """
            batch_size = x.size(0)
            
            # Embed genes: (batch, n_genes, embed_dim)
            x_embed = self.gene_embed(x.unsqueeze(-1))
            
            # Aggregate genes to pathway representations
            # For each pathway: mean of gene embeddings in that pathway
            pathway_reprs = []
            for p_idx in range(self.n_pathways):
                genes_in = self.pathway_mask[:, p_idx].bool()
                n_g = genes_in.sum().item()
                
                if n_g == 0:
                    pathway_reprs.append(torch.zeros(batch_size, self.embed_dim, device=x.device))
                else:
                    # Weighted mean by gene expression (higher expressed genes matter more)
                    gene_weights = F.softmax(x[:, genes_in], dim=1)  # (batch, n_genes_in_pathway)
                    p_repr = torch.einsum('bn,bnd->bd', gene_weights, x_embed[:, genes_in, :])
                    pathway_reprs.append(p_repr)
            
            # Stack: (batch, n_pathways, embed_dim)
            pathway_embeds = torch.stack(pathway_reprs, dim=1)
            
            # Self-attention over pathways (50×50 - tiny!)
            pathway_attn_out, pathway_attn_weights = self.pathway_attention(
                pathway_embeds, pathway_embeds, pathway_embeds,
                need_weights=True, average_attn_weights=True
            )
            
            # MLP
            pathway_processed = self.pathway_mlp(pathway_attn_out) + pathway_attn_out  # residual
            
            # Pathway importance (interpretability)
            importance_scores = self.importance_head(pathway_processed).squeeze(-1)  # (batch, n_pathways)
            pathway_importance = F.softmax(importance_scores, dim=1)
            
            # Global patient representation (attention-weighted sum of pathways)
            patient_repr = torch.einsum('bp,bpd->bd', pathway_importance, pathway_processed)
            
            # Risk score
            risk_score = self.risk_head(patient_repr)
            
            return {
                'risk_score': risk_score,
                'pathway_importance': pathway_importance,
                'pathway_attention': pathway_attn_weights
            }
    
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
    model = PathwayAttentionV3(pathway_mask, embed_dim=64, num_heads=4, dropout=0.2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
    
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("-"*60)
    
    history = {'train_loss': [], 'val_cindex': []}
    best_cindex = 0.5
    patience_counter = 0
    best_state = None
    
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
            
            # Cox loss
            risk = out['risk_score'].squeeze()
            idx = torch.argsort(t, descending=True)
            risk_sorted = risk[idx]
            e_sorted = e[idx]
            log_cum = torch.logcumsumexp(risk_sorted, dim=0)
            event_mask = e_sorted.bool()
            
            if event_mask.sum() > 0:
                loss = -torch.mean(risk_sorted[event_mask] - log_cum[event_mask])
            else:
                loss = torch.tensor(0.0, device=device, requires_grad=True)
            
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1
        
        avg_loss = total_loss / max(n_batches, 1)
        
        # Validate
        model.eval()
        with torch.no_grad():
            x_val = torch.from_numpy(X_val).float().to(device)
            t_val = torch.from_numpy(y_time_val).float().to(device)
            e_val = torch.from_numpy(y_event_val).float().to(device)
            out_val = model(x_val)
            val_cindex = compute_cindex(out_val['risk_score'], t_val, e_val)
        
        scheduler.step(val_cindex)
        
        history['train_loss'].append(avg_loss)
        history['val_cindex'].append(val_cindex)
        
        if epoch % 10 == 0 or val_cindex > best_cindex:
            print(f"Epoch {epoch+1:3d} | Loss: {avg_loss:.4f} | C-index: {val_cindex:.4f}")
        
        if val_cindex > best_cindex:
            best_cindex = val_cindex
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    # Load best + analyze
    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    
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
    
    cancer_relevant = ['MYC', 'E2F', 'G2M', 'DNA_REPAIR', 'APOPTOSIS', 'HYPOXIA', 'P53', 'INFLAMMATORY', 'EPITHELIAL']
    top_5_names = [name for name, _ in top_pathways[:5]]
    relevant = sum(1 for name in top_5_names if any(cp in name for cp in cancer_relevant))
    print(f"Cancer-relevant pathways in top 5: {relevant}/5")
    
    if best_cindex > 0.65:
        print("✅ Strong predictive performance!")
    elif best_cindex > 0.60:
        print("⚠️  Moderate performance")
    else:
        print("❌ Model may be underfitting")
    
    return {
        'best_cindex': best_cindex,
        'epochs_trained': len(history['train_loss']),
        'top_pathways': top_pathways,
        'history': history
    }


@app.local_entrypoint()
def main(epochs: int = 150, lr: float = 5e-4, batch_size: int = 32):
    """Load data locally and train on Modal GPU."""
    import sys
    from pathlib import Path
    import numpy as np
    from sklearn.model_selection import train_test_split
    
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    from data_engineering.tcga_loader import TCGALoader
    
    print("🚀 PathwayAttentionV3 (Memory Efficient) on Modal GPU")
    print("="*60)
    
    print("[1/2] Loading data...")
    loader = TCGALoader("./data/tcga_hnsc")
    loader.load_clinical()
    loader.load_expression()
    loader.create_pathway_mask("./data/pathways/hallmark.gmt")
    aligned = loader.align_data()
    
    n = len(aligned['X'])
    train_idx, val_idx = train_test_split(
        np.arange(n), test_size=0.2, random_state=42, stratify=aligned['y_event']
    )
    
    print(f"   Train: {len(train_idx)}, Val: {len(val_idx)}")
    
    print("[2/2] Training on Modal T4 GPU...")
    results = train_pathway_v3.remote(
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
        lr=lr
    )
    
    print("\n" + "="*60)
    print("📊 TRAINING COMPLETE")
    print("="*60)
    print(f"Best C-index: {results['best_cindex']:.4f}")
    
    print("\nTop 5 Pathways:")
    for name, score in results['top_pathways'][:5]:
        print(f"  • {name}: {score:.4f}")
    
    return results
