"""
Hybrid Pathway-MLP Model

Key insight: Baseline MLP achieves C-index 0.68, pathway models get ~0.53.
The problem is pathway aggregation loses gene-level signal.

Solution: Combine the power of MLP with pathway interpretability.
- Pathway branch: aggregates genes to pathways, provides interpretability
- Gene branch: uses full gene expression, provides prediction power
- Fusion: combines both for final risk score
"""

import modal

app = modal.App("oral-cancer-hybrid")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(["torch", "numpy", "pandas", "scikit-learn"])
)


@app.function(image=image, gpu="T4", timeout=3600)
def train_hybrid(
    X_train, y_time_train, y_event_train,
    X_val, y_time_val, y_event_val,
    pathway_mask, pathway_names,
    epochs: int = 200,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 30
):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    
    print("="*60)
    print("HYBRID PATHWAY-MLP MODEL")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Train: {len(X_train)} ({y_event_train.sum():.0f} events)")
    print(f"Val: {len(X_val)} ({y_event_val.sum():.0f} events)")
    
    n_genes, n_pathways = pathway_mask.shape
    
    class HybridPathwayMLP(nn.Module):
        """
        Combines MLP prediction power with pathway interpretability.
        
        Architecture:
        1. Pathway Branch: Aggregate genes → pathways → pathway scores (interpretable)
        2. Gene Branch: Full MLP on all genes (powerful)
        3. Fusion: Combine both for final risk
        """
        
        def __init__(self, n_genes, pathway_mask, hidden_dim=256, pathway_dim=64, dropout=0.3):
            super().__init__()
            n_genes, n_pathways = pathway_mask.shape
            self.n_genes = n_genes
            self.n_pathways = n_pathways
            
            self.register_buffer('pathway_mask', torch.from_numpy(pathway_mask).float())
            
            # ===== PATHWAY BRANCH (Interpretability) =====
            # Aggregate genes to pathways via weighted mean
            # Learn weights for each pathway
            self.pathway_gene_weights = nn.ParameterList([
                nn.Parameter(torch.ones(int(pathway_mask[:, p].sum())) / pathway_mask[:, p].sum())
                for p in range(n_pathways)
            ])
            
            # Process pathway features
            self.pathway_mlp = nn.Sequential(
                nn.Linear(n_pathways, pathway_dim),
                nn.BatchNorm1d(pathway_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(pathway_dim, pathway_dim),
                nn.ReLU()
            )
            
            # Pathway importance scorer
            self.pathway_importance = nn.Linear(pathway_dim, n_pathways)
            
            # ===== GENE BRANCH (Power) =====
            self.gene_mlp = nn.Sequential(
                nn.Linear(n_genes, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            
            # ===== FUSION =====
            fusion_dim = pathway_dim + hidden_dim // 2
            self.fusion = nn.Sequential(
                nn.Linear(fusion_dim, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1)
            )
            
            self._init_weights()
        
        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        
        def forward(self, x):
            batch_size = x.size(0)
            
            # ===== PATHWAY BRANCH =====
            # Aggregate genes to pathway scores
            pathway_scores = []
            for p_idx in range(self.n_pathways):
                genes_in = self.pathway_mask[:, p_idx].bool()
                if genes_in.sum() == 0:
                    pathway_scores.append(torch.zeros(batch_size, device=x.device))
                else:
                    # Weighted mean of gene expression in pathway
                    gene_vals = x[:, genes_in]  # (batch, n_genes_in_pathway)
                    weights = F.softmax(self.pathway_gene_weights[p_idx], dim=0)
                    pathway_score = (gene_vals * weights).sum(dim=1)
                    pathway_scores.append(pathway_score)
            
            pathway_features = torch.stack(pathway_scores, dim=1)  # (batch, n_pathways)
            pathway_repr = self.pathway_mlp(pathway_features)  # (batch, pathway_dim)
            
            # Pathway importance (for interpretability)
            importance_logits = self.pathway_importance(pathway_repr)  # (batch, n_pathways)
            pathway_importance = F.softmax(importance_logits, dim=1)
            
            # ===== GENE BRANCH =====
            gene_repr = self.gene_mlp(x)  # (batch, hidden_dim//2)
            
            # ===== FUSION =====
            combined = torch.cat([pathway_repr, gene_repr], dim=1)
            risk_score = self.fusion(combined)
            
            return {
                'risk_score': risk_score,
                'pathway_importance': pathway_importance,
                'pathway_features': pathway_features
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
    
    # Create model
    model = HybridPathwayMLP(
        n_genes=n_genes,
        pathway_mask=pathway_mask,
        hidden_dim=256,
        pathway_dim=64,
        dropout=0.3
    ).to(device)
    
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
    
    cancer_keywords = ['MYC', 'E2F', 'G2M', 'DNA_REPAIR', 'APOPTOSIS', 'HYPOXIA', 'P53', 'INFLAMMATORY', 'EPITHELIAL', 'KRAS', 'NOTCH']
    top_5_names = [name for name, _ in top_pathways[:5]]
    relevant = sum(1 for name in top_5_names if any(kw in name for kw in cancer_keywords))
    print(f"Cancer-relevant pathways in top 5: {relevant}/5")
    
    if best_cindex >= 0.65:
        print("✅ Strong predictive performance!")
    elif best_cindex >= 0.60:
        print("⚠️  Moderate performance - approaching baseline MLP")
    else:
        print("❌ Underfitting")
    
    return {
        'best_cindex': best_cindex,
        'epochs_trained': len(history['train_loss']),
        'top_pathways': top_pathways,
        'history': history
    }


@app.local_entrypoint()
def main(epochs: int = 200, lr: float = 1e-3, batch_size: int = 32):
    import sys
    from pathlib import Path
    import numpy as np
    from sklearn.model_selection import train_test_split
    
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    from data_engineering.tcga_loader import TCGALoader
    
    print("🚀 Hybrid Pathway-MLP on Modal GPU")
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
    results = train_hybrid.remote(
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
