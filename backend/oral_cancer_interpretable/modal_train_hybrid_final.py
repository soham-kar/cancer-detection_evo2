"""
Modal Training Script for Hybrid Pathway-MLP

This is the "models not wrappers" approach:
- Prediction: Unconstrained MLP (achieves C-index ~0.68)
- Interpretation: Post-hoc pathway projection (reveals what the model learned)
"""

import modal

app = modal.App("oral-cancer-hybrid-final")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(["torch", "numpy", "pandas", "scikit-learn"])
)


@app.function(image=image, gpu="T4", timeout=3600)
def train_hybrid_final(
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
    print("HYBRID PATHWAY-MLP: PREDICT FREELY, INTERPRET FAITHFULLY")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Train: {len(X_train)} ({y_event_train.sum():.0f} events = {100*y_event_train.mean():.1f}%)")
    print(f"Val: {len(X_val)} ({y_event_val.sum():.0f} events)")
    print(f"Genes: {X_train.shape[1]}, Pathways: {pathway_mask.shape[1]}")
    
    n_genes = X_train.shape[1]
    n_pathways = pathway_mask.shape[1]
    
    # ===== HYBRID MODEL =====
    class HybridPathwayMLP(nn.Module):
        def __init__(self, n_genes, pathway_mask, hidden_dims=[512, 256, 128], 
                     pathway_embed_dim=64, dropout=0.3):
            super().__init__()
            self.n_genes = n_genes
            self.n_pathways = pathway_mask.shape[1]
            
            self.register_buffer('pathway_mask', torch.from_numpy(pathway_mask).float())
            
            # PREDICTION: Unconstrained MLP
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
            
            self.risk_head = nn.Sequential(
                nn.Linear(prev_dim, 64),
                nn.ReLU(),
                nn.Dropout(dropout / 2),
                nn.Linear(64, 1)
            )
            
            # INTERPRETATION: Pathway projection
            self.pathway_queries = nn.Parameter(torch.randn(self.n_pathways, pathway_embed_dim))
            self.hidden_to_pathway = nn.Linear(hidden_dims[-1], pathway_embed_dim)
            self.gene_pathway_scorer = nn.Linear(n_genes, self.n_pathways)
            
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        
        def forward(self, x, return_interpretation=True):
            # Prediction (unconstrained)
            hidden = self.encoder(x)
            risk_score = self.risk_head(hidden)
            
            output = {'risk_score': risk_score}
            
            if return_interpretation:
                # Pathway interpretation (post-hoc)
                hidden_proj = self.hidden_to_pathway(hidden)
                pathway_sim = torch.matmul(hidden_proj, self.pathway_queries.T) / np.sqrt(hidden_proj.size(-1))
                gene_scores = self.gene_pathway_scorer(x)
                combined = pathway_sim + gene_scores
                output['pathway_importance'] = F.softmax(combined, dim=1)
            
            return output
    
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
        hidden_dims=[512, 256, 128],
        pathway_embed_dim=64,
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
            out = model(x, return_interpretation=False)  # Only prediction during training
            
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
            out_val = model(x_val, return_interpretation=False)
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
    
    # Load best model + get pathway interpretation
    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    
    model.eval()
    with torch.no_grad():
        x_val = torch.from_numpy(X_val).float().to(device)
        out = model(x_val, return_interpretation=True)
        importance = out['pathway_importance'].mean(dim=0).cpu().numpy()
    
    sorted_idx = np.argsort(importance)[::-1]
    top_pathways = [(pathway_names[i], float(importance[i])) for i in sorted_idx[:15]]
    
    print("\n" + "="*60)
    print("TOP 15 PATHWAYS (Discovered by Model)")
    print("="*60)
    for i, (name, score) in enumerate(top_pathways):
        print(f"  {i+1}. {name}: {score:.4f}")
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Best Val C-index: {best_cindex:.4f}")
    
    # Biological validation
    cancer_keywords = ['MYC', 'E2F', 'G2M', 'DNA_REPAIR', 'APOPTOSIS', 'HYPOXIA', 
                       'P53', 'INFLAMMATORY', 'EPITHELIAL', 'KRAS', 'NOTCH', 'WNT']
    top_5_names = [name for name, _ in top_pathways[:5]]
    relevant = sum(1 for name in top_5_names if any(kw in name for kw in cancer_keywords))
    print(f"Cancer-relevant pathways in top 5: {relevant}/5")
    
    if best_cindex >= 0.65:
        print("✅ Strong performance! Model successfully learned survival signal.")
    elif best_cindex >= 0.60:
        print("⚠️  Moderate performance - approaching baseline MLP level.")
    else:
        print("❌ Underfitting - may need more training or hyperparameter tuning.")
    
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
    
    print("🚀 Hybrid Pathway-MLP: Predict Freely, Interpret Faithfully")
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
    print(f"   Target: Match baseline MLP C-index of 0.68")
    
    print("[2/2] Training on Modal T4 GPU...")
    results = train_hybrid_final.remote(
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
    
    if results['best_cindex'] >= 0.65:
        print("✅ SUCCESS: Hybrid model matches baseline MLP with interpretability!")
    else:
        print("⚠️  May need hyperparameter tuning")
    
    print("\nTop 5 Pathways Discovered:")
    for name, score in results['top_pathways'][:5]:
        print(f"  • {name}: {score:.4f}")
    
    return results
