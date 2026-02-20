"""
Modal Training Script for Pathway-Attention Model

Runs training on Modal's cloud GPUs - no local CUDA required.
Uses T4 GPU (lowest tier, ~$0.60/hour).

Usage:
    modal run modal_train.py
"""

import modal

# Create Modal app
app = modal.App("oral-cancer-pathway-attention")

# Define the image with all dependencies
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
    gpu="T4",  # Lowest tier GPU
    timeout=3600,
)
def train_on_gpu(
    X_train, y_time_train, y_event_train,
    X_val, y_time_val, y_event_val,
    pathway_mask, pathway_names,
    epochs: int = 50,
    batch_size: int = 16,
    lr: float = 1e-3,
    embed_dim: int = 64,
    dropout: float = 0.3,
    patience: int = 15
):
    """Train PathwayAttention on Modal GPU."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    from sklearn.model_selection import train_test_split
    
    print("="*60)
    print("MODAL GPU TRAINING")
    print("="*60)
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Train: {len(X_train)} patients")
    print(f"Val: {len(X_val)} patients")
    print(f"Genes: {X_train.shape[1]}")
    print(f"Pathways: {pathway_mask.shape[1]}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # ===== DEFINE MODEL INLINE =====
    class PathwayAttention(nn.Module):
        def __init__(self, pathway_mask, embed_dim=64, num_heads=1, dropout=0.1, pathway_names=None):
            super().__init__()
            self.n_genes, self.n_pathways = pathway_mask.shape
            self.embed_dim = embed_dim
            self.pathway_names = pathway_names or [f"P{i}" for i in range(self.n_pathways)]
            
            self.register_buffer('pathway_mask', torch.from_numpy(pathway_mask).float())
            attention_mask_np = (pathway_mask @ pathway_mask.T) > 0
            self.register_buffer('attention_mask_binary', torch.from_numpy(attention_mask_np.astype(np.float32)))
            
            self.gene_embedding = nn.Sequential(
                nn.Linear(1, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            
            self.gene_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
            self.pathway_pool = nn.Linear(embed_dim, embed_dim)
            self.pathway_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
            self.fc_pathway = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Dropout(dropout))
            self.pathway_to_risk = nn.Linear(embed_dim, 1)
            self.global_risk = nn.Linear(self.n_pathways, 1)
            
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        
        def forward(self, x):
            batch_size = x.size(0)
            x_embed = self.gene_embedding(x.unsqueeze(-1))
            
            attn_mask = torch.where(
                self.attention_mask_binary.bool(),
                torch.zeros_like(self.attention_mask_binary),
                torch.full_like(self.attention_mask_binary, float('-inf'))
            )
            
            gene_attn_out, _ = self.gene_attention(x_embed, x_embed, x_embed, attn_mask=attn_mask)
            
            pathway_embeds = []
            for p_idx in range(self.n_pathways):
                genes_in = self.pathway_mask[:, p_idx].bool()
                if genes_in.sum() == 0:
                    pathway_embeds.append(torch.zeros(batch_size, self.embed_dim, device=x.device))
                else:
                    pathway_embeds.append(gene_attn_out[:, genes_in, :].mean(dim=1))
            
            pathway_embeds = torch.stack(pathway_embeds, dim=1)
            pathway_embeds = self.pathway_pool(pathway_embeds)
            
            pathway_attn_out, _ = self.pathway_attention(pathway_embeds, pathway_embeds, pathway_embeds)
            pathway_processed = self.fc_pathway(pathway_attn_out)
            
            pathway_risks = self.pathway_to_risk(pathway_processed).squeeze(-1)
            pathway_importance = F.softmax(pathway_risks, dim=-1)
            risk_score = self.global_risk(pathway_risks)
            
            return {'risk_score': risk_score, 'pathway_importance': pathway_importance}
    
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
    
    # ===== CREATE MODEL =====
    model = PathwayAttention(
        pathway_mask=pathway_mask,
        embed_dim=embed_dim,
        dropout=dropout,
        pathway_names=pathway_names
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = CoxLoss()
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("-"*60)
    
    # ===== TRAINING LOOP =====
    history = {'train_loss': [], 'val_loss': [], 'val_cindex': []}
    best_val_loss = float('inf')
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
            loss = criterion(out['risk_score'], t, e)
            
            if not (torch.isnan(loss) or torch.isinf(loss)):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1
        
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
        
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_cindex'].append(val_cindex)
        
        print(f"Epoch {epoch+1:3d}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | C-index: {val_cindex:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
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
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        x_val = torch.from_numpy(X_val).float().to(device)
        out_val = model(x_val)
        importance = out_val['pathway_importance'].cpu().numpy()
    
    mean_importance = importance.mean(axis=0)
    sorted_idx = np.argsort(mean_importance)[::-1]
    
    print("\n" + "="*60)
    print("TOP 10 PATHWAYS")
    print("="*60)
    for i in sorted_idx[:10]:
        print(f"  {pathway_names[i]}: {mean_importance[i]:.4f}")
    
    results = {
        'final_val_cindex': history['val_cindex'][-1] if history['val_cindex'] else 0.5,
        'best_val_cindex': max(history['val_cindex']) if history['val_cindex'] else 0.5,
        'epochs_trained': len(history['train_loss']),
        'top_pathways': [(pathway_names[i], float(mean_importance[i])) for i in sorted_idx[:10]],
        'history': history
    }
    
    return results


@app.local_entrypoint()
def main(epochs: int = 50, batch_size: int = 16, lr: float = 1e-3):
    """Load data locally and send to Modal for training."""
    import sys
    from pathlib import Path
    
    # Add module to path
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    
    from data_engineering.tcga_loader import TCGALoader
    from sklearn.model_selection import train_test_split
    import numpy as np
    
    print("🚀 Starting Modal GPU Training")
    print("="*60)
    
    # Load data locally
    print("\n[1/2] Loading data locally...")
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
    
    print(f"   Train: {len(train_idx)} patients")
    print(f"   Val: {len(val_idx)} patients")
    
    # Send to Modal
    print("\n[2/2] Training on Modal T4 GPU...")
    results = train_on_gpu.remote(
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
    print(f"Final C-index: {results['final_val_cindex']:.4f}")
    print(f"Best C-index: {results['best_val_cindex']:.4f}")
    print(f"Epochs trained: {results['epochs_trained']}")
    
    print("\nTop 5 Pathways:")
    for name, score in results['top_pathways'][:5]:
        print(f"  • {name}: {score:.4f}")
    
    return results
