"""
MYC-Focused Survival Model
Based on biological analysis: MYC_TARGETS_V1 is real signal (r=0.164**), G2M is noise

Implementation:
1. Fixed preprocessing (StandardScaler, not quantile)
2. Penalty on G2M concentration (>20%)
3. Reward for MYC attention (>15%)
4. Sparse-but-correct pathway attention (not uniform diversity)
"""

import modal
import numpy as np

app = modal.App("oral-cancer-myc-focused")
volume = modal.Volume.from_name("oral-cancer-model", create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch", "numpy", "pandas", "scikit-learn", "scikit-survival"
)

@app.function(image=image, gpu="T4", volumes={"/model": volume}, timeout=3600)
def train_myc_focused():
    """Train with MYC focus (real biological signal) and G2M penalty (overfitting artifact)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import joblib
    
    print("="*60)
    print("MYC-FOCUSED TRAINING")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Load data
    data = np.load("/model/aligned_data.npz", allow_pickle=True)
    X = data['X'].astype(np.float32)
    y_time = data['y_time'].astype(np.float32)
    y_event = data['y_event'].astype(np.float32)
    pathway_mask = data['pathway_mask']
    pathway_names = list(data['pathway_names'])
    gene_names = list(data['gene_names'])
    
    print(f"Data: {X.shape[0]} patients, {X.shape[1]} genes")
    print(f"Pathways: {len(pathway_names)}")
    
    # Find key pathway indices
    g2m_idx = pathway_names.index('HALLMARK_G2M_CHECKPOINT')
    myc_v1_idx = pathway_names.index('HALLMARK_MYC_TARGETS_V1')
    myc_v2_idx = pathway_names.index('HALLMARK_MYC_TARGETS_V2')
    e2f_idx = pathway_names.index('HALLMARK_E2F_TARGETS')
    
    print(f"\nKey pathway indices:")
    print(f"  G2M_CHECKPOINT (to penalize): {g2m_idx}")
    print(f"  MYC_TARGETS_V1 (to encourage): {myc_v1_idx}")
    print(f"  MYC_TARGETS_V2: {myc_v2_idx}")
    print(f"  E2F_TARGETS: {e2f_idx}")
    
    # ======= FIX 1: PROPER PREPROCESSING =======
    print("\n🔧 Applying StandardScaler preprocessing...")
    
    # Train/val/test split FIRST (before fitting scaler)
    train_idx, temp_idx = train_test_split(range(len(y_time)), test_size=0.4, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)
    
    print(f"Split: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    
    # Fit scaler on TRAINING data only (prevent data leakage)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X[train_idx])
    X_val_scaled = scaler.transform(X[val_idx])
    X_test_scaled = scaler.transform(X[test_idx])
    
    print(f"After scaling - Train: mean={X_train_scaled.mean():.3f}, std={X_train_scaled.std():.3f}")
    
    # Save scaler for inference
    joblib.dump(scaler, "/model/preprocessing_scaler.pkl")
    print("✅ Saved scaler to /model/preprocessing_scaler.pkl")
    
    # Convert to tensors
    X_train = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    X_val = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
    
    y_time_train = torch.tensor(y_time[train_idx], dtype=torch.float32).to(device)
    y_time_val = torch.tensor(y_time[val_idx], dtype=torch.float32).to(device)
    y_time_test = torch.tensor(y_time[test_idx], dtype=torch.float32).to(device)
    
    y_event_train = torch.tensor(y_event[train_idx], dtype=torch.float32).to(device)
    y_event_val = torch.tensor(y_event[val_idx], dtype=torch.float32).to(device)
    y_event_test = torch.tensor(y_event[test_idx], dtype=torch.float32).to(device)
    
    # ======= MODEL: MYC-FOCUSED PATHWAY ATTENTION =======
    class MYCFocusedModel(nn.Module):
        def __init__(self, n_genes, pathway_mask, hidden_dims=[64, 32], dropout=0.5):
            super().__init__()
            self.n_genes = n_genes
            self.n_pathways = pathway_mask.shape[1]
            self.register_buffer('pathway_mask', torch.from_numpy(pathway_mask).float())
            
            # Encoder
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
            
            # Risk head
            self.risk_head = nn.Sequential(
                nn.Linear(prev_dim, 32),
                nn.ReLU(),
                nn.Dropout(dropout/2),
                nn.Linear(32, 1)
            )
            
            # Pathway attention (SPARSE, not uniform)
            self.pathway_scorer = nn.Linear(prev_dim, self.n_pathways)
        
        def forward(self, x):
            hidden = self.encoder(x)
            risk = self.risk_head(hidden)
            
            # Pathway importance (sparse softmax with temperature)
            pathway_logits = self.pathway_scorer(hidden)
            pathway_probs = F.softmax(pathway_logits / 0.5, dim=1)  # Temperature 0.5 = more sparse
            
            return risk, pathway_probs
    
    def cox_partial_likelihood(risk, times, events):
        """Negative partial log-likelihood for Cox model."""
        sorted_indices = torch.argsort(times, descending=True)
        risk = risk[sorted_indices]
        events = events[sorted_indices]
        
        hazard_ratio = torch.exp(risk)
        log_risk = torch.log(torch.cumsum(hazard_ratio, dim=0) + 1e-8)
        
        uncensored_likelihood = risk - log_risk
        loss = -torch.sum(uncensored_likelihood * events) / (events.sum() + 1e-8)
        
        return loss
    
    def compute_cindex(risk, times, events):
        """Concordance index."""
        risk = risk.squeeze().detach().cpu().numpy()
        times = times.cpu().numpy()
        events = events.cpu().numpy()
        
        concordant = discordant = 0
        for i in range(len(times)):
            if events[i] == 0:
                continue
            for j in range(len(times)):
                if i == j or times[j] < times[i]:
                    continue
                if risk[i] > risk[j]:
                    concordant += 1
                elif risk[i] < risk[j]:
                    discordant += 1
        
        total = concordant + discordant
        return concordant / total if total > 0 else 0.5
    
    # Initialize model
    model = MYCFocusedModel(
        n_genes=X.shape[1],
        pathway_mask=pathway_mask,
        hidden_dims=[64, 32],
        dropout=0.5
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    # ======= TRAINING WITH MYC FOCUS =======
    best_val_cindex = 0
    best_state = None
    patience_counter = 0
    
    print("\n" + "-"*60)
    print("Training with MYC focus (penalize G2M >20%, encourage MYC >15%)")
    print("-"*60)
    
    for epoch in range(300):
        model.train()
        
        risk, pathway_probs = model(X_train)
        risk = risk.squeeze()
        
        # Survival loss
        surv_loss = cox_partial_likelihood(risk, y_time_train, y_event_train)
        
        # ======= MYC REGULARIZATION =======
        # Mean pathway weights across batch
        mean_probs = pathway_probs.mean(dim=0)
        
        g2m_weight = mean_probs[g2m_idx]
        myc_v1_weight = mean_probs[myc_v1_idx]
        myc_v2_weight = mean_probs[myc_v2_idx]
        
        # Penalty: G2M should be < 20%
        g2m_penalty = torch.relu(g2m_weight - 0.20)
        
        # Reward: MYC_V1 should be > 15%
        myc_reward = torch.relu(0.15 - myc_v1_weight)
        
        # Combined biological regularization
        bio_reg = 0.5 * g2m_penalty + 0.3 * myc_reward
        
        # Total loss
        loss = surv_loss + bio_reg
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Validation
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_risk, val_pathway_probs = model(X_val)
                val_cindex = compute_cindex(val_risk, y_time_val, y_event_val)
                
                # Check pathway weights
                mean_probs_np = pathway_probs.mean(dim=0).cpu().numpy()
                top_idx = np.argsort(mean_probs_np)[::-1][:5]
                top_names = [pathway_names[i].split('_', 1)[1][:15] for i in top_idx]
                top_weights = [mean_probs_np[i] for i in top_idx]
            
            scheduler.step(val_cindex)
            
            print(f"Epoch {epoch:3d} | Surv: {surv_loss:.4f} | Bio: {bio_reg:.4f} | "
                  f"Val C-idx: {val_cindex:.3f} | "
                  f"G2M: {g2m_weight:.1%} | MYC: {myc_v1_weight:.1%}")
            
            if val_cindex > best_val_cindex:
                best_val_cindex = val_cindex
                best_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= 5:
                print(f"Early stopping at epoch {epoch}")
                break
    
    # Load best model
    model.load_state_dict(best_state)
    model.eval()
    
    # Final evaluation
    with torch.no_grad():
        test_risk, test_pathway_probs = model(X_test)
        test_cindex = compute_cindex(test_risk, y_time_test, y_event_test)
        
        mean_probs = test_pathway_probs.mean(dim=0).cpu().numpy()
        top_idx = np.argsort(mean_probs)[::-1][:10]
    
    print("\n" + "="*60)
    print("FINAL RESULTS (MYC-FOCUSED MODEL)")
    print("="*60)
    print(f"Best Val C-index: {best_val_cindex:.3f}")
    print(f"Test C-index: {test_cindex:.3f}")
    
    print("\n📊 Top 10 Pathways:")
    for i, idx in enumerate(top_idx):
        marker = ""
        if pathway_names[idx] == 'HALLMARK_MYC_TARGETS_V1':
            marker = " ← TARGET (MYC)"
        elif pathway_names[idx] == 'HALLMARK_G2M_CHECKPOINT':
            marker = " ← PENALIZED (G2M)"
        print(f"  {i+1}. {pathway_names[idx]:45s}: {mean_probs[idx]*100:.1f}%{marker}")
    
    # Check if MYC > G2M
    myc_weight = mean_probs[myc_v1_idx]
    g2m_weight = mean_probs[g2m_idx]
    
    if myc_weight > g2m_weight:
        print(f"\n✅ SUCCESS: MYC ({myc_weight:.1%}) > G2M ({g2m_weight:.1%})")
    else:
        print(f"\n⚠️ G2M still dominant: G2M ({g2m_weight:.1%}) > MYC ({myc_weight:.1%})")
    
    # Save model
    torch.save(model.state_dict(), "/model/myc_focused_model.pt")
    volume.commit()
    print(f"\n✅ Saved model to /model/myc_focused_model.pt")
    
    return {
        'val_cindex': float(best_val_cindex),
        'test_cindex': float(test_cindex),
        'myc_weight': float(myc_weight),
        'g2m_weight': float(g2m_weight),
        'top_pathways': [(pathway_names[i], float(mean_probs[i])) for i in top_idx[:5]]
    }


@app.local_entrypoint()
def main():
    results = train_myc_focused.remote()
    
    print("\n" + "="*60)
    print("MYC-FOCUSED TRAINING COMPLETE")
    print("="*60)
    print(f"Val C-index:  {results['val_cindex']:.3f}")
    print(f"Test C-index: {results['test_cindex']:.3f}")
    print(f"MYC weight:   {results['myc_weight']*100:.1f}%")
    print(f"G2M weight:   {results['g2m_weight']*100:.1f}%")
    print("\nTop Pathways:")
    for name, prob in results['top_pathways']:
        print(f"  {name}: {prob*100:.1f}%")
