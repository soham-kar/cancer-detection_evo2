"""
Enhanced MYC-Focused Model (Friend's Implementation)
Key improvements:
1. Stronger G2M penalty (2.0x weight, threshold 15%)
2. MYC encouragement (threshold 20%)
3. Sparsity loss (entropy penalty for sparse-but-correct)
4. DNA_REPAIR encouragement
5. Larger hidden dims [512, 256]
"""

import modal
import numpy as np

app = modal.App("oral-cancer-myc-enhanced")
volume = modal.Volume.from_name("oral-cancer-model", create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch", "numpy", "pandas", "scikit-learn", "scikit-survival", "joblib"
)

@app.function(image=image, gpu="T4", volumes={"/model": volume}, timeout=3600)
def train_myc_enhanced():
    """Train with friend's enhanced MYC-focused architecture."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import joblib
    
    print("="*60)
    print("ENHANCED MYC-FOCUSED TRAINING (Friend's Version)")
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
    
    # Key pathway indices
    def get_idx(name):
        try: return pathway_names.index(name)
        except: return -1
    
    g2m_idx = get_idx('HALLMARK_G2M_CHECKPOINT')
    myc_idx = get_idx('HALLMARK_MYC_TARGETS_V1')
    myc_v2_idx = get_idx('HALLMARK_MYC_TARGETS_V2')
    dna_repair_idx = get_idx('HALLMARK_DNA_REPAIR')
    oxphos_idx = get_idx('HALLMARK_OXIDATIVE_PHOSPHORYLATION')
    
    print(f"\nKey pathways: G2M={g2m_idx}, MYC={myc_idx}, MYC_V2={myc_v2_idx}, DNA_REPAIR={dna_repair_idx}")
    
    # Preprocessing - StandardScaler
    train_idx, temp_idx = train_test_split(range(len(y_time)), test_size=0.4, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X[train_idx]).astype(np.float32)
    X_val_scaled = scaler.transform(X[val_idx]).astype(np.float32)
    X_test_scaled = scaler.transform(X[test_idx]).astype(np.float32)
    
    joblib.dump(scaler, "/model/preprocessing_scaler_v2.pkl")
    print(f"\nData split: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    
    # Tensors
    X_train = torch.tensor(X_train_scaled).to(device)
    X_val = torch.tensor(X_val_scaled).to(device)
    X_test = torch.tensor(X_test_scaled).to(device)
    
    y_time_train = torch.tensor(y_time[train_idx]).to(device)
    y_time_val = torch.tensor(y_time[val_idx]).to(device)
    y_time_test = torch.tensor(y_time[test_idx]).to(device)
    
    y_event_train = torch.tensor(y_event[train_idx]).to(device)
    y_event_val = torch.tensor(y_event[val_idx]).to(device)
    y_event_test = torch.tensor(y_event[test_idx]).to(device)
    
    # Enhanced Model Architecture (Friend's version)
    class MYCFocusedPathwayMLP(nn.Module):
        def __init__(self, n_genes, n_pathways, hidden_dims=[512, 256], dropout=0.4):
            super().__init__()
            self.n_genes = n_genes
            self.n_pathways = n_pathways
            
            # MLP Encoder
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
                nn.Linear(prev_dim, 128),
                nn.ReLU(),
                nn.Dropout(dropout/2),
                nn.Linear(128, 1)
            )
            
            # Pathway attention with biological priors
            self.pathway_scorer = nn.Sequential(
                nn.Linear(prev_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, n_pathways)
            )
        
        def forward(self, x):
            hidden = self.encoder(x)
            risk = self.risk_head(hidden)
            pathway_logits = self.pathway_scorer(hidden)
            pathway_probs = F.softmax(pathway_logits, dim=1)
            return risk, pathway_probs
    
    def cox_ph_loss(risk_pred, y_time, y_event):
        """Negative log partial likelihood for Cox PH model."""
        idx = torch.argsort(y_time, descending=True)
        risk_pred = risk_pred[idx]
        y_event = y_event[idx]
        
        exp_risk = torch.exp(risk_pred)
        cumsum_exp = torch.cumsum(exp_risk, dim=0)
        log_lik = risk_pred - torch.log(cumsum_exp + 1e-8)
        event_log_lik = log_lik * y_event
        
        return -torch.sum(event_log_lik) / (torch.sum(y_event) + 1e-8)
    
    def myc_focused_loss(risk_pred, pathway_probs, y_time, y_event):
        """Combined survival + biological prior loss (Friend's implementation)."""
        
        # 1. Survival loss
        surv_loss = cox_ph_loss(risk_pred.squeeze(), y_time, y_event)
        
        # 2. Biological prior loss
        bio_loss = 0.0
        
        # Penalize G2M if exceeds 15% (it's noise)
        if g2m_idx >= 0:
            g2m_weight = pathway_probs[:, g2m_idx].mean()
            g2m_penalty = torch.relu(g2m_weight - 0.15)
            bio_loss = bio_loss + 2.0 * g2m_penalty  # Strong penalty
        
        # Encourage MYC_V1 if below 20% (it's real signal)
        if myc_idx >= 0:
            myc_weight = pathway_probs[:, myc_idx].mean()
            myc_penalty = torch.relu(0.20 - myc_weight)
            bio_loss = bio_loss + 1.0 * myc_penalty
        
        # Encourage MYC_V2
        if myc_v2_idx >= 0:
            myc_v2_weight = pathway_probs[:, myc_v2_idx].mean()
            myc_v2_penalty = torch.relu(0.10 - myc_v2_weight)
            bio_loss = bio_loss + 0.5 * myc_v2_penalty
        
        # Encourage DNA_REPAIR
        if dna_repair_idx >= 0:
            dna_weight = pathway_probs[:, dna_repair_idx].mean()
            dna_penalty = torch.relu(0.10 - dna_weight)
            bio_loss = bio_loss + 0.3 * dna_penalty
        
        # 3. Sparsity loss (entropy - but we WANT low entropy = concentration)
        entropy = -torch.sum(pathway_probs * torch.log(pathway_probs + 1e-8), dim=1).mean()
        sparsity_loss = 0.1 * entropy  # Small penalty to encourage concentration
        
        total_loss = surv_loss + bio_loss + sparsity_loss
        
        return total_loss, {
            'surv': surv_loss.item(),
            'bio': bio_loss.item() if isinstance(bio_loss, torch.Tensor) else 0.0,
            'entropy': entropy.item()
        }
    
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
    model = MYCFocusedPathwayMLP(
        n_genes=X.shape[1],
        n_pathways=len(pathway_names),
        hidden_dims=[512, 256],
        dropout=0.4
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    print("\n" + "-"*60)
    print("Training: G2M<15%, MYC>20%, sparsity penalty")
    print("-"*60)
    
    best_val_cindex = 0
    best_state = None
    patience_counter = 0
    
    for epoch in range(200):
        model.train()
        
        risk, pathway_probs = model(X_train)
        loss, metrics = myc_focused_loss(risk, pathway_probs, y_time_train, y_event_train)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_risk, val_probs = model(X_val)
                val_cindex = compute_cindex(val_risk, y_time_val, y_event_val)
                
                # Check pathway weights
                mean_probs = pathway_probs.mean(dim=0).cpu().numpy()
                g2m_w = mean_probs[g2m_idx] if g2m_idx >= 0 else 0
                myc_w = mean_probs[myc_idx] if myc_idx >= 0 else 0
            
            scheduler.step(-val_cindex)
            
            print(f"Epoch {epoch:3d} | Surv: {metrics['surv']:.3f} | Bio: {metrics['bio']:.3f} | "
                  f"Ent: {metrics['entropy']:.2f} | Val: {val_cindex:.3f} | G2M: {g2m_w:.1%} | MYC: {myc_w:.1%}")
            
            if val_cindex > best_val_cindex:
                best_val_cindex = val_cindex
                best_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= 5:
                print(f"Early stopping at epoch {epoch}")
                break
    
    # Load best
    model.load_state_dict(best_state)
    model.eval()
    
    # Final evaluation
    with torch.no_grad():
        test_risk, test_probs = model(X_test)
        test_cindex = compute_cindex(test_risk, y_time_test, y_event_test)
        
        mean_probs = test_probs.mean(dim=0).cpu().numpy()
        top_idx = np.argsort(mean_probs)[::-1][:10]
    
    print("\n" + "="*60)
    print("FINAL RESULTS (ENHANCED MYC-FOCUSED)")
    print("="*60)
    print(f"Best Val C-index: {best_val_cindex:.3f}")
    print(f"Test C-index: {test_cindex:.3f}")
    
    print("\n📊 Top 10 Pathways:")
    for i, idx in enumerate(top_idx):
        marker = ""
        if pathway_names[idx] == 'HALLMARK_MYC_TARGETS_V1':
            marker = " ← TARGET"
        elif pathway_names[idx] == 'HALLMARK_G2M_CHECKPOINT':
            marker = " ← PENALIZED"
        print(f"  {i+1}. {pathway_names[idx]:45s}: {mean_probs[idx]*100:.1f}%{marker}")
    
    g2m_w = mean_probs[g2m_idx] if g2m_idx >= 0 else 0
    myc_w = mean_probs[myc_idx] if myc_idx >= 0 else 0
    
    print(f"\n🎯 Biological Validation:")
    print(f"   G2M: {g2m_w:.1%} {'✅ <15%' if g2m_w < 0.15 else '❌ Still high'}")
    print(f"   MYC: {myc_w:.1%} {'✅ >20%' if myc_w > 0.20 else '⚠️ Below target'}")
    
    # Save model
    torch.save(model.state_dict(), "/model/myc_enhanced_model.pt")
    volume.commit()
    print(f"\n✅ Saved model to /model/myc_enhanced_model.pt")
    
    return {
        'val_cindex': float(best_val_cindex),
        'test_cindex': float(test_cindex),
        'g2m_weight': float(g2m_w),
        'myc_weight': float(myc_w),
        'top_pathways': [(pathway_names[i], float(mean_probs[i])) for i in top_idx[:5]]
    }


@app.local_entrypoint()
def main():
    results = train_myc_enhanced.remote()
    
    print("\n" + "="*60)
    print("MYC-ENHANCED TRAINING COMPLETE")
    print("="*60)
    print(f"Val C-index:  {results['val_cindex']:.3f}")
    print(f"Test C-index: {results['test_cindex']:.3f}")
    print(f"G2M weight:   {results['g2m_weight']*100:.1f}% (target <15%)")
    print(f"MYC weight:   {results['myc_weight']*100:.1f}% (target >20%)")
