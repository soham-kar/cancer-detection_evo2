"""
Improved Model with Diversity Loss
Based on friend's recommendations - entropy regularization + pathway groups
"""

import modal
import numpy as np

app = modal.App("oral-cancer-diversity-v3")
volume = modal.Volume.from_name("oral-cancer-model", create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch", "numpy", "pandas", "scikit-learn"
)

@app.function(image=image, gpu="T4", volumes={"/model": volume}, timeout=3600)
def train_with_diversity():
    """Train model with entropy diversity loss to prevent pathway concentration."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    from sklearn.model_selection import train_test_split
    
    print("="*60)
    print("TRAINING WITH DIVERSITY LOSS")
    print("="*60)
    
    # Load data from volume
    data = np.load("/model/aligned_data.npz", allow_pickle=True)
    X = data['X']
    y_time = data['y_time']
    y_event = data['y_event']
    pathway_mask = data['pathway_mask']
    pathway_names = list(data['pathway_names'])
    gene_names = list(data['gene_names'])
    
    print(f"Patients: {X.shape[0]}")
    print(f"Genes: {X.shape[1]}")
    print(f"Pathways: {pathway_mask.shape[1]}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Define pathway groups (for group diversity)
    pathway_groups = {
        'proliferation': ['HALLMARK_G2M_CHECKPOINT', 'HALLMARK_E2F_TARGETS', 
                          'HALLMARK_MYC_TARGETS_V1', 'HALLMARK_MYC_TARGETS_V2'],
        'stress': ['HALLMARK_DNA_REPAIR', 'HALLMARK_HYPOXIA', 
                   'HALLMARK_P53_PATHWAY', 'HALLMARK_REACTIVE_OXYGEN_SPECIES_PATHWAY'],
        'metabolism': ['HALLMARK_OXIDATIVE_PHOSPHORYLATION', 'HALLMARK_GLYCOLYSIS',
                       'HALLMARK_FATTY_ACID_METABOLISM', 'HALLMARK_XENOBIOTIC_METABOLISM'],
        'immune': ['HALLMARK_INFLAMMATORY_RESPONSE', 'HALLMARK_INTERFERON_ALPHA_RESPONSE',
                   'HALLMARK_INTERFERON_GAMMA_RESPONSE', 'HALLMARK_ALLOGRAFT_REJECTION'],
        'signaling': ['HALLMARK_NOTCH_SIGNALING', 'HALLMARK_WNT_BETA_CATENIN_SIGNALING',
                      'HALLMARK_HEDGEHOG_SIGNALING', 'HALLMARK_TGF_BETA_SIGNALING']
    }
    
    # Get indices for each group
    group_indices = {}
    for group_name, pathways in pathway_groups.items():
        indices = [pathway_names.index(p) for p in pathways if p in pathway_names]
        if indices:
            group_indices[group_name] = indices
            print(f"  {group_name}: {len(indices)} pathways")
    
    # Model with improved architecture
    class DiversePathwayMLP(nn.Module):
        def __init__(self, n_genes, pathway_mask, hidden_dims=[64, 32], dropout=0.5):
            super().__init__()
            self.n_genes = n_genes
            self.n_pathways = pathway_mask.shape[1]
            self.register_buffer('pathway_mask', torch.from_numpy(pathway_mask).float())
            
            # Encoder (gene-level features)
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
            
            # Risk prediction head
            self.risk_head = nn.Sequential(
                nn.Linear(prev_dim, 32),
                nn.ReLU(),
                nn.Dropout(dropout/2),
                nn.Linear(32, 1)
            )
            
            # Pathway importance scorer
            self.pathway_scorer = nn.Sequential(
                nn.Linear(hidden_dims[-1], self.n_pathways),
                nn.Softmax(dim=1)  # Force valid probability distribution
            )
        
        def forward(self, x):
            hidden = self.encoder(x)
            risk = self.risk_head(hidden)
            pathway_probs = self.pathway_scorer(hidden)
            return risk, pathway_probs
    
    def diversity_loss(pathway_probs, temperature=0.1):
        """
        Encourage diverse pathway selection.
        Higher entropy = more diverse = better.
        """
        # Top-k entropy (focus on top 5 pathways)
        top_k_vals, _ = torch.topk(pathway_probs, k=5, dim=1)
        
        # Normalize to sum to 1 for entropy calculation
        top_k_normalized = top_k_vals / (top_k_vals.sum(dim=1, keepdim=True) + 1e-8)
        
        # Entropy (higher = more diverse)
        entropy = -torch.sum(top_k_normalized * torch.log(top_k_normalized + 1e-8), dim=1)
        
        # Return negative entropy (because we want to MAXIMIZE entropy)
        return -entropy.mean()
    
    def group_diversity_loss(pathway_probs, group_indices):
        """
        Encourage activation across pathway groups.
        """
        group_scores = []
        for group_name, indices in group_indices.items():
            group_score = pathway_probs[:, indices].sum(dim=1)
            group_scores.append(group_score)
        
        # Stack and normalize
        group_scores = torch.stack(group_scores, dim=1)
        group_probs = group_scores / (group_scores.sum(dim=1, keepdim=True) + 1e-8)
        
        # Entropy across groups
        group_entropy = -torch.sum(group_probs * torch.log(group_probs + 1e-8), dim=1)
        
        return -group_entropy.mean()
    
    def cox_partial_likelihood(risk, times, events):
        """Negative partial log-likelihood for Cox model."""
        # Sort by time
        sorted_indices = torch.argsort(times, descending=True)
        risk = risk[sorted_indices]
        events = events[sorted_indices]
        
        # Compute log partial likelihood
        hazard_ratio = torch.exp(risk)
        log_risk = torch.log(torch.cumsum(hazard_ratio, dim=0) + 1e-8)
        
        # Only count events (deaths)
        uncensored_likelihood = risk - log_risk
        loss = -torch.sum(uncensored_likelihood * events) / (events.sum() + 1e-8)
        
        return loss
    
    def compute_cindex(risk, times, events):
        """Compute concordance index."""
        risk = risk.squeeze().detach().cpu().numpy()
        times = times.cpu().numpy()
        events = events.cpu().numpy()
        
        concordant = 0
        discordant = 0
        tied = 0
        
        for i in range(len(times)):
            if events[i] == 0:
                continue
            for j in range(len(times)):
                if i == j:
                    continue
                if times[j] < times[i]:
                    continue  # j must have longer observed time
                
                if risk[i] > risk[j]:
                    concordant += 1
                elif risk[i] < risk[j]:
                    discordant += 1
                else:
                    tied += 0.5
        
        total = concordant + discordant + tied
        if total == 0:
            return 0.5
        return concordant / total
    
    # Train/val/test split
    train_idx, temp_idx = train_test_split(range(len(y_time)), test_size=0.4, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)
    
    print(f"\nSplit: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    
    # Convert to tensors
    X_train = torch.tensor(X[train_idx], dtype=torch.float32).to(device)
    y_time_train = torch.tensor(y_time[train_idx], dtype=torch.float32).to(device)
    y_event_train = torch.tensor(y_event[train_idx], dtype=torch.float32).to(device)
    
    X_val = torch.tensor(X[val_idx], dtype=torch.float32).to(device)
    y_time_val = torch.tensor(y_time[val_idx], dtype=torch.float32).to(device)
    y_event_val = torch.tensor(y_event[val_idx], dtype=torch.float32).to(device)
    
    X_test = torch.tensor(X[test_idx], dtype=torch.float32).to(device)
    y_time_test = torch.tensor(y_time[test_idx], dtype=torch.float32).to(device)
    y_event_test = torch.tensor(y_event[test_idx], dtype=torch.float32).to(device)
    
    # Initialize model
    model = DiversePathwayMLP(
        n_genes=X.shape[1],
        pathway_mask=pathway_mask,
        hidden_dims=[64, 32],
        dropout=0.5
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    
    # Training loop
    best_val_cindex = 0
    best_state = None
    patience_counter = 0
    
    # Hyperparameters (following friend's advice)
    diversity_weight = 0.1  # Weight for entropy regularization
    group_diversity_weight = 0.05  # Weight for group diversity
    
    print("\n" + "-"*60)
    print(f"Training with diversity_weight={diversity_weight}, group_weight={group_diversity_weight}")
    print("-"*60)
    
    for epoch in range(200):
        model.train()
        
        risk, pathway_probs = model(X_train)
        risk = risk.squeeze()
        
        # Survival loss
        surv_loss = cox_partial_likelihood(risk, y_time_train, y_event_train)
        
        # Diversity losses
        div_loss = diversity_loss(pathway_probs)
        group_div_loss = group_diversity_loss(pathway_probs, group_indices)
        
        # Total loss
        loss = surv_loss + diversity_weight * div_loss + group_diversity_weight * group_div_loss
        
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
                
                # Check pathway diversity
                mean_probs = pathway_probs.mean(dim=0).cpu().numpy()
                top_idx = np.argsort(mean_probs)[::-1][:5]
                top_probs = [mean_probs[i] for i in top_idx]
                top_names = [pathway_names[i].split('_')[1] for i in top_idx]
                
                # Entropy of top-5 (higher = more diverse)
                top_entropy = -sum(p * np.log(p + 1e-8) for p in top_probs)
            
            print(f"Epoch {epoch:3d} | Surv: {surv_loss:.4f} | Div: {div_loss:.4f} | "
                  f"Val C-idx: {val_cindex:.3f} | Top-5 Entropy: {top_entropy:.2f}")
            
            if val_cindex > best_val_cindex:
                best_val_cindex = val_cindex
                best_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= 3:  # 30 epochs without improvement
                print(f"Early stopping at epoch {epoch}")
                break
    
    # Load best model
    model.load_state_dict(best_state)
    model.eval()
    
    # Final evaluation
    with torch.no_grad():
        test_risk, test_pathway_probs = model(X_test)
        test_cindex = compute_cindex(test_risk, y_time_test, y_event_test)
        
        # Pathway distribution
        mean_probs = test_pathway_probs.mean(dim=0).cpu().numpy()
        top_idx = np.argsort(mean_probs)[::-1][:10]
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Best Val C-index: {best_val_cindex:.3f}")
    print(f"Test C-index: {test_cindex:.3f}")
    
    print("\n📊 Top 10 Pathways (with diversity):")
    for i, idx in enumerate(top_idx):
        print(f"  {i+1}. {pathway_names[idx]:45s}: {mean_probs[idx]*100:.1f}%")
    
    # Check if diversity worked
    max_pathway_weight = mean_probs.max()
    if max_pathway_weight < 0.30:
        print("\n✅ DIVERSITY SUCCESS: Max pathway weight < 30%")
    elif max_pathway_weight < 0.50:
        print("\n⚠️ MODERATE DIVERSITY: Max pathway weight < 50%")
    else:
        print("\n❌ STILL CONCENTRATED: Max pathway weight > 50%")
    
    # Save model
    torch.save(model.state_dict(), "/model/diverse_model_v3.pt")
    volume.commit()
    print(f"\n✅ Saved model to /model/diverse_model_v3.pt")
    
    return {
        'val_cindex': best_val_cindex,
        'test_cindex': test_cindex,
        'top_pathways': [(pathway_names[i], float(mean_probs[i])) for i in top_idx[:5]],
        'max_pathway_weight': float(max_pathway_weight)
    }


@app.local_entrypoint()
def main():
    results = train_with_diversity.remote()
    print("\n" + "="*60)
    print("REMOTE TRAINING COMPLETE")
    print("="*60)
    print(f"Val C-index: {results['val_cindex']:.3f}")
    print(f"Test C-index: {results['test_cindex']:.3f}")
    print(f"Max Pathway Weight: {results['max_pathway_weight']*100:.1f}%")
    print("\nTop Pathways:")
    for name, prob in results['top_pathways']:
        print(f"  {name}: {prob*100:.1f}%")
