"""
Batch Correction + Deep Model Retraining
Train the HybridPathwayMLP on batch-corrected merged data
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from sklearn.model_selection import train_test_split
from lifelines.utils import concordance_index

# Load batch-corrected data
data = np.load('batch_corrected/batch_corrected_data.npz', allow_pickle=True)

tcga_X = data['tcga_X']
gse_X = data['gse_X']
tcga_y_time = data['tcga_y_time']
tcga_y_event = data['tcga_y_event']
gse_y_time = data['gse_y_time']
gse_y_event = data['gse_y_event']
pathway_mask = data['pathway_mask']
pathway_names = data['pathway_names']
gene_names = data['gene_names']

print(f"TCGA: {tcga_X.shape}, GSE: {gse_X.shape}")
print(f"Pathway mask: {pathway_mask.shape}")

# Need to subset pathway mask to common genes
# Load original aligned data to get original gene list
original_data = np.load('aligned_data.npz', allow_pickle=True)
original_genes = list(original_data['gene_names'])
common_genes = list(gene_names)

gene_to_idx = {g: i for i, g in enumerate(original_genes)}
common_idx = [gene_to_idx[g] for g in common_genes]
pathway_mask_subset = pathway_mask[common_idx, :]

print(f"Pathway mask subset: {pathway_mask_subset.shape}")

# Merge TCGA + GSE for training with stratified hold-out
# Hold out 20% of GSE for final testing
gse_train_idx, gse_test_idx = train_test_split(
    range(len(gse_y_time)), test_size=0.3, random_state=42
)

# Training data: all TCGA + 70% GSE
X_train = np.vstack([tcga_X, gse_X[gse_train_idx]])
y_time_train = np.concatenate([tcga_y_time, gse_y_time[gse_train_idx]])
y_event_train = np.concatenate([tcga_y_event, gse_y_event[gse_train_idx]])

# Test data: 30% GSE (completely held out)
X_test = gse_X[gse_test_idx]
y_time_test = gse_y_time[gse_test_idx]
y_event_test = gse_y_event[gse_test_idx]

print(f"\nTraining: {X_train.shape[0]} patients (TCGA + 70% GSE)")
print(f"Testing: {X_test.shape[0]} patients (30% GSE held-out)")

# Split training into train/val
train_idx, val_idx = train_test_split(
    range(len(y_time_train)), test_size=0.2, random_state=42
)

# Model definition (simplified, more regularization)
class SimplerPathwayMLP(nn.Module):
    def __init__(self, n_genes, pathway_mask, hidden_dim=32, dropout=0.6):
        super().__init__()
        self.n_genes = n_genes
        self.n_pathways = pathway_mask.shape[1]
        self.register_buffer('pathway_mask', torch.from_numpy(pathway_mask).float())
        
        # Pathway-level encoder (sum genes in each pathway)
        self.pathway_encoder = nn.Sequential(
            nn.Linear(self.n_pathways, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Risk predictor
        self.risk_head = nn.Linear(hidden_dim // 2, 1)
        
    def forward(self, x):
        # Aggregate to pathway level first (reduces dimension, more robust)
        pathway_activations = torch.matmul(x, self.pathway_mask)  # (batch, n_pathways)
        
        # Normalize per sample
        pathway_activations = F.layer_norm(pathway_activations, [self.n_pathways])
        
        features = self.pathway_encoder(pathway_activations)
        risk = self.risk_head(features)
        
        return {'risk_score': risk, 'pathway_importance': pathway_activations}


def compute_cindex(risk, times, events):
    risk = risk.squeeze().detach().cpu().numpy()
    times = times.cpu().numpy()
    events = events.cpu().numpy()
    
    # Filter valid
    valid = ~(np.isnan(times) | np.isnan(events))
    if valid.sum() < 10:
        return 0.5
    
    try:
        return concordance_index(times[valid], -risk[valid], events[valid])
    except:
        return 0.5


def train_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    # Data tensors
    X_train_t = torch.tensor(X_train[train_idx], dtype=torch.float32)
    y_time_train_t = torch.tensor(y_time_train[train_idx], dtype=torch.float32)
    y_event_train_t = torch.tensor(y_event_train[train_idx], dtype=torch.float32)
    
    X_val_t = torch.tensor(X_train[val_idx], dtype=torch.float32)
    y_time_val_t = torch.tensor(y_time_train[val_idx], dtype=torch.float32)
    y_event_val_t = torch.tensor(y_event_train[val_idx], dtype=torch.float32)
    
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_time_test_t = torch.tensor(y_time_test, dtype=torch.float32)
    y_event_test_t = torch.tensor(y_event_test, dtype=torch.float32)
    
    # Model
    model = SimplerPathwayMLP(
        n_genes=len(common_genes),
        pathway_mask=pathway_mask_subset,
        hidden_dim=32,
        dropout=0.6
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    
    # Training loop
    best_val_cindex = 0
    best_model_state = None
    patience = 20
    no_improve = 0
    
    for epoch in range(200):
        model.train()
        
        # Forward
        X_batch = X_train_t.to(device)
        y_time_batch = y_time_train_t.to(device)
        y_event_batch = y_event_train_t.to(device)
        
        output = model(X_batch)
        risk = output['risk_score'].squeeze()
        
        # Cox partial likelihood loss approximation
        # For events: want risk to be higher if survival is lower
        loss = 0
        events = y_event_batch.bool()
        if events.sum() > 0:
            for i in torch.where(events)[0]:
                # Risk set: all samples with time >= time_i
                risk_set = y_time_batch >= y_time_batch[i]
                if risk_set.sum() > 0:
                    log_risk = torch.log(torch.exp(risk[risk_set]).sum() + 1e-8)
                    loss += -risk[i] + log_risk
            loss = loss / events.sum()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Validation
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_output = model(X_val_t.to(device))
                val_risk = val_output['risk_score']
                val_cindex = compute_cindex(val_risk, y_time_val_t, y_event_val_t)
                
                test_output = model(X_test_t.to(device))
                test_risk = test_output['risk_score']
                test_cindex = compute_cindex(test_risk, y_time_test_t, y_event_test_t)
            
            print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f} | Val C-idx: {val_cindex:.3f} | Test C-idx: {test_cindex:.3f}")
            
            if val_cindex > best_val_cindex:
                best_val_cindex = val_cindex
                best_model_state = model.state_dict().copy()
                no_improve = 0
            else:
                no_improve += 1
            
            if no_improve >= patience // 10:
                print(f"Early stopping at epoch {epoch}")
                break
    
    # Load best model
    model.load_state_dict(best_model_state)
    model.eval()
    
    # Final evaluation
    with torch.no_grad():
        test_output = model(X_test_t.to(device))
        test_risk = test_output['risk_score']
        final_cindex = compute_cindex(test_risk, y_time_test_t, y_event_test_t)
    
    print("\n" + "="*60)
    print("🎯 FINAL RESULTS (Deep Model on Batch-Corrected Data)")
    print("="*60)
    print(f"   Best Val C-index:  {best_val_cindex:.3f}")
    print(f"   Test C-index:      {final_cindex:.3f}")
    print()
    
    if final_cindex > 0.60:
        print("   ✅ EXCELLENT: Deep model works after correction!")
    elif final_cindex > 0.55:
        print("   ✅ GOOD: Acceptable cross-platform performance")
    elif final_cindex > 0.50:
        print("   ⚠️ MARGINAL: Still weak")
    else:
        print("   ❌ FAILED: Deep model doesn't generalize")
    
    print()
    print("   Comparison to original:")
    print(f"   Original (no correction): 0.521")
    print(f"   After batch correction:   {final_cindex:.3f}")
    print(f"   Improvement:              {final_cindex - 0.521:+.3f}")
    print("="*60)
    
    return model, final_cindex


if __name__ == "__main__":
    model, cindex = train_model()
