"""
Modal Validation Suite v2 (Fixing Overfitting)

Hypothesis:
1. Previous model minimized entropy -> Forced 1-hot pathway selection (overfitting).
2. Model size (2.5M params) >> Training data (153 samples).

Changes:
- Removed entropy minimization term.
- Reduced model capacity: [512,256,128] -> [64, 32].
- Increased Dropout: 0.3 -> 0.5.
"""

import modal
import sys
import numpy as np

app = modal.App("oral-cancer-validation-v2")
model_volume = modal.Volume.from_name("oral-cancer-model", create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.11").pip_install(["torch", "numpy", "scikit-learn", "prettytable", "scipy"])

@app.function(image=image, gpu="T4", volumes={"/model": model_volume}, timeout=3600)
def run_validation_v2():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.model_selection import train_test_split
    from prettytable import PrettyTable
    from scipy.stats import entropy
    
    print("="*60)
    print("VALIDATION SUITE V2: SMALLER & DIVERSE")
    print("="*60)
    
    # Load Data
    data = np.load('/model/aligned_data.npz', allow_pickle=True)
    X, y_time, y_event = data['X'], data['y_time'], data['y_event']
    pathway_mask = data['pathway_mask']
    pathway_names = data['pathway_names']
    gene_names = data['gene_names']
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Strictly same split
    indices = np.arange(len(X))
    train_val_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=y_event)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.25, random_state=42, stratify=y_event[train_val_idx])
    
    print(f"Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")
    
    def to_tensor(idx):
        return (
            torch.from_numpy(X[idx]).float().to(device),
            torch.from_numpy(y_time[idx]).float().to(device),
            torch.from_numpy(y_event[idx]).float().to(device)
        )
    
    x_train, t_train, e_train = to_tensor(train_idx)
    x_val, t_val, e_val = to_tensor(val_idx)
    x_test, t_test, e_test = to_tensor(test_idx)
    
    class SmallHybridMLP(nn.Module):
        def __init__(self, n_genes, n_pathways, hidden_dims=[64, 32], dropout=0.5):
            super().__init__()
            self.n_pathways = n_pathways
            
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
            
            self.risk_head = nn.Sequential(nn.Linear(prev_dim, 1))
            
            # Interpretability branch
            self.hidden_to_pathway = nn.Linear(hidden_dims[-1], n_pathways)
            
        def forward(self, x, return_interpretation=True):
            hidden = self.encoder(x)
            risk = self.risk_head(hidden)
            output = {'risk_score': risk}
            
            if return_interpretation:
                # Direct projection from latent space to pathways
                output['pathway_importance'] = F.softmax(self.hidden_to_pathway(hidden), dim=1)
            return output

    def compute_cindex(risk, times, events):
        risk, times, events = risk.squeeze().cpu().numpy(), times.cpu().numpy(), events.cpu().numpy()
        concordant = discordant = tied = 0
        n = len(risk)
        for i in range(n):
            if events[i] == 0: continue
            for j in range(n):
                if i == j or times[i] >= times[j]: continue
                if risk[i] > risk[j]: concordant += 1
                elif risk[i] < risk[j]: discordant += 1
                else: tied += 1
        total = concordant + discordant + tied
        return (concordant + 0.5 * tied) / total if total > 0 else 0.5
    
    def train(name, hidden_dims, reg_weight=0.0):
        print(f"\nTraining {name}...")
        model = SmallHybridMLP(len(gene_names), len(pathway_names), hidden_dims).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01) # Strong L2
        
        best_c, best_state = 0.5, None
        
        for epoch in range(150):
            model.train()
            idx = torch.randperm(len(x_train))
            for i in range(0, len(x_train), 32):
                bx, bt, be = x_train[idx[i:i+32]], t_train[idx[i:i+32]], e_train[idx[i:i+32]]
                
                optimizer.zero_grad()
                out = model(bx)
                risk = out['risk_score'].squeeze()
                
                # Cox Loss
                idx_sorted = torch.argsort(bt, descending=True)
                r, e = risk[idx_sorted], be[idx_sorted]
                mask = e.bool()
                if mask.sum() > 0:
                    loss = -torch.mean(r[mask] - torch.logcumsumexp(r, dim=0)[mask])
                    loss.backward()
                    optimizer.step()
            
            model.eval()
            with torch.no_grad():
                val_c = compute_cindex(model(x_val, False)['risk_score'], t_val, e_val)
            
            if val_c > best_c:
                best_c = val_c
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        # Test
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        model.eval()
        with torch.no_grad():
            out = model(x_test, True)
            test_c = compute_cindex(out['risk_score'], t_test, e_test)
            imp = out['pathway_importance'].cpu().numpy().mean(axis=0)
            ent = entropy(imp)
            top = pathway_names[np.argmax(imp)]
            top_p = np.max(imp)
            
        return {'val': best_c, 'test': test_c, 'ent': ent, 'top': top, 'top_p': top_p}

    r1 = train("Model C (Small, NoReg)", [64, 32])
    r2 = train("Model D (Tiny, NoReg)", [32, 16])
    
    t = PrettyTable(['Metric', 'Model C [64,32]', 'Model D [32,16]'])
    t.add_row(['Val C-index', f"{r1['val']:.4f}", f"{r2['val']:.4f}"])
    t.add_row(['Test C-index', f"{r1['test']:.4f}", f"{r2['test']:.4f}"])
    t.add_row(['Entropy', f"{r1['ent']:.3f}", f"{r2['ent']:.3f}"])
    t.add_row(['Top Pathway', r1['top'][:20], r2['top'][:20]])
    t.add_row(['Top %', f"{r1['top_p']:.1%}", f"{r2['top_p']:.1%}"])
    print(t)

@app.local_entrypoint()
def main():
    run_validation_v2.remote()
