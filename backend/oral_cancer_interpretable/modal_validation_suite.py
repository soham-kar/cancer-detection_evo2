"""
Modal Validation Suite for Oral Cancer Model (Phase 4)

Objectives:
1. Validated on held-out Test set (20%) - strict no-peeking.
2. Diversity Analysis: Test if reducing pathway_reg_weight improves diversity.
3. Compare Model A (Baseline) vs Model B (Low Reg).
"""

import modal
import sys
from pathlib import Path

app = modal.App("oral-cancer-validation")

# Reuse volume
model_volume = modal.Volume.from_name("oral-cancer-model", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(["torch", "numpy", "pandas", "scikit-learn", "prettytable", "scipy"])
)


@app.function(image=image, gpu="T4", volumes={"/model": model_volume}, timeout=3600)
def run_validation_suite():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    from sklearn.model_selection import train_test_split
    from prettytable import PrettyTable
    from scipy.stats import entropy
    
    print("="*60)
    print("PHASE 4: RIGOROUS VALIDATION SUITE")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # === DATA LOADING ===
    # We load from volume to ensure we use the exact same aligned data as production
    print("\n[1/4] Loading data from volume...")
    try:
        data = np.load('/model/aligned_data.npz', allow_pickle=True)
        print(f"   Available keys: {data.files}")
        X = data['X']
        pathway_mask = data['pathway_mask']
        pathway_names = data['pathway_names']
        gene_names = data['gene_names']
        
        # Robust loading for y_time/y_event
        if 'y_time' in data:
            y_time = data['y_time']
        elif 'survival_time' in data:
            y_time = data['survival_time']
        else:
            raise KeyError("y_time not found")

        if 'y_event' in data:
            y_event = data['y_event']
        elif 'event' in data:
            y_event = data['event']
        else:
            raise KeyError("y_event not found")
            
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    print(f"   Patients: {len(X)}")
    print(f"   Genes: {len(gene_names)}")
    print(f"   Pathways: {len(pathway_names)}")
    
    # === STRICT SPLIT ===
    # 60% Train, 20% Val, 20% Test
    print("\n[2/4] Splitting data (60/20/20)...")
    
    # First split: Train+Val (80%) vs Test (20%)
    indices = np.arange(len(X))
    train_val_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=y_event
    )
    
    # Second split: Train (75% of 80% = 60%) vs Val (25% of 80% = 20%)
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=0.25, random_state=42, stratify=y_event[train_val_idx]
    )
    
    print(f"   Train: {len(train_idx)} (60%)")
    print(f"   Val:   {len(val_idx)} (20%)")
    print(f"   Test:  {len(test_idx)} (20%) - HELD OUT")
    
    # Prepare tensors
    def to_tensor(idx):
        return (
            torch.from_numpy(X[idx]).float().to(device),
            torch.from_numpy(y_time[idx]).float().to(device),
            torch.from_numpy(y_event[idx]).float().to(device)
        )
    
    x_train, t_train, e_train = to_tensor(train_idx)
    x_val, t_val, e_val = to_tensor(val_idx)
    x_test, t_test, e_test = to_tensor(test_idx)
    pathway_mask_t = torch.from_numpy(pathway_mask).float().to(device)
    
    # === EXPERIMENT DEFINITION ===
    
    class HybridPathwayMLP(nn.Module):
        def __init__(self, n_genes, pathway_mask, hidden_dims=[512, 256, 128], 
                     pathway_embed_dim=64, dropout=0.3):
            super().__init__()
            self.n_genes = n_genes
            self.n_pathways = pathway_mask.shape[1]
            self.register_buffer('pathway_mask', torch.from_numpy(pathway_mask).float())
            
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
            self.risk_head = nn.Sequential(nn.Linear(prev_dim, 64), nn.ReLU(), nn.Dropout(dropout/2), nn.Linear(64, 1))
            self.pathway_queries = nn.Parameter(torch.randn(self.n_pathways, pathway_embed_dim))
            self.hidden_to_pathway = nn.Linear(hidden_dims[-1], pathway_embed_dim)
            self.gene_pathway_scorer = nn.Linear(n_genes, self.n_pathways)
        
        def forward(self, x, return_interpretation=True):
            hidden = self.encoder(x)
            risk_score = self.risk_head(hidden)
            output = {'risk_score': risk_score}
            if return_interpretation:
                hp = self.hidden_to_pathway(hidden)
                pathway_sim = torch.matmul(hp, self.pathway_queries.T) / np.sqrt(hp.size(-1))
                gene_scores = self.gene_pathway_scorer(x)
                output['pathway_importance'] = F.softmax(pathway_sim + gene_scores, dim=1)
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
    
    def train_experiment(name, reg_weight):
        print(f"\nTraining Experiment: {name} (Reg={reg_weight})")
        
        model = HybridPathwayMLP(len(gene_names), pathway_mask).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        
        best_c = 0.5
        best_state = None
        
        for epoch in range(100):
            model.train()
            indices = torch.randperm(len(x_train))
            
            for i in range(0, len(x_train), 32):
                idx = indices[i:i+32]
                bx, bt, be = x_train[idx], t_train[idx], e_train[idx]
                
                optimizer.zero_grad()
                out = model(bx, return_interpretation=True)
                risk = out['risk_score'].squeeze()
                
                # Cox Loss
                idx_sorted = torch.argsort(bt, descending=True)
                risk_s, e_s = risk[idx_sorted], be[idx_sorted]
                mask = e_s.bool()
                if mask.sum() > 0:
                    cox_loss = -torch.mean(risk_s[mask] - torch.logcumsumexp(risk_s, dim=0)[mask])
                else:
                    cox_loss = torch.tensor(0.0, device=device, requires_grad=True)
                
                # Pathway Regularization (Maximize concentration = Minimize Entropy)
                imp = out['pathway_importance']
                pathway_entropy = -torch.sum(imp * torch.log(imp + 1e-10), dim=1).mean()
                reg_loss = pathway_entropy * reg_weight
                
                # Total loss
                loss = cox_loss + reg_loss
                
                if torch.isfinite(loss):
                    loss.backward()
                    optimizer.step()
            
            # Val
            model.eval()
            with torch.no_grad():
                out = model(x_val, return_interpretation=False)
                val_c = compute_cindex(out['risk_score'], t_val, e_val)
                
            if val_c > best_c:
                best_c = val_c
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        # Load best and evaluate on Test
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        model.eval()
        with torch.no_grad():
            out_test = model(x_test, return_interpretation=True)
            test_c = compute_cindex(out_test['risk_score'], t_test, e_test)
            
            # Diversity analysis
            imps = out_test['pathway_importance'].cpu().numpy()
            mean_imp = imps.mean(axis=0)
            avg_entropy = entropy(mean_imp)
            top_pathway = pathway_names[np.argmax(mean_imp)]
            top_score = np.max(mean_imp)
        
        return {
            'val_c': best_c,
            'test_c': test_c,
            'entropy': avg_entropy,
            'top_pathway': top_pathway,
            'top_score': top_score,
            'model': best_state
        }

    # === RUN EXPERIMENTS ===
    print("\n[3/4] Running Experiments...")
    
    results_a = train_experiment("Model A (Baseline)", reg_weight=0.01)
    results_b = train_experiment("Model B (Diversity)", reg_weight=0.001)
    
    # === RESULTS TABLE ===
    print("\n[4/4] FINAL RESULTS")
    t = PrettyTable(['Metric', 'Model A (Reg=0.01)', 'Model B (Reg=0.001)'])
    t.add_row(['Val C-index', f"{results_a['val_c']:.4f}", f"{results_b['val_c']:.4f}"])
    t.add_row(['Test C-index (Held-out)', f"{results_a['test_c']:.4f}", f"{results_b['test_c']:.4f}"])
    t.add_row(['Pathway Entropy', f"{results_a['entropy']:.3f}", f"{results_b['entropy']:.3f}"])
    t.add_row(['Top Pathway', results_a['top_pathway'][:20], results_b['top_pathway'][:20]])
    t.add_row(['Top Pathway %', f"{results_a['top_score']:.1%}", f"{results_b['top_score']:.1%}"])
    print(t)
    
    # Recommendation
    winner = None
    if results_b['test_c'] >= results_a['test_c'] - 0.02 and results_b['entropy'] > results_a['entropy'] + 0.5:
        print("\n🏆 Recommendation: Model B (Diversity)")
        print("   Reason: Better interpretability with similar performance.")
        winner = results_b
    elif results_a['test_c'] > results_b['test_c'] + 0.02:
        print("\n🏆 Recommendation: Model A (Baseline)")
        print("   Reason: Significantly better prediction performance.")
        winner = results_a
    else:
        print("\n🏆 Recommendation: Inconclusive / Tie")
    
    return "Validation Complete"

@app.local_entrypoint()
def main():
    print("🚀 Starting Modal Validation Suite...")
    run_validation_suite.remote()
