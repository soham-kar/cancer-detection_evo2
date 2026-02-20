"""
Test Diversity Model on External Validation (GSE65858)
"""

import modal
import numpy as np

app = modal.App("test-diversity-model")
volume = modal.Volume.from_name("oral-cancer-model")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch", "numpy", "pandas", "scikit-learn"
)

@app.function(image=image, gpu="T4", volumes={"/model": volume}, timeout=600)
def test_on_gse65858(gse_X: np.ndarray, gse_y_time: np.ndarray, gse_y_event: np.ndarray, gene_names: list):
    """Test diversity model on GSE65858 external data."""
    import torch
    import torch.nn as nn
    import numpy as np
    
    print("="*60)
    print("TESTING DIVERSITY MODEL ON GSE65858")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Load training data for gene alignment
    data = np.load("/model/aligned_data.npz", allow_pickle=True)
    train_gene_names = list(data['gene_names'])
    pathway_mask = data['pathway_mask']
    pathway_names = list(data['pathway_names'])
    
    print(f"Model genes: {len(train_gene_names)}")
    print(f"GSE genes: {len(gene_names)}")
    
    # Find common genes
    common_genes = [g for g in train_gene_names if g in gene_names]
    print(f"Common genes: {len(common_genes)}")
    
    # Build aligned GSE matrix
    train_gene_to_idx = {g: i for i, g in enumerate(train_gene_names)}
    gse_gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    
    n_patients = gse_X.shape[0]
    n_genes = len(train_gene_names)
    X_aligned = np.zeros((n_patients, n_genes), dtype=np.float32)
    
    for gene in common_genes:
        train_idx = train_gene_to_idx[gene]
        gse_idx = gse_gene_to_idx[gene]
        X_aligned[:, train_idx] = gse_X[:, gse_idx]
    
    print(f"Aligned GSE matrix: {X_aligned.shape}")
    
    # Define model architecture (must match training)
    class DiversePathwayMLP(nn.Module):
        def __init__(self, n_genes, pathway_mask, hidden_dims=[64, 32], dropout=0.5):
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
            
            self.risk_head = nn.Sequential(
                nn.Linear(prev_dim, 32),
                nn.ReLU(),
                nn.Dropout(dropout/2),
                nn.Linear(32, 1)
            )
            
            self.pathway_scorer = nn.Sequential(
                nn.Linear(hidden_dims[-1], self.n_pathways),
                nn.Softmax(dim=1)
            )
        
        def forward(self, x):
            hidden = self.encoder(x)
            risk = self.risk_head(hidden)
            pathway_probs = self.pathway_scorer(hidden)
            return risk, pathway_probs
    
    # Load model
    model = DiversePathwayMLP(
        n_genes=len(train_gene_names),
        pathway_mask=pathway_mask,
        hidden_dims=[64, 32],
        dropout=0.5
    ).to(device)
    
    model.load_state_dict(torch.load("/model/diverse_model_v3.pt", map_location=device))
    model.eval()
    print("✅ Loaded diverse_model_v3.pt")
    
    # Run inference
    X_tensor = torch.tensor(X_aligned, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        risk_scores, pathway_probs = model(X_tensor)
    
    risk_scores = risk_scores.squeeze().cpu().numpy()
    pathway_probs = pathway_probs.cpu().numpy()
    
    # Calculate C-index
    def compute_cindex(risk, times, events):
        concordant = 0
        discordant = 0
        
        for i in range(len(times)):
            if np.isnan(times[i]) or np.isnan(events[i]) or events[i] == 0:
                continue
            for j in range(len(times)):
                if i == j or np.isnan(times[j]):
                    continue
                if times[j] < times[i]:
                    continue
                
                if risk[i] > risk[j]:
                    concordant += 1
                elif risk[i] < risk[j]:
                    discordant += 1
        
        total = concordant + discordant
        if total == 0:
            return 0.5
        return concordant / total
    
    c_index = compute_cindex(risk_scores, gse_y_time, gse_y_event)
    
    # Pathway distribution
    mean_probs = pathway_probs.mean(axis=0)
    top_idx = np.argsort(mean_probs)[::-1][:10]
    
    print("\n" + "="*60)
    print("EXTERNAL VALIDATION RESULTS (DIVERSITY MODEL)")
    print("="*60)
    print(f"GSE65858 C-index: {c_index:.3f}")
    print()
    
    if c_index > 0.60:
        print("✅ EXCELLENT: Diversity model generalizes well!")
    elif c_index > 0.55:
        print("✅ GOOD: Improved generalization")
    elif c_index > 0.52:
        print("⚠️ SLIGHT IMPROVEMENT over baseline (0.521)")
    else:
        print("❌ NO IMPROVEMENT over baseline")
    
    print(f"\nComparison:")
    print(f"  Original model (no diversity):  0.521")
    print(f"  Batch-corrected deep model:     0.552")
    print(f"  Diversity model (this):         {c_index:.3f}")
    print(f"  Change from original:           {c_index - 0.521:+.3f}")
    
    print("\n📊 Top 10 Pathways (Diversity Model):")
    for i, idx in enumerate(top_idx):
        print(f"  {i+1}. {pathway_names[idx]:45s}: {mean_probs[idx]*100:.1f}%")
    
    max_weight = mean_probs.max()
    print(f"\nMax pathway weight: {max_weight*100:.1f}%")
    
    return {
        'c_index': c_index,
        'max_pathway_weight': float(max_weight),
        'top_pathways': [(pathway_names[i], float(mean_probs[i])) for i in top_idx[:5]]
    }


@app.local_entrypoint()
def main():
    import pandas as pd
    from pathlib import Path
    
    # Load GSE65858
    gse_expr = pd.read_csv("D:/project/biotech-evo2/data/external/GSE65858_expression.csv", index_col=0)
    gse_clin = pd.read_csv("D:/project/biotech-evo2/data/external/GSE65858_clinical.csv", index_col=0)
    
    print(f"GSE65858 Expression: {gse_expr.shape}")
    print(f"GSE65858 Clinical: {gse_clin.shape}")
    
    # Prepare data
    gse_X = gse_expr.values
    gse_y_time = gse_clin['time_numeric'].values
    gse_y_event = gse_clin['event'].values
    gene_names = list(gse_expr.columns)
    
    # Run test
    results = test_on_gse65858.remote(gse_X, gse_y_time, gse_y_event, gene_names)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"GSE65858 C-index: {results['c_index']:.3f}")
    print(f"Max pathway weight: {results['max_pathway_weight']*100:.1f}%")
