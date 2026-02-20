"""
Modal Batch Validation for GSE65858
Runs all patients in a single GPU container for speed.
"""

import modal
import numpy as np

app = modal.App("oral-cancer-batch-validation")
volume = modal.Volume.from_name("oral-cancer-model", create_if_missing=False)

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch", "numpy", "pandas", "scikit-learn"
)

@app.function(
    image=image,
    volumes={"/model": volume},
    gpu="T4",
    timeout=600,
)
def batch_validate(expression_dict: dict, gene_list: list, patient_ids: list):
    """
    Run batch inference on all patients at once.
    
    Args:
        expression_dict: {patient_id: {gene: value, ...}, ...}
        gene_list: List of gene names (columns)
        patient_ids: List of patient IDs
    
    Returns:
        List of predictions for each patient
    """
    import torch
    import torch.nn as nn
    import numpy as np
    import pandas as pd
    
    print(f"🚀 Starting batch validation on {len(patient_ids)} patients")
    
    # Load model artifacts
    data = np.load("/model/aligned_data.npz", allow_pickle=True)
    pathway_mask = data['pathway_mask']
    pathway_names = data['pathway_names']
    gene_names = list(data['gene_names'])
    
    print(f"   Model genes: {len(gene_names)}")
    print(f"   Input genes: {len(gene_list)}")
    
    # Build gene mapping
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    
    # Prepare input matrix
    n_patients = len(patient_ids)
    n_genes = len(gene_names)
    X = np.zeros((n_patients, n_genes), dtype=np.float32)
    
    matched_genes = 0
    for i, pid in enumerate(patient_ids):
        patient_data = expression_dict[pid]
        for gene, value in patient_data.items():
            if gene in gene_to_idx:
                X[i, gene_to_idx[gene]] = value
                if i == 0:
                    matched_genes += 1
    
    print(f"   Matched genes: {matched_genes}/{len(gene_names)}")
    
    # Define model architecture (MUST match training exactly)
    class HybridPathwayMLP(nn.Module):
        def __init__(self, n_genes, pathway_mask, hidden_dims=[64, 32], 
                     pathway_embed_dim=64, dropout=0.5):
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
                output['pathway_importance'] = torch.softmax(pathway_sim + gene_scores, dim=1)
            return output
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Device: {device}")
    
    model = HybridPathwayMLP(
        n_genes=len(gene_names),
        pathway_mask=pathway_mask,
        hidden_dims=[64, 32],
        dropout=0.5
    ).to(device)
    
    model.load_state_dict(torch.load("/model/best_model.pt", map_location=device))
    model.eval()
    
    # Run inference
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        output = model(X_tensor, return_interpretation=True)
    
    risk_scores = output['risk_score'].squeeze().cpu().numpy()
    pathway_importance = output['pathway_importance'].cpu().numpy()
    
    # Format results
    results = []
    for i, pid in enumerate(patient_ids):
        # Get top pathways for this patient
        patient_pathway_imp = pathway_importance[i]
        top_indices = np.argsort(patient_pathway_imp)[::-1][:5]
        
        results.append({
            'Patient_ID': pid,
            'Risk_Score': float(risk_scores[i]),
            'Top_Pathway_1': pathway_names[top_indices[0]],
            'Top_Pathway_1_Score': float(patient_pathway_imp[top_indices[0]]),
            'Top_Pathway_2': pathway_names[top_indices[1]],
            'Top_Pathway_2_Score': float(patient_pathway_imp[top_indices[1]]),
            'Top_Pathway_3': pathway_names[top_indices[2]],
            'Top_Pathway_3_Score': float(patient_pathway_imp[top_indices[2]]),
        })
    
    print(f"✅ Completed batch validation")
    print(f"   Risk score range: [{min(risk_scores):.3f}, {max(risk_scores):.3f}]")
    
    return results


@app.local_entrypoint()
def main():
    import pandas as pd
    from pathlib import Path
    
    # Load GSE65858 data
    expr_path = Path("D:/project/biotech-evo2/data/external/GSE65858_expression.csv")
    clin_path = Path("D:/project/biotech-evo2/data/external/GSE65858_clinical.csv")
    
    print(f"📂 Loading {expr_path}...")
    expr_df = pd.read_csv(expr_path, index_col=0)
    clin_df = pd.read_csv(clin_path, index_col=0)
    
    print(f"   Expression: {expr_df.shape}")
    print(f"   Clinical: {clin_df.shape}")
    
    # Prepare data for Modal
    patient_ids = list(expr_df.index)
    gene_list = list(expr_df.columns)
    expression_dict = {pid: row.to_dict() for pid, row in expr_df.iterrows()}
    
    # Run batch validation
    print("🚀 Sending to Modal...")
    results = batch_validate.remote(expression_dict, gene_list, patient_ids)
    
    # Save results
    results_df = pd.DataFrame(results)
    output_path = "gse65858_predictions.csv"
    results_df.to_csv(output_path, index=False)
    print(f"✅ Saved predictions to {output_path}")
    
    # Calculate C-index
    from sklearn.utils import check_consistent_length
    
    # Merge with clinical data
    merged = results_df.set_index('Patient_ID').join(clin_df[['time_numeric', 'event']])
    
    # Calculate concordance index
    from lifelines.utils import concordance_index
    
    valid = merged.dropna(subset=['time_numeric', 'event', 'Risk_Score'])
    c_index = concordance_index(
        valid['time_numeric'],
        -valid['Risk_Score'],  # Negative: higher risk = lower survival
        valid['event']
    )
    
    print("\n" + "="*50)
    print("🎯 EXTERNAL VALIDATION RESULTS")
    print("="*50)
    print(f"   Dataset: GSE65858 (Swedish Oral Cancer Cohort)")
    print(f"   Patients: {len(valid)}")
    print(f"   Events: {int(valid['event'].sum())}")
    print(f"")
    print(f"   📊 C-INDEX: {c_index:.3f}")
    print(f"")
    
    if c_index > 0.60:
        print("   ✅ EXCELLENT: Model generalizes well!")
    elif c_index > 0.55:
        print("   ✅ GOOD: Model generalizes acceptably")
    elif c_index > 0.50:
        print("   ⚠️ MARGINAL: Barely better than random")
    else:
        print("   ❌ FAILED: Model does not generalize")
    
    print(f"\n   Comparison:")
    print(f"   - TCGA Test Set:  0.641")
    print(f"   - GSE65858:       {c_index:.3f}")
    print(f"   - Drop:           {0.641 - c_index:.3f}")
    print("="*50)
    
    # Top pathways analysis
    print("\n📊 Top Pathways in External Cohort:")
    pathway_counts = results_df['Top_Pathway_1'].value_counts().head(10)
    for pathway, count in pathway_counts.items():
        print(f"   {pathway}: {count} patients ({count/len(results_df)*100:.1f}%)")
