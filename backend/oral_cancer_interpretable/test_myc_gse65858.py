"""
Test MYC-Enhanced Model on GSE65858
Critical test: Does the MYC signal transfer across platforms?
"""

import modal
import numpy as np

app = modal.App("test-myc-gse65858")
volume = modal.Volume.from_name("oral-cancer-model")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch", "numpy", "pandas", "scikit-learn", "scikit-survival", "joblib"
)

@app.function(image=image, gpu="T4", volumes={"/model": volume}, timeout=600)
def test_myc_on_gse65858(gse_X, gse_y_time, gse_y_event, gene_names):
    """Test MYC-focused model on GSE65858 external data."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    import joblib
    
    print("="*60)
    print("MYC-ENHANCED MODEL → GSE65858 EXTERNAL VALIDATION")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Load training data for alignment
    data = np.load("/model/aligned_data.npz", allow_pickle=True)
    train_gene_names = list(data['gene_names'])
    pathway_mask = data['pathway_mask']
    pathway_names = list(data['pathway_names'])
    X_train = data['X']
    
    print(f"Model genes: {len(train_gene_names)}")
    print(f"GSE genes: {len(gene_names)}")
    
    # Find common genes
    common_genes = [g for g in train_gene_names if g in gene_names]
    print(f"Common genes: {len(common_genes)}")
    
    # Align GSE to model gene order
    train_gene_to_idx = {g: i for i, g in enumerate(train_gene_names)}
    gse_gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    
    n_patients = gse_X.shape[0]
    n_genes = len(train_gene_names)
    X_aligned = np.zeros((n_patients, n_genes), dtype=np.float32)
    
    for gene in common_genes:
        train_idx = train_gene_to_idx[gene]
        gse_idx = gse_gene_to_idx[gene]
        X_aligned[:, train_idx] = gse_X[:, gse_idx]
    
    print(f"Aligned shape: {X_aligned.shape}")
    
    # Apply preprocessing (StandardScaler fitted on training data)
    try:
        scaler = joblib.load("/model/preprocessing_scaler_v2.pkl")
        X_scaled = scaler.transform(X_aligned)
        print("✅ Loaded preprocessing scaler v2")
    except:
        # Fallback: fit scaler on training data
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_scaled = scaler.transform(X_aligned)
        print("⚠️ Fitted scaler on training data (fallback)")
    
    print(f"After scaling - Mean: {X_scaled.mean():.3f}, Std: {X_scaled.std():.3f}")
    
    # Model architecture (must match training)
    class MYCFocusedPathwayMLP(nn.Module):
        def __init__(self, n_genes, n_pathways, hidden_dims=[512, 256], dropout=0.4):
            super().__init__()
            
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
                nn.Linear(prev_dim, 128),
                nn.ReLU(),
                nn.Dropout(dropout/2),
                nn.Linear(128, 1)
            )
            
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
    
    # Load model
    model = MYCFocusedPathwayMLP(
        n_genes=len(train_gene_names),
        n_pathways=len(pathway_names),
        hidden_dims=[512, 256],
        dropout=0.4
    ).to(device)
    
    model.load_state_dict(torch.load("/model/myc_enhanced_model.pt", map_location=device))
    model.eval()
    print("✅ Loaded myc_enhanced_model.pt")
    
    # Inference
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        risk_scores, pathway_probs = model(X_tensor)
    
    risk_scores = risk_scores.squeeze().cpu().numpy()
    pathway_probs = pathway_probs.cpu().numpy()
    
    # Calculate C-index
    from sksurv.metrics import concordance_index_censored
    
    valid_mask = ~(np.isnan(gse_y_time) | np.isnan(gse_y_event))
    c_index = concordance_index_censored(
        gse_y_event[valid_mask].astype(bool),
        gse_y_time[valid_mask],
        risk_scores[valid_mask]
    )[0]
    
    # Pathway distribution
    mean_probs = pathway_probs.mean(axis=0)
    top_idx = np.argsort(mean_probs)[::-1][:15]
    
    print("\n" + "="*60)
    print("RESULTS: MYC-ENHANCED MODEL ON GSE65858")
    print("="*60)
    print(f"\n🎯 C-INDEX: {c_index:.3f}")
    
    if c_index > 0.58:
        print("   ✅ EXCELLENT: MYC signal generalizes well!")
    elif c_index > 0.55:
        print("   ✅ GOOD: MYC helps, some platform shift remains")
    elif c_index > 0.52:
        print("   ⚠️ MARGINAL: Some signal but weak")
    else:
        print("   ❌ POOR: Platform gap dominates")
    
    print(f"\nComparison:")
    print(f"   Original model (G2M):     0.521")
    print(f"   Diversity model:          0.517")
    print(f"   MYC-enhanced (this):      {c_index:.3f}")
    print(f"   Improvement:              {c_index - 0.521:+.3f}")
    
    # Pathway distribution
    print("\n📊 Pathway Distribution on GSE65858:")
    myc_idx = pathway_names.index('HALLMARK_MYC_TARGETS_V1') if 'HALLMARK_MYC_TARGETS_V1' in pathway_names else -1
    g2m_idx = pathway_names.index('HALLMARK_G2M_CHECKPOINT') if 'HALLMARK_G2M_CHECKPOINT' in pathway_names else -1
    
    for i, idx in enumerate(top_idx):
        if mean_probs[idx] < 0.005:
            continue
        marker = ""
        if pathway_names[idx] == 'HALLMARK_MYC_TARGETS_V1':
            marker = " ← TARGET 🎯"
        elif pathway_names[idx] == 'HALLMARK_G2M_CHECKPOINT':
            marker = " ← PENALIZED ❌"
        print(f"   {pathway_names[idx]:45s}: {mean_probs[idx]*100:.1f}%{marker}")
    
    # Risk direction check
    print("\n🔍 Risk Direction Check:")
    try:
        risk_quartiles = pd.qcut(risk_scores[valid_mask], 4, labels=['Q1','Q2','Q3','Q4'], duplicates='drop')
        event_rates = pd.Series(gse_y_event[valid_mask]).groupby(risk_quartiles).mean()
        print(f"   Event rates by quartile: {event_rates.to_dict()}")
        
        if 'Q4' in event_rates and 'Q1' in event_rates:
            if event_rates['Q4'] > event_rates['Q1']:
                print("   ✅ Risk direction CORRECT (Q4 > Q1)")
            else:
                print("   ⚠️ Risk direction may be inverted")
    except Exception as e:
        print(f"   Could not compute quartiles: {e}")
    
    # Key pathway weights
    myc_w = mean_probs[myc_idx] if myc_idx >= 0 else 0
    g2m_w = mean_probs[g2m_idx] if g2m_idx >= 0 else 0
    
    print(f"\n🧬 Key Pathway Weights:")
    print(f"   MYC_TARGETS_V1: {myc_w:.1%} {'✅' if myc_w > 0.20 else '⚠️'}")
    print(f"   G2M_CHECKPOINT: {g2m_w:.1%} {'✅' if g2m_w < 0.15 else '❌'}")
    
    return {
        'c_index': float(c_index),
        'myc_weight': float(myc_w),
        'g2m_weight': float(g2m_w),
        'improvement': float(c_index - 0.521)
    }


@app.local_entrypoint()
def main():
    import pandas as pd
    
    # Load GSE65858
    gse_expr = pd.read_csv("D:/project/biotech-evo2/data/external/GSE65858_expression.csv", index_col=0)
    gse_clin = pd.read_csv("D:/project/biotech-evo2/data/external/GSE65858_clinical.csv", index_col=0)
    
    print(f"GSE65858: {gse_expr.shape}")
    
    gse_X = gse_expr.values.astype(np.float32)
    gse_y_time = gse_clin['time_numeric'].values.astype(np.float32)
    gse_y_event = gse_clin['event'].values.astype(np.float32)
    gene_names = list(gse_expr.columns)
    
    results = test_myc_on_gse65858.remote(gse_X, gse_y_time, gse_y_event, gene_names)
    
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"GSE65858 C-index: {results['c_index']:.3f}")
    print(f"MYC weight: {results['myc_weight']*100:.1f}%")
    print(f"G2M weight: {results['g2m_weight']*100:.1f}%")
    print(f"Improvement over baseline: {results['improvement']:+.3f}")
