"""
All-in-One Modal Training + Deployment

This script:
1. Trains the hybrid model on GPU
2. Saves model + data to Modal Volume
3. Deploys the prediction API

Run: modal run modal_train_and_deploy.py
Deploy: modal deploy modal_train_and_deploy.py
"""

import modal

app = modal.App("oral-cancer-complete")

# Shared volume for model storage
model_volume = modal.Volume.from_name("oral-cancer-model", create_if_missing=True)

# Image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch",
        "numpy",
        "pandas",
        "scikit-learn",
        "fastapi",
        "pydantic"
    ])
)


# ================================
# TRAINING FUNCTION
# ================================
@app.function(
    image=image,
    gpu="T4",
    volumes={"/model": model_volume},
    timeout=3600
)
def train_and_save():
    """Train model and save to persistent volume."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    import pickle
    
    print("="*60)
    print("TRAINING + SAVING TO MODAL VOLUME")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    # === LOAD DATA (inline - no external dependencies) ===
    print("\n[1/5] Loading TCGA data from UCSC Xena format...")
    
    # Load clinical
    clinical = pd.read_csv('/model/raw_data/clinical.tsv', sep='\t')
    
    # Filter oral cavity
    site_col = 'primary_site' if 'primary_site' in clinical.columns else 'icd_o_3_site'
    if site_col in clinical.columns:
        oral_codes = ['C02', 'C03', 'C04', 'C05', 'C06']
        mask = clinical[site_col].astype(str).str[:3].isin(oral_codes)
        clinical = clinical[mask]
    
    # Extract survival
    if 'days_to_death' in clinical.columns:
        clinical['survival_time'] = clinical['days_to_death']
    if 'days_to_last_followup' in clinical.columns:
        mask = clinical['survival_time'].isna()
        clinical.loc[mask, 'survival_time'] = pd.to_numeric(
            clinical.loc[mask, 'days_to_last_followup'], errors='coerce'
        )
    
    clinical['event'] = (clinical['vital_status'].str.upper() == 'DECEASED').astype(float)
    clinical = clinical[clinical['survival_time'].notna() & (clinical['survival_time'] > 0)]
    
    print(f"   Clinical: {len(clinical)} patients")
    
    # Load expression
    expr_df = pd.read_csv('/model/raw_data/expression.tsv', sep='\t', index_col=0)
    print(f"   Expression: {expr_df.shape}")
    
    # Load pathways
    pathways = {}
    with open('/model/raw_data/hallmark.gmt', 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            name = parts[0]
            genes = parts[2:]
            pathways[name] = genes
    
    print(f"   Pathways: {len(pathways)}")
    
    # Align data
    pathway_genes = set()
    for genes in pathways.values():
        pathway_genes.update(genes)
    
    common_genes = list(set(expr_df.index) & pathway_genes)
    expr_df = expr_df.loc[common_genes]
    
    common_patients = list(set(expr_df.columns) & set(clinical['submitter_id']))
    expr_df = expr_df[common_patients]
    clinical = clinical[clinical['submitter_id'].isin(common_patients)]
    clinical = clinical.set_index('submitter_id').loc[common_patients].reset_index()
    
    # Create matrices
    X = expr_df.T.values.astype(np.float32)
    y_time = clinical['survival_time'].values.astype(np.float32)
    y_event = clinical['event'].values.astype(np.float32)
    gene_names = list(expr_df.index)
    
    # Create pathway mask
    pathway_names = list(pathways.keys())
    pathway_mask = np.zeros((len(gene_names), len(pathway_names)), dtype=np.float32)
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    for p_idx, (p_name, p_genes) in enumerate(pathways.items()):
        for gene in p_genes:
            if gene in gene_to_idx:
                pathway_mask[gene_to_idx[gene], p_idx] = 1.0
    
    print(f"\n   Final: {len(X)} patients, {len(gene_names)} genes, {len(pathway_names)} pathways")
    print(f"   Events: {y_event.sum():.0f}/{len(y_event)} ({100*y_event.mean():.1f}%)")
    
    # === MODEL ===
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
            
            self.risk_head = nn.Sequential(
                nn.Linear(prev_dim, 64),
                nn.ReLU(),
                nn.Dropout(dropout / 2),
                nn.Linear(64, 1)
            )
            
            self.pathway_queries = nn.Parameter(torch.randn(self.n_pathways, pathway_embed_dim))
            self.hidden_to_pathway = nn.Linear(hidden_dims[-1], pathway_embed_dim)
            self.gene_pathway_scorer = nn.Linear(n_genes, self.n_pathways)
        
        def forward(self, x, return_interpretation=True):
            hidden = self.encoder(x)
            risk_score = self.risk_head(hidden)
            
            output = {'risk_score': risk_score}
            
            if return_interpretation:
                hidden_proj = self.hidden_to_pathway(hidden)
                pathway_sim = torch.matmul(hidden_proj, self.pathway_queries.T) / np.sqrt(hidden_proj.size(-1))
                gene_scores = self.gene_pathway_scorer(x)
                combined = pathway_sim + gene_scores
                output['pathway_importance'] = F.softmax(combined, dim=1)
            
            return output
    
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
    
    # === TRAIN ===
    print("\n[2/5] Training model...")
    
    train_idx, val_idx = train_test_split(
        np.arange(len(X)), test_size=0.2, random_state=42, stratify=y_event
    )
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_time_train, y_time_val = y_time[train_idx], y_time[val_idx]
    y_event_train, y_event_val = y_event[train_idx], y_event[val_idx]
    
    model = HybridPathwayMLP(
        n_genes=len(gene_names),
        pathway_mask=pathway_mask,
        hidden_dims=[512, 256, 128],
        dropout=0.3
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
    
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    best_cindex = 0.5
    best_state = None
    patience = 30
    patience_counter = 0
    batch_size = 32
    epochs = 200
    
    for epoch in range(epochs):
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
            out = model(x, return_interpretation=False)
            
            risk = out['risk_score'].squeeze()
            idx = torch.argsort(t, descending=True)
            risk_sorted = risk[idx]
            e_sorted = e[idx]
            log_cum = torch.logcumsumexp(risk_sorted, dim=0)
            event_mask = e_sorted.bool()
            
            if event_mask.sum() > 0:
                loss = -torch.mean(risk_sorted[event_mask] - log_cum[event_mask])
            else:
                loss = torch.tensor(0.0, device=device, requires_grad=True)
            
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1
        
        # Validate
        model.eval()
        with torch.no_grad():
            x_val = torch.from_numpy(X_val).float().to(device)
            t_val = torch.from_numpy(y_time_val).float().to(device)
            e_val = torch.from_numpy(y_event_val).float().to(device)
            out_val = model(x_val, return_interpretation=False)
            val_cindex = compute_cindex(out_val['risk_score'], t_val, e_val)
        
        scheduler.step(val_cindex)
        
        if epoch % 20 == 0 or val_cindex > best_cindex:
            print(f"   Epoch {epoch+1:3d} | Loss: {total_loss/max(n_batches,1):.4f} | C-index: {val_cindex:.4f}")
        
        if val_cindex > best_cindex:
            best_cindex = val_cindex
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"   Early stopping at epoch {epoch+1}")
                break
    
    print(f"\n   Best C-index: {best_cindex:.4f}")
    
    # === SAVE TO VOLUME ===
    print("\n[3/5] Saving model to volume...")
    
    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    
    torch.save(best_state, '/model/best_model.pt')
    print("   Saved: /model/best_model.pt")
    
    print("\n[4/5] Saving data to volume...")
    
    np.savez(
        '/model/aligned_data.npz',
        X=X,
        y_time=y_time,
        y_event=y_event,
        pathway_mask=pathway_mask,
        pathway_names=np.array(pathway_names),
        gene_names=np.array(gene_names)
    )
    print("   Saved: /model/aligned_data.npz")
    
    # Commit volume
    model_volume.commit()
    
    print("\n[5/5] Getting pathway importance...")
    
    model.eval()
    with torch.no_grad():
        x_val = torch.from_numpy(X_val).float().to(device)
        out = model(x_val, return_interpretation=True)
        importance = out['pathway_importance'].mean(dim=0).cpu().numpy()
    
    sorted_idx = np.argsort(importance)[::-1]
    
    print("\n" + "="*60)
    print("TOP PATHWAYS")
    print("="*60)
    for i in range(5):
        idx = sorted_idx[i]
        print(f"  {i+1}. {pathway_names[idx]}: {importance[idx]:.4f}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best C-index: {best_cindex:.4f}")
    print("Model saved to Modal Volume: oral-cancer-model")
    
    return {'best_cindex': best_cindex, 'epochs': epoch+1}


# ================================
# PREDICTION API
# ================================
@app.cls(
    image=image,
    gpu="T4",
    volumes={"/model": model_volume},
    keep_warm=1,
    timeout=120
)
class OralCancerPredictor:
    """Production predictor loaded from volume."""
    
    @modal.enter()
    def load_model(self):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import numpy as np
        
        print("Loading model from volume...")
        
        # Load data config
        self.data = np.load('/model/aligned_data.npz', allow_pickle=True)
        self.pathway_names = list(self.data['pathway_names'])
        self.gene_names = list(self.data['gene_names'])
        self.pathway_mask = self.data['pathway_mask']
        
        # Define model
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
                
                self.risk_head = nn.Sequential(
                    nn.Linear(prev_dim, 64),
                    nn.ReLU(),
                    nn.Dropout(dropout / 2),
                    nn.Linear(64, 1)
                )
                
                self.pathway_queries = nn.Parameter(torch.randn(self.n_pathways, pathway_embed_dim))
                self.hidden_to_pathway = nn.Linear(hidden_dims[-1], pathway_embed_dim)
                self.gene_pathway_scorer = nn.Linear(n_genes, self.n_pathways)
            
            def forward(self, x, return_interpretation=True):
                hidden = self.encoder(x)
                risk_score = self.risk_head(hidden)
                
                output = {'risk_score': risk_score}
                
                if return_interpretation:
                    hidden_proj = self.hidden_to_pathway(hidden)
                    pathway_sim = torch.matmul(hidden_proj, self.pathway_queries.T) / np.sqrt(hidden_proj.size(-1))
                    gene_scores = self.gene_pathway_scorer(x)
                    combined = pathway_sim + gene_scores
                    output['pathway_importance'] = F.softmax(combined, dim=1)
                
                return output
        
        self.model = HybridPathwayMLP(
            n_genes=len(self.gene_names),
            pathway_mask=self.pathway_mask
        )
        
        state_dict = torch.load('/model/best_model.pt', map_location='cpu')
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)
        
        print(f"Loaded on {self.device}: {len(self.gene_names)} genes, {len(self.pathway_names)} pathways")
    
    @modal.method()
    def predict(self, expression_dict: dict) -> dict:
        import torch
        import numpy as np
        
        x = np.zeros(len(self.gene_names))
        matched = 0
        for i, gene in enumerate(self.gene_names):
            if gene in expression_dict:
                x[i] = expression_dict[gene]
                matched += 1
            else:
                x[i] = np.median(self.data['X'][:, i])
        
        x_tensor = torch.tensor([x]).float().to(self.device)
        
        with torch.no_grad():
            output = self.model(x_tensor, return_interpretation=True)
            raw_risk = output['risk_score'].item()
            risk_prob = 1 / (1 + np.exp(-raw_risk))
            pathway_imp = output['pathway_importance'][0].cpu().numpy()
            top_idx = np.argsort(pathway_imp)[-5:][::-1]
        
        category = 'High' if risk_prob > 0.7 else 'Medium' if risk_prob > 0.4 else 'Low'
        
        return {
            'risk_score': round(risk_prob, 3),
            'risk_category': category,
            'genes_matched': matched,
            'pathways': [
                {'name': self.pathway_names[i].replace('HALLMARK_', ''), 
                 'importance': round(float(pathway_imp[i]), 4)}
                for i in top_idx
            ],
            'primary_pathway': self.pathway_names[top_idx[0]].replace('HALLMARK_', '')
        }
    
    @modal.method()
    def health(self) -> dict:
        return {
            'status': 'healthy',
            'genes': len(self.gene_names),
            'pathways': len(self.pathway_names)
        }


# ================================
# LOCAL ENTRY POINT
# ================================
@app.local_entrypoint()
def main(action: str = "train"):
    """
    Run training or test prediction.
    
    Usage:
        modal run modal_train_and_deploy.py --action train
        modal run modal_train_and_deploy.py --action test
    """
    import sys
    from pathlib import Path
    
    if action == "train":
        # First, upload raw data to volume
        print("📤 Uploading raw data to Modal volume...")
        
        # Check for local data
        data_dir = Path("./data")
        if not (data_dir / "tcga_hnsc" / "clinical.tsv").exists():
            print("ERROR: Local data not found. Run download_real_data.py first.")
            return
        
        # Upload using Modal CLI (manual step)
        print("\n⚠️  First, upload raw data to volume:")
        print("   modal volume put oral-cancer-model ./data/tcga_hnsc/clinical.tsv raw_data/clinical.tsv")
        print("   modal volume put oral-cancer-model ./data/tcga_hnsc/expression.tsv raw_data/expression.tsv")
        print("   modal volume put oral-cancer-model ./data/pathways/hallmark.gmt raw_data/hallmark.gmt")
        print("\nThen run: modal run modal_train_and_deploy.py --action train_remote")
        
    elif action == "train_remote":
        print("🚀 Training on Modal GPU...")
        result = train_and_save.remote()
        print(f"\n✅ Training complete! C-index: {result['best_cindex']:.4f}")
        print("\nTo deploy: modal deploy modal_train_and_deploy.py")
        
    elif action == "test":
        print("🧪 Testing prediction API...")
        
        # Create sample expression
        import numpy as np
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        from data_engineering.tcga_loader import TCGALoader
        
        loader = TCGALoader("./data/tcga_hnsc")
        loader.load_clinical()
        loader.load_expression()
        loader.create_pathway_mask("./data/pathways/hallmark.gmt")
        aligned = loader.align_data()
        
        sample_expr = {
            aligned['gene_names'][i]: float(aligned['X'][0, i])
            for i in range(len(aligned['gene_names']))
        }
        
        predictor = OralCancerPredictor()
        result = predictor.predict.remote(sample_expr)
        
        print("\n" + "="*50)
        print("PREDICTION RESULT")
        print("="*50)
        print(f"Risk Score: {result['risk_score']}")
        print(f"Category: {result['risk_category']}")
        print(f"Primary Pathway: {result['primary_pathway']}")
        print(f"Genes Matched: {result['genes_matched']}")
        
    else:
        print(f"Unknown action: {action}")
        print("Use: train, train_remote, or test")
