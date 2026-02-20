"""
All-in-One Modal Training + Deployment (Robust Version)

This version splits training and deployment into separate steps to avoid race conditions.

Step 1: modal run modal_complete_v2.py --action train
Step 2: modal deploy modal_complete_v2.py
"""

import modal
import os

app = modal.App("oral-cancer-api-v2")

# Persistent volume for model storage
model_volume = modal.Volume.from_name("oral-cancer-model", create_if_missing=True)

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


@app.function(image=image, gpu="T4", volumes={"/model": model_volume}, timeout=3600)
def train_and_save(X, y_time, y_event, pathway_mask, pathway_names, gene_names):
    """Train model and save to persistent volume."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    from sklearn.model_selection import train_test_split
    
    print("="*60)
    print("TRAINING HYBRID PATHWAY-MLP")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Model definition
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
    
    # Train
    train_idx, val_idx = train_test_split(np.arange(len(X)), test_size=0.2, random_state=42, stratify=y_event)
    X_train, X_val, y_time_train, y_time_val, y_event_train, y_event_val = (
        X[train_idx], X[val_idx], y_time[train_idx], y_time[val_idx], y_event[train_idx], y_event[val_idx]
    )
    
    model = HybridPathwayMLP(len(gene_names), pathway_mask).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_cindex, best_state = 0.5, None
    
    for epoch in range(100):  # Reduced epochs for stability
        model.train()
        indices = np.random.permutation(len(X_train))
        for i in range(0, len(X_train), 32):
            batch_idx = indices[i:i+32]
            if len(batch_idx) < 2: continue  # Skip batch size 1 explicitly (fix BatchNorm error)
            
            x = torch.from_numpy(X_train[batch_idx]).float().to(device)
            t = torch.from_numpy(y_time_train[batch_idx]).float().to(device)
            e = torch.from_numpy(y_event_train[batch_idx]).float().to(device)
            
            optimizer.zero_grad()
            risk = model(x, return_interpretation=False)['risk_score'].squeeze()
            idx = torch.argsort(t, descending=True)
            r, ev = risk[idx], e[idx]
            if ev.sum() > 0:
                loss = -torch.mean(r[ev.bool()] - torch.logcumsumexp(r, dim=0)[ev.bool()])
                loss.backward()
                optimizer.step()
        
        model.eval()
        with torch.no_grad():
            out = model(torch.from_numpy(X_val).float().to(device), return_interpretation=False)
            val_c = compute_cindex(out['risk_score'], torch.from_numpy(y_time_val).float(), torch.from_numpy(y_event_val).float())
        
        if val_c > best_cindex:
            best_cindex = val_c
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        if epoch % 10 == 0: print(f"Epoch {epoch+1} | C-index: {val_c:.4f}")
    
    print(f"Final Best C-index: {best_cindex:.4f}")
    
    # SAVE FILES
    print("Saving files to /model volume...")
    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    
    torch.save(best_state, '/model/best_model.pt')
    np.savez('/model/aligned_data.npz', X=X, y_time=y_time, y_event=y_event,
             pathway_mask=pathway_mask, 
             pathway_names=np.array(pathway_names), gene_names=np.array(gene_names))
    
    # Explicitly commit volume to be safe
    model_volume.commit()
    print("Volume committed.")
    return best_cindex


@app.cls(image=image, gpu="T4", volumes={"/model": model_volume}, min_containers=0)
class OralCancerPredictor:
    @modal.enter()
    def load(self):
        import torch, torch.nn as nn, torch.nn.functional as F, numpy as np
        
        if not os.path.exists('/model/aligned_data.npz'):
            raise FileNotFoundError("Model files not found! Did you run --action train?")
            
        self.data = np.load('/model/aligned_data.npz', allow_pickle=True)
        self.pathway_names = list(self.data['pathway_names'])
        self.gene_names = list(self.data['gene_names'])
        
        class HybridPathwayMLP(nn.Module):
            def __init__(self, n_genes, pathway_mask, hidden_dims=[64, 32], pathway_embed_dim=64, dropout=0.5):
                super().__init__()
                self.n_genes, self.n_pathways = n_genes, pathway_mask.shape[1]
                self.register_buffer('pathway_mask', torch.from_numpy(pathway_mask).float())
                layers = []
                prev_dim = n_genes
                for hd in hidden_dims:
                    layers.extend([nn.Linear(prev_dim, hd), nn.BatchNorm1d(hd), nn.ReLU(), nn.Dropout(dropout)])
                    prev_dim = hd
                self.encoder = nn.Sequential(*layers)
                self.risk_head = nn.Sequential(nn.Linear(prev_dim, 64), nn.ReLU(), nn.Dropout(dropout/2), nn.Linear(64, 1))
                self.pathway_queries = nn.Parameter(torch.randn(self.n_pathways, pathway_embed_dim))
                self.hidden_to_pathway = nn.Linear(hidden_dims[-1], pathway_embed_dim)
                self.gene_pathway_scorer = nn.Linear(n_genes, self.n_pathways)
            def forward(self, x, return_interpretation=True):
                hidden = self.encoder(x)
                output = {'risk_score': self.risk_head(hidden)}
                if return_interpretation:
                    hp = self.hidden_to_pathway(hidden)
                    output['pathway_importance'] = F.softmax(torch.matmul(hp, self.pathway_queries.T)/np.sqrt(hp.size(-1)) + self.gene_pathway_scorer(x), dim=1)
                return output
        
        self.model = HybridPathwayMLP(len(self.gene_names), self.data['pathway_mask'])
        self.model.load_state_dict(torch.load('/model/best_model.pt', map_location='cpu'))
        self.model.eval()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)
        print(f"Loaded: {len(self.gene_names)} genes, {len(self.pathway_names)} pathways")
    
    @modal.method()
    def predict(self, expression_dict: dict) -> dict:
        import torch, numpy as np
        x = np.array([expression_dict.get(g, np.median(self.data['X'][:,i])) for i,g in enumerate(self.gene_names)])
        with torch.no_grad():
            out = self.model(torch.tensor([x]).float().to(self.device), return_interpretation=True)
            risk = 1/(1+np.exp(-out['risk_score'].item()))
            imp = out['pathway_importance'][0].cpu().numpy()
            top_idx = np.argsort(imp)[-5:][::-1]
        return {
            'risk_score': round(risk, 3),
            'risk_category': 'High' if risk > 0.7 else 'Medium' if risk > 0.4 else 'Low',
            'primary_pathway': self.pathway_names[top_idx[0]].replace('HALLMARK_', ''),
            'pathways': [{'name': self.pathway_names[i].replace('HALLMARK_',''), 'importance': round(float(imp[i]),4)} for i in top_idx]
        }


@app.local_entrypoint()
def main(action: str = "train"):
    import sys, numpy as np
    from pathlib import Path
    
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    from data_engineering.tcga_loader import TCGALoader
    
    if action == "train":
        print("📂 Loading data...")
        loader = TCGALoader("./data/tcga_hnsc")
        loader.load_clinical()
        loader.load_expression()
        loader.create_pathway_mask("./data/pathways/hallmark.gmt")
        aligned = loader.align_data()
        
        print("🚀 Training & Saving...")
        best_c = train_and_save.remote(
            X=aligned['X'], y_time=aligned['y_time'], y_event=aligned['y_event'],
            pathway_mask=aligned['pathway_mask'], pathway_names=aligned['pathway_names'],
            gene_names=aligned['gene_names']
        )
        print(f"✅ Training Done. C-index: {best_c:.4f}")
        print("To deploy: modal deploy modal_complete_v2.py")
        
    elif action == "test":
        print("🧪 Testing API...")
        predictor = OralCancerPredictor()
        # Mock data for quick test
        res = predictor.predict.remote({"TP53": 5.0})
        print(res)
