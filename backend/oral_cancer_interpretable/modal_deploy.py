"""
Modal Deployment for Oral Cancer Risk Prediction API

Production-ready endpoint for clinical decision support.
Deploy with: modal deploy modal_deploy.py
"""

import modal
from typing import Dict, List, Any

app = modal.App("oral-cancer-risk-api")

# Production image with all dependencies
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

# Model files volume (upload once, use forever)
model_volume = modal.Volume.from_name("oral-cancer-model", create_if_missing=True)


@app.cls(
    image=image,
    gpu="T4",
    volumes={"/model": model_volume},
    keep_warm=1,  # Keep one instance warm for fast responses
    timeout=120
)
class OralCancerPredictor:
    """Production predictor with loaded model and clinical interpretation."""
    
    @modal.enter()
    def load_model(self):
        """Load model on container startup."""
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import numpy as np
        
        print("Loading oral cancer risk model...")
        
        # Load aligned data (for gene names and pathway info)
        self.data = np.load('/model/aligned_data.npz', allow_pickle=True)
        self.pathway_names = list(self.data['pathway_names'])
        self.gene_names = list(self.data['gene_names'])
        self.pathway_mask = self.data['pathway_mask']
        
        # Define model architecture (must match training)
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
        
        # Load model
        self.model = HybridPathwayMLP(
            n_genes=len(self.gene_names),
            pathway_mask=self.pathway_mask,
            hidden_dims=[512, 256, 128],
            pathway_embed_dim=64,
            dropout=0.3
        )
        
        state_dict = torch.load('/model/best_hybrid_model.pt', map_location='cpu')
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        # Move to GPU
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)
        
        print(f"Model loaded on {self.device}")
        print(f"Features: {len(self.gene_names)} genes, {len(self.pathway_names)} pathways")
    
    @modal.method()
    def predict(self, expression_dict: Dict[str, float]) -> Dict[str, Any]:
        """
        Predict oral cancer risk from gene expression.
        
        Args:
            expression_dict: {gene_name: expression_value, ...}
            
        Returns:
            Clinical risk assessment with pathway analysis
        """
        import torch
        import numpy as np
        
        # Convert dict to vector
        x = np.zeros(len(self.gene_names))
        matched_genes = 0
        
        for i, gene in enumerate(self.gene_names):
            if gene in expression_dict:
                x[i] = expression_dict[gene]
                matched_genes += 1
            else:
                # Use median from training data for missing genes
                x[i] = np.median(self.data['X'][:, i])
        
        x_tensor = torch.tensor([x]).float().to(self.device)
        
        with torch.no_grad():
            output = self.model(x_tensor, return_interpretation=True)
            
            raw_risk = output['risk_score'].item()
            risk_prob = 1 / (1 + np.exp(-raw_risk))
            
            pathway_imp = output['pathway_importance'][0].cpu().numpy()
            top_idx = np.argsort(pathway_imp)[-5:][::-1]
        
        # Risk category
        if risk_prob > 0.7:
            category = 'High'
            color = 'red'
        elif risk_prob > 0.4:
            category = 'Medium'
            color = 'yellow'
        else:
            category = 'Low'
            color = 'green'
        
        return {
            'risk_score': round(risk_prob, 3),
            'risk_category': category,
            'risk_color': color,
            'genes_matched': matched_genes,
            'genes_total': len(self.gene_names),
            'pathways': [
                {
                    'name': self.pathway_names[i].replace('HALLMARK_', ''),
                    'importance': round(float(pathway_imp[i]), 4)
                }
                for i in top_idx
            ],
            'primary_pathway': self.pathway_names[top_idx[0]].replace('HALLMARK_', ''),
            'model_confidence': 'High' if pathway_imp[top_idx[0]] > 0.5 else 'Medium'
        }
    
    @modal.method()
    def batch_predict(self, patients: List[Dict[str, float]]) -> List[Dict[str, Any]]:
        """Predict for multiple patients."""
        return [self.predict(patient) for patient in patients]
    
    @modal.method()
    def get_feature_names(self) -> Dict[str, List[str]]:
        """Return expected input features."""
        return {
            'gene_names': self.gene_names[:100],  # First 100 for API docs
            'total_genes': len(self.gene_names),
            'pathway_names': [p.replace('HALLMARK_', '') for p in self.pathway_names]
        }


# FastAPI web endpoint
@app.function(image=image)
@modal.asgi_app()
def web_app():
    """FastAPI web endpoint for REST API access."""
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    from typing import Dict
    
    api = FastAPI(
        title="Oral Cancer Risk API",
        description="AI-powered oral cancer survival risk prediction with pathway interpretation",
        version="1.0.0"
    )
    
    class PredictionRequest(BaseModel):
        expression: Dict[str, float]
        patient_id: str = "anonymous"
    
    class BatchRequest(BaseModel):
        patients: List[Dict[str, float]]
    
    @api.get("/")
    async def root():
        return {
            "service": "Oral Cancer Risk Prediction",
            "version": "1.0.0",
            "endpoints": ["/predict", "/batch", "/features"]
        }
    
    @api.post("/predict")
    async def predict(request: PredictionRequest):
        try:
            predictor = OralCancerPredictor()
            result = predictor.predict.remote(request.expression)
            result['patient_id'] = request.patient_id
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @api.post("/batch")
    async def batch_predict(request: BatchRequest):
        try:
            predictor = OralCancerPredictor()
            return predictor.batch_predict.remote(request.patients)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @api.get("/features")
    async def get_features():
        predictor = OralCancerPredictor()
        return predictor.get_feature_names.remote()
    
    @api.get("/health")
    async def health():
        return {"status": "healthy"}
    
    return api


@app.local_entrypoint()
def test():
    """Test the deployment locally."""
    import numpy as np
    from pathlib import Path
    import sys
    
    # Load sample data
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    from data_engineering.tcga_loader import TCGALoader
    
    print("Testing Oral Cancer Risk API...")
    
    loader = TCGALoader("./data/tcga_hnsc")
    loader.load_clinical()
    loader.load_expression()
    loader.create_pathway_mask("./data/pathways/hallmark.gmt")
    aligned = loader.align_data()
    
    # Create sample expression dict
    sample_expr = {
        aligned['gene_names'][i]: float(aligned['X'][0, i])
        for i in range(len(aligned['gene_names']))
    }
    
    # Test prediction
    predictor = OralCancerPredictor()
    result = predictor.predict.remote(sample_expr)
    
    print("\n" + "="*50)
    print("TEST PREDICTION RESULT")
    print("="*50)
    print(f"Risk Score: {result['risk_score']}")
    print(f"Category: {result['risk_category']}")
    print(f"Primary Pathway: {result['primary_pathway']}")
    print(f"Genes Matched: {result['genes_matched']}/{result['genes_total']}")
    print("\nTop Pathways:")
    for p in result['pathways'][:3]:
        print(f"  • {p['name']}: {p['importance']:.1%}")
    
    print("\n✅ Test completed successfully!")
