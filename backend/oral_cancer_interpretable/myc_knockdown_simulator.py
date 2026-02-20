"""
MYC Knockdown Simulator for Survival-Guided CRISPR

Simulates the transcriptional effect of CRISPR-mediated MYC knockdown,
including downstream cascade effects on MYC target genes.

Used to calculate "Survival Benefit" for therapeutic index.
"""

import modal
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

app = modal.App("myc-knockdown-simulator")
volume = modal.Volume.from_name("oral-cancer-model")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch", "numpy", "pandas", "scipy", "scikit-learn"
)

# MYC target genes from MSigDB Hallmark and literature
MYC_TARGETS = {
    'direct': ['CCND1', 'CCND2', 'CDK4', 'BCL2', 'MCL1', 'LDHA', 
              'PTTG1', 'BIRC5', 'AURKA', 'PLK1', 'FOXM1', 'EZH2'],
    'metabolic': ['HK2', 'PFKM', 'PKM', 'LDHA', 'SLC2A1', 'SLC2A3', 
                 'FASN', 'ACLY', 'SREBF1', 'SREBF2'],
    'ribosomal': ['RPL5', 'RPL11', 'RPL23', 'RPS14', 'RPS19', 
                 'RPS7', 'RPL32', 'RPS6', 'EIF4E', 'EIF4G1'],
    'cell_cycle': ['CDC25A', 'CDK1', 'CCNB1', 'CCNB2', 'CDC20', 
                  'AURKB', 'BUB1', 'MAD2L1'],
    'repressed': ['CDKN1A', 'CDKN2B', 'GADD45A', 'GADD45B', 'TP53']  # MYC represses these
}


def find_gene_index(gene_list: List[str], target: str) -> Optional[int]:
    """Find gene index with alias handling."""
    aliases = {
        'MYC': ['MYC', 'c-Myc', 'MYC_HUMAN'],
        'CCND1': ['CCND1', 'Cyclin_D1', 'BCL1'],
        'CDKN1A': ['CDKN1A', 'p21', 'CIP1', 'WAF1'],
        'SLC2A1': ['SLC2A1', 'GLUT1'],
        'SLC2A3': ['SLC2A3', 'GLUT3']
    }
    
    candidates = aliases.get(target, [target])
    for candidate in candidates:
        if candidate in gene_list:
            return gene_list.index(candidate)
    return None


@app.function(image=image, volumes={"/model": volume}, timeout=300)
def simulate_myc_knockdown(patient_expression: np.ndarray, 
                          gene_names: List[str],
                          knockdown_efficiency: float = 0.9,
                          cascade_strength: float = 0.5) -> Dict:
    """
    Simulate MYC CRISPR knockdown with downstream transcriptional effects.
    
    Args:
        patient_expression: log2(TPM+1) expression vector
        gene_names: List of gene symbols aligned with expression
        knockdown_efficiency: How much to reduce MYC (0.9 = 90% reduction)
        cascade_strength: How much MYC targets are affected (0.5 = 50%)
    
    Returns:
        Dictionary with modified expression and perturbation statistics
    """
    print("="*60)
    print("MYC KNOCKDOWN SIMULATION")
    print("="*60)
    
    # Find MYC index
    myc_idx = find_gene_index(gene_names, 'MYC')
    if myc_idx is None:
        raise ValueError("MYC not found in gene list!")
    
    original_myc = patient_expression[myc_idx]
    print(f"\nPatient MYC expression: {original_myc:.2f}")
    
    # Find target indices
    target_indices = {}
    present_targets = {}
    
    for category, genes in MYC_TARGETS.items():
        target_indices[category] = []
        present_targets[category] = []
        
        for gene in genes:
            idx = find_gene_index(gene_names, gene)
            if idx is not None:
                target_indices[category].append(idx)
                present_targets[category].append(gene)
    
    # Count targets found
    total_targets = sum(len(v) for v in present_targets.values())
    print(f"MYC target genes found: {total_targets}")
    for cat, genes in present_targets.items():
        if genes:
            print(f"  {cat}: {len(genes)} genes")
    
    # Create modified expression
    modified_expression = patient_expression.copy()
    perturbation_log = []
    
    # 1. Direct MYC suppression
    modified_expression[myc_idx] *= (1 - knockdown_efficiency)
    new_myc = modified_expression[myc_idx]
    
    perturbation_log.append({
        'gene': 'MYC',
        'original': float(original_myc),
        'modified': float(new_myc),
        'change_pct': float(-knockdown_efficiency * 100),
        'mechanism': 'direct_crispr'
    })
    
    print(f"\nDirect suppression: MYC {original_myc:.2f} → {new_myc:.2f} (-{knockdown_efficiency*100:.0f}%)")
    
    # 2. Cascade effects on activated targets
    myc_reduction = knockdown_efficiency
    
    for category in ['direct', 'metabolic', 'ribosomal', 'cell_cycle']:
        indices = target_indices.get(category, [])
        if not indices:
            continue
        
        # Cascade effect = MYC reduction × cascade_strength
        cascade_effect = myc_reduction * cascade_strength
        
        for idx in indices:
            original = modified_expression[idx]
            modified_expression[idx] *= (1 - cascade_effect)
            new_val = modified_expression[idx]
            
            perturbation_log.append({
                'gene': gene_names[idx],
                'original': float(original),
                'modified': float(new_val),
                'change_pct': float(-cascade_effect * 100),
                'mechanism': f'{category}_cascade'
            })
    
    # 3. De-repress MYC-repressed genes (they should INCREASE)
    repress_effect = myc_reduction * cascade_strength * 0.5
    
    for idx in target_indices.get('repressed', []):
        original = modified_expression[idx]
        modified_expression[idx] *= (1 + repress_effect)
        new_val = modified_expression[idx]
        
        perturbation_log.append({
            'gene': gene_names[idx],
            'original': float(original),
            'modified': float(new_val),
            'change_pct': float(repress_effect * 100),
            'mechanism': 'derepression'
        })
    
    # Calculate summary statistics
    genes_affected = len(perturbation_log)
    
    # MYC pathway activity score (mean of target expression)
    def calc_pathway_score(expr):
        scores = []
        for cat in ['direct', 'metabolic', 'cell_cycle']:
            for idx in target_indices.get(cat, []):
                scores.append(expr[idx])
        return np.mean(scores) if scores else 0
    
    pathway_before = calc_pathway_score(patient_expression)
    pathway_after = calc_pathway_score(modified_expression)
    
    print(f"\nMYC Pathway Score: {pathway_before:.2f} → {pathway_after:.2f}")
    print(f"Pathway reduction: {(1 - pathway_after/pathway_before)*100:.1f}%")
    print(f"Total genes affected: {genes_affected}")
    
    return {
        'original_expression': patient_expression.tolist(),
        'modified_expression': modified_expression.tolist(),
        'myc_index': myc_idx,
        'original_myc': float(original_myc),
        'new_myc': float(new_myc),
        'pathway_score_before': float(pathway_before),
        'pathway_score_after': float(pathway_after),
        'pathway_reduction_pct': float((1 - pathway_after/pathway_before)*100),
        'genes_affected': genes_affected,
        'perturbation_log': perturbation_log,
        'knockdown_efficiency': knockdown_efficiency,
        'cascade_strength': cascade_strength
    }


@app.function(image=image, gpu="T4", volumes={"/model": volume}, timeout=600)
def test_knockdown_on_tcga() -> Dict:
    """
    Test MYC knockdown simulation on real TCGA patients.
    Compares survival risk before and after simulated knockdown.
    """
    print("="*60)
    print("TESTING MYC KNOCKDOWN ON TCGA PATIENTS")
    print("="*60)
    
    # Load TCGA data
    data = np.load("/model/aligned_data.npz", allow_pickle=True)
    X = data['X']
    gene_names = list(data['gene_names'])
    pathway_names = list(data['pathway_names'])
    y_time = data['y_time']
    y_event = data['y_event']
    
    print(f"\nLoaded: {X.shape[0]} patients, {X.shape[1]} genes")
    
    # Find MYC index
    myc_idx = find_gene_index(gene_names, 'MYC')
    if myc_idx is None:
        return {"error": "MYC not found"}
    
    myc_levels = X[:, myc_idx]
    print(f"MYC expression range: {myc_levels.min():.2f} - {myc_levels.max():.2f}")
    print(f"MYC median: {np.median(myc_levels):.2f}")
    
    # Load survival model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
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
    
    model = MYCFocusedPathwayMLP(len(gene_names), len(pathway_names)).to(device)
    model.load_state_dict(torch.load("/model/myc_enhanced_model.pt", map_location=device))
    model.eval()
    print("✅ Loaded MYC survival model")
    
    # Select high-MYC patients (top 20%)
    threshold = np.percentile(myc_levels, 80)
    high_myc_mask = myc_levels > threshold
    high_myc_indices = np.where(high_myc_mask)[0]
    
    print(f"\nTesting on {len(high_myc_indices)} high-MYC patients (top 20%)")
    
    results = []
    
    # Temperature scaling to prevent saturation
    TEMPERATURE = 10.0  # Higher = softer probabilities
    
    for i, patient_idx in enumerate(high_myc_indices[:20]):  # Test 20 patients
        patient_expr = X[patient_idx]
        
        # Baseline risk (raw logit)
        with torch.no_grad():
            patient_tensor = torch.tensor(patient_expr, dtype=torch.float32).unsqueeze(0).to(device)
            base_risk_raw, base_pathway = model(patient_tensor)
            base_logit = base_risk_raw.item()
            # Apply temperature scaling for usable probabilities
            base_risk_scaled = torch.sigmoid(torch.tensor(base_logit / TEMPERATURE)).item()
        
        # Simulate knockdown
        sim = simulate_myc_knockdown.local(
            patient_expr, gene_names,
            knockdown_efficiency=0.9,
            cascade_strength=0.5
        )
        
        modified_expr = np.array(sim['modified_expression'])
        
        # Post-knockdown risk
        with torch.no_grad():
            modified_tensor = torch.tensor(modified_expr, dtype=torch.float32).unsqueeze(0).to(device)
            new_risk_raw, new_pathway = model(modified_tensor)
            new_logit = new_risk_raw.item()
            new_risk_scaled = torch.sigmoid(torch.tensor(new_logit / TEMPERATURE)).item()
        
        # Survival benefit = logit difference (positive = risk decreased = good)
        benefit_logit = base_logit - new_logit
        benefit_prob = base_risk_scaled - new_risk_scaled
        
        results.append({
            'patient_idx': int(patient_idx),
            'original_myc': sim['original_myc'],
            'base_logit': base_logit,
            'new_logit': new_logit,
            'base_risk_scaled': base_risk_scaled,
            'new_risk_scaled': new_risk_scaled,
            'survival_benefit_logit': benefit_logit,
            'survival_benefit_prob': benefit_prob,
            'pathway_reduction_pct': sim['pathway_reduction_pct'],
            'actual_event': int(y_event[patient_idx]),
            'actual_time': float(y_time[patient_idx])
        })
        
        if i < 5:
            print(f"\n  Patient {patient_idx}:")
            print(f"    MYC: {sim['original_myc']:.2f} → {sim['new_myc']:.2f}")
            print(f"    Raw Logit: {base_logit:.2f} → {new_logit:.2f}")
            print(f"    Scaled Risk (T={TEMPERATURE}): {base_risk_scaled:.3f} → {new_risk_scaled:.3f}")
            print(f"    Benefit (logit): {benefit_logit:.3f}")


    
    # Summary statistics
    results_df = pd.DataFrame(results)
    
    print("\n" + "="*60)
    print("SUMMARY: MYC KNOCKDOWN SIMULATION")
    print("="*60)
    print(f"\nPatients tested: {len(results)}")
    print(f"Temperature scaling: {TEMPERATURE}")
    print(f"Average baseline logit: {results_df['base_logit'].mean():.2f}")
    print(f"Average post-KD logit: {results_df['new_logit'].mean():.2f}")
    print(f"Average survival benefit (logit diff): {results_df['survival_benefit_logit'].mean():.3f}")
    print(f"Average benefit (probability): {results_df['survival_benefit_prob'].mean():.3f}")
    print(f"Positive benefit rate: {(results_df['survival_benefit_logit'] > 0).mean()*100:.1f}%")
    
    # Correlation: Does higher MYC → more benefit?
    from scipy.stats import pearsonr
    if results_df['survival_benefit_logit'].std() > 0:
        corr, pval = pearsonr(results_df['original_myc'], results_df['survival_benefit_logit'])
        print(f"\nMYC vs Benefit correlation: r={corr:.3f}, p={pval:.4f}")
    else:
        corr, pval = 0.0, 1.0
        print("\nMYC vs Benefit correlation: No variance in benefit")
    
    if corr > 0.2:
        print("✅ Higher MYC expression → More benefit from knockdown (expected)")
    elif results_df['survival_benefit_logit'].mean() > 0:
        print("✅ Positive average benefit from MYC knockdown!")
    else:
        print("⚠️ Check simulation parameters")
    
    return {
        'n_patients': len(results),
        'temperature': TEMPERATURE,
        'avg_baseline_logit': float(results_df['base_logit'].mean()),
        'avg_post_kd_logit': float(results_df['new_logit'].mean()),
        'avg_survival_benefit_logit': float(results_df['survival_benefit_logit'].mean()),
        'avg_survival_benefit_prob': float(results_df['survival_benefit_prob'].mean()),
        'positive_benefit_rate': float((results_df['survival_benefit_logit'] > 0).mean()),
        'myc_benefit_correlation': float(corr) if not np.isnan(corr) else 0.0,
        'correlation_pval': float(pval) if not np.isnan(pval) else 1.0,
        'patient_results': results
    }


@app.local_entrypoint()
def main():
    """Test MYC knockdown simulation on TCGA patients."""
    print("\n" + "="*60)
    print("MYC KNOCKDOWN SIMULATION + SURVIVAL MODEL TEST")
    print("="*60)
    
    result = test_knockdown_on_tcga.remote()
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    print(f"\nTemperature Scaling: {result['temperature']}")
    print(f"Average Baseline Logit: {result['avg_baseline_logit']:.2f}")
    print(f"Average Post-KD Logit: {result['avg_post_kd_logit']:.2f}")
    print(f"Average Survival Benefit (logit): {result['avg_survival_benefit_logit']:.3f}")
    print(f"Positive Benefit Rate: {result['positive_benefit_rate']*100:.1f}%")
    print(f"MYC-Benefit Correlation: r={result['myc_benefit_correlation']:.3f}")
    
    if result['avg_survival_benefit_logit'] > 0.5:
        print("\n✅ MYC knockdown shows significant survival benefit!")
        print("   Ready for therapeutic index calculation.")
    elif result['avg_survival_benefit_logit'] > 0.1:
        print("\n✅ Moderate benefit from MYC knockdown")
    else:
        print("\n⚠️ Low benefit - check simulation parameters")

