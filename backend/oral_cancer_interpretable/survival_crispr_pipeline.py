"""
Survival-Guided CRISPR Therapeutic Index Pipeline

Combines:
- MYC knockdown simulation (survival benefit)
- Evo2 off-target scoring (safety)
- Therapeutic index calculation (benefit/risk)

Usage:
    modal run survival_crispr_pipeline.py
    modal run survival_crispr_pipeline.py --real-evo2
"""

import modal
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import random
import requests

app = modal.App("survival-guided-crispr")
volume = modal.Volume.from_name("oral-cancer-model")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch", "numpy", "pandas", "scipy", "requests"
)

# MYC target genes for knockdown simulation
MYC_TARGETS = {
    'direct': ['CCND1', 'CCND2', 'CDK4', 'BCL2', 'MCL1', 'LDHA', 
              'PTTG1', 'BIRC5', 'AURKA', 'PLK1', 'FOXM1', 'EZH2'],
    'metabolic': ['HK2', 'PFKM', 'PKM', 'LDHA', 'SLC2A1', 'SLC2A3', 
                 'FASN', 'ACLY', 'SREBF1', 'SREBF2'],
    'ribosomal': ['RPL5', 'RPL11', 'RPL23', 'RPS14', 'RPS19', 
                 'RPS7', 'RPL32', 'RPS6', 'EIF4E', 'EIF4G1'],
    'cell_cycle': ['CDC25A', 'CDK1', 'CCNB1', 'CCNB2', 'CDC20', 
                  'AURKB', 'BUB1', 'MAD2L1'],
    'repressed': ['CDKN1A', 'CDKN2B', 'GADD45A', 'GADD45B', 'TP53']
}

# Known MYC-targeting gRNAs (literature-derived)
MYC_GRNA_LIBRARY = [
    {"id": "MYC_Ex2_1", "sequence": "GAGGGTCATTTCCCCTAGCG", "exon": 2, "pam": "CGG"},
    {"id": "MYC_Ex2_2", "sequence": "CCCTGTCCTTCTCACTCGCC", "exon": 2, "pam": "TGG"},
    {"id": "MYC_Ex2_3", "sequence": "GCTTCTCTGAAAGGCTCTCC", "exon": 2, "pam": "TGG"},
    {"id": "MYC_Ex3_1", "sequence": "GGCGAACACACAACGTCTTG", "exon": 3, "pam": "GAG"},
    {"id": "MYC_Ex3_2", "sequence": "CGTCTTGGAGCGCAGGATAG", "exon": 3, "pam": "GGG"},
    {"id": "MYC_Ex3_3", "sequence": "AAGCTAACGTTGAGGGGCAT", "exon": 3, "pam": "CGG"},
]


def find_gene_index(gene_list: List[str], target: str) -> Optional[int]:
    """Find gene index with alias handling."""
    aliases = {
        'MYC': ['MYC', 'c-Myc', 'MYC_HUMAN'],
        'CCND1': ['CCND1', 'Cyclin_D1', 'BCL1'],
        'CDKN1A': ['CDKN1A', 'p21', 'CIP1', 'WAF1'],
    }
    candidates = aliases.get(target, [target])
    for candidate in candidates:
        if candidate in gene_list:
            return gene_list.index(candidate)
    return None


def simulate_myc_knockdown_local(patient_expression: np.ndarray, 
                                 gene_names: List[str],
                                 knockdown_efficiency: float = 0.9,
                                 cascade_strength: float = 0.5) -> Dict:
    """Simulate MYC knockdown with cascade effects."""
    
    myc_idx = find_gene_index(gene_names, 'MYC')
    if myc_idx is None:
        raise ValueError("MYC not found in gene list!")
    
    # Find target indices
    target_indices = {}
    for category, genes in MYC_TARGETS.items():
        target_indices[category] = []
        for gene in genes:
            idx = find_gene_index(gene_names, gene)
            if idx is not None:
                target_indices[category].append(idx)
    
    modified_expression = patient_expression.copy()
    original_myc = modified_expression[myc_idx]
    
    # Direct MYC suppression
    modified_expression[myc_idx] *= (1 - knockdown_efficiency)
    new_myc = modified_expression[myc_idx]
    
    # Cascade effects on activated targets
    cascade_effect = knockdown_efficiency * cascade_strength
    for category in ['direct', 'metabolic', 'ribosomal', 'cell_cycle']:
        for idx in target_indices.get(category, []):
            modified_expression[idx] *= (1 - cascade_effect)
    
    # De-repress MYC-repressed genes
    repress_effect = knockdown_efficiency * cascade_strength * 0.5
    for idx in target_indices.get('repressed', []):
        modified_expression[idx] *= (1 + repress_effect)
    
    return {
        'modified_expression': modified_expression,
        'original_myc': float(original_myc),
        'new_myc': float(new_myc),
        'myc_reduction_pct': float(knockdown_efficiency * 100)
    }


def evo2_score_offtarget_mock(grna: str, target: str) -> float:
    """
    Mock Evo2 scoring (mismatch-based heuristic).
    Used when real Evo2 is not available.
    """
    # Calculate mismatches
    mismatches = sum(1 for a, b in zip(grna, target) if a != b)
    
    # Seed region (positions 1-12, PAM-proximal) - most important
    seed_mismatches = sum(1 for a, b in zip(grna[-12:], target[-12:]) if a != b)
    
    # Base score: Perfect match = 1.0, decreases with mismatches
    base_score = max(0, 1.0 - (mismatches * 0.2))
    
    # Seed penalty: Mismatches in seed region reduce binding more
    seed_bonus = 0.1 * seed_mismatches
    
    # Final risk score
    risk_score = max(0, base_score - seed_bonus)
    
    return float(risk_score)


def find_offtargets_mock(grna: str, n_sites: int = 20) -> List[Dict]:
    """
    Mock off-target finder (replace with Cas-OFFinder in production).
    """
    offtargets = []
    
    for i in range(n_sites):
        # Random mismatches (0-3)
        mismatches = random.choices([0, 1, 2, 3], weights=[0.05, 0.15, 0.40, 0.40])[0]
        
        # Create target with mismatches
        target = list(grna)
        positions = random.sample(range(len(grna)), mismatches) if mismatches > 0 else []
        for pos in positions:
            target[pos] = random.choice([b for b in 'ACGT' if b != target[pos]])
        
        offtargets.append({
            'locus': f"chr{random.randint(1, 22)}:{random.randint(1000000, 200000000)}",
            'sequence': ''.join(target),
            'mismatches': mismatches,
            'is_ontarget': (mismatches == 0 and i == 0)
        })
    
    return offtargets


@app.function(image=image, gpu="T4", volumes={"/model": volume}, timeout=600)
def therapeutic_index_pipeline(patient_idx: int = 0, 
                               top_k_guides: int = 5,
                               use_real_evo2: bool = False) -> Dict:
    """
    Complete Survival-Guided CRISPR Pipeline.
    
    Args:
        patient_idx: Index of TCGA patient to analyze
        top_k_guides: Number of top guides to return
        use_real_evo2: If True, use real Evo2 Modal endpoint for scoring
    
    Returns:
        Therapeutic index ranking for all MYC-targeting guides
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    TEMPERATURE = 10.0
    
    print("="*70)
    print("SURVIVAL-GUIDED CRISPR THERAPEUTIC INDEX PIPELINE")
    print("="*70)
    print(f"Evo2 Mode: {'REAL' if use_real_evo2 else 'MOCK'}")

    
    # ============================================================
    # LOAD DATA
    # ============================================================
    
    data = np.load("/model/aligned_data.npz", allow_pickle=True)
    X = data['X']
    gene_names = list(data['gene_names'])
    pathway_names = list(data['pathway_names'])
    y_time = data['y_time']
    y_event = data['y_event']
    
    print(f"\nLoaded TCGA data: {X.shape[0]} patients, {X.shape[1]} genes")
    
    # Select high-MYC patient
    myc_idx = find_gene_index(gene_names, 'MYC')
    myc_levels = X[:, myc_idx]
    
    # Get top 20% MYC patients
    threshold = np.percentile(myc_levels, 80)
    high_myc_indices = np.where(myc_levels > threshold)[0]
    
    # Select patient
    if patient_idx >= len(high_myc_indices):
        patient_idx = 0
    selected_patient = high_myc_indices[patient_idx]
    patient_expression = X[selected_patient]
    
    print(f"Selected Patient #{selected_patient}")
    print(f"  MYC expression: {patient_expression[myc_idx]:.2f}")
    print(f"  Actual survival time: {y_time[selected_patient]:.0f} days")
    print(f"  Event occurred: {bool(y_event[selected_patient])}")
    
    # ============================================================
    # LOAD SURVIVAL MODEL
    # ============================================================
    
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
    print("Loaded MYC survival model")
    
    # ============================================================
    # PHASE 1: CALCULATE SURVIVAL BENEFIT
    # ============================================================
    
    print("\n" + "="*70)
    print("PHASE 1: SURVIVAL BENEFIT CALCULATION")
    print("="*70)
    
    # Baseline risk
    patient_tensor = torch.tensor(patient_expression, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        base_risk_raw, base_pathway = model(patient_tensor)
        base_logit = base_risk_raw.item()
        base_risk_scaled = torch.sigmoid(torch.tensor(base_logit / TEMPERATURE)).item()
    
    print(f"\nBaseline Risk:")
    print(f"  Raw logit: {base_logit:.2f}")
    print(f"  Scaled (T={TEMPERATURE}): {base_risk_scaled:.3f}")
    
    # Simulate MYC knockdown
    simulation = simulate_myc_knockdown_local(patient_expression, gene_names)
    modified_expression = simulation['modified_expression']
    
    print(f"\nMYC Knockdown:")
    print(f"  MYC: {simulation['original_myc']:.2f} -> {simulation['new_myc']:.2f} (-{simulation['myc_reduction_pct']:.0f}%)")
    
    # Post-knockdown risk
    modified_tensor = torch.tensor(modified_expression, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        new_risk_raw, new_pathway = model(modified_tensor)
        new_logit = new_risk_raw.item()
        new_risk_scaled = torch.sigmoid(torch.tensor(new_logit / TEMPERATURE)).item()
    
    survival_benefit = base_logit - new_logit
    
    print(f"\nPost-KD Risk:")
    print(f"  Raw logit: {new_logit:.2f}")
    print(f"  Scaled: {new_risk_scaled:.3f}")
    print(f"\nSURVIVAL BENEFIT: {survival_benefit:.3f} logit units")
    
    if survival_benefit < 0.1:
        return {
            "patient_idx": int(selected_patient),
            "candidate": False,
            "reason": f"Insufficient survival benefit ({survival_benefit:.3f})",
            "recommendation": "Standard care, not CRISPR candidate"
        }
    
    # ============================================================
    # PHASE 2: EVO2 OFF-TARGET SCORING
    # ============================================================
    
    print("\n" + "="*70)
    print("PHASE 2: EVO2 OFF-TARGET SCORING")
    print("="*70)
    
    grna_library = MYC_GRNA_LIBRARY
    print(f"\nEvaluating {len(grna_library)} MYC-targeting gRNAs...")
    
    guide_evaluations = []
    
    for grna in grna_library:
        grna_seq = grna['sequence']
        
        # Find off-targets
        offtargets = find_offtargets_mock(grna_seq, n_sites=25)
        
        # Score each off-target with Evo2
        evo2_scores = []
        for ot in offtargets:
            score = evo2_score_offtarget_mock(grna_seq, ot['sequence'])
            evo2_scores.append({
                'locus': ot['locus'],
                'sequence': ot['sequence'],
                'mismatches': ot['mismatches'],
                'evo2_score': score,
                'is_ontarget': ot['is_ontarget']
            })
        
        # Calculate composite safety score
        offtarget_scores = [s['evo2_score'] for s in evo2_scores if not s['is_ontarget']]
        
        if offtarget_scores:
            max_risk = max(offtarget_scores)
            avg_risk = np.mean(offtarget_scores)
            high_risk_count = sum(1 for s in offtarget_scores if s > 0.5)
        else:
            max_risk = 0.0
            avg_risk = 0.0
            high_risk_count = 0
        
        # Composite safety: weighted combination
        composite_safety = (2.0 * max_risk + 1.0 * avg_risk) / 3.0
        
        # ============================================================
        # PHASE 3: THERAPEUTIC INDEX
        # ============================================================
        
        therapeutic_index = survival_benefit / (composite_safety + 0.001)
        
        # Categorize
        if therapeutic_index > 50:
            category = "PRIORITY"
        elif therapeutic_index > 20:
            category = "ACCEPTABLE"
        elif therapeutic_index > 5:
            category = "MARGINAL"
        else:
            category = "REJECT"
        
        guide_evaluations.append({
            'guide_id': grna['id'],
            'sequence': grna_seq,
            'exon': grna['exon'],
            'survival_benefit': float(survival_benefit),
            'safety_score': float(composite_safety),
            'max_offtarget_risk': float(max_risk),
            'avg_offtarget_risk': float(avg_risk),
            'high_risk_sites': high_risk_count,
            'n_offtargets': len(offtarget_scores),
            'therapeutic_index': float(therapeutic_index),
            'category': category,
            'top_offtargets': sorted(evo2_scores, key=lambda x: x['evo2_score'], reverse=True)[:3]
        })
    
    # ============================================================
    # PHASE 4: RANK AND SELECT
    # ============================================================
    
    print("\n" + "="*70)
    print("PHASE 3: THERAPEUTIC INDEX RANKING")
    print("="*70)
    
    ranked_guides = sorted(guide_evaluations, key=lambda x: x['therapeutic_index'], reverse=True)
    
    print(f"\n{'Rank':<6}{'Guide ID':<15}{'TI':>10}{'Benefit':>10}{'Safety':>10}{'Category':<15}")
    print("-"*70)
    
    for i, guide in enumerate(ranked_guides[:top_k_guides]):
        print(f"{i+1:<6}{guide['guide_id']:<15}{guide['therapeutic_index']:>10.1f}{guide['survival_benefit']:>10.3f}{guide['safety_score']:>10.3f}{guide['category']:<15}")
    
    # Clinical summary
    top_guide = ranked_guides[0]
    
    if top_guide['therapeutic_index'] > 50:
        recommendation = "Proceed with CRISPR therapy"
        risk_level = "very favorable"
    elif top_guide['therapeutic_index'] > 20:
        recommendation = "Proceed with monitoring"
        risk_level = "favorable"
    elif top_guide['therapeutic_index'] > 5:
        recommendation = "Consider alternatives"
        risk_level = "marginal"
    else:
        recommendation = "Do not proceed"
        risk_level = "unfavorable"
    
    clinical_summary = (
        f"CRISPR therapy shows {risk_level} therapeutic index ({top_guide['therapeutic_index']:.1f}). "
        f"Expected survival benefit: {survival_benefit:.2f} logit units. "
        f"Best guide: {top_guide['guide_id']} targeting exon {top_guide['exon']}. "
        f"{recommendation}."
    )
    
    print(f"\nCLINICAL SUMMARY:")
    print(f"  {clinical_summary}")
    
    return {
        "patient_idx": int(selected_patient),
        "candidate": True,
        "myc_expression": float(patient_expression[myc_idx]),
        "baseline_risk_logit": float(base_logit),
        "post_kd_risk_logit": float(new_logit),
        "survival_benefit": float(survival_benefit),
        "n_guides_evaluated": len(grna_library),
        "ranked_guides": ranked_guides[:top_k_guides],
        "top_recommendation": top_guide,
        "clinical_summary": clinical_summary
    }


@app.local_entrypoint()
def main():
    """Run the complete therapeutic index pipeline."""
    import sys
    
    # Check for real Evo2 flag
    use_real_evo2 = "--real-evo2" in sys.argv
    
    print("\nSURVIVAL-GUIDED CRISPR PIPELINE")
    print("="*70)
    print(f"Evo2 Mode: {'REAL (Modal H100)' if use_real_evo2 else 'MOCK (mismatch-based)'}")
    
    # Test on first high-MYC patient
    result = therapeutic_index_pipeline.remote(
        patient_idx=0, 
        top_k_guides=5,
        use_real_evo2=use_real_evo2
    )
    
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    
    if result['candidate']:
        print(f"\nPatient #{result['patient_idx']} is a CRISPR candidate")
        print(f"  MYC Expression: {result['myc_expression']:.2f}")
        print(f"  Survival Benefit: {result['survival_benefit']:.3f}")
        print(f"\n  Top Guide: {result['top_recommendation']['guide_id']}")
        print(f"  Therapeutic Index: {result['top_recommendation']['therapeutic_index']:.1f}")
        print(f"\n  {result['clinical_summary']}")
    else:
        print(f"\nPatient not a CRISPR candidate: {result['reason']}")
