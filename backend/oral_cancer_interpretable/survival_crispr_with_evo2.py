"""
Survival-Guided CRISPR Pipeline with Real Evo2 Integration

This file combines:
1. MYC knockdown simulation (survival benefit)
2. Real Evo2 off-target scoring (safety) - runs on H100
3. Therapeutic index calculation (benefit/risk)

Both components are in the SAME Modal app for inter-function calls.

Usage:
    modal run survival_crispr_with_evo2.py
"""

import modal
import subprocess
import sys
import os

# ===================================================================
# EVO2 IMAGE SETUP (Heavy - requires H100)
# ===================================================================

def build_cuda_kernels():
    """Compile flash-attn and transformer-engine on a GPU machine."""
    os.environ["MAX_JOBS"] = "6"
    for pkg in ("flash-attn==2.8.0.post2",
                "transformer_engine[pytorch]==2.8.0"):
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", pkg, "--no-build-isolation"
        ])

evo2_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12"
    )
    .apt_install(
        "build-essential", "cmake", "ninja-build", "libcudnn8",
        "libcudnn8-dev", "git", "gcc", "g++"
    )
    .env({"CC": "/usr/bin/gcc", "CXX": "/usr/bin/g++", "BUILD_ID": "crispr-evo2-v1"})
    .pip_install("packaging", "wheel", "setuptools", "ninja")
    .run_commands(
        "git clone --recurse-submodules https://github.com/ArcInstitute/evo2.git && "
        "cd evo2 && pip install ."
    )
    .run_function(                  
        build_cuda_kernels,
        gpu="L40S",
        memory=32768,
        cpu=8,
        timeout=3600
    )
    .pip_install(
        "torch", 
        "vtx>=0.0.8",
        "pandas",
        "numpy",
        "scipy"
    )
)

# ===================================================================
# SURVIVAL MODEL IMAGE (Lighter - runs on T4)
# ===================================================================

survival_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch", "numpy", "pandas", "scipy", "requests"
)

# ===================================================================
# CREATE COMBINED APP
# ===================================================================

app = modal.App("survival-guided-crispr-evo2")
oral_cancer_volume = modal.Volume.from_name("oral-cancer-model")
hf_cache_volume = modal.Volume.from_name("hf_cache", create_if_missing=True)

# ===================================================================
# EVO2 SCORER CLASS (H100)
# ===================================================================

@app.cls(image=evo2_image, gpu="H100", volumes={"/root/.cache/huggingface": hf_cache_volume})
class Evo2OffTargetScorer:
    """Evo2 scorer for CRISPR off-target risk assessment"""
    
    @modal.enter()
    def load_model(self):
        from evo2 import Evo2
        print("Loading Evo2-7B model for CRISPR off-target scoring...")
        self.model = Evo2('evo2_7b')
        print("Evo2-7B loaded successfully")
    
    def _score_offtarget(self, grna_seq: str, target_seq: str, pam: str = "NGG") -> dict:
        """
        Real Evo2 off-target scoring using delta_log_likelihood.
        
        Methodology:
        1. Identify mismatches between gRNA (intended) and Target (off-target).
        2. Treat gRNA as the 'reference' (ideal) and Target as 'alt'.
        3. Sum delta_log_likelihood for each mismatch (independence assumption).
        4. Lower delta_ll = Lower binding probability (Safe).
           Near zero delta_ll = High binding probability (Risk).
        """
        import numpy as np
        
        try:
            # Ensure equal length
            min_len = min(len(grna_seq), len(target_seq))
            grna_seq = grna_seq[:min_len]
            target_seq = target_seq[:min_len]
            
            mismatches = []
            seed_mismatches = 0
            
            # Identify mismatch positions
            for i, (g, t) in enumerate(zip(grna_seq, target_seq)):
                if g != t:
                    mismatches.append(i)
                    if i >= min_len - 12:  # Seed region (last 12bp)
                        seed_mismatches += 1
            
            # Calculate Delta LL
            total_delta_ll = 0.0
            
            if not mismatches:
                total_delta_ll = 0.0 # Perfect match
            else:
                # Sum delta_ll for each mismatch position
                for pos in mismatches:
                    # Construct sequences for single-point query
                    # Ref: gRNA sequence
                    # Alt: gRNA sequence with ONLY this mutation applied
                    # This isolates the effect of this specific mismatch
                    
                    # Note: We use the full context if possible, but here we just use the seqs we have.
                    # Ideally we would use the gRNA seq as ref, and mutate one base to match target.
                    
                    # Construct single-mutant Alt
                    seq_list = list(grna_seq)
                    seq_list[pos] = target_seq[pos]
                    single_mutant_alt = "".join(seq_list)
                    
                    try:
                        # Call Evo2
                        dll = self.model.delta_log_likelihood(grna_seq, single_mutant_alt, position=pos)
                        total_delta_ll += float(dll)
                    except Exception as e_inner:
                        print(f"Error scoring pos {pos}: {e_inner}")
                        total_delta_ll += -1.0 # Penalty for error
            
            # Risk Level Classification
            # delta_ll is usually negative.
            # -10 = very unlikely (Safe)
            # -1 = somewhat unlikely
            # -0.1 = very likely (Risk)
            
            # Thresholds (calibrated loosely)
            if total_delta_ll > -2.0:
                risk_level = "HIGH"
            elif total_delta_ll > -5.0:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"
                
            # Convert to [0,1] probability-like score for compatibility
            # exp(delta_ll) is a likelihood ratio
            risk_score = np.exp(total_delta_ll / 2.0) # Scaling factor
            risk_score = min(0.99, max(0.01, risk_score))
            
            return {
                'evo2_score': float(risk_score),
                'delta_ll': float(total_delta_ll),
                'mismatches': len(mismatches),
                'seed_mismatches': seed_mismatches,
                'risk_level': risk_level,
                'grna': grna_seq,
                'target': target_seq,
                'method': 'real_evo2_dll'
            }
            
        except Exception as e:
            print(f"Error scoring off-target: {e}")
            return {
                'evo2_score': 0.0,
                'delta_ll': -99.9,
                'mismatches': 0,
                'seed_mismatches': 0,
                'risk_level': "ERROR",
                'method': 'error'
            }

    
    @modal.method()
    def score_offtarget(self, grna_seq: str, target_seq: str, pam: str = "NGG") -> dict:
        """Exposed remote method."""
        return self._score_offtarget(grna_seq, target_seq, pam)
    
    @modal.method()
    def score_batch(self, offtargets: list, grna_seq: str) -> list:
        """Score a batch of off-targets for efficiency."""
        results = []
        for ot in offtargets:
            score = self._score_offtarget(grna_seq, ot['sequence'])
            score['locus'] = ot.get('locus', 'unknown')
            results.append(score)
        return results


# ===================================================================
# CONSTANTS
# ===================================================================

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

MYC_GRNA_LIBRARY = [
    {"id": "MYC_Ex2_1", "sequence": "GAGGGTCATTTCCCCTAGCG", "exon": 2, "pam": "CGG"},
    {"id": "MYC_Ex2_2", "sequence": "CCCTGTCCTTCTCACTCGCC", "exon": 2, "pam": "TGG"},
    {"id": "MYC_Ex2_3", "sequence": "GCTTCTCTGAAAGGCTCTCC", "exon": 2, "pam": "TGG"},
    {"id": "MYC_Ex3_1", "sequence": "GGCGAACACACAACGTCTTG", "exon": 3, "pam": "GAG"},
    {"id": "MYC_Ex3_2", "sequence": "CGTCTTGGAGCGCAGGATAG", "exon": 3, "pam": "GGG"},
    {"id": "MYC_Ex3_3", "sequence": "AAGCTAACGTTGAGGGGCAT", "exon": 3, "pam": "CGG"},
]


# ===================================================================
# SURVIVAL MODEL PIPELINE (T4)
# ===================================================================

@app.function(image=survival_image, gpu="T4", volumes={"/model": oral_cancer_volume}, timeout=1200)
def therapeutic_index_pipeline_real_evo2(patient_idx: int = 0, top_k_guides: int = 5) -> dict:
    """
    Complete Survival-Guided CRISPR Pipeline with REAL Evo2.
    
    This function:
    1. Loads patient data and survival model (T4 GPU)
    2. Calculates survival benefit from MYC knockdown
    3. Calls Evo2OffTargetScorer (H100) for each gRNA's off-targets
    4. Computes therapeutic index and ranks guides
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    import random
    from typing import List, Dict, Optional
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    TEMPERATURE = 10.0
    
    print("="*70)
    print("SURVIVAL-GUIDED CRISPR PIPELINE (REAL EVO2)")
    print("="*70)
    
    # Helper functions
    def find_gene_index(gene_list, target):
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
    
    def simulate_myc_knockdown(patient_expression, gene_names, 
                               knockdown_efficiency=0.9, cascade_strength=0.5):
        myc_idx = find_gene_index(gene_names, 'MYC')
        if myc_idx is None:
            raise ValueError("MYC not found!")
        
        target_indices = {}
        for category, genes in MYC_TARGETS.items():
            target_indices[category] = []
            for gene in genes:
                idx = find_gene_index(gene_names, gene)
                if idx is not None:
                    target_indices[category].append(idx)
        
        modified = patient_expression.copy()
        original_myc = modified[myc_idx]
        modified[myc_idx] *= (1 - knockdown_efficiency)
        
        cascade_effect = knockdown_efficiency * cascade_strength
        for category in ['direct', 'metabolic', 'ribosomal', 'cell_cycle']:
            for idx in target_indices.get(category, []):
                modified[idx] *= (1 - cascade_effect)
        
        for idx in target_indices.get('repressed', []):
            modified[idx] *= (1 + cascade_effect * 0.5)
        
        return {
            'modified_expression': modified,
            'original_myc': float(original_myc),
            'new_myc': float(modified[myc_idx])
        }
    
    def generate_mock_offtargets(grna_seq, n_sites=15):
        """
        Generate realistic mock off-targets (simulating Cas-OFFinder results).
        Creates a mix of high-fidelity sites and noise to test safety scoring.
        """
        offtargets = []
        import random

        # 1. On-target (perfect match) - always present
        offtargets.append({
            'locus': f"chr8:{random.randint(127000000, 129000000)}",  # MYC locus
            'sequence': grna_seq,
            'mismatches': 0,
            'is_ontarget': True
        })
        
        # 2. Realistic off-targets (1-3 mismatches, weighted toward more mismatches)
        # In reality, good guides have few 0-1 mismatch sites.
        # Most "potential" off-targets have 2-3 mismatches.
        for i in range(n_sites - 1):
            # Weight toward more mismatches (safer)
            # 1mm: 15%, 2mm: 45%, 3mm: 40%
            mismatches = random.choices([1, 2, 3], weights=[0.15, 0.45, 0.40])[0]
            
            target = list(grna_seq)
            positions = random.sample(range(len(grna_seq)), mismatches)
            for pos in positions:
                # Changes in seed region (last 12 bp) are biologically common for off-targets
                # but effectively reduce cutting (safer).
                target[pos] = random.choice([b for b in 'ACGT' if b != target[pos]])
            
            offtargets.append({
                'locus': f"chr{random.randint(1, 22)}:{random.randint(1000000, 200000000)}",
                'sequence': ''.join(target),
                'mismatches': mismatches,
                'is_ontarget': False
            })
        
        return offtargets
    
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
    threshold = np.percentile(myc_levels, 80)
    high_myc_indices = np.where(myc_levels > threshold)[0]
    
    if patient_idx >= len(high_myc_indices):
        patient_idx = 0
    selected_patient = high_myc_indices[patient_idx]
    patient_expression = X[selected_patient]
    
    print(f"Selected Patient #{selected_patient}")
    print(f"  MYC expression: {patient_expression[myc_idx]:.2f}")
    
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
                nn.Linear(prev_dim, 128), nn.ReLU(), nn.Dropout(dropout/2), nn.Linear(128, 1)
            )
            self.pathway_scorer = nn.Sequential(
                nn.Linear(prev_dim, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, n_pathways)
            )
        
        def forward(self, x):
            hidden = self.encoder(x)
            risk = self.risk_head(hidden)
            pathway_probs = F.softmax(self.pathway_scorer(hidden), dim=1)
            return risk, pathway_probs
    
    model = MYCFocusedPathwayMLP(len(gene_names), len(pathway_names)).to(device)
    model.load_state_dict(torch.load("/model/myc_enhanced_model.pt", map_location=device))
    model.eval()
    print("Loaded MYC survival model")
    
    # ============================================================
    # PHASE 1: SURVIVAL BENEFIT
    # ============================================================
    
    print("\n" + "="*70)
    print("PHASE 1: SURVIVAL BENEFIT CALCULATION")
    print("="*70)
    
    patient_tensor = torch.tensor(patient_expression, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        base_logit = model(patient_tensor)[0].item()
    
    simulation = simulate_myc_knockdown(patient_expression, gene_names)
    modified_tensor = torch.tensor(simulation['modified_expression'], dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        new_logit = model(modified_tensor)[0].item()
    
    survival_benefit = base_logit - new_logit
    
    print(f"\nBaseline Risk Logit: {base_logit:.2f}")
    print(f"Post-KD Risk Logit: {new_logit:.2f}")
    print(f"SURVIVAL BENEFIT: {survival_benefit:.3f} logit units")
    
    if survival_benefit < 0.1:
        return {"candidate": False, "reason": "Insufficient survival benefit"}
    
    # ============================================================
    # PHASE 2: EVO2 OFF-TARGET SCORING (REAL)
    # ============================================================
    
    print("\n" + "="*70)
    print("PHASE 2: REAL EVO2 OFF-TARGET SCORING (H100)")
    print("="*70)
    
    # Lookup Evo2 scorer from same app (runs on H100)
    # Important: Use Cls.from_name since this function runs on different image
    import modal as modal_lib
    print("Connecting to Evo2 scorer (H100)...")
    Evo2Scorer = modal_lib.Cls.from_name("survival-guided-crispr-evo2", "Evo2OffTargetScorer")
    
    guide_evaluations = []
    
    for grna in MYC_GRNA_LIBRARY:
        grna_seq = grna['sequence']
        print(f"\nScoring {grna['id']}...")
        
        # Generate off-targets (use Cas-OFFinder in production)
        offtargets = generate_mock_offtargets(grna_seq, n_sites=10)
        
        # Score with REAL Evo2 (batch call to H100)
        evo2_results = Evo2Scorer().score_batch.remote(offtargets, grna_seq)
        
        # Calculate composite safety
        scores = [r['evo2_score'] for r in evo2_results]
        max_risk = max(scores) if scores else 0.0
        avg_risk = np.mean(scores) if scores else 0.0
        composite_safety = (2.0 * max_risk + 1.0 * avg_risk) / 3.0
        
        # Therapeutic index
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
        
        print(f"  Safety: {composite_safety:.3f}, TI: {therapeutic_index:.1f} [{category}]")
        
        guide_evaluations.append({
            'guide_id': grna['id'],
            'sequence': grna_seq,
            'exon': grna['exon'],
            'survival_benefit': float(survival_benefit),
            'safety_score': float(composite_safety),
            'max_risk': float(max_risk),
            'avg_risk': float(avg_risk),
            'therapeutic_index': float(therapeutic_index),
            'category': category,
            'evo2_details': evo2_results[:3]  # Top 3 off-targets
        })
    
    # ============================================================
    # PHASE 3: RANK AND SELECT
    # ============================================================
    
    print("\n" + "="*70)
    print("PHASE 3: THERAPEUTIC INDEX RANKING")
    print("="*70)
    
    ranked = sorted(guide_evaluations, key=lambda x: x['therapeutic_index'], reverse=True)
    
    print(f"\n{'Rank':<6}{'Guide':<15}{'TI':>10}{'Safety':>10}{'Category':<12}")
    print("-"*55)
    for i, g in enumerate(ranked[:top_k_guides]):
        print(f"{i+1:<6}{g['guide_id']:<15}{g['therapeutic_index']:>10.1f}{g['safety_score']:>10.3f}{g['category']:<12}")
    
    top = ranked[0]
    if top['therapeutic_index'] > 50:
        recommendation = "Proceed with CRISPR therapy"
    elif top['therapeutic_index'] > 20:
        recommendation = "Proceed with monitoring"
    elif top['therapeutic_index'] > 5:
        recommendation = "Consider alternatives"
    else:
        recommendation = "Do not proceed"
    
    clinical_summary = (
        f"Therapeutic index: {top['therapeutic_index']:.1f}. "
        f"Survival benefit: {survival_benefit:.2f}. "
        f"Best guide: {top['guide_id']}. {recommendation}."
    )
    
    print(f"\nCLINICAL SUMMARY: {clinical_summary}")
    
    return {
        "patient_idx": int(selected_patient),
        "candidate": True,
        "survival_benefit": float(survival_benefit),
        "ranked_guides": ranked[:top_k_guides],
        "top_recommendation": top,
        "clinical_summary": clinical_summary
    }


@app.local_entrypoint()
def main():
    """Run the complete pipeline with real Evo2."""
    print("\n" + "="*70)
    print("SURVIVAL-GUIDED CRISPR PIPELINE (REAL EVO2 INTEGRATION)")
    print("="*70)
    print("This runs on:")
    print("  - Survival model: T4 GPU")
    print("  - Evo2 scoring: H100 GPU")
    print("="*70)
    
    result = therapeutic_index_pipeline_real_evo2.remote(patient_idx=0, top_k_guides=5)
    
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    
    if result['candidate']:
        print(f"\nPatient #{result['patient_idx']} is a CRISPR candidate")
        print(f"Survival Benefit: {result['survival_benefit']:.3f}")
        print(f"\nTop Guide: {result['top_recommendation']['guide_id']}")
        print(f"Therapeutic Index: {result['top_recommendation']['therapeutic_index']:.1f}")
        print(f"\n{result['clinical_summary']}")
    else:
        print(f"\nPatient not a candidate: {result.get('reason', 'Unknown')}")
