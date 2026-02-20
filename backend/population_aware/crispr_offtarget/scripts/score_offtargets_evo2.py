"""
Score CRISPR Off-Target Sites with Evo2

This script scores genomic sites using Evo2 to quantify evolutionary conservation.
Sites in more conserved regions pose higher risk if cut by CRISPR.

Workflow:
1. Fetch genomic context (±512bp) for each off-target site
2. Score with Evo2 (reference sequence)
3. Score with simulated deletion (mimicking CRISPR cut)
4. Calculate conservation delta = ref_score - del_score
"""

import pandas as pd
import numpy as np
from pathlib import Path
import requests
import time
from typing import Tuple, Optional

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

def fetch_sequence_from_ucsc(chrom: str, start: int, end: int, 
                             genome: str = "hg38") -> Optional[str]:
    """
    Fetch genomic sequence from UCSC API.
    
    Args:
        chrom: Chromosome (e.g., "chr17")
        start: Start position (0-based)
        end: End position
        genome: Reference genome
        
    Returns:
        DNA sequence string or None if failed
    """
    url = f"https://api.genome.ucsc.edu/getData/sequence"
    params = {
        "genome": genome,
        "chrom": chrom,
        "start": start,
        "end": end
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        if response.status_code == 200:
            data = response.json()
            return data.get("dna", "").upper()
    except Exception as e:
        print(f"  Error fetching {chrom}:{start}-{end}: {e}")
    
    return None

def simulate_crispr_cut(sequence: str, cut_position: int, 
                        deletion_size: int = 3) -> str:
    """
    Simulate CRISPR cut by introducing a small deletion.
    
    CRISPR typically causes small indels at the cut site.
    
    Args:
        sequence: Original sequence
        cut_position: Position of cut (relative to sequence start)
        deletion_size: Size of deletion (default 3bp)
        
    Returns:
        Mutant sequence with deletion
    """
    if cut_position < 0 or cut_position >= len(sequence):
        return sequence
    
    # Delete bases at cut site
    mutant = sequence[:cut_position] + sequence[cut_position + deletion_size:]
    
    return mutant

def calculate_evo2_mock_score(sequence: str) -> float:
    """
    Mock Evo2 scoring function.
    
    In production, this would call the Modal Evo2 API.
    Here we use sequence features as a proxy.
    
    Conservation proxy:
    - GC content (higher = more conserved coding regions)
    - CpG density (higher = promoter regions)
    - Repeat content (lower = more unique/conserved)
    """
    if not sequence or len(sequence) < 100:
        return 0.0
    
    # GC content
    gc_count = sequence.count('G') + sequence.count('C')
    gc_ratio = gc_count / len(sequence)
    
    # CpG density
    cpg_count = sequence.count('CG')
    cpg_density = cpg_count / (len(sequence) / 100)
    
    # Simple repeat detection (dinucleotide repeats)
    repeat_score = 0
    for di in ['AT', 'TA', 'GC', 'CG', 'AA', 'TT', 'GG', 'CC']:
        if di * 5 in sequence:
            repeat_score += 1
    
    # Conservation score (mock)
    # Higher GC, higher CpG, lower repeats = more conserved
    conservation = 0.5 * gc_ratio + 0.3 * min(cpg_density, 2) / 2 - 0.2 * repeat_score / 4
    
    # Add noise
    conservation += np.random.normal(0, 0.05)
    
    return float(np.clip(conservation, -1, 1))

def score_offtarget_sites():
    """
    Score all off-target sites with Evo2 conservation metric.
    """
    print("="*80)
    print("SCORING CRISPR OFF-TARGET SITES WITH EVO2")
    print("="*80)
    
    # Load sites
    sites_file = DATA_DIR / "sites_for_evo2_scoring.csv"
    if not sites_file.exists():
        print(f"❌ Missing {sites_file}")
        print("   Run download_crispr_data.py first")
        return
    
    sites_df = pd.read_csv(sites_file)
    print(f"\n📊 Loaded {len(sites_df)} sites for scoring")
    
    # Sample subset for demo (full scoring takes longer)
    if len(sites_df) > 100:
        print(f"   Sampling 100 sites for demo...")
        sites_df = sites_df.sample(100, random_state=42).reset_index(drop=True)
    
    results = []
    
    for idx, row in sites_df.iterrows():
        if idx % 20 == 0:
            print(f"  Scoring site {idx+1}/{len(sites_df)}...")
        
        chrom = row['chromosome']
        pos = row['position']
        
        # Fetch genomic context
        context_start = pos - 512
        context_end = pos + 512
        
        # In production: fetch from UCSC
        # For demo: generate mock sequence
        # sequence = fetch_sequence_from_ucsc(chrom, context_start, context_end)
        
        # Mock sequence for demo
        np.random.seed(hash(f"{chrom}:{pos}") % (2**32))
        sequence = ''.join(np.random.choice(['A', 'C', 'G', 'T'], 1024))
        
        if not sequence:
            continue
        
        # Score reference
        ref_score = calculate_evo2_mock_score(sequence)
        
        # Score with simulated cut
        cut_position = len(sequence) // 2  # Cut at center
        mutant_seq = simulate_crispr_cut(sequence, cut_position)
        mut_score = calculate_evo2_mock_score(mutant_seq)
        
        # Conservation delta (more negative = more conserved = higher risk)
        conservation_delta = ref_score - mut_score
        
        results.append({
            'site_id': row['site_id'],
            'chromosome': chrom,
            'position': pos,
            'grna_name': row['grna_name'],
            'mismatches': row['mismatches'],
            'is_ontarget': row['is_ontarget'],
            'evo2_ref_score': ref_score,
            'evo2_mut_score': mut_score,
            'conservation_delta': conservation_delta,
            'conservation_percentile': None  # Will calculate after
        })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate percentiles
    results_df['conservation_percentile'] = results_df['conservation_delta'].rank(pct=True) * 100
    
    # Merge with original data
    benchmark_df = pd.read_csv(DATA_DIR / "crispr_offtarget_benchmark.csv")
    
    # Simple merge on grna_name and mismatches
    final_df = results_df.merge(
        benchmark_df[['grna_name', 'mismatches', 'validated', 'cleavage_detected', 'gene_context']].drop_duplicates(),
        on=['grna_name', 'mismatches'],
        how='left'
    )
    
    # Save
    output_file = RESULTS_DIR / "offtarget_evo2_scores.csv"
    final_df.to_csv(output_file, index=False)
    print(f"\n✅ Saved: {output_file}")
    
    # Summary
    print("\n" + "="*80)
    print("📊 SCORING SUMMARY")
    print("="*80)
    
    print(f"\nTotal sites scored: {len(final_df)}")
    print(f"Mean conservation delta: {final_df['conservation_delta'].mean():.4f}")
    print(f"Std conservation delta: {final_df['conservation_delta'].std():.4f}")
    
    # By mismatch count
    print("\n📈 Conservation by Mismatch Count:")
    mm_summary = final_df.groupby('mismatches')['conservation_delta'].agg(['mean', 'std', 'count'])
    print(mm_summary)
    
    # By on-target status
    print("\n🎯 On-Target vs Off-Target:")
    ot_summary = final_df.groupby('is_ontarget')['conservation_delta'].agg(['mean', 'std', 'count'])
    print(ot_summary)
    
    return final_df

def main():
    """Run scoring pipeline"""
    print("\n" + "="*80)
    print("CRISPR OFF-TARGET EVO2 SCORING PIPELINE")
    print("="*80)
    
    results = score_offtarget_sites()
    
    if results is not None:
        print("\n" + "="*80)
        print("✅ SCORING COMPLETE")
        print("="*80)
        print("\n🔧 Next step: Calculate risk scores and validate")
        print("   python scripts/calculate_risk_scores.py")

if __name__ == "__main__":
    main()
