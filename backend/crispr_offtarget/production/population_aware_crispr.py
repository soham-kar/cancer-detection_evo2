"""
Population-Aware CRISPR Off-Target Scoring

Applies population-specific calibration to CRISPR off-target predictions.
Based on thesis findings: gnomAD PAM site variants affect cleavage efficiency.

Key Features:
1. Population-specific PAM penalties
2. gnomAD variant integration
3. Ancestry-aware risk stratification
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path

# Import config
from config import config, get_population_penalty


# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"


def query_gnomad_pam_variants(
    chrom: str,
    start: int,
    end: int,
    population: str
) -> pd.DataFrame:
    """
    Query gnomAD v4 for variants in PAM region.
    
    Adapted from thesis: population_thresholds.py
    
    Args:
        chrom: Chromosome (e.g., "chr17")
        start: Start position
        end: End position
        population: Population code (AFR, EUR, EAS, etc.)
        
    Returns:
        DataFrame with variants in region
    """
    # Mock implementation for demo
    # In production, connect to gnomAD API or local VCF
    
    # Generate deterministic mock data
    np.random.seed(hash(f"{chrom}{start}{end}{population}") % (2**32))
    
    n_variants = np.random.poisson(0.5)  # Average 0.5 variants per region
    
    if n_variants == 0:
        return pd.DataFrame(columns=['pos', 'ref', 'alt', 'af', 'consequence'])
    
    variants = []
    for i in range(n_variants):
        af_base = {
            'AFR': 0.08, 'AMR': 0.05, 'EAS': 0.03, 
            'EUR': 0.02, 'SAS': 0.04
        }.get(population.upper(), 0.03)
        
        variants.append({
            'pos': start + np.random.randint(0, end - start),
            'ref': np.random.choice(['A', 'C', 'G', 'T']),
            'alt': np.random.choice(['A', 'C', 'G', 'T']),
            'af': af_base * np.random.uniform(0.5, 2.0),
            'consequence': np.random.choice(
                ['PAM_disruption', 'noncoding', 'synonymous'],
                p=[0.3, 0.5, 0.2]
            )
        })
    
    return pd.DataFrame(variants)


def calculate_pam_disruption_penalty(variants: pd.DataFrame) -> float:
    """
    Calculate penalty for population-specific PAM variants.
    
    Higher penalty = more likely to disrupt CRISPR binding.
    
    Args:
        variants: DataFrame with PAM region variants
        
    Returns:
        Penalty score (0.0 to ~0.1)
    """
    if len(variants) == 0:
        return 0.0
    
    penalty = 0.0
    
    for _, var in variants.iterrows():
        af = var['af']
        consequence = var.get('consequence', 'unknown')
        
        # Weight by allele frequency and consequence
        if consequence == "PAM_disruption":
            penalty += af * 0.1  # High impact
        elif consequence == "noncoding":
            penalty += af * 0.01  # Low impact
        elif consequence == "synonymous":
            penalty += af * 0.005  # Minimal impact
    
    return min(penalty, 0.1)  # Cap at 0.1


def population_aware_score(
    target_seq: str,
    off_target_seq: str,
    mismatch_positions: List[int],
    population: str,
    chrom: Optional[str] = None,
    genomic_start: Optional[int] = None,
) -> Dict[str, float]:
    """
    Score off-target with population-specific calibration.
    
    Combines:
    1. Base Evo2 score
    2. Population-specific PAM penalty
    
    Args:
        target_seq: gRNA sequence
        off_target_seq: Off-target sequence
        mismatch_positions: List of mismatch positions
        population: Population code (AFR, EUR, EAS, etc.)
        chrom: Chromosome (optional, for PAM lookup)
        genomic_start: Genomic start position (optional)
        
    Returns:
        Dictionary with scores and population adjustments
    """
    # Import scorer
    from crispr_scorer import score_crispr_pair_mock
    
    # Get base Evo2 score
    base_score = score_crispr_pair_mock(
        target_seq, off_target_seq, mismatch_positions
    )
    
    # Get population penalty from config
    pop_config_penalty = get_population_penalty(population)
    
    # Get PAM-specific penalty (if genomic coordinates provided)
    pam_penalty = 0.0
    if chrom and genomic_start:
        pam_start = genomic_start + len(target_seq) - 3  # PAM at 3' end
        variants = query_gnomad_pam_variants(
            chrom, pam_start, pam_start + 3, population
        )
        pam_penalty = calculate_pam_disruption_penalty(variants)
    
    # Total population penalty
    total_pop_penalty = pop_config_penalty + pam_penalty
    
    # Final score
    final_score = base_score['weighted_delta_ll'] - total_pop_penalty
    
    return {
        **base_score,
        "population": population,
        "config_penalty": pop_config_penalty,
        "pam_penalty": pam_penalty,
        "total_population_penalty": total_pop_penalty,
        "final_score": float(final_score)
    }


def score_dataset_population_aware(
    df: pd.DataFrame,
    population: str,
    include_genomic: bool = False
) -> pd.DataFrame:
    """
    Score entire dataset with population-specific calibration.
    
    Args:
        df: DataFrame with off-target data
        population: Population code
        include_genomic: Whether to include genomic PAM lookup
        
    Returns:
        DataFrame with population-aware scores
    """
    print(f"\nScoring dataset for population: {population}")
    
    results = []
    
    for idx, row in df.iterrows():
        if idx % 100 == 0:
            print(f"  Processing {idx}/{len(df)}...")
        
        # Parse mismatch positions
        positions = row.get('mismatch_positions', [])
        if isinstance(positions, str):
            try:
                positions = eval(positions)
            except:
                positions = []
        
        # Get genomic coordinates if available
        chrom = row.get('chromosome', None) if include_genomic else None
        start = row.get('position', None) if include_genomic else None
        
        # Score
        score = population_aware_score(
            row['grna_sequence'],
            row['target_sequence'],
            positions,
            population,
            chrom,
            start
        )
        
        results.append({
            'grna_name': row.get('grna_name', ''),
            'target_sequence': row.get('target_sequence', ''),
            'population': population,
            'base_score': score['weighted_delta_ll'],
            'pop_penalty': score['total_population_penalty'],
            'final_score': score['final_score'],
            'confidence': score['confidence']
        })
    
    return pd.DataFrame(results)


def analyze_population_specificity(
    grna: str,
    target_locus: str,
    populations: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Analyze how gRNA score varies across populations.
    
    Replicates thesis: population_thresholds.py analysis pattern.
    
    Args:
        grna: gRNA sequence
        target_locus: Genomic locus (e.g., "chr13:27629200-27629220")
        populations: List of populations to analyze
        
    Returns:
        DataFrame with per-population scores
    """
    if populations is None:
        populations = list(config.POPULATION_PENALTIES.keys())
    
    # Parse genomic coordinates
    try:
        chrom, pos_str = target_locus.split(':')
        start, end = map(int, pos_str.split('-'))
    except:
        chrom, start, end = None, None, None
    
    results = []
    
    for pop in populations:
        # Get config penalty
        config_penalty = get_population_penalty(pop)
        
        # Get PAM variants
        if chrom and start:
            pam_start = start + 20  # After gRNA
            variants = query_gnomad_pam_variants(chrom, pam_start, pam_start + 3, pop)
            pam_penalty = calculate_pam_disruption_penalty(variants)
            n_variants = len(variants)
        else:
            pam_penalty = 0.0
            n_variants = 0
        
        results.append({
            "population": pop,
            "config_penalty": config_penalty,
            "pam_penalty": pam_penalty,
            "total_penalty": config_penalty + pam_penalty,
            "n_pam_variants": n_variants
        })
    
    return pd.DataFrame(results)


def generate_population_figure(
    df_population: pd.DataFrame,
    grna: str,
    output_path: Optional[str] = None
):
    """
    Generate population-specific scoring figure.
    
    Args:
        df_population: DataFrame from analyze_population_specificity
        grna: gRNA sequence (for title)
        output_path: Output file path
    """
    import matplotlib.pyplot as plt
    
    if output_path is None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = RESULTS_DIR / "figures" / "figure3_population.png"
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(df_population)))
    
    bars = ax.bar(
        df_population['population'],
        df_population['total_penalty'],
        color=colors,
        edgecolor='black',
        alpha=0.8
    )
    
    ax.set_ylabel('Population-Specific Penalty', fontsize=12, fontweight='bold')
    ax.set_xlabel('Population', fontsize=12, fontweight='bold')
    ax.set_title(f'gRNA Score Variation Across Populations\n{grna[:20]}...', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, df_population['total_penalty']):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {output_path}")


def main():
    """Test population-aware scoring"""
    print("="*60)
    print("POPULATION-AWARE CRISPR SCORING")
    print("="*60)
    
    # Test single scoring
    target = "GAGTCCGAGCAGAAGAAGAAGG"
    off_target = "GAGTCCGAGCAGAAGtAGAAGG"
    mismatches = [15]
    
    print("\n1. Single pair scoring:")
    for pop in ['AFR', 'EUR', 'EAS']:
        result = population_aware_score(
            target, off_target, mismatches, pop,
            chrom="chr13", genomic_start=27629200
        )
        print(f"\n   {pop}:")
        print(f"     Base score: {result['weighted_delta_ll']:.4f}")
        print(f"     Pop penalty: {result['total_population_penalty']:.4f}")
        print(f"     Final score: {result['final_score']:.4f}")
    
    # Analyze population specificity
    print("\n2. Population specificity analysis:")
    grna = "GAGTCCGAGCAGAAGAAGAA"
    pop_analysis = analyze_population_specificity(
        grna, "chr13:27629200-27629220"
    )
    print(pop_analysis)
    
    # Generate figure
    print("\n3. Generating population figure...")
    generate_population_figure(pop_analysis, grna)
    
    print("\n✅ Population-aware scoring complete!")


if __name__ == "__main__":
    main()
