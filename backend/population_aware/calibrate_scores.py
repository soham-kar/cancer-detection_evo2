"""
Population-Aware Score Calibration

Combines raw Evo2 scores with population-specific allele frequencies
to create calibrated scores that reduce population bias.

Methodology:
- Uses gnomAD allele frequencies (af_afr, af_eur/af_nfe)
- Applies population-specific threshold adjustment
- Generates calibrated scores that account for population variation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_population_bias_score(af_afr, af_eur):
    """
    Calculate population bias indicator.
    
    High bias = variant common in one population but rare in another
    Low bias = similar frequency across populations
    
    Returns:
        float: Bias score (0 = no bias, 1 = maximum bias)
    """
    # Handle missing values
    af_afr = af_afr if pd.notna(af_afr) else 0
    af_eur = af_eur if pd.notna(af_eur) else 0
    
    # Avoid division by zero
    if af_afr == 0 and af_eur == 0:
        return 0
    
    # Calculate fold-difference
    max_af = max(af_afr, af_eur)
    min_af = min(af_afr, af_eur) + 1e-10  # Add small constant
    
    fold_diff = max_af / min_af
    
    # Normalize to 0-1 scale (log scale)
    # fold_diff = 1  --> bias = 0
    # fold_diff = 10 --> bias = 0.5
    # fold_diff = 100 --> bias = 0.75
    bias_score = np.log10(fold_diff) / (np.log10(fold_diff) + 2)
    
    return bias_score

def calibrate_score(evo2_score, af_afr, af_eur, population='neutral'):
    """
    Calibrate Evo2 score based on population allele frequencies.
    
    Args:
        evo2_score: Raw Evo2 delta log-likelihood
        af_afr: African allele frequency
        af_eur: European allele frequency  
        population: Target population ('afr', 'eur', or 'neutral')
    
    Returns:
        float: Calibrated score
    """
    # Calculate bias
    bias = calculate_population_bias_score(af_afr, af_eur)
    
    # If no bias or no AF data, return raw score
    if bias < 0.1 or (pd.isna(af_afr) and pd.isna(af_eur)):
        return evo2_score
    
    # Population-specific adjustment
    if population == 'afr':
        # For African populations: downweight if rare in AFR but common in EUR
        if af_afr < af_eur:
            adjustment = -bias * 0.3  # Reduce severity if AFR-rare
        else:
            adjustment = 0
    elif population == 'eur':
        # For European populations: downweight if rare in EUR but common in AFR
        if af_eur < af_afr:
            adjustment = -bias * 0.3  # Reduce severity if EUR-rare
        else:
            adjustment = 0
    else:  # neutral
        # For population-neutral: no adjustment (report raw score)
        adjustment = 0
    
    calibrated = evo2_score + adjustment
    
    return calibrated

def add_calibrated_scores(df, gene_name='BRCA1'):
    """
    Add calibrated scores to dataframe.
    
    Args:
        df: DataFrame with evo2_score, af_afr, af_nfe columns
        gene_name: Gene name for metadata
    
    Returns:
        DataFrame with added calibration columns
    """
    df = df.copy()
    
    # Rename af_nfe to af_eur for clarity
    if 'af_nfe' in df.columns and 'af_eur' not in df.columns:
       df['af_eur'] = df['af_nfe']
    
    # Calculate population bias score
    df['pop_bias'] = df.apply(
        lambda row: calculate_population_bias_score(
            row.get('af_afr', 0), 
            row.get('af_eur', 0)
        ), 
        axis=1
    )
    
    # Calculate calibrated scores for each population
    df['calibrated_afr'] = df.apply(
        lambda row: calibrate_score(
            row['evo2_score'], 
            row.get('af_afr', 0), 
            row.get('af_eur', 0), 
            population='afr'
        ),
        axis=1
    )
    
    df['calibrated_eur'] = df.apply(
        lambda row: calibrate_score(
            row['evo2_score'], 
            row.get('af_afr', 0), 
            row.get('af_eur', 0), 
            population='eur'
        ),
        axis=1
    )
    
    df['calibrated_neutral'] = df.apply(
        lambda row: calibrate_score(
            row['evo2_score'], 
            row.get('af_afr', 0), 
            row.get('af_eur', 0), 
            population='neutral'
        ),
        axis=1
    )
    
    return df

def visualize_calibration(df, output_path='results/calibration_comparison.png'):
    """
    Create visualization comparing raw vs calibrated scores.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Score distributions before/after calibration
    ax = axes[0, 0]
    ax.hist(df['evo2_score'], bins=50, alpha=0.5, label='Raw Evo2', color='blue')
    ax.hist(df['calibrated_afr'], bins=50, alpha=0.5, label='Calibrated (AFR)', color='red')
    ax.hist(df['calibrated_eur'], bins=50, alpha=0.5, label='Calibrated (EUR)', color='green')
    ax.set_xlabel('Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Score Distributions: Raw vs Calibrated')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Population bias distribution
    ax = axes[0, 1]
    ax.hist(df['pop_bias'], bins=50, color='purple', alpha=0.7)
    ax.set_xlabel('Population Bias Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Population Bias Distribution')
    ax.axvline(0.3, color='red', linestyle='--', label='High bias threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Calibration effect (scatter)
    ax = axes[1, 0]
    high_bias = df[df['pop_bias'] > 0.3]
    ax.scatter(df['evo2_score'], df['calibrated_afr'], alpha=0.3, s=10, label='All variants')
    ax.scatter(high_bias['evo2_score'], high_bias['calibrated_afr'], 
               alpha=0.8, s=20, color='red', label='High bias variants')
    ax.plot([-0.05, 0], [-0.05, 0], 'k--', alpha=0.5, label='No change')
    ax.set_xlabel('Raw Evo2 Score')
    ax.set_ylabel('Calibrated Score (AFR)')
    ax.set_title('Calibration Effect on High-Bias Variants')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. AF correlation
    ax = axes[1, 1]
    valid = df[(df['af_afr'] > 0) & (df['af_eur'] > 0)]
    ax.scatter(valid['af_eur'], valid['af_afr'], alpha=0.5, s=10)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.plot([1e-6, 1], [1e-6, 1], 'k--', alpha=0.5, label='Equal frequency')
    ax.set_xlabel('EUR Allele Frequency')
    ax.set_ylabel('AFR Allele Frequency')
    ax.set_title('Population Allele Frequency Correlation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Calibration visualization saved to {output_path}")
    plt.close()

def main():
    print("="*70)
    print("POPULATION-AWARE SCORE CALIBRATION")
    print("="*70)
    
    # Load BRCA1 Evo2 scores
    evo2_file = "results/brca1_evo2_scores_REAL.csv"
    gnomad_file = "results/brca1_with_gnomad.csv"
    
    try:
        df_evo2 = pd.read_csv(evo2_file)
        print(f"\n📊 Loaded {len(df_evo2)} BRCA1 Evo2 scores")
        
        df_gnomad = pd.read_csv(gnomad_file)
        print(f"📊 Loaded {len(df_gnomad)} BRCA1 gnomAD variants")
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        return
    
    # Merge on variant key
    print("\n🔗 Merging Evo2 scores with gnomAD frequencies...")
    df = pd.merge(
        df_evo2,
        df_gnomad[['chrom', 'pos_hg38', 'ref', 'alt', 'af_nfe', 'af_afr', 'af_eas', 'af_sas', 'af_amr']],
        on=['chrom', 'pos_hg38', 'ref', 'alt'],
        how='left'
    )
    
    print(f"   Merged: {len(df)} variants")
    print(f"   With AF data: {df['af_nfe'].notna().sum()}")
    
    # Add calibrated scores
    print("\n🔧 Calculating calibrated scores...")
    df_calibrated = add_calibrated_scores(df)
    
    # Save calibrated results
    output_file = "results/brca1_evo2_calibrated.csv"
    df_calibrated.to_csv(output_file, index=False)
    print(f"✅ Calibrated scores saved to {output_file}")
    
    # Summary statistics
    print("\n📈 Calibration Summary:")
    print(f"   Total variants: {len(df_calibrated)}")
    with_af = df_calibrated[df_calibrated['af_nfe'].notna() | df_calibrated['af_afr'].notna()]
    print(f"   With AF data: {len(with_af)}")
    
    if len(with_af) > 0:
        high_bias = (with_af['pop_bias'] > 0.3).sum()
        print(f"   High bias variants (>0.3): {high_bias}")
        print(f"   Mean bias score: {with_af['pop_bias'].mean():.3f}")
        
        # Score changes
        score_change_afr = (with_af['calibrated_afr'] - with_af['evo2_score']).abs()
        score_change_eur = (with_af['calibrated_eur'] - with_af['evo2_score']).abs()
        
        print(f"\n📊 Score Adjustments:")
        print(f"   AFR calibration: mean change = {score_change_afr.mean():.6f}")
        print(f"   EUR calibration: mean change = {score_change_eur.mean():.6f}")
        print(f"   Variants adjusted (AFR): {(score_change_afr > 0.001).sum()}")
        print(f"   Variants adjusted (EUR): {(score_change_eur > 0.001).sum()}")
        
        # Create visualization
        print("\n📊 Creating calibration visualizations...")
        visualize_calibration(with_af)
    else:
        print("⚠️ No variants with AF data for visualization")
    
    print("\n" + "="*70)
    print("✅ CALIBRATION COMPLETE!")
    print("="*70)

if __name__ == "__main__":
    main()
