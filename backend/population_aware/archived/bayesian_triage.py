"""
Hybrid Bayesian Uncertainty Quantification for Genomic Foundation Models

Integrates Evo2 AI scores with population allele frequency priors using
Bayesian inference to:
1. Compute posterior pathogenicity probabilities
2. Quantify epistemic uncertainty
3. Identify high-confidence vs. uncertain predictions

Gap 2C Implementation: Bayesian Triage System
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ==============================================================================
# CONFIGURATION
# ==============================================================================
INPUT_FILE = "results/brca1_global_calibration.csv"
OUTPUT_DIR = Path("results/bayesian_triage")

# Prior probabilities based on disease prevalence
# BRCA1 pathogenic variants are rare in general population (~1 in 1000)
PRIOR_PATHOGENIC_BASELINE = 0.001

def likelihood_pathogenic(score):
    """
    P(Score | Pathogenic)
    
    Based on observed distribution of pathogenic variants.
    More negative scores are more likely for pathogenic variants.
    """
    return stats.norm.pdf(score, loc=-0.015, scale=0.08)

def likelihood_benign(score):
    """
    P(Score | Benign)
    
    Based on observed distribution of benign variants.
    Scores closer to zero are more likely for benign variants.
    """
    return stats.norm.pdf(score, loc=-0.005, scale=0.04)

def calculate_population_prior(row):
    """
    Calculate prior P(Pathogenic) from population allele frequencies.
    
    Key principle: Common variants are almost certainly benign.
    Rare variants have higher prior probability of being pathogenic.
    
    Returns:
        float: Prior probability that variant is pathogenic (0-1)
    """
    # Get maximum AF across all populations (conservative)
    af_cols = ['af_afr', 'af_nfe', 'af_sas', 'af_eas']
    afs = [row.get(col, 0) for col in af_cols]
    afs = [af if pd.notna(af) else 0 for af in afs]
    max_af = max(afs) if afs else 0
    
    # Apply frequency-based prior adjustment
    if max_af > 0.005:  # >0.5% frequency - very common
        return 0.00001  # Almost certainly benign
    elif max_af > 0.001:  # 0.1-0.5% - common
        return 0.0001
    elif max_af > 0.0001:  # 0.01-0.1% - uncommon
        return 0.001
    elif max_af == 0:  # Absent from gnomAD - very rare
        return 0.01  # Higher suspicion for unseen variants
    else:
        return PRIOR_PATHOGENIC_BASELINE

def compute_bayesian_posterior(score, prior_path):
    """
    Apply Bayes' Theorem to compute posterior probability.
    
    P(Pathogenic | Score, AF) = [P(Score | Path) × P(Path | AF)] / P(Score)
    
    Args:
        score: Evo2 score
        prior_path: Prior P(Pathogenic) from population data
        
    Returns:
        tuple: (posterior_pathogenic, uncertainty_entropy)
    """
    # Prior probabilities
    prior_benign = 1 - prior_path
    
    # Likelihoods
    lik_path = likelihood_pathogenic(score)
    lik_benign = likelihood_benign(score)
    
    # Posterior computation (Bayes' Theorem)
    numerator_path = lik_path * prior_path
    numerator_benign = lik_benign * prior_benign
    evidence = numerator_path + numerator_benign
    
    if evidence == 0:
        posterior_path = 0
    else:
        posterior_path = numerator_path / evidence
    
    # Uncertainty quantification (Shannon Entropy)
    # H = -p*log(p) - (1-p)*log(1-p)
    # Range: 0 (certain) to 0.693 (maximum uncertainty at p=0.5)
    p = posterior_path
    if p == 0 or p == 1:
        entropy = 0
    else:
        entropy = -(p * np.log(p) + (1-p) * np.log(1-p))
    
    return posterior_path, entropy

def classify_triage(posterior, entropy):
    """
    Classify variants into triage categories based on posterior and uncertainty.
    
    Args:
        posterior: Posterior probability of being pathogenic
        entropy: Uncertainty (Shannon entropy)
        
    Returns:
        str: Triage category
    """
    CONFIDENCE_THRESHOLD = 0.3  # Entropy threshold for "high confidence"
    
    if entropy < CONFIDENCE_THRESHOLD:
        if posterior > 0.90:
            return 'High Conf Pathogenic'
        elif posterior < 0.10:
            return 'High Conf Benign'
        else:
            return 'Confident VUS'
    else:
        return 'Uncertain'

def create_confidence_tent_plot(df, output_dir):
    """
    Create the "Confidence Tent" visualization.
    
    Shows relationship between Evo2 score, uncertainty, and triage classification.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Color scheme
    colors = {
        'High Conf Pathogenic': '#e74c3c',
        'High Conf Benign': '#2ecc71',
        'Confident VUS': '#f39c12',
        'Uncertain': '#95a5a6'
    }
    
    # Scatter plot
    for category in colors.keys():
        subset = df[df['triage_class'] == category]
        ax.scatter(subset['evo2_score'], subset['uncertainty_entropy'],
                   c=colors[category], label=category, 
                   alpha=0.6, s=30, edgecolors='black', linewidths=0.3)
    
    # Confidence threshold line
    ax.axhline(y=0.3, color='black', linestyle='--', linewidth=2, 
               alpha=0.7, label='Confidence Threshold (H=0.3)')
    
    # Formatting
    ax.set_xlabel('Evo2 Score (More Negative = More Pathogenic)', 
                  fontsize=12, fontweight='bold')
    ax.set_ylabel('Epistemic Uncertainty (Shannon Entropy)', 
                  fontsize=12, fontweight='bold')
    ax.set_title('Bayesian Uncertainty Quantification: The "Confidence Tent"\n'
                 'Population-Aware AI with Uncertainty Estimation', 
                 fontsize=14, fontweight='bold', pad=15)
    
    ax.legend(loc='upper right', frameon=True, fontsize=10, shadow=True)
    ax.grid(True, alpha=0.3)
    
    # Add annotations
    ax.text(0.02, 0.98, 'High Uncertainty →', 
            transform=ax.transAxes, ha='left', va='top',
            fontsize=10, style='italic', color='gray')
    ax.text(0.02, 0.02, 'High Confidence →', 
            transform=ax.transAxes, ha='left', va='bottom',
            fontsize=10, style='italic', color='gray')
    
    plt.tight_layout()
    
    output_path = output_dir / "confidence_tent_plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()

def create_posterior_distribution_plot(df, output_dir):
    """
    Show distribution of posterior probabilities.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Posterior probability histogram
    ax = axes[0]
    ax.hist(df['bayesian_prob_pathogenic'], bins=30, 
            color='#3498db', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.axvline(0.5, color='red', linestyle='--', linewidth=2, 
               label='Decision Boundary (0.5)')
    ax.set_xlabel('Posterior P(Pathogenic)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Number of Variants', fontsize=11, fontweight='bold')
    ax.set_title('Distribution of Bayesian Posterior Probabilities', 
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Uncertainty distribution
    ax = axes[1]
    ax.hist(df['uncertainty_entropy'], bins=30, 
            color='#e74c3c', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.axvline(0.3, color='orange', linestyle='--', linewidth=2,
               label='Confidence Threshold')
    ax.set_xlabel('Epistemic Uncertainty (Entropy)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Number of Variants', fontsize=11, fontweight='bold')
    ax.set_title('Distribution of Uncertainty Scores', 
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_path = output_dir / "posterior_distributions.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()

def main():
    print("="*80)
    print("BAYESIAN UNCERTAINTY QUANTIFICATION - GAP 2C")
    print("="*80)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 Output directory: {OUTPUT_DIR}")
    
    # Load calibrated data
    print(f"\n📊 Loading calibrated data from: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    print(f"   Loaded {len(df)} variants")
    
    # Apply Bayesian inference
    print("\n⚙️  Computing Bayesian posteriors and uncertainties...")
    
    posteriors = []
    uncertainties = []
    
    for idx, row in df.iterrows():
        score = row['evo2_score']
        prior_path = calculate_population_prior(row)
        posterior, entropy = compute_bayesian_posterior(score, prior_path)
        
        posteriors.append(posterior)
        uncertainties.append(entropy)
    
    df['bayesian_prob_pathogenic'] = posteriors
    df['uncertainty_entropy'] = uncertainties
    
    print(f"   ✅ Computed {len(posteriors)} posterior probabilities")
    
    # Classify into triage categories
    print("\n🔧 Classifying variants into triage categories...")
    df['triage_class'] = df.apply(
        lambda row: classify_triage(row['bayesian_prob_pathogenic'], 
                                     row['uncertainty_entropy']),
        axis=1
    )
    
    # Save results
    output_file = OUTPUT_DIR / "brca1_bayesian_triage.csv"
    df.to_csv(output_file, index=False)
    print(f"✅ Saved triage results to: {output_file}")
    
    # Generate visualizations
    print("\n📊 Creating visualizations...")
    create_confidence_tent_plot(df, OUTPUT_DIR)
    create_posterior_distribution_plot(df, OUTPUT_DIR)
    
    # Summary statistics
    print("\n" + "="*80)
    print("🏆 BAYESIAN TRIAGE SUMMARY")
    print("="*80)
    
    triage_counts = df['triage_class'].value_counts()
    print("\n📊 Variant Classification:")
    for category, count in triage_counts.items():
        pct = count / len(df) * 100
        print(f"   {category:<25} {count:>5} ({pct:>5.1f}%)")
    
    # Additional statistics
    print(f"\n📈 Posterior Probability Statistics:")
    print(f"   Mean: {df['bayesian_prob_pathogenic'].mean():.4f}")
    print(f"   Median: {df['bayesian_prob_pathogenic'].median():.4f}")
    print(f"   High confidence (>0.9 or <0.1): {((df['bayesian_prob_pathogenic'] > 0.9) | (df['bayesian_prob_pathogenic'] < 0.1)).sum()}")
    
    print(f"\n📉 Uncertainty Statistics:")
    print(f"   Mean entropy: {df['uncertainty_entropy'].mean():.4f}")
    print(f"   High confidence (H<0.3): {(df['uncertainty_entropy'] < 0.3).sum()}")
    print(f"   Uncertain (H>0.3): {(df['uncertainty_entropy'] > 0.3).sum()}")
    
    # Identify rescue cases
    print(f"\n🚀 'Rescue' Cases (AI correction via Bayesian inference):")
    # Cases where raw score suggests pathogenic but posterior corrects to benign
    rescue_cases = df[
        (df['evo2_score'] < -0.01) &  # Raw score pathogenic
        (df['bayesian_prob_pathogenic'] < 0.1) &  # Posterior says benign
        (df['uncertainty_entropy'] < 0.3)  # High confidence
    ]
    print(f"   Found {len(rescue_cases)} rescue cases")
    
    if len(rescue_cases) > 0:
        print("\n   Top 3 rescue examples:")
        for idx, row in rescue_cases.head(3).iterrows():
            variant_id = f"{row.get('chrom', 'N/A')}:{row.get('pos_hg38', 'N/A')}{row.get('ref', 'N/A')}>{row.get('alt', 'N/A')}"
            print(f"     {variant_id}")
            print(f"       Raw Score: {row['evo2_score']:.6f}")
            print(f"       Posterior P(Path): {row['bayesian_prob_pathogenic']:.6f}")
            print(f"       Uncertainty: {row['uncertainty_entropy']:.3f}")
    
    print("\n" + "="*80)
    print("✅ BAYESIAN ANALYSIS COMPLETE!")
    print("="*80)
    
    print("\n💡 Key Outputs:")
    print(f"   1. {output_file.name} - Full triage results")
    print(f"   2. confidence_tent_plot.png - Main visualization")
    print(f"   3. posterior_distributions.png - Statistical distributions")

if __name__ == "__main__":
    main()
