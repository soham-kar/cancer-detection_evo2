"""
Enhanced Bayesian Uncertainty Framework with Monte Carlo Sampling

Production-ready implementation incorporating:
- Beta distribution priors from allele frequencies  
- Sigmoid transformation of Evo2 scores to likelihoods
- Monte Carlo sampling for 95% credible intervals
- VUS flagging based on uncertainty width
- Clinical decision support outputs

Gap 2C: Hybrid Bayesian Framework
"""

import pandas as pd
import numpy as np
from scipy.stats import beta, norm
from scipy.special import expit  # sigmoid
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

# ==============================================================================
# BAYESIAN FRAMEWORK CONFIGURATION
# ==============================================================================

class BayesianConfig:
    """Hyperparameters for Bayesian framework"""
    
    # Evo2 score transformation
    EVO2_SCALE = 100  # Scaling factor for sigmoid (k parameter)
    
    # Disease prevalence prior
    BRCA1_PREVALENCE = 0.0025  # ~1 in 400 carry pathogenic BRCA1 variant
    
    # gnomAD parameters
    GNOMAD_SAMPLE_SIZE = 125748  # v3.1.2 exome samples
    
    # Uncertainty thresholds
    CONFIDENCE_THRESHOLD = 0.3  # CI width threshold for "high confidence"
    VUS_POSTERIOR_RANGE = (0.45, 0.55)  # Posterior range defining VUS
    
    # Monte Carlo sampling
    N_MC_SAMPLES = 1000  # Number of samples for credible interval

# ==============================================================================
# CORE BAYESIAN FUNCTIONS
# ==============================================================================

def evo2_to_likelihood(evo2_score: float, k: float = BayesianConfig.EVO2_SCALE) -> float:
    """
    Convert Evo2 delta score to likelihood P(Score | Pathogenic).
    
    Uses sigmoid transformation: L = 1 / (1 + exp(k * score))
    
    Rationale:
    - Pathogenic variants have negative scores (delta < 0)
    - More negative → Higher likelihood of pathogenicity
    - Sigmoid provides smooth, interpretable mapping to [0,1]
    
    Args:
        evo2_score: Raw Evo2 delta score (typically -0.2 to +0.1)
        k: Scaling factor (higher = sharper transitions)
        
    Returns:
        float: Likelihood in [0, 1]
    """
    return float(expit(-k * evo2_score))  # Negative because lower scores = more pathogenic

def allele_frequency_prior(af: float, 
                           sample_size: int = BayesianConfig.GNOMAD_SAMPLE_SIZE) -> float:
    """
    Calculate prior P(Pathogenic | AF) using Beta distribution.
    
    Beta(α=0.5, β=N*AF) models uncertainty in allele frequency estimation.
    
    Rationale:
    - Common variants (high AF) are almost always benign
    - Rare variants (low AF) have higher prior for pathogenicity
    - Zero AF (not in gnomAD) suggests rarity but uncertainty
    
    Args:
        af: Allele frequency (0-1)
        sample_size: gnomAD sample size for Beta parameters
        
    Returns:
        float: Prior probability that variant is pathogenic
    """
    if af == 0:
        # Not observed in gnomAD - very rare, but could be artifact
        return 0.8
    
    if af > 0.01:
        # Common variants (>1%) are almost certainly benign
        return 0.00001
    
    # Beta distribution: probability that true AF < 0.1% (pathogenic threshold)
    alpha = 0.5  # Jeffrey's prior
    beta_param = sample_size * af
    
    # P(true AF < 0.001 | observed AF)
    prior = beta.cdf(0.001, alpha, beta_param)
    
    return float(prior)

def compute_posterior(likelihood: float, 
                     prior: float, 
                     prevalence: float = BayesianConfig.BRCA1_PREVALENCE) -> float:
    """
    Apply Bayes' Theorem to compute posterior probability.
    
    P(Path | Score, AF) = [L(Score|Path) × P(Path|AF) × P(Path)] / Evidence
    
    Args:
        likelihood: P(Score | Pathogenic) from Evo2
        prior: P(Pathogenic | AF) from population data
        prevalence: Base rate P(Pathogenic) in population
        
    Returns:
        float: Posterior P(Pathogenic | Score, AF)
    """
    # Bayesian update
    posterior_path = likelihood * prior * prevalence
    posterior_benign = (1 - likelihood) * (1 - prior) * (1 - prevalence)
    
    evidence = posterior_path + posterior_benign
    
    if evidence == 0:
        return 0.5  # Maximum uncertainty
    
    return float(posterior_path / evidence)

def monte_carlo_credible_interval(evo2_score: float,
                                  af_values: list,
                                  n_samples: int = BayesianConfig.N_MC_SAMPLES) -> tuple:
    """
    Compute 95% credible interval using Monte Carlo sampling.
    
    Accounts for uncertainty in allele frequency estimates by sampling
    from Beta posteriors and recomputing Bayesian inference.
    
    Args:
        evo2_score: Evo2 delta score (fixed)
        af_values: List of AFs from different populations
        n_samples: Number of Monte Carlo samples
        
    Returns:
        tuple: (ci_lower, ci_upper, posterior_samples)
    """
    likelihood = evo2_to_likelihood(evo2_score)
    posterior_samples = []
    
    for _ in range(n_samples):
        # Sample AF from Beta distribution for each population
        sampled_priors = []
        
        for af in af_values:
            if af > 0:
                # Sample from Beta(α=0.5, β=N*AF)
                alpha = 0.5
                beta_param = BayesianConfig.GNOMAD_SAMPLE_SIZE * af
                sampled_af = beta.rvs(alpha, beta_param)
            else:
                sampled_af = 0
            
            sampled_priors.append(allele_frequency_prior(sampled_af))
        
        # Average prior across populations
        mean_prior = np.mean(sampled_priors)
        
        # Compute posterior with sampled prior
        posterior = compute_posterior(likelihood, mean_prior)
        posterior_samples.append(posterior)
    
    # Compute 95% credible interval
    ci_lower, ci_upper = np.percentile(posterior_samples, [2.5, 97.5])
    
    return ci_lower, ci_upper, posterior_samples

def classify_variant(posterior: float, 
                    ci_width: float,
                    likelihood: float) -> dict:
    """
    Classify variant and provide clinical recommendation.
    
    Args:
        posterior: Posterior probability of pathogenicity
        ci_width: Width of 95% credible interval
        likelihood: Evo2 likelihood score
        
    Returns:
        dict: Classification and recommendation
    """
    is_vus = (
        (BayesianConfig.VUS_POSTERIOR_RANGE[0] <= posterior <= BayesianConfig.VUS_POSTERIOR_RANGE[1]) or
        (ci_width > BayesianConfig.CONFIDENCE_THRESHOLD) or
        (0.4 < likelihood < 0.6)
    )
    
    if is_vus:
        classification = 'VUS'
        recommendation = 'REVIEW - Requires expert interpretation'
    elif posterior > 0.8:
        classification = 'Likely Pathogenic'
        recommendation = 'HIGH CONFIDENCE - Consider clinical action'
    elif posterior < 0.2:
        classification = 'Likely Benign'
        recommendation = 'HIGH CONFIDENCE - Low clinical concern'
    else:
        classification = 'Uncertain'
        recommendation = 'MODERATE CONFIDENCE - Monitor or retest'
    
    return {
        'classification': classification,
        'recommendation': recommendation,
        'is_vus': is_vus,
        'confidence': 'HIGH' if ci_width < 0.15 else ('MODERATE' if ci_width < 0.3 else 'LOW')
    }

# ==============================================================================
# BATCH PROCESSING
# ==============================================================================

def process_variants_bayesian(input_file: str, output_dir: Path):
    """
    Apply Bayesian framework to all variants in dataset.
    
    Args:
        input_file: Path to CSV with Evo2 scores and gnomAD AFs
        output_dir: Directory for outputs
    """
    print("="*80)
    print("ENHANCED BAYESIAN UNCERTAINTY QUANTIFICATION")
    print("="*80)
    
    # Load data
    print(f"\n📊 Loading data from: {input_file}")
    df = pd.read_csv(input_file)
    print(f"   Loaded {len(df)} variants")
    
    # Verify required columns
    required_cols = ['evo2_score']
    af_cols = ['af_afr', 'af_nfe', 'af_sas', 'af_eas']
    
    missing = [col for col in required_cols + af_cols if col not in df.columns]
    if missing:
        print(f"⚠️  Warning: Missing columns {missing}")
        print("   Will proceed with available data")
    
    # Process each variant
    print("\n⚙️  Computing Bayesian posteriors with Monte Carlo sampling...")
    
    results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing variants"):
        evo2_score = row['evo2_score']
        
        # Get allele frequencies
        af_values = []
        for col in af_cols:
            if col in df.columns:
                af = row.get(col, 0)
                af_values.append(af if pd.notna(af) else 0)
        
        if not af_values:
            af_values = [0]  # Default if no AF data
        
        # Compute likelihood
        likelihood = evo2_to_likelihood(evo2_score)
        
        # Compute prior (average across populations)
        priors = [allele_frequency_prior(af) for af in af_values]
        mean_prior = np.mean(priors)
        
        # Compute posterior
        posterior = compute_posterior(likelihood, mean_prior)
        
        # Monte Carlo credible interval
        ci_lower, ci_upper, samples = monte_carlo_credible_interval(
            evo2_score, af_values
        )
        ci_width = ci_upper - ci_lower
        
        # Classify
        classification = classify_variant(posterior, ci_width, likelihood)
        
        # Store results
        results.append({
            **row.to_dict(),
            'likelihood': likelihood,
            'prior': mean_prior,
            'posterior': posterior,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'ci_width': ci_width,
            **classification
        })
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # Save
    output_file = output_dir / "brca1_bayesian_enhanced.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n✅ Saved results to: {output_file}")
    
    return results_df

# ==============================================================================
# VISUALIZATION
# ==============================================================================

def create_enhanced_visualizations(df: pd.DataFrame, output_dir: Path):
    """Create publication-quality visualizations."""
    
    print("\n📊 Creating enhanced visualizations...")
    
    # Figure 1: Confidence Tent with CI bars
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Panel A: Scatter with error bars
    ax = axes[0]
    
    colors = {
        'VUS': '#f39c12',
        'Likely Pathogenic': '#e74c3c',
        'Likely Benign': '#2ecc71',
        'Uncertain': '#95a5a6'
    }
    
    for cls in colors.keys():
        subset = df[df['classification'] == cls]
        ax.scatter(subset['evo2_score'], subset['posterior'],
                   c=colors[cls], label=cls, alpha=0.6, s=40,
                   edgecolors='black', linewidths=0.3)
    
    ax.set_xlabel('Evo2 Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Posterior P(Pathogenic)', fontsize=12, fontweight='bold')
    ax.set_title('Enhanced Bayesian Framework\nPosterior Probabilities', 
                 fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(0.5, color='black', linestyle='--', alpha=0.5)
    ax.text(-0.15, 1.05, "A", transform=ax.transAxes, 
            size=22, weight='bold', va='top')
    
    # Panel B: CI width distribution
    ax = axes[1]
    
    ax.hist(df['ci_width'], bins=30, color='#3498db', 
            alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.axvline(0.3, color='red', linestyle='--', linewidth=2,
               label='Confidence Threshold')
    ax.set_xlabel('Credible Interval Width', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Variants', fontsize=12, fontweight='bold')
    ax.set_title('Uncertainty Quantification\n95% CI Width Distribution',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.text(-0.15, 1.05, "B", transform=ax.transAxes,
            size=22, weight='bold', va='top')
    
    plt.tight_layout()
    
    output_path = output_dir / "enhanced_bayesian_framework.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()

# ==============================================================================
# VALIDATION & STATISTICS
# ==============================================================================

def generate_statistics(df: pd.DataFrame, output_dir: Path):
    """Generate summary statistics and validation metrics."""
    
    print("\n" + "="*80)
    print("📊 ENHANCED BAYESIAN FRAMEWORK - SUMMARY STATISTICS")
    print("="*80)
    
    # Classification breakdown
    print("\n📋 Variant Classification:")
    class_counts = df['classification'].value_counts()
    for cls, count in class_counts.items():
        pct = count / len(df) * 100
        print(f"   {cls:<25} {count:>5} ({pct:>5.1f}%)")
    
    # Confidence breakdown
    print("\n🎯 Confidence Levels:")
    conf_counts = df['confidence'].value_counts()
    for conf, count in conf_counts.items():
        pct = count / len(df) * 100
        print(f"   {conf:<10} {count:>5} ({pct:>5.1f}%)")
    
    # Posterior statistics
    print("\n📈 Posterior Probability Statistics:")
    print(f"   Mean: {df['posterior'].mean():.4f}")
    print(f"   Median: {df['posterior'].median():.4f}")
    print(f"   Std Dev: {df['posterior'].std():.4f}")
    
    # CI statistics
    print("\n📉 Credible Interval Statistics:")
    print(f"   Mean CI width: {df['ci_width'].mean():.4f}")
    print(f"   Median CI width: {df['ci_width'].median():.4f}")
    print(f"   High confidence (<0.15): {(df['ci_width'] < 0.15).sum()}")
    print(f"   Medium confidence (0.15-0.3): {((df['ci_width'] >= 0.15) & (df['ci_width'] < 0.3)).sum()}")
    print(f"   Low confidence (>0.3): {(df['ci_width'] >= 0.3).sum()}")
    
    # VUS flagging
    vus_flagged = df[df['is_vus'] == True]
    print(f"\n🚨 VUS Flagging for Expert Review:")
    print(f"   Total VUS: {len(vus_flagged)} ({len(vus_flagged)/len(df)*100:.1f}%)")
    print(f"   Mean CI width (VUS): {vus_flagged['ci_width'].mean():.4f}")
    print(f"   Mean posterior (VUS): {vus_flagged['posterior'].mean():.4f}")
    
    # Export summary
    summary = {
        'Total_Variants': len(df),
        'VUS_Count': len(vus_flagged),
        'VUS_Percent': len(vus_flagged)/len(df)*100,
        'High_Conf_Pathogenic': len(df[(df['posterior'] > 0.8) & (df['ci_width'] < 0.15)]),
        'High_Conf_Benign': len(df[(df['posterior'] < 0.2) & (df['ci_width'] < 0.15)]),
        'Mean_Posterior': df['posterior'].mean(),
        'Mean_CI_Width': df['ci_width'].mean()
    }
    
    summary_df = pd.DataFrame([summary])
    summary_file = output_dir / "bayesian_summary_stats.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"\n✅ Summary statistics saved to: {summary_file}")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    # Configuration
    input_file = "results/brca1_global_calibration.csv"
    output_dir = Path("results/bayesian_enhanced")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process variants
    results_df = process_variants_bayesian(input_file, output_dir)
    
    # Generate visualizations
    create_enhanced_visualizations(results_df, output_dir)
    
    # Generate statistics
    generate_statistics(results_df, output_dir)
    
    print("\n" + "="*80)
    print("✅ ENHANCED BAYESIAN FRAMEWORK COMPLETE!")
    print("="*80)
    
    print("\n💡 Key Outputs:")
    print(f"   1. brca1_bayesian_enhanced.csv - Full results with CIs")
    print(f"   2. enhanced_bayesian_framework.png - Publication figure")
    print(f"   3. bayesian_summary_stats.csv - Summary statistics")

if __name__ == "__main__":
    main()
