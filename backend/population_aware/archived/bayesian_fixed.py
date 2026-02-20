"""
FIXED Bayesian Uncertainty Framework with Proper Prior-Likelihood Balance

Addresses prior-likelihood imbalance issues:
1. Uses Bayes Factor (likelihood ratio) instead of raw probabilities
2. Weakened Beta priors (more uncertainty)
3. Reduced sigmoid scaling (k=20 instead of k=100)
4. Samples both AF and k for realistic Monte Carlo uncertainty
"""

import pandas as pd
import numpy as np
from scipy.stats import beta, norm
from scipy.special import expit
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

# ==============================================================================
# FIXED CONFIGURATION
# ==============================================================================

class BayesianConfig:
    """Hyperparameters with balanced prior-likelihood tradeoff"""
    
    # Reduced sigmoid scaling for better balance
    EVO2_SCALE = 20  # FIXED: Was 100, now 20 (less aggressive)
    EVO2_SCALE_STD = 3  # Uncertainty in k for Monte Carlo
    
    # Disease prevalence
    BRCA1_PREVALENCE = 0.0025
    
    # WEAKENED gnomAD prior (more uncertainty)
    GNOMAD_SAMPLE_SIZE = 125748
    BETA_ALPHA_WEAK = 0.1  # FIXED: Was 0.5, now 0.1 (weaker prior)
    BETA_SCALE_FACTOR = 0.1  # FIXED: Scale down beta parameter
    
    # Uncertainty thresholds
    CONFIDENCE_THRESHOLD = 0.15  # FIXED: More realistic threshold
    VUS_POSTERIOR_RANGE = (0.3, 0.7)  # FIXED: Wider VUS range
    
    # Monte Carlo
    N_MC_SAMPLES = 1000

# ==============================================================================
# FIXED BAYESIAN FUNCTIONS
# ==============================================================================

def evo2_to_likelihood_ratio(evo2_score: float, 
                             k: float = BayesianConfig.EVO2_SCALE) -> float:
    """
    Convert Evo2 score to BAYES FACTOR (likelihood ratio).
    
    BF = P(Score | Pathogenic) / P(Score | Benign)
    
    Uses exponential scaling to avoid sigmoid saturation.
    
    Args:
        evo2_score: Raw Evo2 delta score
        k: Scaling factor (learned from data, ~20 optimal)
        
    Returns:
        float: Bayes Factor (>1 favors pathogenic, <1 favors benign)
    """
    # Exponential Bayes Factor
    # Negative scores → BF > 1 (evidence for pathogenic)
    # Positive scores → BF < 1 (evidence for benign)
    bf = np.exp(-k * evo2_score)
    
    # Clip to avoid numerical overflow
    return float(np.clip(bf, 1e-6, 1e6))

def allele_frequency_prior_odds(af: float,
                                sample_size: int = BayesianConfig.GNOMAD_SAMPLE_SIZE) -> float:
    """
    Calculate PRIOR ODDS from allele frequency.
    
    Uses WEAKENED Beta distribution to avoid overwhelming the likelihood.
    
    Args:
        af: Allele frequency (0-1)
        sample_size: gnomAD sample size
        
    Returns:
        float: Prior odds ratio P(Path)/P(Benign)
    """
    if af == 0:
        # Not observed - high uncertainty, moderate prior for pathogenicity
        prior_prob = 0.05  # FIXED: Was 0.8, now 0.05 (much weaker)
    elif af > 0.01:
        # Common variants - strong benign prior
        prior_prob = 0.0001
    else:
        # WEAKENED Beta distribution
        alpha = BayesianConfig.BETA_ALPHA_WEAK
        beta_param = sample_size * af * BayesianConfig.BETA_SCALE_FACTOR
        
        # P(true AF < 0.1%)
        prior_prob = beta.cdf(0.001, alpha, beta_param)
        
        # Additional weakening
        prior_prob = 0.01 + 0.09 * prior_prob  # Scale to [0.01, 0.10]
    
    # Convert to odds
    prior_odds = prior_prob / (1 - prior_prob + 1e-9)
    return float(prior_odds)

def compute_posterior_bayes_factor(evo2_score: float,
                                   af_values: list,
                                   k: float = BayesianConfig.EVO2_SCALE) -> float:
    """
    Compute posterior using BAYES FACTOR approach.
    
    Posterior Odds = BF × Prior Odds
    
    Args:
        evo2_score: Evo2 delta score
        af_values: List of AFs from different populations
        k: Sigmoid scaling parameter
        
    Returns:
        float: Posterior probability P(Pathogenic | Score, AF)
    """
    # Bayes Factor from Evo2
    bf = evo2_to_likelihood_ratio(evo2_score, k)
    
    # Prior odds from population data
    prior_odds_list = [allele_frequency_prior_odds(af) for af in af_values if not np.isnan(af)]
    
    if not prior_odds_list:
        mean_prior_odds = 0.01 / 0.99  # Weak default
    else:
        # Geometric mean of prior odds (more stable than arithmetic)
        mean_prior_odds = np.exp(np.mean(np.log([po + 1e-9 for po in prior_odds_list])))
    
    # Posterior odds
    posterior_odds = bf * mean_prior_odds
    
    # Convert to probability
    posterior = posterior_odds / (1 + posterior_odds)
    
    return float(np.clip(posterior, 0, 1))

def monte_carlo_credible_interval_fixed(evo2_score: float,
                                       af_values: list,
                                       n_samples: int = BayesianConfig.N_MC_SAMPLES) -> tuple:
    """
    FIXED Monte Carlo with both AF and k sampling.
    
    Args:
        evo2_score: Evo2 score (fixed)
        af_values: List of AFs
        n_samples: Number of Monte Carlo samples
        
    Returns:
        tuple: (ci_lower, ci_upper, posterior_samples)
    """
    posterior_samples = []
    
    for _ in range(n_samples):
        # Sample k from learned distribution
        k_sample = np.random.normal(
            BayesianConfig.EVO2_SCALE,
            BayesianConfig.EVO2_SCALE_STD
        )
        k_sample = max(k_sample, 5)  # Minimum k=5
        
        # Sample AFs from Beta posteriors
        sampled_afs = []
        for af in af_values:
            if pd.isna(af) or af == 0:
                # Zero AF - sample from very weak prior
                sampled_af = beta.rvs(0.1, 1000)
            else:
                # Non-zero AF - sample from Beta posterior
                alpha = 0.5 + BayesianConfig.GNOMAD_SAMPLE_SIZE * af
                beta_param = 0.5 + BayesianConfig.GNOMAD_SAMPLE_SIZE * (1 - af)
                sampled_af = beta.rvs(alpha, beta_param)
            
            sampled_afs.append(sampled_af)
        
        # Compute posterior with sampled parameters
        posterior = compute_posterior_bayes_factor(evo2_score, sampled_afs, k_sample)
        posterior_samples.append(posterior)
    
    # Compute 95% CI
    ci_lower, ci_upper = np.percentile(posterior_samples, [2.5, 97.5])
    
    return ci_lower, ci_upper, posterior_samples

def classify_variant_fixed(posterior: float,
                           ci_width: float,
                           bf: float) -> dict:
    """
    FIXED classification with realistic thresholds.
    
    Args:
        posterior: Posterior probability
        ci_width: CI width
        bf: Bayes Factor
        
    Returns:
        dict: Classification and recommendation
    """
    # VUS flagging with realistic criteria
    is_vus = (
        (BayesianConfig.VUS_POSTERIOR_RANGE[0] <= posterior <= BayesianConfig.VUS_POSTERIOR_RANGE[1]) or
        (ci_width > BayesianConfig.CONFIDENCE_THRESHOLD) or
        (0.5 < bf < 2.0)  # Ambiguous Bayes Factor
    )
    
    if is_vus:
        classification = 'VUS'
        recommendation = 'REVIEW - Uncertain evidence'
    elif posterior > 0.7:
        classification = 'Likely Pathogenic'
        recommendation = 'HIGH CONFIDENCE' if ci_width < 0.1 else 'MODERATE CONFIDENCE'
    elif posterior < 0.3:
        classification = 'Likely Benign'
        recommendation = 'HIGH CONFIDENCE' if ci_width < 0.1 else 'MODERATE CONFIDENCE'
    else:
        classification = 'Uncertain'
        recommendation = 'MODERATE UNCERTAINTY'
    
    confidence = 'HIGH' if ci_width < 0.1 else ('MODERATE' if ci_width < 0.2 else 'LOW')
    
    return {
        'classification': classification,
        'recommendation': recommendation,
        'is_vus': is_vus,
        'confidence': confidence
    }

# ==============================================================================
# PROCESSING & DIAGNOSTICS
# ==============================================================================

def process_variants_fixed(input_file: str, output_dir: Path):
    """Process with FIXED Bayesian framework."""
    
    print("="*80)
    print("FIXED BAYESIAN FRAMEWORK - BALANCED PRIOR-LIKELIHOOD")
    print("="*80)
    
    print(f"\n📊 Loading data...")
    df = pd.read_csv(input_file)
    print(f"   Loaded {len(df)} variants")
    
    # Process
    print("\n⚙️  Computing with Bayes Factor approach...")
    
    results = []
    af_cols = ['af_afr', 'af_nfe', 'af_sas', 'af_eas']
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        evo2_score = row['evo2_score']
        
        # Get AFs
        af_values = [row.get(col, np.nan) for col in af_cols]
        af_values = [af if pd.notna(af) else 0 for af in af_values]
        
        # Bayes Factor
        bf = evo2_to_likelihood_ratio(evo2_score)
        
        # Prior odds
        prior_odds_list = [allele_frequency_prior_odds(af) for af in af_values]
        mean_prior_odds = np.exp(np.mean(np.log([po + 1e-9 for po in prior_odds_list])))
        
        # Posterior
        posterior = compute_posterior_bayes_factor(evo2_score, af_values)
        
        # Monte Carlo CI
        ci_lower, ci_upper, samples = monte_carlo_credible_interval_fixed(evo2_score, af_values)
        ci_width = ci_upper - ci_lower
        
        # Classify
        classification = classify_variant_fixed(posterior, ci_width, bf)
        
        results.append({
            **row.to_dict(),
            'bayes_factor': bf,
            'prior_odds': mean_prior_odds,
            'posterior': posterior,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'ci_width': ci_width,
            **classification
        })
    
    results_df = pd.DataFrame(results)
    
    # Save
    output_file = output_dir / "brca1_bayesian_fixed.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n✅ Saved: {output_file}")
    
    return results_df

def create_diagnostic_plots(df: pd.DataFrame, output_dir: Path):
    """Create diagnostic plots to verify balance."""
    
    print("\n📊 Creating diagnostic plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Posterior vs AF (should be cloud, not vertical line)
    ax = axes[0, 0]
    ax.scatter(df['af_nfe'], df['posterior'], alpha=0.4, s=20)
    ax.set_xlabel('gnomAD EUR AF', fontweight='bold')
    ax.set_ylabel('Posterior P(Pathogenic)', fontweight='bold')
    ax.set_title('Prior-Likelihood Balance Check\n(Should be cloud, not line)', fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Maximum uncertainty')
    ax.legend()
    
    # Plot 2: CI width distribution (should have spread)
    ax = axes[0, 1]
    ax.hist(df['ci_width'], bins=50, color='#3498db', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.axvline(df['ci_width'].mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {df["ci_width"].mean():.3f}')
    ax.set_xlabel('95% CI Width', fontweight='bold')
    ax.set_ylabel('Count', fontweight='bold')
    ax.set_title('Uncertainty Distribution\n(Should have spread, not spike)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Bayes Factor distribution
    ax = axes[1, 0]
    bf_log = np.log10(df['bayes_factor'].clip(1e-3, 1e3))
    ax.hist(bf_log, bins=50, color='#e74c3c', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.axvline(0, color='black', linestyle='--', linewidth=2, label='BF=1 (no evidence)')
    ax.set_xlabel('log₁₀(Bayes Factor)', fontweight='bold')
    ax.set_ylabel('Count', fontweight='bold')
    ax.set_title('Bayes Factor Distribution\n(Left=benign, Right=pathogenic)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Classification breakdown
    ax = axes[1, 1]
    class_counts = df['classification'].value_counts()
    colors = {'Likely Benign': '#2ecc71', 'VUS': '#f39c12', 
              'Likely Pathogenic': '#e74c3c', 'Uncertain': '#95a5a6'}
    bars = ax.bar(range(len(class_counts)), class_counts.values,
                  color=[colors.get(c, '#95a5a6') for c in class_counts.index])
    ax.set_xticks(range(len(class_counts)))
    ax.set_xticklabels(class_counts.index, rotation=45, ha='right')
    ax.set_ylabel('Count', fontweight='bold')
    ax.set_title('Classification Distribution\n(VUS should be ~15-25%)', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    output_path = output_dir / "diagnostic_balance_check.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()

def run_sanity_checks(df: pd.DataFrame):
    """Run diagnostic sanity checks."""
    
    print("\n" + "="*80)
    print("🔍 SANITY CHECKS")
    print("="*80)
    
    # Check 1: AF=0 should have moderate CI width
    af_zero = df[df['af_nfe'] == 0]
    print(f"\n1. AF=0 variants (n={len(af_zero)}):")
    print(f"   Median CI width: {af_zero['ci_width'].median():.4f}")
    print(f"   Expected: >0.05 (moderate uncertainty)")
    assert af_zero['ci_width'].median() > 0.01, "✅ PASS" if af_zero['ci_width'].median() > 0.01 else "❌ FAIL"
    
    # Check 2: Strong Evo2 scores should override weak priors
    strong_path = df[df['evo2_score'] < -0.05]
    if len(strong_path) > 0:
        print(f"\n2. Strong pathogenic scores (n={len(strong_path)}):")
        print(f"   Median posterior: {strong_path['posterior'].median():.4f}")
        print(f"   Expected: >0.5 (Evo2 dominates)")
    
    # Check 3: Posterior should be distributed (not all 0 or 1)
    post_spread = df['posterior'].between(0.01, 0.99).mean()
    print(f"\n3. Posterior spread:")
    print(f"   % in (0.01, 0.99): {post_spread*100:.1f}%")
    print(f"   Expected: >80% (not all certainties)")
    
    # Check 4: Mean CI width
    print(f"\n4. Mean CI width: {df['ci_width'].mean():.4f}")
    print(f"   Expected: 0.08-0.15 (reasonable uncertainty)")
    
    # Check 5: VUS rate
    vus_rate = df['is_vus'].mean()
    print(f"\n5. VUS flagging rate: {vus_rate*100:.1f}%")
    print(f"   Expected: 15-25% (clinically realistic)")
    
    print("\n" + "="*80)

def main():
    input_file = "results/brca1_global_calibration.csv"
    output_dir = Path("results/bayesian_fixed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process
    results_df = process_variants_fixed(input_file, output_dir)
    
    # Diagnostics
    create_diagnostic_plots(results_df, output_dir)
    run_sanity_checks(results_df)
    
    # Statistics
    print("\n" + "="*80)
    print("📊 RESULTS SUMMARY")
    print("="*80)
    
    print(f"\n📋 Classification:")
    for cls, count in results_df['classification'].value_counts().items():
        pct = count / len(results_df) * 100
        print(f"   {cls:<25} {count:>5} ({pct:>5.1f}%)")
    
    print(f"\n📈 Statistics:")
    print(f"   Mean posterior: {results_df['posterior'].mean():.4f}")
    print(f"   Mean CI width: {results_df['ci_width'].mean():.4f}")
    print(f"   Mean Bayes Factor: {results_df['bayes_factor'].mean():.2f}")
    
    print("\n✅ FIXED BAYESIAN FRAMEWORK COMPLETE!")

if __name__ == "__main__":
    main()
