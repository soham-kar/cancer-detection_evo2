"""
TUNED Bayesian Framework - Balanced Prior-Likelihood

Fixes applied:
1. k=40 (2x stronger Evo2 influence)
2. AF=0 prior = 0.01 (weaker, not 0.05)
3. VUS threshold (0.2 < BF < 5.0) - wider range
4. Built-in validation with functional class checks
"""

import pandas as pd
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

# ==============================================================================
# TUNED CONFIGURATION
# ==============================================================================

class BayesianConfig:
    """TUNED hyperparameters for balanced prior-likelihood"""
    
    # FIXED: k=40 (2x stronger than before)
    EVO2_SCALE = 40  # Was 20, now 40 - sweet spot
    EVO2_SCALE_STD = 5  # Slightly wider uncertainty
    
    BRCA1_PREVALENCE = 0.0025
    GNOMAD_SAMPLE_SIZE = 125748
    
    # Weakened priors
    BETA_ALPHA_WEAK = 0.1
    BETA_SCALE_FACTOR = 0.1
    
    # Realistic thresholds
    CONFIDENCE_THRESHOLD = 0.15
    VUS_POSTERIOR_RANGE = (0.3, 0.7)
    
    # Optimized Monte Carlo
    N_MC_SAMPLES = 100

# ==============================================================================
# CORE FUNCTIONS
# ==============================================================================

def evo2_to_likelihood_ratio(evo2_score: float, k: float = BayesianConfig.EVO2_SCALE) -> float:
    """Bayes Factor with k=40"""
    bf = np.exp(-k * evo2_score)
    return float(np.clip(bf, 1e-6, 1e6))

def allele_frequency_prior_odds(af: float) -> float:
    """TUNED: Weakened prior for AF=0"""
    if af == 0:
        prior_prob = 0.01  # FIXED: Was 0.05, now 0.01 (much weaker)
    elif af > 0.01:
        prior_prob = 0.0001
    else:
        alpha = BayesianConfig.BETA_ALPHA_WEAK
        beta_param = BayesianConfig.GNOMAD_SAMPLE_SIZE * af * BayesianConfig.BETA_SCALE_FACTOR
        prior_prob = beta.cdf(0.001, alpha, beta_param)
        prior_prob = 0.01 + 0.09 * prior_prob
    
    return float(prior_prob / (1 - prior_prob + 1e-9))

def compute_posterior_bayes_factor(evo2_score: float, af_values: list,
                                   k: float = BayesianConfig.EVO2_SCALE) -> float:
    """Posterior with k=40"""
    bf = evo2_to_likelihood_ratio(evo2_score, k)
    
    prior_odds_list = [allele_frequency_prior_odds(af) 
                       for af in af_values if not np.isnan(af) and af > 0]
    
    if not prior_odds_list:
        mean_prior_odds = 0.01 / 0.99
    else:
        mean_prior_odds = np.exp(np.mean(np.log([po + 1e-9 for po in prior_odds_list])))
    
    posterior_odds = bf * mean_prior_odds
    posterior = posterior_odds / (1 + posterior_odds)
    return float(np.clip(posterior, 0, 1))

def monte_carlo_credible_interval(evo2_score: float, af_values: list,
                                  n_samples: int = BayesianConfig.N_MC_SAMPLES) -> tuple:
    """Monte Carlo with 100 samples"""
    posterior_samples = []
    
    for _ in range(n_samples):
        k_sample = max(np.random.normal(BayesianConfig.EVO2_SCALE, 
                                        BayesianConfig.EVO2_SCALE_STD), 10)
        
        sampled_afs = []
        for af in af_values:
            if pd.isna(af) or af == 0:
                sampled_af = beta.rvs(0.1, 1000)
            else:
                alpha = 0.5 + BayesianConfig.GNOMAD_SAMPLE_SIZE * af
                beta_param = 0.5 + BayesianConfig.GNOMAD_SAMPLE_SIZE * (1 - af)
                sampled_af = beta.rvs(alpha, beta_param)
            sampled_afs.append(sampled_af)
        
        posterior = compute_posterior_bayes_factor(evo2_score, sampled_afs, k_sample)
        posterior_samples.append(posterior)
    
    ci_lower, ci_upper = np.percentile(posterior_samples, [2.5, 97.5])
    return ci_lower, ci_upper, posterior_samples

def classify_variant(posterior: float, ci_width: float, bf: float) -> dict:
    """TUNED: Fixed VUS threshold"""
    # FIXED: Wider BF range for VUS (0.2 < BF < 5.0)
    is_vus = (
        (BayesianConfig.VUS_POSTERIOR_RANGE[0] <= posterior <= BayesianConfig.VUS_POSTERIOR_RANGE[1]) or
        (ci_width > BayesianConfig.CONFIDENCE_THRESHOLD) or
        (0.2 < bf < 5.0)  # FIXED: Was (0.5, 2.0), now (0.2, 5.0)
    )
    
    if is_vus:
        classification = 'VUS'
        recommendation = 'REVIEW'
    elif posterior > 0.7:
        classification = 'Likely Pathogenic'
        recommendation = 'HIGH CONFIDENCE' if ci_width < 0.1 else 'MODERATE'
    elif posterior < 0.3:
        classification = 'Likely Benign'
        recommendation = 'HIGH CONFIDENCE' if ci_width < 0.1 else 'MODERATE'
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
# PROCESSING
# ==============================================================================

def process_variants_tuned(input_file: str, output_dir: Path):
    """Process with TUNED parameters"""
    
    print("="*80)
    print("TUNED BAYESIAN FRAMEWORK - k=40, WEAKENED PRIORS")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  k (Evo2 scale): {BayesianConfig.EVO2_SCALE}")
    print(f"  AF=0 prior: 0.01 (weak)")
    print(f"  VUS BF range: (0.2, 5.0)")
    print(f"  MC samples: {BayesianConfig.N_MC_SAMPLES}")
    
    df = pd.read_csv(input_file)
    print(f"\n📊 Loaded {len(df)} variants")
    
    # Handle AF columns
    af_cols = []
    for col in ['af_afr', 'af_nfe', 'af_eas', 'af_sas', 'af_amr']:
        if col in df.columns:
            af_cols.append(col)
        elif col == 'af_nfe' and 'af_eur' in df.columns:
            df['af_nfe'] = df['af_eur']
            af_cols.append('af_nfe')
    
    print(f"   AF columns: {af_cols}")
    
    if 'evo2_score' not in df.columns:
        print("❌ Missing evo2_score column")
        return None
    
    # Check Evo2 score distribution
    print(f"\n📈 Evo2 Score Distribution:")
    print(df['evo2_score'].describe())
    
    # Process
    print(f"\n⚙️  Processing with k={BayesianConfig.EVO2_SCALE}...")
    
    results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        evo2_score = row['evo2_score']
        
        af_values = [row.get(col, np.nan) for col in af_cols]
        af_values = [af if pd.notna(af) else 0 for af in af_values]
        
        bf = evo2_to_likelihood_ratio(evo2_score)
        
        prior_odds_list = [allele_frequency_prior_odds(af) for af in af_values if af > 0]
        mean_prior_odds = np.exp(np.mean(np.log([po + 1e-9 for po in prior_odds_list]))) if prior_odds_list else 0.01/0.99
        
        posterior = compute_posterior_bayes_factor(evo2_score, af_values)
        
        ci_lower, ci_upper, samples = monte_carlo_credible_interval(evo2_score, af_values)
        ci_width = ci_upper - ci_lower
        
        classification = classify_variant(posterior, ci_width, bf)
        
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
    
    output_file = output_dir / "brca1_bayesian_tuned.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n✅ Saved: {output_file}")
    
    return results_df

# ==============================================================================
# VALIDATION
# ==============================================================================

def validate_bayesian_balance(df: pd.DataFrame):
    """COMPREHENSIVE validation with functional class checks"""
    
    print("\n" + "="*80)
    print("🔍 COMPREHENSIVE VALIDATION")
    print("="*80)
    
    checks_passed = 0
    checks_total = 0
    
    # Check 1: Mean BF
    print(f"\n1. Mean Bayes Factor: {df['bayes_factor'].mean():.2f}")
    print(f"   Expected: >2.0 (moderate evidence)")
    if df['bayes_factor'].mean() > 2.0:
        print("   ✅ PASS")
        checks_passed += 1
    else:
        print("   ❌ FAIL")
    checks_total += 1
    
    # Check 2: VUS rate
    vus_rate = df['is_vus'].mean()
    print(f"\n2. VUS rate: {vus_rate*100:.1f}%")
    print(f"   Expected: 10-30%")
    if 0.10 < vus_rate < 0.30:
        print("   ✅ PASS")
        checks_passed += 1
    else:
        print("   ❌ FAIL")
    checks_total += 1
    
    # Check 3: Mean CI width
    mean_ci = df['ci_width'].mean()
    print(f"\n3. Mean CI width: {mean_ci:.4f}")
    print(f"   Expected: 0.05-0.20")
    if 0.05 < mean_ci < 0.20:
        print("   ✅ PASS")
        checks_passed += 1
    else:
        print("   ❌ FAIL")
    checks_total += 1
    
    # Check 4: AF=0 uncertainty
    af_zero = df[df.get('af_nfe', df.get('af_eur', 0)) == 0]
    if len(af_zero) > 0:
        median_ci_zero = af_zero['ci_width'].median()
        print(f"\n4. AF=0 median CI width: {median_ci_zero:.4f}")
        print(f"   Expected: >0.05")
        if median_ci_zero > 0.05:
            print("   ✅ PASS")
            checks_passed += 1
        else:
            print("   ❌ FAIL")
        checks_total += 1
    
    # Check 5: Strong pathogenic variants (if func_class available)
    if 'func_class' in df.columns:
        pathogenic = df[df['func_class'].isin(['LOF', 'Pathogenic'])]
        if len(pathogenic) > 10:
            median_bf_path = pathogenic['bayes_factor'].median()
            median_post_path = pathogenic['posterior'].median()
            print(f"\n5a. Pathogenic median BF: {median_bf_path:.2f}")
            print(f"    Expected: >2.0")
            if median_bf_path > 2.0:
                print("    ✅ PASS")
                checks_passed += 1
            else:
                print("    ❌ FAIL")
            checks_total += 1
            
            print(f"\n5b. Pathogenic median posterior: {median_post_path:.3f}")
            print(f"    Expected: >0.5")
            if median_post_path > 0.5:
                print("    ✅ PASS")
                checks_passed += 1
            else:
                print("    ❌ FAIL")
            checks_total += 1
    
    # Check 6: Strong Evo2 scores
    strong_path = df[df['evo2_score'] < -0.05]
    if len(strong_path) > 10:
        median_post_strong = strong_path['posterior'].median()
        print(f"\n6. Strong pathogenic (score<-0.05) median posterior: {median_post_strong:.3f}")
        print(f"   Expected: >0.6")
        if median_post_strong > 0.6:
            print("   ✅ PASS")
            checks_passed += 1
        else:
            print("   ❌ FAIL")
        checks_total += 1
    
    # Check 7: Posterior spread
    post_spread = df['posterior'].between(0.01, 0.99).mean()
    print(f"\n7. Posterior spread: {post_spread*100:.1f}% in (0.01, 0.99)")
    print(f"   Expected: >80%")
    if post_spread > 0.80:
        print("   ✅ PASS")
        checks_passed += 1
    else:
        print("   ❌ FAIL")
    checks_total += 1
    
    print("\n" + "="*80)
    print(f"FINAL SCORE: {checks_passed}/{checks_total} checks passed")
    if checks_passed >= checks_total * 0.7:
        print("✅ VALIDATION PASSED - Framework is balanced!")
    else:
        print(f"⚠️  {checks_total - checks_passed} check(s) failed")
    print("="*80)
    
    return checks_passed, checks_total

def create_diagnostic_plots(df: pd.DataFrame, output_dir: Path):
    """Enhanced diagnostic plots"""
    
    print("\n📊 Creating diagnostic plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Plot 1: Posterior vs AF
    ax = axes[0, 0]
    af_col = 'af_nfe' if 'af_nfe' in df.columns else 'af_eur'
    if af_col in df.columns:
        mask = df[af_col] > 0
        ax.scatter(df.loc[mask, af_col], df.loc[mask, 'posterior'], alpha=0.3, s=10)
        ax.set_xscale('log')
        ax.set_xlabel('gnomAD AF (log)', fontweight='bold')
        ax.set_ylabel('Posterior P(Pathogenic)', fontweight='bold')
        ax.set_title('Prior-Likelihood Balance\n(Cloud = Good)', fontweight='bold')
        ax.axhline(0.5, color='red', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)
    
    # Plot 2: CI width
    ax = axes[0, 1]
    ax.hist(df['ci_width'], bins=50, color='#3498db', alpha=0.7, edgecolor='black')
    ax.axvline(df['ci_width'].mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {df["ci_width"].mean():.3f}')
    ax.axvline(0.15, color='orange', linestyle='--', alpha=0.7, label='Threshold')
    ax.set_xlabel('95% CI Width', fontweight='bold')
    ax.set_ylabel('Count', fontweight='bold')
    ax.set_title('Uncertainty Distribution', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Bayes Factor
    ax = axes[0, 2]
    bf_log = np.log10(df['bayes_factor'].clip(1e-2, 1e2))
    ax.hist(bf_log, bins=50, color='#e74c3c', alpha=0.7, edgecolor='black')
    ax.axvline(0, color='black', linestyle='--', linewidth=2, label='BF=1')
    ax.axvline(np.log10(df['bayes_factor'].mean()), color='green', 
               linestyle='--', linewidth=2, label=f'Mean BF={df["bayes_factor"].mean():.2f}')
    ax.set_xlabel('log₁₀(Bayes Factor)', fontweight='bold')
    ax.set_ylabel('Count', fontweight='bold')
    ax.set_title('Bayes Factor Distribution', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Classification
    ax = axes[1, 0]
    class_counts = df['classification'].value_counts()
    colors = {'Likely Benign': '#2ecc71', 'VUS': '#f39c12',
              'Likely Pathogenic': '#e74c3c', 'Uncertain': '#95a5a6'}
    bars = ax.bar(range(len(class_counts)), class_counts.values,
                  color=[colors.get(c, '#95a5a6') for c in class_counts.index])
    ax.set_xticks(range(len(class_counts)))
    ax.set_xticklabels(class_counts.index, rotation=45, ha='right')
    ax.set_ylabel('Count', fontweight='bold')
    ax.set_title(f'Classification (VUS={df["is_vus"].mean()*100:.1f}%)', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 5: Evo2 score distribution
    ax = axes[1, 1]
    ax.hist(df['evo2_score'], bins=50, color='#9b59b6', alpha=0.7, edgecolor='black')
    ax.axvline(df['evo2_score'].mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {df["evo2_score"].mean():.4f}')
    ax.set_xlabel('Evo2 Score', fontweight='bold')
    ax.set_ylabel('Count', fontweight='bold')
    ax.set_title('Evo2 Score Distribution', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 6: Posterior vs Evo2 score
    ax = axes[1, 2]
    ax.scatter(df['evo2_score'], df['posterior'], alpha=0.3, s=10, c=df['bayes_factor'],
               cmap='RdYlGn', norm= plt.Normalize(vmin=0.1, vmax=10))
    ax.set_xlabel('Evo2 Score', fontweight='bold')
    ax.set_ylabel('Posterior P(Pathogenic)', fontweight='bold')
    ax.set_title('Posterior vs Evo2 Score', fontweight='bold')
    ax.axhline(0.5, color='black', linestyle='--', alpha=0.5)
    ax.axvline(0, color='black', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label('Bayes Factor', fontweight='bold')
    
    plt.tight_layout()
    
    output_path = output_dir / "diagnostic_tuned.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    input_file = "results/brca1_global_calibration.csv"
    output_dir = Path("results/bayesian_tuned")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process
    results_df = process_variants_tuned(input_file, output_dir)
    
    if results_df is not None:
        # Validation
        checks_passed, checks_total = validate_bayesian_balance(results_df)
        
        # Diagnostics
        create_diagnostic_plots(results_df, output_dir)
        
        # Summary
        print("\n" + "="*80)
        print("📊 SUMMARY")
        print("="*80)
        
        print(f"\n📋 Classification:")
        for cls, count in results_df['classification'].value_counts().items():
            pct = count / len(results_df) * 100
            print(f"   {cls:<25} {count:>5} ({pct:>5.1f}%)")
        
        print(f"\n📈 Key Metrics:")
        print(f"   Mean posterior: {results_df['posterior'].mean():.4f}")
        print(f"   Mean CI width: {results_df['ci_width'].mean():.4f}")
        print(f"   Mean Bayes Factor: {results_df['bayes_factor'].mean():.2f}")
        print(f"   VUS rate: {results_df['is_vus'].mean()*100:.1f}%")
        
        print(f"\n🎯 Validation: {checks_passed}/{checks_total} checks passed")
        
        print("\n✅ TUNED BAYESIAN FRAMEWORK COMPLETE!")

if __name__ == "__main__":
    main()
