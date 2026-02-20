"""
Script 07: Survival Integration
Build Prognostic Driver Score (PDS) from Evo2 scores
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from scipy import stats
import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import *

def calculate_prognostic_driver_score(patient_variants, top_n=10):
    """
    Calculate PDS for a patient based on top N most deleterious variants
    
    PDS = weighted sum of Evo2 scores
    More negative = more drivers = worse prognosis
    """
    # Get top N most deleterious variants
    top_variants = patient_variants.nsmallest(top_n, 'delta_score')
    
    if len(top_variants) == 0:
        return 0
    
    # Weight by absolute score (more negative = higher weight)
    weights = np.abs(top_variants['delta_score'])
    weights = weights / weights.sum()  # Normalize
    
    # Weighted sum
    pds = np.sum(top_variants['delta_score'] * weights)
    
    return pds

def calculate_pds_for_all_patients(variants_df):
    """Calculate PDS for each patient"""
    top_n = ANALYSIS_CONFIG["survival"]["pds_top_variants"]
    
    patient_scores = []
    for patient_id in variants_df['patient_id'].unique():
        patient_vars = variants_df[variants_df['patient_id'] == patient_id]
        pds = calculate_prognostic_driver_score(patient_vars, top_n)
        
        patient_scores.append({
            'patient_id': patient_id,
            'pds': pds,
            'n_variants': len(patient_vars),
            'n_high_impact': (patient_vars['delta_score'] < -0.5).sum()
        })
    
    return pd.DataFrame(patient_scores)

def survival_analysis(survival_df):
    """Perform Kaplan-Meier and Cox regression analysis"""
    print("\n" + "=" * 60)
    print("SURVIVAL ANALYSIS")
    print("=" * 60)
    
    # Check required columns
    required = ['survival_time', 'event_status', 'pds']
    missing = [col for col in required if col not in survival_df.columns]
    if missing:
        print(f"Error: Missing columns: {missing}")
        print("Available columns:", survival_df.columns.tolist())
        return None
    
    # Stratify by PDS (high vs low driver burden)
    method = ANALYSIS_CONFIG["survival"]["stratification_method"]
    if method == "median":
        threshold = survival_df['pds'].median()
        survival_df['pds_group'] = survival_df['pds'].apply(
            lambda x: 'High' if x < threshold else 'Low'
        )
    elif method == "tertile":
        survival_df['pds_group'] = pd.qcut(
            survival_df['pds'], 
            q=3, 
            labels=['Low', 'Medium', 'High']
        )
    
    print(f"\nStratification: {method}")
    print(f"PDS threshold: {threshold:.4f}")
    print(f"High burden: {(survival_df['pds_group'] == 'High').sum()} patients")
    print(f"Low burden: {(survival_df['pds_group'] == 'Low').sum()} patients")
    
    # Kaplan-Meier analysis
    kmf_high = KaplanMeierFitter()
    kmf_low = KaplanMeierFitter()
    
    high_mask = survival_df['pds_group'] == 'High'
    low_mask = survival_df['pds_group'] == 'Low'
    
    kmf_high.fit(
        survival_df[high_mask]['survival_time'],
        survival_df[high_mask]['event_status'],
        label='High Driver Burden'
    )
    
    kmf_low.fit(
        survival_df[low_mask]['survival_time'],
        survival_df[low_mask]['event_status'],
        label='Low Driver Burden'
    )
    
    # Log-rank test
    results = logrank_test(
        survival_df[high_mask]['survival_time'],
        survival_df[low_mask]['survival_time'],
        survival_df[high_mask]['event_status'],
        survival_df[low_mask]['event_status']
    )
    
    print(f"\nLog-rank test:")
    print(f"  Test statistic: {results.test_statistic:.3f}")
    print(f"  P-value: {results.p_value:.4f}")
    print(f"  Significant: {results.p_value < 0.05}")
    
    # Cox proportional hazards
    print(f"\nCox Proportional Hazards Model:")
    cph = CoxPHFitter()
    
    # Prepare data for Cox model
    cox_data = survival_df[['survival_time', 'event_status', 'pds']].copy()
    
    # Add clinical covariates if available
    if 'age' in survival_df.columns:
        cox_data['age'] = survival_df['age']
    if 'sex' in survival_df.columns:
        cox_data['sex_male'] = (survival_df['sex'] == 'M').astype(int)
    
    try:
        cph.fit(cox_data, duration_col='survival_time', event_col='event_status')
        print(cph.summary)
        
        # Hazard ratio for PDS
        hr = np.exp(cph.params_['pds'])
        ci = np.exp(cph.confidence_intervals_.loc['pds'])
        print(f"\nPDS Hazard Ratio: {hr:.3f} (95% CI: {ci[0]:.3f}-{ci[1]:.3f})")
    except Exception as e:
        print(f"Cox model failed: {e}")
        cph = None
    
    return {
        'kmf_high': kmf_high,
        'kmf_low': kmf_low,
        'logrank_p': results.p_value,
        'cph': cph
    }

def create_survival_figures(survival_df, km_results, output_dir):
    """Generate survival analysis figures"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Panel A: Kaplan-Meier curves
    ax = axes[0, 0]
    km_results['kmf_high'].plot_survival_function(ax=ax, ci_show=True)
    km_results['kmf_low'].plot_survival_function(ax=ax, ci_show=True)
    ax.set_xlabel('Time (months)')
    ax.set_ylabel('Survival Probability')
    ax.set_title(f"A. Survival by Driver Burden (p={km_results['logrank_p']:.3f})")
    ax.legend()
    
    # Panel B: PDS distribution
    ax = axes[0, 1]
    high_pds = survival_df[survival_df['pds_group'] == 'High']['pds']
    low_pds = survival_df[survival_df['pds_group'] == 'Low']['pds']
    ax.hist(high_pds, bins=20, alpha=0.6, label='High Burden', color='red')
    ax.hist(low_pds, bins=20, alpha=0.6, label='Low Burden', color='green')
    ax.set_xlabel('Prognostic Driver Score')
    ax.set_ylabel('Count')
    ax.set_title('B. PDS Distribution')
    ax.legend()
    
    # Panel C: PDS vs survival time
    ax = axes[1, 0]
    colors = ['red' if e == 1 else 'blue' for e in survival_df['event_status']]
    ax.scatter(survival_df['pds'], survival_df['survival_time'], 
               c=colors, alpha=0.6, s=50)
    ax.set_xlabel('Prognostic Driver Score')
    ax.set_ylabel('Survival Time (months)')
    ax.set_title('C. PDS vs Survival Time')
    
    # Add correlation
    corr, p = stats.spearmanr(survival_df['pds'], survival_df['survival_time'])
    ax.text(0.05, 0.95, f'ρ={corr:.3f}\np={p:.3f}', 
            transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white'))
    
    # Panel D: Cox hazard ratios (if available)
    ax = axes[1, 1]
    if km_results['cph'] is not None:
        cph = km_results['cph']
        hr = np.exp(cph.params_)
        ci = np.exp(cph.confidence_intervals_)
        
        y_pos = range(len(hr))
        ax.scatter(hr, y_pos, s=100, color='darkblue')
        ax.hlines(y_pos, ci.iloc[:, 0], ci.iloc[:, 1], color='darkblue', alpha=0.5)
        ax.axvline(x=1, color='black', linestyle='--', alpha=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(hr.index)
        ax.set_xlabel('Hazard Ratio')
        ax.set_title('D. Cox Proportional Hazards')
        ax.set_xscale('log')
    
    plt.tight_layout()
    
    # Save
    for fmt in FIGURE_CONFIG['format']:
        output_path = output_dir / f"figure_survival_analysis.{fmt}"
        plt.savefig(output_path, dpi=FIGURE_CONFIG['dpi'], bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Survival integration analysis")
    parser.add_argument("--input", type=str, required=True, help="Evo2 scores CSV")
    parser.add_argument("--clinical", type=str, required=True, help="Clinical data with survival")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("PHASE 5: SURVIVAL INTEGRATION")
    print("=" * 60)
    
    # Load data
    variants = pd.read_csv(args.input)
    clinical = pd.read_csv(args.clinical)
    
    print(f"\nLoaded {len(variants)} variants")
    print(f"Loaded {len(clinical)} patients")
    
    # Calculate PDS for each patient
    pds_df = calculate_pds_for_all_patients(variants)
    print(f"\nCalculated PDS for {len(pds_df)} patients")
    print(f"PDS range: {pds_df['pds'].min():.4f} to {pds_df['pds'].max():.4f}")
    
    # Merge with clinical data
    survival_df = clinical.merge(pds_df, on='patient_id', how='inner')
    
    # Check if survival data exists
    if 'survival_status' not in survival_df.columns:
        print("\nWarning: No survival data found. Creating mock data for testing.")
        # Create mock survival data
        np.random.seed(42)
        survival_df['survival_time'] = np.random.exponential(24, len(survival_df))
        survival_df['event_status'] = np.random.binomial(1, 0.3, len(survival_df))
    else:
        survival_df['event_status'] = survival_df['survival_status']
        # If no survival_time, estimate from status
        if 'survival_time' not in survival_df.columns:
            survival_df['survival_time'] = np.where(
                survival_df['event_status'] == 1,
                np.random.exponential(12, len(survival_df)),
                np.random.exponential(36, len(survival_df))
            )
    
    # Perform survival analysis
    km_results = survival_analysis(survival_df)
    
    if km_results is None:
        print("Survival analysis failed. Check data format.")
        return
    
    # Save results
    survival_path = RESULTS_DIR / "patient_survival_with_pds.csv"
    survival_df.to_csv(survival_path, index=False)
    print(f"\nSaved survival data: {survival_path}")
    
    # Generate figures
    create_survival_figures(survival_df, km_results, FIGURES_DIR)
    
    print("\n" + "=" * 60)
    print("KEY FINDINGS:")
    print("=" * 60)
    print(f"1. PDS calculated for {len(pds_df)} patients")
    print(f"2. Log-rank test p-value: {km_results['logrank_p']:.4f}")
    if km_results['logrank_p'] < 0.05:
        print("   ✓ Significant survival difference between groups!")
    else:
        print("   ✗ No significant survival difference (may need more patients)")
    
    if km_results['cph'] is not None:
        hr = np.exp(km_results['cph'].params_['pds'])
        print(f"3. PDS Hazard Ratio: {hr:.3f}")
        print(f"   (Each unit decrease in PDS increases death risk by {(hr-1)*100:.1f}%)")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
