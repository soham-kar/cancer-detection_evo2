"""
Three-Population Calibration Analysis (AFR, EUR, EAS)

Enhanced population-aware calibration including East Asian populations
to demonstrate global applicability of the methodology.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_tri_population_bias(af_afr, af_eur, af_eas):
    """
    Calculate maximum population bias across three populations.
    
    Returns the largest fold-difference between any two populations,
    indicating the most extreme population-specific effect.
    """
    populations = {'AFR': af_afr, 'EUR': af_eur, 'EAS': af_eas}
    
    # Handle missing values
    populations = {k: (v if pd.notna(v) and v > 0 else 1e-10) 
                   for k, v in populations.items()}
    
    # Calculate all pairwise fold differences
    max_fold = 1
    bias_pair = None
    
    for pop1 in ['AFR', 'EUR', 'EAS']:
        for pop2 in ['AFR', 'EUR', 'EAS']:
            if pop1 < pop2:  # Avoid duplicates
                fold = max(populations[pop1], populations[pop2]) / min(populations[pop1], populations[pop2])
                if fold > max_fold:
                    max_fold = fold
                    bias_pair = f"{pop1} vs {pop2}"
    
    # Convert to 0-1 scale
    bias_score = np.log10(max_fold) / (np.log10(max_fold) + 2)
    
    return bias_score, bias_pair, max_fold

def create_three_population_heatmap(df):
    """
    Create heatmap showing AF patterns across three populations.
    """
    # Filter for variants with data in all three populations
    complete = df[(df['af_afr'] > 0) & (df['af_eur'] > 0) & (df['af_eas'] > 0)]
    
    if len(complete) == 0:
        print("⚠️ No variants with complete tri-population data")
        return
    
    print(f"\n📊 {len(complete)} variants with complete AFR+EUR+EAS data")
    
    # Prepare data for heatmap
    af_data = complete[['af_afr', 'af_eur', 'af_eas']].values
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Correlation heatmap
    ax = axes[0]
    corr_matrix = complete[['af_afr', 'af_eur', 'af_eas']].corr()
   
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdYlBu_r',
                center=0, vmin=-1, vmax=1, ax=ax,
                xticklabels=['African', 'European', 'East Asian'],
                yticklabels=['African', 'European', 'East Asian'],
                cbar_kws={'label': 'Correlation'})
    ax.set_title('Population Allele Frequency Correlations', fontsize=13, weight='bold')
    
    # Plot 2: Pairwise scatter
    ax = axes[1]
    
    # AFR vs EUR
    ax.scatter(complete['af_eur'], complete['af_afr'], 
               alpha=0.5, s=40, label='AFR vs EUR', color='#e74c3c')
    
    # EUR vs EAS  
    ax.scatter(complete['af_eur'], complete['af_eas'],
               alpha=0.5, s=40, label='EAS vs EUR', color='#3498db')
    
    # AFR vs EAS
    ax.scatter(complete['af_afr'], complete['af_eas'],
               alpha=0.5, s=40, label='EAS vs AFR', color='#2ecc71')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.plot([1e-6, 1], [1e-6, 1], 'k--', alpha=0.5, linewidth=2, label='Equal frequency')
    ax.set_xlabel('Reference Population AF', fontsize=12, weight='bold')
    ax.set_ylabel('Comparison Population AF', fontsize=12, weight='bold')
    ax.set_title('Pairwise Population AF Comparisons', fontsize=13, weight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    
    output_path = "results/three_population_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Three-population comparison saved to {output_path}")
    plt.close()

def analyze_population_specific_variants(df):
    """
    Find variants that are population-specific (present in one, rare/absent in others).
    """
    print("\n" + "="*80)
    print("POPULATION-SPECIFIC VARIANT ANALYSIS")
    print("="*80)
    
    threshold_common = 0.001  # 0.1% - consider "common"
    threshold_rare = 0.0001   # 0.01% - consider "rare/absent"
    
    # AFR-specific
    afr_specific = df[
        (df['af_afr'] > threshold_common) &
        (df['af_eur'] < threshold_rare) &
        (df['af_eas'] < threshold_rare)
    ]
    
    # EUR-specific
    eur_specific = df[
        (df['af_eur'] > threshold_common) &
        (df['af_afr'] < threshold_rare) &
        (df['af_eas'] < threshold_rare)
    ]
    
    # EAS-specific
    eas_specific = df[
        (df['af_eas'] > threshold_common) &
        (df['af_afr'] < threshold_rare) &
        (df['af_eur'] < threshold_rare)
    ]
    
    print(f"\n📊 Population-Specific Variants (>0.1% in one pop, <0.01% in others):")
    print(f"   AFR-specific: {len(afr_specific)}")
    print(f"   EUR-specific: {len(eur_specific)}")
    print(f"   EAS-specific: {len(eas_specific)}")
    
    # Show examples
    if len(afr_specific) > 0:
        print(f"\n🔍 Top AFR-specific variant:")
        top = afr_specific.nlargest(1, 'af_afr').iloc[0]
        print(f"   {top['variant_id']}: AF_AFR={top['af_afr']:.6f}, "
              f"AF_EUR={top.get('af_eur', 0):.6f}, AF_EAS={top.get('af_eas', 0):.6f}")
    
    if len(eas_specific) > 0:
        print(f"\n🔍 Top EAS-specific variant:")
        top = eas_specific.nlargest(1, 'af_eas').iloc[0]
        print(f"   {top['variant_id']}: AF_EAS={top['af_eas']:.6f}, "
              f"AF_AFR={top.get('af_afr', 0):.6f}, AF_EUR={top.get('af_eur', 0):.6f}")
    
    return {
        'AFR': afr_specific,
        'EUR': eur_specific,
        'EAS': eas_specific
    }

def main():
    print("="*80)
    print("THREE-POPULATION CALIBRATION ANALYSIS")
    print("="*80)
    
    # Load calibrated data
    df = pd.read_csv("results/brca1_evo2_calibrated.csv")
    df['variant_id'] = df['chrom'].astype(str) + ':' + df['pos_hg38'].astype(str) + df['ref'] + '>' + df['alt']
    
    # Filter for variants with any AF data
    df_with_af = df[(df['af_afr'].notna()) | (df['af_eur'].notna()) | (df['af_eas'].notna())]
    
    print(f"\n📊 Dataset Summary:")
    print(f"   Total variants: {len(df)}")
    print(f"   With any AF data: {len(df_with_af)}")
    print(f"   With AFR data: {df['af_afr'].notna().sum()}")
    print(f"   With EUR data: {df['af_eur'].notna().sum()}")
    print(f"   With EAS data: {df['af_eas'].notna().sum()}")
    
    # Calculate tri-population bias
    print("\n🔧 Calculating tri-population bias scores...")
    
    results = []
    for idx, row in df_with_af.iterrows():
        bias_score, bias_pair, max_fold = calculate_tri_population_bias(
            row.get('af_afr', 0),
            row.get('af_eur', 0),
            row.get('af_eas', 0)
        )
        results.append({
            'index': idx,
            'tri_pop_bias': bias_score,
            'bias_pair': bias_pair,
            'max_fold_diff': max_fold
        })
    
    df_bias = pd.DataFrame(results)
    df_with_af = df_with_af.merge(df_bias.set_index('index'), left_index=True, right_index=True, how='left')
    
    # Top tri-population bias variants
    print("\n🔍 TOP 5 VARIANTS WITH HIGHEST TRI-POPULATION BIAS:")
    print("-" * 100)
    print(f"{'Variant':<25} {'Bias':<8} {'Max Fold':<10} {'Comparison':<15} {'AF_AFR':<10} {'AF_EUR':<10} {'AF_EAS':<10}")
    print("-" * 100)
    
    top_bias = df_with_af.nlargest(5, 'tri_pop_bias')
    for _, row in top_bias.iterrows():
        print(f"{row['variant_id']:<25} {row['tri_pop_bias']:<8.3f} {row['max_fold_diff']:<10.1f} "
              f"{row.get('bias_pair', 'N/A'):<15} {row.get('af_afr', 0):<10.6f} "
              f"{row.get('af_eur', 0):<10.6f} {row.get('af_eas', 0):<10.6f}")
    
    # Create visualizations
    create_three_population_heatmap(df_with_af)
    
    # Find population-specific variants
    pop_specific = analyze_population_specific_variants(df_with_af)
    
    # Export summary
    summary = {
        'Total_Variants': len(df),
        'With_AF_Data': len(df_with_af),
        'High_TriPop_Bias_05': (df_with_af['tri_pop_bias'] > 0.5).sum(),
        'Extreme_TriPop_Bias_07': (df_with_af['tri_pop_bias'] > 0.7).sum(),
        'AFR_Specific': len(pop_specific['AFR']),
        'EUR_Specific': len(pop_specific['EUR']),
        'EAS_Specific': len(pop_specific['EAS']),
        'Mean_TriPop_Bias': df_with_af['tri_pop_bias'].mean()
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv("results/three_population_summary.csv", index=False)
    
    print("\n" + "="*80)
    print("✅ THREE-POPULATION ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\n📊 Summary:")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.3f}")
        else:
            print(f"   {key}: {value}")

if __name__ == "__main__":
    main()
