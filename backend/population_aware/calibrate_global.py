"""
Global 4-Population Calibration System

Calibrates Evo2 scores for African (AFR), European (EUR), 
South Asian (SAS), and East Asian (EAS) populations.

Methodology:
- Each population compared against average of all others
- Bias score quantifies population-specific frequency differences
- Calibration adjusts scores to reduce false positives in underrepresented populations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_global_bias(row, target_pop, other_pops):
    """
    Calculate bias for target population vs others.
    
    Args:
        row: DataFrame row with AF columns
        target_pop: Target population AF column name
        other_pops: List of other population AF column names
    
    Returns:
        float: Bias score (0-1)
    """
    target_af = row.get(target_pop, 0)
    target_af = target_af if pd.notna(target_af) else 0
    
    # Calculate mean AF in other populations
    other_afs = [row.get(p, 0) for p in other_pops]
    other_afs = [af if pd.notna(af) else 0 for af in other_afs]
    rest_af = np.mean(other_afs) if other_afs else 0
    
    # Calculate bias
    epsilon = 1e-9
    if target_af == 0 and rest_af == 0:
        return 0, 0  # No bias, no difference
    
    # Fold difference
    max_af = max(target_af, rest_af)
    min_af = min(target_af, rest_af) + epsilon
    fold_diff = max_af / min_af
    
    # Bias score (0-1 scale)
    bias = np.log10(fold_diff) / (np.log10(fold_diff) + 2)
    
    # Direction: positive if target > rest (common here, rare elsewhere)
    direction = 1 if target_af > rest_af else -1
    
    return bias * direction, fold_diff

def calibrate_for_population(evo2_score, bias_score, calibration_strength=0.3):
    """
    Calibrate score based on population-specific bias.
    
    Args:
        evo2_score: Raw Evo2 score
        bias_score: Population bias score (with direction)
        calibration_strength: Adjustment weight (0.3 = 30% adjustment)
    
    Returns:
        float: Calibrated score
    """
    # If variant is common in this population (positive bias),
    # shift score towards benign (less negative/more positive)
    adjustment = bias_score * calibration_strength
    
    return evo2_score + adjustment

def main():
    print("="*80)
    print("GLOBAL 4-POPULATION CALIBRATION SYSTEM")
    print("="*80)
    
    # Load data
    df = pd.read_csv("results/brca1_evo2_calibrated.csv")
    
    print(f"\n📊 Loaded {len(df)} BRCA1 variants")
    
    # Define populations
    populations = {
        'AFR': 'af_afr',
        'EUR': 'af_nfe',  # Non-Finnish European
        'SAS': 'af_sas',  # South Asian
        'EAS': 'af_eas'   # East Asian
    }
    
    print(f"\n🌍 Calibrating for {len(populations)} global populations:")
    for pop, col in populations.items():
        if col in df.columns:
            n_with_data = df[col].notna().sum()
            print(f"   {pop}: {n_with_data} variants with data")
        else:
            print(f"   ⚠️  {pop}: Column {col} not found!")
    
    # Calculate bias and calibration for each population
    print("\n🔧 Calculating population-specific bias scores...")
    
    for pop_name, pop_col in populations.items():
        if pop_col not in df.columns:
            print(f"   ⚠️  Skipping {pop_name} (no data)")
            continue
        
        # Other populations for comparison
        other_pops = [col for name, col in populations.items() 
                      if col != pop_col and col in df.columns]
        
       # Calculate bias for each variant
        bias_results = df.apply(
            lambda row: calculate_global_bias(row, pop_col, other_pops),
            axis=1
        )
        
        df[f'bias_{pop_name.lower()}'] = [b[0] for b in bias_results]
        df[f'fold_{pop_name.lower()}'] = [b[1] for b in bias_results]
        
        # Calibrate score
        df[f'calibrated_{pop_name.lower()}'] = df.apply(
            lambda row: calibrate_for_population(
                row['evo2_score'],
                row[f'bias_{pop_name.lower()}']
            ),
            axis=1
        )
        
        # Statistics
        high_bias = (df[f'bias_{pop_name.lower()}'].abs() > 0.5).sum()
        mean_bias = df[f'bias_{pop_name.lower()}'].mean()
        
        print(f"   ✅ {pop_name}: {high_bias} high-bias variants, mean bias = {mean_bias:.3f}")
    
    # Save global calibration
    output_file = "results/brca1_global_calibration.csv"
    df.to_csv(output_file, index=False)
    print(f"\n✅ Global calibration saved to {output_file}")
    
    # ===================================================================
    # SUMMARY STATISTICS
    # ===================================================================
    print("\n" + "="*80)
    print("📊 GLOBAL CALIBRATION SUMMARY")
    print("="*80)
    
    with_af = df[df['af_nfe'].notna() | df['af_afr'].notna() | 
                  df['af_sas'].notna() | df['af_eas'].notna()]
    
    print(f"\nTotal variants: {len(df)}")
    print(f"With any population data: {len(with_af)}")
    
    for pop in ['AFR', 'EUR', 'SAS', 'EAS']:
        if f'bias_{pop.lower()}' in df.columns:
            high_pos_bias = (df[f'bias_{pop.lower()}'] > 0.5).sum()
            high_neg_bias = (df[f'bias_{pop.lower()}'] < -0.5).sum()
            
            print(f"\n{pop}:")
            print(f"  High positive bias (common here): {high_pos_bias}")
            print(f"  High negative bias (rare here): {high_neg_bias}")
            
            # Calculate mean adjustment
            adjustments = (df[f'calibrated_{pop.lower()}'] - df['evo2_score']).abs()
            print(f"  Mean adjustment magnitude: {adjustments.mean():.6f}")
            print(f"  Max adjustment: {adjustments.max():.6f}")
    
    # ===================================================================
    # POPULATION-SPECIFIC HERO VARIANTS
    # ===================================================================
    print("\n" + "="*80)
    print("🏆 POPULATION-SPECIFIC HERO VARIANTS")
    print("="*80)
    
    heroes = {}
    
    for pop in ['AFR', 'EUR', 'SAS', 'EAS']:
        bias_col = f'bias_{pop.lower()}'
        if bias_col not in df.columns:
            continue
        
        # Find top positive bias (common in this pop, rare elsewhere)
        top = df.nlargest(1, bias_col)
        
        if len(top) > 0 and top.iloc[0][bias_col] > 0.3:
            heroes[pop] = top.iloc[0]
            
            print(f"\n🌍 {pop} Hero Variant:")
            variant = heroes[pop]
            print(f"   Variant: {variant.get('chrom', '')}:{variant.get('pos_hg38', '')}"
                  f"{variant.get('ref', '')}>{variant.get('alt', '')}")
            print(f"   Bias Score: {variant[bias_col]:.3f}")
            print(f"   Fold Difference: {variant[f'fold_{pop.lower()}']:.1f}x")
            print(f"   Raw Evo2: {variant['evo2_score']:.6f}")
            print(f"   Calibrated: {variant[f'calibrated_{pop.lower()}']:.6f}")
            
            # Show AFs
            for p in ['AFR', 'EUR', 'SAS', 'EAS']:
                col = populations.get(p)
                if col and col in df.columns:
                    af = variant.get(col, 0)
                    print(f"   AF_{p}: {af:.6f}")
    
    # ===================================================================
    # VISUALIZATION
    # ===================================================================
    print("\n📊 Creating global calibration visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot bias distributions for each population
    for idx, pop in enumerate(['AFR', 'EUR', 'SAS', 'EAS']):
        ax = axes[idx // 2, idx % 2]
        bias_col = f'bias_{pop.lower()}'
        
        if bias_col in df.columns:
            df[bias_col].hist(bins=50, ax=ax, alpha=0.7, color=['#e74c3c', '#3498db', '#2ecc71', '#f39c12'][idx])
            ax.axvline(0, color='black', linestyle='--', linewidth=2, label='Neutral')
            ax.axvline(0.5, color='red', linestyle='--', linewidth=1, label='High bias')
            ax.axvline(-0.5, color='red', linestyle='--', linewidth=1)
            ax.set_xlabel('Population Bias Score', fontsize=11, weight='bold')
            ax.set_ylabel('Frequency', fontsize=11, weight='bold')
            ax.set_title(f'{pop} Population Bias Distribution', fontsize=12, weight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_path = "results/global_population_bias_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Global bias distributions saved to {output_path}")
    plt.close()
    
    print("\n" + "="*80)
    print("✅ GLOBAL 4-POPULATION CALIBRATION COMPLETE!")
    print("="*80)
    
    print("\n💡 Key Outputs:")
    print("   1. brca1_global_calibration.csv - Full calibration results")
    print("   2. global_population_bias_distribution.png - Bias distributions")
    print("\n   Each variant now has 4 calibrated scores:")
    print("      - calibrated_afr (African-optimized)")
    print("      - calibrated_eur (European-optimized)")
    print("      - calibrated_sas (South Asian-optimized)")
    print("      - calibrated_eas (East Asian-optimized)")

if __name__ == "__main__":
    main()
