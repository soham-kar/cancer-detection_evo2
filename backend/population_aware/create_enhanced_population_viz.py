"""
Enhanced Three-Population Comparison Visualization

Creates publication-quality visualization showing allele frequency
relationships across global populations with improved aesthetics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

def main():
    print("="*80)
    print("CREATING ENHANCED GLOBAL POPULATION COMPARISON")
    print("="*80)
    
    # Load data
    df = pd.read_csv("results/brca1_global_calibration.csv")
    
    # Filter for variants with data in all populations
    df_complete = df[
        (df['af_afr'].notna()) & (df['af_afr'] > 0) &
        (df['af_nfe'].notna()) & (df['af_nfe'] > 0) &
        (df['af_eas'].notna()) & (df['af_eas'] > 0) &
        (df['af_sas'].notna()) & (df['af_sas'] > 0)
    ]
    
    # All variants with any data
    df_any = df[
        (df['af_afr'].notna()) | (df['af_nfe'].notna()) |
        (df['af_eas'].notna()) | (df['af_sas'].notna())
    ]
    
    print(f"\n📊 Data Summary:")
    print(f"   Complete (all 4 pops): {len(df_complete)} variants")
    print(f"   Any population data: {len(df_any)} variants")
    
    # Create enhanced figure
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)
    
    populations = {
        'AFR': {'col': 'af_afr', 'color': '#FFD93D', 'name': 'African'},
        'EUR': {'col': 'af_nfe', 'color': '#6C5CE7', 'name': 'European'},
        'EAS': {'col': 'af_eas', 'color': '#4ECDC4', 'name': 'East Asian'},
        'SAS': {'col': 'af_sas', 'color': '#FF6B6B', 'name': 'South Asian'}
    }
    
    # ===================================================================
    # Plot 1: Correlation Heatmap (Top Left)
    # ===================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    
    corr_data = df_any[['af_afr', 'af_nfe', 'af_eas', 'af_sas']].fillna(0).corr()
    
    sns.heatmap(corr_data, annot=True, fmt='.3f', cmap='RdYlBu_r',
                center=0, vmin=-1, vmax=1, ax=ax1,
                xticklabels=['AFR', 'EUR', 'EAS', 'SAS'],
                yticklabels=['AFR', 'EUR', 'EAS', 'SAS'],
                cbar_kws={'label': 'Pearson Correlation', 'shrink': 0.8},
                square=True, linewidths=1, linecolor='white')
    
    ax1.set_title('Population AF Correlation Matrix', fontsize=13, weight='bold', pad=10)
    
    # ===================================================================
    # Plot 2-4: Pairwise Scatter Plots (Top Middle & Right, Middle Left)
    # ===================================================================
    pairwise_comparisons = [
        ('EUR', 'AFR', gs[0, 1]),
        ('EUR', 'EAS', gs[0, 2]),
        ('EUR', 'SAS', gs[1, 0])
    ]
    
    for pop1, pop2, position in pairwise_comparisons:
        ax = fig.add_subplot(position)
        
        col1 = populations[pop1]['col']
        col2 = populations[pop2]['col']
        
        # Get data with both populations
        data = df_any[(df_any[col1] > 0) & (df_any[col2] > 0)]
        
        if len(data) > 0:
            # Scatter plot
            ax.scatter(data[col1], data[col2], 
                       alpha=0.6, s=60, 
                       color=populations[pop2]['color'],
                       edgecolors='black', linewidths=0.5)
            
            # Equal frequency line
            min_val = min(data[col1].min(), data[col2].min())
            max_val = max(data[col1].max(), data[col2].max())
            ax.plot([min_val, max_val], [min_val, max_val], 
                    'k--', alpha=0.5, linewidth=2, label='Equal frequency')
            
            # Log scale
            ax.set_xscale('log')
            ax.set_yscale('log')
            
            # Labels
            ax.set_xlabel(f'{populations[pop1]["name"]} AF', fontsize=11, weight='bold')
            ax.set_ylabel(f'{populations[pop2]["name"]} AF', fontsize=11, weight='bold')
            ax.set_title(f'{pop1} vs {pop2} (n={len(data)})', fontsize=12, weight='bold')
            
            ax.grid(True, alpha=0.3, which='both')
            ax.legend(fontsize=9, loc='upper left')
    
    # ===================================================================
    # Plot 5: Population-Specific Variant Distribution (Middle Center)
    # ===================================================================
    ax5 = fig.add_subplot(gs[1, 1])
    
    # Count population-specific variants
    threshold_common = 0.001  # 0.1%
    threshold_rare = 0.0001   # 0.01%
    
    pop_specific_counts = {}
    
    for pop_name, pop_info in populations.items():
        col = pop_info['col']
        other_cols = [p['col'] for k, p in populations.items() if k != pop_name]
        
        # Common in this pop, rare in others
        mask = (df_any[col] > threshold_common)
        for other_col in other_cols:
            mask &= (df_any[other_col].fillna(0) < threshold_rare)
        
        pop_specific_counts[pop_name] = mask.sum()
    
    bars = ax5.bar(range(len(pop_specific_counts)), 
                   list(pop_specific_counts.values()),
                   color=[populations[p]['color'] for p in pop_specific_counts.keys()],
                   edgecolor='black', linewidth=1.5, alpha=0.8)
    
    ax5.set_xticks(range(len(pop_specific_counts)))
    ax5.set_xticklabels(list(pop_specific_counts.keys()), fontsize=11, weight='bold')
    ax5.set_ylabel('Number of Variants', fontsize=11, weight='bold')
    ax5.set_title('Population-Specific Variants\n(>0.1% in one, <0.01% in others)', 
                  fontsize=12, weight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, pop_specific_counts.values())):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}',
                ha='center', va='bottom', fontsize=10, weight='bold')
    
    # ===================================================================
    # Plot 6: AF Distribution by Population (Middle Right)
    # ===================================================================
    ax6 = fig.add_subplot(gs[1, 2])
    
    violin_data = []
    for pop_name in ['AFR', 'EUR', 'EAS', 'SAS']:
        col = populations[pop_name]['col']
        values = df_any[df_any[col] > 0][col]
        if len(values) > 0:
            # Log transform for better visualization
            log_values = np.log10(values + 1e-10)
            for val in log_values:
                violin_data.append({'Population': pop_name, 'Log10(AF)': val})
    
    violin_df = pd.DataFrame(violin_data)
    
    sns.violinplot(data=violin_df, x='Population', y='Log10(AF)', 
                   palette=[populations[p]['color'] for p in ['AFR', 'EUR', 'EAS', 'SAS']],
                   ax=ax6, inner='quartile')
    
    ax6.set_ylabel('log₁₀(Allele Frequency)', fontsize=11, weight='bold')
    ax6.set_xlabel('Population', fontsize=11, weight='bold')
    ax6.set_title('AF Distribution Across Populations', fontsize=12, weight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    ax6.axhline(np.log10(0.001), color='red', linestyle='--', 
                linewidth=1, alpha=0.5, label='0.1% threshold')
    
    # ===================================================================
    # Plot 7: 3D Scatter (Bottom Left - EUR vs AFR vs EAS)
    # ===================================================================
    from mpl_toolkits.mplot3d import Axes3D
    
    ax7 = fig.add_subplot(gs[2, 0], projection='3d')
    
    data_3d = df_any[
        (df_any['af_nfe'] > 0) & 
        (df_any['af_afr'] > 0) & 
        (df_any['af_eas'] > 0)
    ]
    
    if len(data_3d) > 0:
        scatter = ax7.scatter(np.log10(data_3d['af_nfe'] + 1e-10),
                             np.log10(data_3d['af_afr'] + 1e-10),
                             np.log10(data_3d['af_eas'] + 1e-10),
                             c=data_3d['bias_eur'].fillna(0), 
                             cmap='RdYlBu_r', s=100, alpha=0.7,
                             edgecolors='black', linewidths=0.5)
        
        ax7.set_xlabel('log₁₀(AF EUR)', fontsize=10, weight='bold')
        ax7.set_ylabel('log₁₀(AF AFR)', fontsize=10, weight='bold')
        ax7.set_zlabel('log₁₀(AF EAS)', fontsize=10, weight='bold')
        ax7.set_title('3D Population AF Space\n(colored by EUR bias)', 
                      fontsize=11, weight='bold')
        
        cbar = plt.colorbar(scatter, ax=ax7, shrink=0.5, aspect=5)
        cbar.set_label('EUR Bias Score', fontsize=9)
    
    # ===================================================================
    # Plot 8: Fold-Difference Distribution (Bottom Center)
    # ===================================================================
    ax8 = fig.add_subplot(gs[2, 1])
    
    # Calculate max fold differences
    fold_diffs = []
    for _, row in df_any.iterrows():
        afs = [row.get(populations[p]['col'], 0) for p in populations.keys()]
        afs = [af if pd.notna(af) and af > 0 else 1e-10 for af in afs]
        if max(afs) > 1e-9:
            max_fold = max(afs) / min(afs)
            fold_diffs.append(np.log10(max_fold))
    
    ax8.hist(fold_diffs, bins=30, color='#3498db', alpha=0.7, 
             edgecolor='black', linewidth=1)
    ax8.set_xlabel('log₁₀(Maximum Fold Difference)', fontsize=11, weight='bold')
    ax8.set_ylabel('Number of Variants', fontsize=11, weight='bold')
    ax8.set_title('Population Frequency Divergence', fontsize=12, weight='bold')
    ax8.axvline(np.log10(10), color='orange', linestyle='--', 
                linewidth=2, label='10x threshold')
    ax8.axvline(np.log10(100), color='red', linestyle='--', 
                linewidth=2, label='100x threshold')
    ax8.legend(fontsize=9)
    ax8.grid(True, alpha=0.3, axis='y')
    
    # ===================================================================
    # Plot 9: Summary Statistics Table (Bottom Right)
    # ===================================================================
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    # Create summary stats
    summary_text = "📊 GLOBAL POPULATION SUMMARY\n\n"
    summary_text += f"Total Variants: {len(df)}\n"
    summary_text += f"With Population Data: {len(df_any)}\n\n"
    
    summary_text += "Population Coverage:\n"
    for pop_name, pop_info in populations.items():
        count = df_any[pop_info['col']].notna().sum()
        pct = count / len(df_any) * 100
        summary_text += f"  {pop_name}: {count} ({pct:.1f}%)\n"
    
    summary_text += f"\nComplete Data (all 4): {len(df_complete)}\n\n"
    
    summary_text += "Population-Specific:\n"
    for pop, count in pop_specific_counts.items():
        summary_text += f"  {pop}: {count} variants\n"
    
    summary_text += f"\nExtreme Divergence:\n"
    extreme = sum(1 for fd in fold_diffs if fd > np.log10(100))
    summary_text += f"  >100x: {extreme} variants\n"
    
    ax9.text(0.1, 0.95, summary_text, transform=ax9.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=1', facecolor='wheat', alpha=0.8))
    
    # Overall title
    fig.suptitle('Global Population Allele Frequency Analysis\nBRCA1 Variants Across 4 Major Ancestries', 
                 fontsize=16, weight='bold', y=0.98)
    
    # Save
    output_path = "results/global_population_analysis/three_population_comparison_enhanced.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"\n✅ Enhanced visualization saved to:")
    print(f"   {output_path}")
    
    # Also save to main results folder
    output_path2 = "results/three_population_comparison.png"
    plt.savefig(output_path2, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"   {output_path2}")
    
    plt.close()
    
    print("\n" + "="*80)
    print("✅ ENHANCED POPULATION COMPARISON COMPLETE!")
    print("="*80)
    
    print("\n📊 Visualization includes:")
    print("   1. Correlation heatmap (4x4)")
    print("   2. Pairwise scatter plots (EUR vs others)")
    print("   3. Population-specific variant counts")
    print("   4. AF distribution violin plots")
    print("   5. 3D scatter plot (EUR-AFR-EAS space)")
    print("   6. Fold-difference histogram")
    print("   7. Summary statistics panel")

if __name__ == "__main__":
    main()
