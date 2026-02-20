"""
Publication-Quality Statistical Panel Visualization

Multi-panel figure with:
- Panel labeling (A, B, C, D)
- Fixed correlation heatmap
- Constrained layout preventing overlap
- Consistent axis ranges
- High DPI for print quality
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.gridspec import GridSpec

def main():
    print("="*80)
    print("CREATING PUBLICATION-QUALITY STATISTICAL PANEL")
    print("="*80)
    
    # Load real data
    df = pd.read_csv("results/brca1_global_calibration.csv")
    
    print(f"\n📊 Loaded {len(df)} variants")
    
    # Filter for visualization
    df_with_af = df[
        (df['af_afr'].notna()) | (df['af_nfe'].notna()) |
        (df['af_sas'].notna()) | (df['af_eas'].notna())
    ]
    
    print(f"   Variants with AF data: {len(df_with_af)}")
    
    # Setup figure with GridSpec and constrained layout
    plt.rcParams.update({'font.size': 11, 'font.family': 'sans-serif'})
    
    fig = plt.figure(figsize=(16, 12), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # ==================================================================
    # PANEL A: Correlation Matrix (FIXED)
    # ==================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Calculate correlation on actual AF data
    af_cols = ['af_afr', 'af_nfe', 'af_sas', 'af_eas']
    corr_data = df_with_af[af_cols].fillna(0).corr()
    
    # Create heatmap with proper data
    sns.heatmap(corr_data, annot=True, fmt=".3f", 
                cmap='RdYlBu_r', center=0, vmin=-1, vmax=1,
                linewidths=1.5, linecolor='white',
                cbar_kws={'label': 'Pearson Correlation', 'shrink': 0.8},
                square=True, ax=ax1,
                xticklabels=['AFR', 'EUR', 'SAS', 'EAS'],
                yticklabels=['AFR', 'EUR', 'SAS', 'EAS'])
    
    ax1.set_title("Population AF Correlation Matrix", 
                  fontweight='bold', fontsize=13, pad=15)
    
    # Panel label
    ax1.text(-0.18, 1.05, "A", transform=ax1.transAxes, 
             size=22, weight='bold', va='top')
    
    # ==================================================================
    # PANEL B: EUR vs SAS Scatter (Key Population Divergence)
    # ==================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Get data with both populations
    scatter_data = df_with_af[
        (df_with_af['af_nfe'] > 0) & (df_with_af['af_sas'] > 0)
    ]
    
    if len(scatter_data) > 0:
        # Scatter plot on log scale
        ax2.scatter(scatter_data['af_nfe'], scatter_data['af_sas'],
                    alpha=0.6, color='#9b59b6', s=80, 
                    edgecolors='black', linewidths=0.5)
        
        # Equal frequency diagonal
        min_val = min(scatter_data['af_nfe'].min(), scatter_data['af_sas'].min())
        max_val = max(scatter_data['af_nfe'].max(), scatter_data['af_sas'].max())
        ax2.plot([min_val, max_val], [min_val, max_val], 
                 'k--', linewidth=2, alpha=0.6, label='Equal Frequency')
        
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        
        ax2.set_xlabel("European Allele Frequency (log)", 
                       fontweight='bold', fontsize=12)
        ax2.set_ylabel("South Asian Allele Frequency (log)", 
                       fontweight='bold', fontsize=12)
        ax2.set_title(f"Population Divergence: EUR vs SAS (n={len(scatter_data)})", 
                      fontweight='bold', fontsize=13, pad=15)
        
        ax2.legend(loc='upper left', frameon=True, fontsize=10)
        ax2.grid(True, which="both", ls="--", alpha=0.3)
    
    # Panel label
    ax2.text(-0.18, 1.05, "B", transform=ax2.transAxes, 
             size=22, weight='bold', va='top')
    
    # ==================================================================
    # PANEL C: Population Bias Distribution
    # ==================================================================
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Get bias scores (using EUR bias as example)
    bias_data = df_with_af['bias_eur'].dropna()
    
    if len(bias_data) > 0:
        # Histogram with KDE
        sns.histplot(bias_data, bins=25, kde=True, color='#3498db', 
                     ax=ax3, edgecolor='black', linewidth=0.5, alpha=0.7)
        
        # Threshold lines
        ax3.axvline(0.5, color='orange', linestyle='--', linewidth=2, 
                    label='High Bias (>0.5)', alpha=0.8)
        ax3.axvline(-0.5, color='orange', linestyle='--', linewidth=2, 
                    alpha=0.8)
        ax3.axvline(0.7, color='red', linestyle='--', linewidth=2, 
                    label='Extreme Bias (>0.7)', alpha=0.8)
        ax3.axvline(-0.7, color='red', linestyle='--', linewidth=2, 
                    alpha=0.8)
        
        ax3.set_xlabel("Population Bias Score", 
                       fontweight='bold', fontsize=12)
        ax3.set_ylabel("Number of Variants", 
                       fontweight='bold', fontsize=12)
        ax3.set_title("Distribution of EUR Population Bias", 
                      fontweight='bold', fontsize=13, pad=15)
        
        ax3.legend(loc='upper right', frameon=True, fontsize=10)
        ax3.grid(True, alpha=0.3, axis='y')
    
    # Panel label
    ax3.text(-0.18, 1.05, "C", transform=ax3.transAxes, 
             size=22, weight='bold', va='top')
    
    # ==================================================================
    # PANEL D: AF Distribution by Population (Violin Plot)
    # ==================================================================
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Prepare data for violin plot
    violin_data = []
    for pop, col in [('AFR', 'af_afr'), ('EUR', 'af_nfe'), 
                     ('SAS', 'af_sas'), ('EAS', 'af_eas')]:
        values = df_with_af[df_with_af[col] > 0][col]
        for val in values:
            if val > 0:
                violin_data.append({
                    'Population': pop,
                    'log10_AF': np.log10(val)
                })
    
    violin_df = pd.DataFrame(violin_data)
    
    if len(violin_df) > 0:
        # Violin plot
        sns.violinplot(data=violin_df, x='Population', y='log10_AF',
                       palette=['#FFD93D', '#6C5CE7', '#FF6B6B', '#4ECDC4'],
                       ax=ax4, inner='quartile')
        
        ax4.set_ylabel("log₁₀(Allele Frequency)", 
                       fontweight='bold', fontsize=12)
        ax4.set_xlabel("Population", fontweight='bold', fontsize=12)
        ax4.set_title("AF Distribution Across Populations", 
                      fontweight='bold', fontsize=13, pad=15)
        
        # Reference line for common variant threshold (0.1%)
        ax4.axhline(np.log10(0.001), color='red', linestyle='--', 
                    linewidth=1.5, alpha=0.5, label='0.1% threshold')
        
        ax4.legend(loc='upper right', frameon=True, fontsize=10)
        ax4.grid(True, alpha=0.3, axis='y')
    
    # Panel label
    ax4.text(-0.18, 1.05, "D", transform=ax4.transAxes, 
             size=22, weight='bold', va='top')
    
    # Overall title removed for cleaner publication look
    # fig.suptitle('Global Population Genome Analysis: BRCA1 Variants', 
    #              fontsize=17, fontweight='bold', y=0.995)
    
    # Save high-resolution outputs
    output_paths = [
        "results/global_population_analysis/FINAL_statistical_panel.png",
        "results/FINAL_statistical_panel.png"
    ]
    
    for output_path in output_paths:
        plt.savefig(output_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"✅ Saved to: {output_path}")
    
    plt.close()
    
    print("\n" + "="*80)
    print("✅ PUBLICATION-QUALITY STATISTICAL PANEL COMPLETE!")
    print("="*80)
    
    print("\n📊 Panel Contents:")
    print("   A: Correlation matrix (4x4) - Shows AF relationships")
    print("   B: EUR vs SAS scatter - Key divergence example")
    print("   C: Bias distribution - Statistical overview")
    print("   D: AF violin plots - Distribution by population")

if __name__ == "__main__":
    main()
