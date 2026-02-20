"""
Hero Variant Visualization - Publication Figure

Creates dumbbell plot showing before/after calibration impact
on key variants that demonstrate population bias correction.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

def create_hero_variant_plot():
    """
    Create publication-ready dumbbell plot showing calibration impact.
    """
    
    # Load case studies (top variants identified by analysis)
    df_cases = pd.read_csv("results/calibration_case_studies.csv")
    
    # Select top 2 most impactful variants (one AFR, one EUR)
    top_afr = df_cases.nlargest(1, 'delta_afr').iloc[0]
    top_eur = df_cases.nlargest(1, 'delta_eur').iloc[0]
    
    print("="*80)
    print("CREATING HERO VARIANT VISUALIZATION")
    print("="*80)
    
    print(f"\n📊 Top AFR Impact:")
    print(f"   Variant: {top_afr['variant_id']}")
    print(f"   Raw Score: {top_afr['evo2_score']:.6f}")
    print(f"   Calibrated: {top_afr['calibrated_afr']:.6f}")
    print(f"   Delta: {top_afr['delta_afr']:.6f}")
    
    print(f"\n📊 Top EUR Impact:")
    print(f"   Variant: {top_eur['variant_id']}")
    print(f"   Raw Score: {top_eur['evo2_score']:.6f}")
    print(f"   Calibrated: {top_eur['calibrated_eur']:.6f}")
    print(f"   Delta: {top_eur['delta_eur']:.6f}")
    
    # Create plot data
    plot_data = []
    
    # AFR variant
    afr_label = f"{top_afr['variant_id']}\n(AF_AFR={top_afr['af_afr']:.4f}, AF_EUR={top_afr['af_eur']:.4f})"
    plot_data.extend([
        {'Variant': afr_label, 'Type': 'Raw Evo2', 'Score': top_afr['evo2_score'], 'Population': 'AFR'},
        {'Variant': afr_label, 'Type': 'Calibrated', 'Score': top_afr['calibrated_afr'], 'Population': 'AFR'}
    ])
    
    # EUR variant  
    eur_label = f"{top_eur['variant_id']}\n(AF_AFR={top_eur['af_afr']:.4f}, AF_EUR={top_eur['af_eur']:.4f})"
    plot_data.extend([
        {'Variant': eur_label, 'Type': 'Raw Evo2', 'Score': top_eur['evo2_score'], 'Population': 'EUR'},
        {'Variant': eur_label, 'Type': 'Calibrated', 'Score': top_eur['calibrated_eur'], 'Population': 'EUR'}
    ])
    
    df_plot = pd.DataFrame(plot_data)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.set_style("whitegrid")
    
    # Draw connecting lines (dumbbell handles)
    variants = df_plot['Variant'].unique()
    y_positions = {v: i for i, v in enumerate(variants)}
    
    for variant in variants:
        subset = df_plot[df_plot['Variant'] == variant]
        raw = subset[subset['Type'] == 'Raw Evo2']['Score'].values[0]
        calib = subset[subset['Type'] == 'Calibrated']['Score'].values[0]
        y = y_positions[variant]
        
        # Draw line
        ax.plot([raw, calib], [y, y], color='black', linewidth=2.5, zorder=1, alpha=0.6)
        
        # Add arrow showing direction
        arrow_start = raw
        arrow_length = (calib - raw) * 0.85
        ax.annotate('', xy=(arrow_start + arrow_length, y), xytext=(arrow_start, y),
                    arrowprops=dict(arrowstyle='->', lw=2.5, color='black', alpha=0.6))
        
        # Add delta annotation
        delta = abs(calib - raw)
        mid_x = (raw + calib) / 2
        ax.text(mid_x, y + 0.15, f'Δ = {delta:.3f}', 
                ha='center', va='bottom', fontsize=11, weight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # Draw points
    colors = {'Raw Evo2': '#95a5a6', 'Calibrated': '#e74c3c'}
    for idx, row in df_plot.iterrows():
        y = y_positions[row['Variant']]
        color = colors[row['Type']]
        marker = 'o' if row['Type'] == 'Raw Evo2' else 's'
        ax.scatter(row['Score'], y, s=250, color=color, marker=marker, 
                   zorder=3, edgecolors='black', linewidths=1.5,
                   label=row['Type'] if idx < 2 else '')
    
    # Formatting
    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels(list(y_positions.keys()), fontsize=10)
    ax.set_xlabel('Evo2 Pathogenicity Score\n(More Negative = More Pathogenic)', fontsize=12, weight='bold')
    ax.set_title('Population-Aware Calibration Impact on "Hero" Variants', 
                 fontsize=14, weight='bold', pad=20)
    
    # Add reference line at 0
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5, linewidth=1.5, label='Neutral')
    
    # Add population context annotations
    ax.text(0.02, 0.98, '← More Pathogenic', transform=ax.transAxes,
            ha='left', va='top', fontsize=10, style='italic', color='red')
    ax.text(0.98, 0.98, 'More Benign →', transform=ax.transAxes,
            ha='right', va='top', fontsize=10, style='italic', color='green')
    
    # Legend
    handles, labels = ax.get_legend_handles_labels()
    # Remove duplicates
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), 
              loc='upper right', fontsize=11, frameon=True, shadow=True)
    
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    
    # Save
    output_path = "results/hero_variant_shift.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Hero variant visualization saved to {output_path}")
    plt.close()
    
    return output_path

def create_bias_distribution_plot():
    """
    Create supplementary figure showing population bias distribution.
    """
    df = pd.read_csv("results/brca1_evo2_calibrated.csv")
    df_with_af = df[df['pop_bias'] > 0]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Bias distribution
    ax = axes[0]
    ax.hist(df_with_af['pop_bias'], bins=30, color='#3498db', alpha=0.7, edgecolor='black')
    ax.axvline(0.5, color='orange', linestyle='--', linewidth=2, label='High bias threshold')
    ax.axvline(0.7, color='red', linestyle='--', linewidth=2, label='Extreme bias threshold')
    ax.set_xlabel('Population Bias Score', fontsize=12, weight='bold')
    ax.set_ylabel('Number of Variants', fontsize=12, weight='bold')
    ax.set_title('Distribution of Population Bias Scores', fontsize=13, weight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: AF correlation
    ax = axes[1]
    valid = df_with_af[(df_with_af['af_afr'] > 0) & (df_with_af['af_eur'] > 0)]
    
    if len(valid) > 0:
        ax.scatter(valid['af_eur'], valid['af_afr'], alpha=0.6, s=50, color='#9b59b6')
        ax.plot([1e-6, 1], [1e-6, 1], 'k--', alpha=0.5, linewidth=2, label='Equal frequency')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('EUR Allele Frequency', fontsize=12, weight='bold')
        ax.set_ylabel('AFR Allele Frequency', fontsize=12, weight='bold')
        ax.set_title('Population Allele Frequency Correlation', fontsize=13, weight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    
    output_path = "results/population_bias_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Bias distribution plot saved to {output_path}")
    plt.close()
    
    return output_path

def main():
    print("\n" + "="*80)
    print("GENERATING PUBLICATION FIGURES")
    print("="*80 + "\n")
    
    # Create main hero variant plot
    fig1 = create_hero_variant_plot()
    
    # Create supplementary bias distribution plot
    fig2 = create_bias_distribution_plot()
    
    print("\n" + "="*80)
    print("✅ ALL PUBLICATION FIGURES GENERATED!")
    print("="*80)
    print(f"\n📊 Figures created:")
    print(f"   1. {fig1} - Main hero variant comparison")
    print(f"   2. {fig2} - Population bias distribution")

if __name__ == "__main__":
    main()
