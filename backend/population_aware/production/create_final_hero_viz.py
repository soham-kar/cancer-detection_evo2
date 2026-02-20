"""
Publication-Quality Global Hero Variants Visualization

Enhanced dumbbell plot with:
- Multi-line Y-axis labels (Population + Variant + AF)
- Proper spacing and non-overlapping annotations
- Professional aesthetics with constrained layout
- High DPI for print quality
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

def main():
    print("="*80)
    print("CREATING PUBLICATION-QUALITY HERO VARIANTS VISUALIZATION")
    print("="*80)
    
    # Data from global calibration results
    data = [
        {
            'Pop': 'South Asian (SAS)', 
            'Variant': 'chr17:43051089T>C', 
            'AF': '0.145%',
            'Raw': -0.017945, 
            'Calib': 0.208545, 
            'Delta': 0.226
        },
        {
            'Pop': 'East Asian (EAS)',  
            'Variant': 'chr17:43124118T>C', 
            'AF': '0.154%',
            'Raw': -0.000506, 
            'Calib': 0.226214, 
            'Delta': 0.227
        },
        {
            'Pop': 'African (AFR)',     
            'Variant': 'chr17:43115791G>C', 
            'AF': '0.222%',
            'Raw': -0.003335, 
            'Calib': 0.224772, 
            'Delta': 0.228
        },
        {
            'Pop': 'European (EUR)',    
            'Variant': 'chr17:43047635C>T', 
            'AF': '0.015%',
            'Raw': -0.001280, 
            'Calib': 0.215006, 
            'Delta': 0.216
        }
    ]
    
    df = pd.DataFrame(data)
    
    print(f"\n📊 Creating visualization for {len(df)} hero variants...")
    
    # Set global font sizes for readability
    plt.rcParams.update({
        'font.size': 12, 
        'font.family': 'sans-serif',
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11
    })
    
    # Create figure with constrained layout to prevent overlap
    fig, ax = plt.subplots(figsize=(14, 7), constrained_layout=True)
    
    # Create multi-line Y-labels (Population + Variant + AF)
    y_labels = []
    for _, row in df.iterrows():
        # Format: Population (bold) / Variant (AF)
        label = f"{row['Pop']}\n{row['Variant']}\n(AF: {row['AF']})"
        y_labels.append(label)
    
    # Draw dumbbells
    for i, row in df.iterrows():
        # The connecting line
        ax.plot([row['Raw'], row['Calib']], [i, i], 
                color='#34495e', linewidth=3, zorder=1, solid_capstyle='round')
        
        # Arrow head showing direction
        arrow_start = row['Raw']
        arrow_length = (row['Calib'] - row['Raw']) * 0.92
        ax.annotate('', xy=(arrow_start + arrow_length, i), 
                    xytext=(arrow_start, i),
                    arrowprops=dict(arrowstyle='->', lw=3, color='#34495e'),
                    zorder=1)
        
        # Delta annotation box (uniform positioning)
        mid_point = (row['Raw'] + row['Calib']) / 2
        ax.text(mid_point, i + 0.32, f"Δ = +{row['Delta']:.3f}", 
                ha='center', va='bottom', 
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.4", 
                          facecolor='#f1c40f', 
                          edgecolor='none', 
                          alpha=0.9),
                zorder=3)
    
    # Scatter points (circles for raw, squares for calibrated)
    ax.scatter(df['Raw'], range(len(df)), 
               color='#95a5a6', s=350, marker='o',
               label='Raw AI Score', zorder=2, 
               edgecolors='black', linewidths=2)
    
    ax.scatter(df['Calib'], range(len(df)), 
               color='#2ecc71', s=350, marker='s',
               label='Calibrated Score', zorder=2, 
               edgecolors='black', linewidths=2)
    
    # Y-axis formatting
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(y_labels, fontsize=10.5, linespacing=1.2)
    
    # X-axis formatting
    ax.set_xlabel("Pathogenicity Score", fontsize=13, fontweight='bold', labelpad=10)
    
    # Title
    ax.set_title("Global Population-Aware Calibration: Rescuing False Positives\nHero Variants from 4 Global Ancestries", 
                 fontsize=15, fontweight='bold', pad=20)
    
    # Reference line at zero
    ax.axvline(0, color='gray', linestyle='--', linewidth=2, alpha=0.7, zorder=0)
    ax.text(0, -0.65, "Neutral\nThreshold", 
            ha='center', va='top', fontsize=10, color='gray', 
            style='italic', weight='bold')
    
    # Directional labels on X-axis
    x_min = df['Raw'].min() - 0.01
    x_max = df['Calib'].max() + 0.01
    
    ax.text(x_min, -0.85, "← Predicted Pathogenic", 
            color='#c0392b', ha='left', va='top',
            fontsize=11, fontweight='bold', style='italic')
    
    ax.text(x_max, -0.85, "Predicted Benign →", 
            color='#27ae60', ha='right', va='top',
            fontsize=11, fontweight='bold', style='italic')
    
    # Legend outside plot area
    ax.legend(loc='lower right', frameon=True, fontsize=12, 
              shadow=True, fancybox=True, framealpha=0.95)
    
    # Grid for readability
    ax.grid(True, alpha=0.3, axis='x', linestyle=':', linewidth=1)
    
    # Set appropriate limits
    ax.set_xlim(x_min - 0.005, x_max + 0.005)
    ax.set_ylim(-1.1, len(df) - 0.5)
    
    # Save high-resolution output
    output_paths = [
        "results/global_population_analysis/FINAL_global_hero_variants.png",
        "results/FINAL_global_hero_variants.png"
    ]
    
    for output_path in output_paths:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        print(f"✅ Saved to: {output_path}")
    
    plt.close()
    
    print("\n" + "="*80)
    print("✅ PUBLICATION-QUALITY HERO VARIANTS VISUALIZATION COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()
