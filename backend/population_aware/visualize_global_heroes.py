"""
Global Hero Variants Visualization

Creates 4-way dumbbell plot showing calibration impact
across African, European, South Asian, and East Asian populations.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def main():
    print("="*80)
    print("CREATING GLOBAL HERO VARIANTS VISUALIZATION")
    print("="*80)
    
    # Data from global calibration results
    data = [
        {
            'Pop': 'South Asian (SAS)', 
            'Variant': 'chr17:43051089T>C', 
            'Raw': -0.017945, 
            'Calib': 0.208545, 
            'AF': 'AF_SAS=0.145%',
            'Delta': 0.226
        },
        {
            'Pop': 'East Asian (EAS)',  
            'Variant': 'chr17:43124118T>C', 
            'Raw': -0.000506, 
            'Calib': 0.226214, 
            'AF': 'AF_EAS=0.154%',
            'Delta': 0.227
        },
        {
            'Pop': 'African (AFR)',     
            'Variant': 'chr17:43115791G>C', 
            'Raw': -0.003335, 
            'Calib': 0.224772, 
            'AF': 'AF_AFR=0.222%',
            'Delta': 0.228
        },
        {
            'Pop': 'European (EUR)',    
            'Variant': 'chr17:43047635C>T', 
            'Raw': -0.001280, 
            'Calib': 0.215006, 
            'AF': 'AF_EUR=0.015%',
            'Delta': 0.216
        }
    ]
    
    df_plot = pd.DataFrame(data)
    
    print("\n🏆 Global Hero Variants:")
    for _, row in df_plot.iterrows():
        print(f"   {row['Pop']}: {row['Variant']}")
        print(f"      Raw: {row['Raw']:.6f} → Calibrated: {row['Calib']:.6f}")
        print(f"      Correction: +{row['Delta']:.3f} ({row['AF']})")
    
    # Setup Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.set_style("whitegrid")
    
    # Color scheme for populations
    colors = {
        'South Asian (SAS)': '#FF6B6B',  # Red
        'East Asian (EAS)': '#4ECDC4',   # Teal
        'African (AFR)': '#FFD93D',      # Yellow
        'European (EUR)': '#6C5CE7'      # Purple
    }
    
    # Create Dumbbell lines
    for i, row in df_plot.iterrows():
        pop_color = colors.get(row['Pop'], 'black')
        
        # Draw line connecting Raw and Calibrated
        ax.plot([row['Raw'], row['Calib']], [i, i], 
                color='black', linewidth=2.5, zorder=1, alpha=0.6)
        
        # Add Arrow showing direction
        arrow_start = row['Raw']
        arrow_length = (row['Calib'] - row['Raw']) * 0.85
        ax.annotate('', xy=(arrow_start + arrow_length, i), 
                    xytext=(arrow_start, i),
                    arrowprops=dict(arrowstyle='->', lw=2.5, 
                                    color='black', alpha=0.6))
        
        # Add delta annotation
        mid_x = (row['Raw'] + row['Calib']) / 2
        ax.text(mid_x, i + 0.2, f'Δ = +{row["Delta"]:.3f}', 
                ha='center', va='bottom', fontsize=11, weight='bold',
                bbox=dict(boxstyle='round,pad=0.4', 
                          facecolor='yellow', alpha=0.8))
    
    # Plot Points
    for i, row in df_plot.iterrows():
        pop_color = colors.get(row['Pop'], 'black')
        
        # Raw score (circle)
        ax.scatter(row['Raw'], i, color='#95a5a6', s=300, 
                   marker='o', zorder=3, edgecolors='black', 
                   linewidths=2, label='Raw AI Score' if i == 0 else '')
        
        # Calibrated score (square)
        ax.scatter(row['Calib'], i, color=pop_color, s=300, 
                   marker='s', zorder=3, edgecolors='black', 
                   linewidths=2, label='Calibrated Score' if i == 0 else '')
    
    # Y-axis labels with variant IDs and AF
    y_labels = [f"{row['Pop']}\n{row['Variant']}\n({row['AF']})" 
                for _, row in df_plot.iterrows()]
    ax.set_yticks(range(len(df_plot)))
    ax.set_yticklabels(y_labels, fontsize=10)
    
    # X-axis
    ax.set_xlabel('Evo2 Pathogenicity Score\n(← More Pathogenic | More Benign →)', 
                  fontsize=12, weight='bold')
    
    # Title
    ax.set_title('Global Population-Aware Calibration: Rescuing False Positives\n'
                 'Hero Variants from 4 Global Populations', 
                 fontsize=14, weight='bold', pad=20)
    
    # Reference line at 0
    ax.axvline(x=0, color='gray', linestyle='--', 
               linewidth=2, alpha=0.7, label='Neutral Threshold')
    
    # Annotations
    ax.text(0.02, 0.98, '← Predicted Pathogenic', 
            transform=ax.transAxes, ha='left', va='top', 
            fontsize=10, style='italic', color='red', weight='bold')
    
    ax.text(0.98, 0.98, 'Predicted Benign →', 
            transform=ax.transAxes, ha='right', va='top', 
            fontsize=10, style='italic', color='green', weight='bold')
    
    # Legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), 
              loc='lower right', fontsize=11, 
              frameon=True, shadow=True, fancybox=True)
    
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    # Save
    output_path = "results/global_hero_variants.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    print(f"\n✅ Global hero variants visualization saved to {output_path}")
    
    plt.close()
    
    # Also create a summary table for the manuscript
    summary_df = df_plot[['Pop', 'Variant', 'Raw', 'Calib', 'Delta', 'AF']].copy()
    summary_df = summary_df.rename(columns={
        'Pop': 'Population',
        'Variant': 'Variant ID',
        'Raw': 'Raw Evo2 Score',
        'Calib': 'Calibrated Score',
        'Delta': 'Correction Magnitude',
        'AF': 'Population Frequency'
    })
    
    summary_df['Raw Evo2 Score'] = summary_df['Raw Evo2 Score'].apply(lambda x: f"{x:.6f}")
    summary_df['Calibrated Score'] = summary_df['Calibrated Score'].apply(lambda x: f"{x:.6f}")
    summary_df['Correction Magnitude'] = summary_df['Correction Magnitude'].apply(lambda x: f"+{x:.3f}")
    
    summary_df.to_csv("results/table4_global_hero_variants.csv", index=False)
    
    print(f"✅ Summary table saved to results/table4_global_hero_variants.csv")
    
    print("\n" + "="*80)
    print("✅ GLOBAL VISUALIZATION COMPLETE!")
    print("="*80)
    
    print("\n💡 Key Insight:")
    print("   All 4 hero variants show 21-23% score corrections,")
    print("   transforming pathogenic predictions into benign classifications")
    print("   when accounting for population-specific allele frequencies.")
    print("\n   This demonstrates the critical need for population-aware AI")
    print("   in clinical genomics to prevent false positives for diverse patients.")

if __name__ == "__main__":
    main()
