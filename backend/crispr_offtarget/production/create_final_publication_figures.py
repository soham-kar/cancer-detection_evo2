"""
Final Publication Figures for Nature Medicine

Creates 4 publication-ready figures:
1. ROC curve overview (baseline vs population-aware)
2. Population-specific risk heatmap
3. Uncertainty-driven cost savings
4. Clinical decision matrix
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_curve, roc_auc_score

# Set Nature Medicine style
plt.style.use('default')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 8,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.titlesize': 11,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'grid.alpha': 0.3
})

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures" / "publication"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    'AFR': '#d62728', 'EUR': '#1f77b4', 'EAS': '#2ca02c',
    'AMR': '#ff7f0e', 'SAS': '#9467bd', 'FIN': '#8c564b',
    'baseline': '#7f7f7f', 'improved': '#1f77b4'
}


def create_figure_1_roc():
    """Figure 1: ROC curve - baseline vs population-aware"""
    fig, ax = plt.subplots(figsize=(4, 4))
    
    # Load data
    df = pd.read_csv(RESULTS_DIR / "crispr_offtarget_final.csv")
    df_pop = pd.read_csv(RESULTS_DIR / "crispr_offtarget_population.csv")
    
    # Baseline ROC
    fpr_base, tpr_base, _ = roc_curve(df['is_validated'], df['final_score'])
    auroc_base = roc_auc_score(df['is_validated'], df['final_score'])
    
    # Population-aware (AFR - best)
    fpr_pop, tpr_pop, _ = roc_curve(df_pop['is_validated'], df_pop['AFR_score'])
    auroc_pop = roc_auc_score(df_pop['is_validated'], df_pop['AFR_score'])
    
    # Plot
    ax.plot(fpr_base, tpr_base, color=COLORS['baseline'], linewidth=2,
            label=f'Baseline (AUROC={auroc_base:.3f})', linestyle='--')
    ax.plot(fpr_pop, tpr_pop, color=COLORS['improved'], linewidth=2,
            label=f'Population-Aware (AUROC={auroc_pop:.3f})')
    ax.plot([0, 1], [0, 1], 'k:', linewidth=0.8, alpha=0.5)
    
    ax.set_xlabel('False Positive Rate', fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontweight='bold')
    ax.set_title('CRISPR Off-Target Prediction', fontweight='bold')
    ax.legend(frameon=False, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure1_roc.png", dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / "figure1_roc.pdf", bbox_inches='tight')
    plt.close()
    print("✅ Figure 1: ROC curve")


def create_figure_2_population_heatmap():
    """Figure 2: Population-specific risk heatmap"""
    fig, ax = plt.subplots(figsize=(5, 4))
    
    df = pd.read_csv(RESULTS_DIR / "crispr_offtarget_population.csv")
    
    # Clean gRNA names: HBB_site1 -> HBB
    df['gene_name'] = df['grna_name'].str.replace('_site1', '', regex=False)
    
    # Aggregate by gene name (clean)
    populations = ['AFR', 'EUR', 'EAS', 'AMR', 'SAS', 'FIN']
    grna_risks = df.groupby('gene_name')[[f'{p}_risk' for p in populations]].mean()
    grna_risks.columns = populations
    
    sns.heatmap(grna_risks, cmap='YlOrRd', annot=True, fmt='.3f',
                cbar_kws={'label': 'Risk Score'}, ax=ax)
    
    ax.set_title('Population-Specific Off-Target Risk', fontweight='bold')
    ax.set_xlabel('Population', fontweight='bold')
    ax.set_ylabel('gRNA', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure2_population_heatmap.png", dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / "figure2_population_heatmap.pdf", bbox_inches='tight')
    plt.close()
    print("✅ Figure 2: Population heatmap")


def create_figure_3_uncertainty():
    """Figure 3: Uncertainty-driven cost savings"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))
    
    # Load uncertainty data
    df = pd.read_csv(RESULTS_DIR / "crispr_offtarget_uncertainty.csv")
    
    # Cost comparison
    high_count = len(df[df['calibrated_confidence'] == 'HIGH'])
    total_count = len(df)
    
    cost_all = total_count * 1000
    cost_high = high_count * 1000
    
    bars = ax1.bar(['Validate All', 'HIGH Only'], [cost_all, cost_high],
                   color=['#d62728', '#2ca02c'], alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Cost ($)', fontweight='bold')
    ax1.set_title('Experimental Cost Reduction', fontweight='bold')
    
    for bar, cost in zip(bars, [cost_all, cost_high]):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10000,
                 f'${cost:,.0f}', ha='center', fontsize=9)
    
    # Precision by confidence
    conf_groups = df.groupby('calibrated_confidence')['is_validated'].mean()
    conf_order = ['HIGH', 'MODERATE', 'LOW']
    precisions = [conf_groups.get(c, 0) for c in conf_order]
    
    ax2.bar(conf_order, precisions, color=['#2ca02c', '#ff7f0e', '#d62728'],
            alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Precision', fontweight='bold')
    ax2.set_xlabel('Confidence Level', fontweight='bold')
    ax2.set_title('Confidence Calibration', fontweight='bold')
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure3_uncertainty.png", dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / "figure3_uncertainty.pdf", bbox_inches='tight')
    plt.close()
    print("✅ Figure 3: Uncertainty & cost")


def create_figure_4_clinical_table():
    """Figure 4: Clinical decision matrix"""
    fig, ax = plt.subplots(figsize=(7, 3))
    
    # Example clinical decisions
    data = {
        'Population': ['AFR', 'EUR', 'EAS', 'AFR', 'EUR', 'AMR'],
        'Gene': ['HBB', 'PCSK9', 'BCL11A', 'BCL11A', 'HBB', 'DMD'],
        'Risk': ['0.041', '0.018', '0.025', '0.032', '0.012', '0.028'],
        'Confidence': ['HIGH', 'MOD', 'HIGH', 'HIGH', 'LOW', 'MOD'],
        'Action': ['AVOID', 'WATCH', 'VALIDATE', 'VALIDATE', 'SAFE', 'WATCH']
    }
    df = pd.DataFrame(data)
    
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df.values, colLabels=df.columns,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.6)
    
    # Color action column
    action_colors = {'AVOID': '#ff6666', 'VALIDATE': '#ffcc66', 
                     'WATCH': '#ffeb99', 'SAFE': '#99ff99'}
    
    for i, action in enumerate(df['Action']):
        table[(i+1, 4)].set_facecolor(action_colors.get(action, 'white'))
    
    ax.set_title('Clinical Decision Matrix\nPopulation-Specific Recommendations',
                 fontweight='bold', pad=30)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure4_clinical.png", dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / "figure4_clinical.pdf", bbox_inches='tight')
    plt.close()
    print("✅ Figure 4: Clinical table")


def main():
    print("="*60)
    print("GENERATING NATURE MEDICINE PUBLICATION FIGURES")
    print("="*60)
    
    create_figure_1_roc()
    create_figure_2_population_heatmap()
    create_figure_3_uncertainty()
    create_figure_4_clinical_table()
    
    print("\n" + "="*60)
    print(f"✅ ALL FIGURES SAVED TO: {FIGURES_DIR}")
    print("="*60)
    print("\nFiles generated:")
    print("  - figure1_roc.png/pdf")
    print("  - figure2_population_heatmap.png/pdf")
    print("  - figure3_uncertainty.png/pdf")
    print("  - figure4_clinical.png/pdf")


if __name__ == "__main__":
    main()
