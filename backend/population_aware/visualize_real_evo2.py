"""
Real Evo2 Atlas Visualization

Creates comprehensive visual atlas of Evo2 predictions
Output: results/evo2_results/
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def main():
    print("="*70)
    print("REAL EVO2 ATLAS VISUALIZATION")
    print("="*70)
    
    # Load data
    df = pd.read_csv("results/evo2_results/brca1_evo2_scores.csv")
    print(f"\n📂 Loaded {len(df)} variants")
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 300
    
    # 1. Score Distributions by Class
    print("\n📊 Creating score distribution plot...")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for cls, color in [('FUNC', 'green'), ('INT', 'orange'), ('LOF', 'red')]:
        subset = df[df['func_class'] == cls]
        if len(subset) > 0:
            ax.hist(subset['evo2_score'], bins=50, alpha=0.6, label=f'{cls} (n={len(subset)})', color=color)
    
    ax.axvline(x=-0.007162, color='black', linestyle='--', linewidth=2, label='Optimal Threshold')
    ax.set_xlabel('Evo2 Delta Score', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Real Evo2 Score Distributions by Functional Class', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/evo2_results/score_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved score_distributions.png")
    
    # 2. Confusion Matrix Heatmap
    print("\n🔲 Creating confusion matrix...")
    df_bin = df[df['func_class'].isin(['FUNC', 'LOF'])].copy()
    df_bin['label'] = df_bin['func_class'].map({'LOF': 1, 'FUNC': 0})
    df_bin['prediction'] = (df_bin['evo2_score'] < -0.007162).astype(int)
    
    cm = confusion_matrix(df_bin['label'], df_bin['prediction'])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'},
                xticklabels=['Pred Benign', 'Pred Pathogenic'],
                yticklabels=['True Benign', 'True Pathogenic'],
                ax=ax, annot_kws={'size': 14})
    ax.set_title('Confusion Matrix - Real Evo2 on BRCA1', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/evo2_results/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved confusion_matrix.png")
    
    # 3. Evo2 vs Findlay Scatter
    print("\n📈 Creating Evo2 vs Findlay scatter plot...")
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for cls, color, marker in [('FUNC', 'green', 'o'), ('INT', 'orange', 's'), ('LOF', 'red', '^')]:
        subset = df[df['func_class'] == cls]
        ax.scatter(subset['func_score'], subset['evo2_score'], 
                  alpha=0.5, s=20, c=color, marker=marker, label=cls)
    
    ax.axhline(y=-0.007162, color='black', linestyle='--', alpha=0.5, label='Evo2 Threshold')
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlabel('Findlay Functional Score', fontsize=12)
    ax.set_ylabel('Evo2 Delta Score', fontsize=12)
    ax.set_title('Evo2 vs Findlay Scores (Correlation = 0.451)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/evo2_results/evo2_vs_findlay_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved evo2_vs_findlay_scatter.png")
    
    # 4. Position-based Heatmap
    print("\n🗺️  Creating genomic position heatmap...")
    
    # Bin positions for visualization
    df['pos_bin'] = (df['pos_hg38'] // 1000) * 1000
    
    # Create pivot table
    pivot = df.groupby(['pos_bin', 'func_class'])['evo2_score'].mean().unstack(fill_value=np.nan)
    
    if len(pivot) > 0:
        fig, ax = plt.subplots(figsize=(16, 6))
        sns.heatmap(pivot.T, cmap='RdBu_r', center=0, 
                   cbar_kws={'label': 'Mean Evo2 Score'},
                   ax=ax, robust=True)
        ax.set_xlabel('Genomic Position (kb bins)', fontsize=12)
        ax.set_ylabel('Functional Class', fontsize=12)
        ax.set_title('BRCA1 Mutation Atlas - Real Evo2 Scores by Position', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('results/evo2_results/brca1_atlas_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✅ Saved brca1_atlas_heatmap.png")
    
    # 5. Summary Statistics
    print("\n📋 Creating summary statistics...")
    summary = df.groupby('func_class')['evo2_score'].agg(['count', 'mean', 'std', 'min', 'max'])
    summary.to_csv('results/evo2_results/summary_statistics.csv')
    print("   ✅ Saved summary_statistics.csv")
    
    print("\n" + "="*70)
    print("✅ ATLAS VISUALIZATION COMPLETE!")
    print("="*70)
    print("\nGenerated files in results/evo2_results/:")
    print("  - score_distributions.png")
    print("  - confusion_matrix.png")
    print("  - evo2_vs_findlay_scatter.png")
    print("  - brca1_atlas_heatmap.png")
    print("  - summary_statistics.csv")
    print("="*70)

if __name__ == "__main__":
    main()
