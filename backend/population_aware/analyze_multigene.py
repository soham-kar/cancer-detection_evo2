"""
Multi-Gene Comparative Analysis: BRCA1 vs BRCA2 vs PALB2

Compares Evo2 performance across three DNA repair genes to:
1. Prove generalization capability
2. Identify gene-specific threshold differences
3. Justify population-aware calibration methodology
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import os

# Configuration
FILE_BRCA1 = "results/brca1_evo2_scores_REAL.csv"
FILE_MULTI = "multi_gene/combined_sample_scores.csv"
OUTPUT_METRICS = "results/multigene_metrics.csv"

def main():
    print("🚀 Starting Multi-Gene Comparative Analysis...")
    
    # 1. Load Datasets
    try:
        df_brca1 = pd.read_csv(FILE_BRCA1)
        df_brca1['gene'] = 'BRCA1'
        print(f"   Loaded BRCA1: {len(df_brca1)} variants")
        
        df_multi = pd.read_csv(FILE_MULTI)
        print(f"   Loaded Multi-Gene: {len(df_multi)} variants")
        
    except FileNotFoundError as e:
        print(f"❌ Error loading files: {e}")
        print("   Make sure you are in the right directory!")
        return

    # 2. Standardization
    # Process BRCA1 (LOF=pathogenic=1, FUNC=benign=0)
    df_b1_clean = df_brca1[df_brca1['func_class'].isin(['LOF', 'FUNC'])].copy()
    df_b1_clean['label'] = df_b1_clean['func_class'].apply(lambda x: 1 if x == 'LOF' else 0)
    df_b1_clean = df_b1_clean[['gene', 'evo2_score', 'label']].dropna()
    
    # Process Multi-Gene (Pathogenic=1, Benign=0)
    df_multi_clean = df_multi.copy()
    df_multi_clean['label'] = df_multi_clean['func_class'].apply(
        lambda x: 1 if str(x).lower() in ['pathogenic', 'lof'] else 0
    )
    df_multi_clean = df_multi_clean[['gene', 'evo2_score', 'label']].dropna()
    
    # Combine
    full_df = pd.concat([df_b1_clean, df_multi_clean], ignore_index=True)
    
    print(f"\n📊 Combined dataset: {len(full_df)} variants")
    print(f"   - BRCA1: {len(df_b1_clean)}")
    print(f"   - Multi-gene: {len(df_multi_clean)}")
    
    # 3. Calculate Metrics per Gene
    results = []
    
    plt.figure(figsize=(10, 8))
    colors = {'BRCA1': '#1f77b4', 'BRCA2': '#ff7f0e', 'PALB2': '#2ca02c'}
    
    print("\n📊 Gene-Specific Performance:")
    print(f"{'Gene':<10} {'N':<6} {'AUC':<8} {'Sensitivity':<12} {'Specificity':<12} {'Threshold':<12}")
    print("-" * 75)
    
    for gene in ['BRCA1', 'BRCA2', 'PALB2']:
        subset = full_df[full_df['gene'] == gene]
        if len(subset) == 0: 
            print(f"{gene:<10} No data")
            continue
        
        # ROC Calculation (negative score = pathogenic)
        fpr, tpr, thresholds = roc_curve(subset['label'], -subset['evo2_score'])
        roc_auc = auc(fpr, tpr)
        
        # Optimal Threshold (Youden's J statistic)
        J = tpr - fpr
        ix = np.argmax(J)
        best_thresh = thresholds[ix]
        
        # Metrics at optimal threshold
        sens = tpr[ix]
        spec = 1 - fpr[ix]
        
        print(f"{gene:<10} {len(subset):<6} {roc_auc:.3f}    {sens:.1%}\t    {spec:.1%}\t  {-best_thresh:.6f}")
        
        results.append({
            'Gene': gene, 
            'N': len(subset), 
            'AUC': roc_auc, 
            'Sensitivity': sens, 
            'Specificity': spec, 
            'Threshold': -best_thresh
        })
        
        # Add to ROC Plot
        plt.plot(fpr, tpr, lw=2, label=f'{gene} (AUC = {roc_auc:.3f})', color=colors.get(gene, 'black'))

    # 4. Finalize ROC Plot
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title('Evo2 Performance Across DNA Repair Genes', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    os.makedirs("results", exist_ok=True)
    plot_path = "results/multigene_roc_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ ROC Plot saved to {plot_path}")
    plt.close()
    
    # 5. Save Metrics Table
    res_df = pd.DataFrame(results)
    res_df.to_csv(OUTPUT_METRICS, index=False)
    print(f"✅ Metrics table saved to {OUTPUT_METRICS}")
    
    # 6. Distribution Plot (Violin)
    plt.figure(figsize=(14, 7))
    
    # Create custom palette
    palette = {0: 'lightgreen', 1: 'salmon'}
    
    sns.violinplot(data=full_df, x='gene', y='evo2_score', hue='label', 
                   split=True, inner='quartile', palette=palette)
    
    plt.title('Evo2 Score Distributions: Pathogenic vs Benign', fontsize=14, fontweight='bold')
    plt.ylabel('Evo2 Delta Log-Likelihood', fontsize=12)
    plt.xlabel('Gene', fontsize=12)
    plt.axhline(0, color='black', linestyle='--', alpha=0.5, label='Neutral')
    
    # Custom Legend
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color='lightgreen', lw=4),
        Line2D([0], [0], color='salmon', lw=4)
    ]
    plt.legend(custom_lines, ['Benign', 'Pathogenic'], loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')
    
    dist_path = "results/multigene_distribution.png"
    plt.savefig(dist_path, dpi=300, bbox_inches='tight')
    print(f"✅ Distribution plot saved to {dist_path}")
    plt.close()
    
    # 7. Summary Statistics
    print("\n📈 Summary Statistics:")
    for gene in ['BRCA1', 'BRCA2', 'PALB2']:
        subset = full_df[full_df['gene'] == gene]
        if len(subset) == 0: continue
        
        path = subset[subset['label'] == 1]['evo2_score']
        benign = subset[subset['label'] == 0]['evo2_score']
        
        print(f"\n{gene}:")
        print(f"  Pathogenic: mean={path.mean():.6f}, std={path.std():.6f}")
        print(f"  Benign:     mean={benign.mean():.6f}, std={benign.std():.6f}")
        print(f"  Separation: {abs(path.mean() - benign.mean()):.6f}")
    
    print("\n" + "="*75)
    print("✅ MULTI-GENE ANALYSIS COMPLETE!")
    print("="*75)

if __name__ == "__main__":
    main()
