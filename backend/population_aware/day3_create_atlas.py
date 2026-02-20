"""
Day 3: Atlas Visualization
Creates comprehensive visualizations of BRCA1 mutation landscape
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

def main():
    print("Loading Evo2 scores...")
    df = pd.read_csv("results/brca1_evo2_scores.csv")
    
    # Load validation metrics
    with open("results/validation_metrics.json", "r") as f:
        metrics = json.load(f)
    
    threshold = metrics['optimal_threshold']
    
    print(f"Loaded {len(df)} variants")
    print(f"Using threshold: {threshold:.6f}")
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'sans-serif'
    
    # 1. Score distribution by functional class
    print("\nCreating score distribution plot...")
    plt.figure(figsize=(12, 6))
    
    for cls, color in [('FUNC', 'green'), ('LOF', 'red'), ('INT', 'orange')]:
        subset = df[df['func_class'] == cls]
        if len(subset) > 0:
            sns.kdeplot(data=subset, x='evo2_score', label=f'{cls} (n={len(subset)})', 
                       fill=True, alpha=0.3, color=color, linewidth=2)
    
    plt.axvline(x=threshold, color='black', linestyle='--', linewidth=2, 
                label=f'Threshold ({threshold:.4f})')
    plt.xlabel('Evo2 ΔL Score', fontsize=13)
    plt.ylabel('Density', fontsize=13)
    plt.title('BRCA1 Variant Score Distribution by Functional Class', fontsize=15, fontweight='bold')
    plt.legend(fontsize=11, loc='upper left')
    plt.tight_layout()
    plt.savefig('results/score_distributions.png', dpi=300, bbox_inches='tight')
    print("✅ Saved score_distributions.png")
    plt.close()
    
    # 2. Confusion matrix heatmap
    print("Creating confusion matrix...")
    cm = metrics['confusion_matrix']
    cm_array = np.array([
        [cm['true_negatives'], cm['false_positives']],
        [cm['false_negatives'], cm['true_positives']]
    ])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_array, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted Benign', 'Predicted Pathogenic'],
                yticklabels=['Actual Benign', 'Actual Pathogenic'],
                cbar_kws={'label': 'Count'}, annot_kws={'size': 14})
    plt.title('Confusion Matrix: Evo2 vs Findlay Labels', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("✅ Saved confusion_matrix.png")
    plt.close()
    
    # 3. Score vs Functional Score scatter
    if 'func_score' in df.columns:
        print("Creating Evo2 vs Findlay scatter plot...")
        plt.figure(figsize=(10, 8))
        
        for cls, color, marker in [('FUNC', 'green', 'o'), ('LOF', 'red', 's'), ('INT', 'orange', '^')]:
            subset = df[df['func_class'] == cls]
            if len(subset) > 0:
                plt.scatter(subset['func_score'], subset['evo2_score'], 
                           c=color, label=cls, alpha=0.6, s=20, marker=marker)
        
        plt.xlabel('Findlay Functional Score', fontsize=13)
        plt.ylabel('Evo2 ΔL Score', fontsize=13)
        plt.title('Evo2 vs Findlay Functional Scores', fontsize=15, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('results/evo2_vs_findlay_scatter.png', dpi=300, bbox_inches='tight')
        print("✅ Saved evo2_vs_findlay_scatter.png")
        plt.close()
    
    # 4. Position-based heatmap (if position data available)
    if 'pos_hg38' in df.columns:
        print("Creating position-based mutation atlas...")
        
        # Bin positions into windows
        df['pos_bin'] = pd.cut(df['pos_hg38'], bins=50)
        
        # Average score per position bin and class
        pivot_data = df.groupby(['pos_bin', 'func_class'])['evo2_score'].mean().unstack(fill_value=np.nan)
        
        if len(pivot_data) > 0:
            plt.figure(figsize=(14, 8))
            sns.heatmap(pivot_data.T, cmap='RdBu_r', center=0, cbar_kws={'label': 'Mean Evo2 Score'})
            plt.xlabel('BRCA1 Position Bin', fontsize=13)
            plt.ylabel('Functional Class', fontsize=13)
            plt.title('BRCA1 Mutation Atlas: Mean Evo2 Scores by Position', fontsize=15, fontweight='bold')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig('results/brca1_atlas_heatmap.png', dpi=300, bbox_inches='tight')
            print("✅ Saved brca1_atlas_heatmap.png")
            plt.close()
    
    # 5. Summary statistics table
    print("\nCreating summary statistics...")
    summary = df.groupby('func_class').agg({
        'evo2_score': ['count', 'mean', 'std', 'min', 'max']
    }).round(4)
    
    print("\n" + "="*60)
    print("SUMMARY STATISTICS BY FUNCTIONAL CLASS")
    print("="*60)
    print(summary)
    print("="*60)
    
    summary.to_csv('results/summary_statistics.csv')
    print("\n✅ Saved summary_statistics.csv")
    
    print("\n" + "="*60)
    print("ATLAS VISUALIZATION COMPLETE!")
    print("="*60)
    print("Generated files:")
    print("  - score_distributions.png")
    print("  - confusion_matrix.png")
    print("  - evo2_vs_findlay_scatter.png")
    print("  - brca1_atlas_heatmap.png")
    print("  - summary_statistics.csv")
    print("="*60)

if __name__ == "__main__":
    main()
