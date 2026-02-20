"""
CORRECTED Validation Strategy - Cross-Validation for BRCA2/PALB2

CRITICAL FIX: Use k-fold cross-validation instead of single train/test split
- More robust (mean ± std AUROC)
- Prevents overfitting
- Standard for clinical ML papers
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import json

def cross_validate_gene(gene_name, scores_file, n_splits=5):
    """
    Perform stratified k-fold cross-validation
    
    Args:
        gene_name: Gene name (BRCA2, PALB2)
        scores_file: CSV with evo2_score and label columns
        n_splits: Number of CV folds (default 5)
    
    Returns:
        Dictionary with mean AUROC, std, and per-fold results
    """
    print(f"\n{'='*70}")
    print(f"{gene_name} - {n_splits}-Fold Cross-Validation")
    print(f"{'='*70}")
    
    # Load data
    df = pd.read_csv(scores_file)
    
    # Filter to usable variants (exclude VUS)
    df_clean = df[df['label_binary'].notna()].copy()
    
    print(f"\nDataset:")
    print(f"  Total usable: {len(df_clean)}")
    print(f"  Pathogenic: {(df_clean['label_binary']==1).sum()}")
    print(f"  Benign: {(df_clean['label_binary']==0).sum()}")
    
    # Check minimum size
    if len(df_clean) < 200:
        print(f"\n❌ ERROR: Sample size {len(df_clean)} too small for CV")
        print(f"   Minimum required: 200 (40 per fold)")
        return None
    
    # Stratified k-fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold_results = []
    all_fpr = []
    all_tpr = []
    
    print(f"\nRunning {n_splits}-fold cross-validation...")
    
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(df_clean, df_clean['label_binary'])):
        # Split data
        test_df = df_clean.iloc[test_idx]
        
        # Calculate AUROC on test fold
        auroc = roc_auc_score(test_df['label_binary'], -test_df['evo2_score'])
        
        # Calculate other metrics
        threshold = df_clean.iloc[train_idx]['evo2_score'].median()  # Use train median as threshold
        predictions = (test_df['evo2_score'] < threshold).astype(int)
        f1 = f1_score(test_df['label_binary'], predictions)
        
        # ROC curve for this fold
        fpr, tpr, _ = roc_curve(test_df['label_binary'], -test_df['evo2_score'])
        all_fpr.append(fpr)
        all_tpr.append(tpr)
        
        fold_results.append({
            'fold': fold_idx + 1,
            'auroc': auroc,
            'f1': f1,
            'n_test': len(test_df)
        })
        
        print(f"  Fold {fold_idx+1}: AUROC = {auroc:.3f}, F1 = {f1:.3f}")
    
    # Aggregate results
    aurocs = [r['auroc'] for r in fold_results]
    mean_auroc = np.mean(aurocs)
    std_auroc = np.std(aurocs)
    
    print(f"\n{'='*70}")
    print(f"CROSS-VALIDATION RESULTS")
    print(f"{'='*70}")
    print(f"Mean AUROC: {mean_auroc:.3f} ± {std_auroc:.3f}")
    print(f"Range: [{min(aurocs):.3f}, {max(aurocs):.3f}]")
    
    # Statistical test (is this better than random?)
    from scipy import stats
    t_stat, p_value = stats.ttest_1samp(aurocs, 0.5)
    print(f"\nt-test vs random (0.5):")
    print(f"  t-statistic: {t_stat:.3f}")
    print(f"  p-value: {p_value:.2e}")
    
    if p_value < 0.001:
        print(f"  ✅ Significantly better than random (p < 0.001)")
    elif p_value < 0.05:
        print(f"  ✅ Significantly better than random (p < 0.05)")
    else:
        print(f"  ❌ NOT significantly better than random")
    
    results = {
        'gene': gene_name,
        'mean_auroc': float(mean_auroc),
        'std_auroc': float(std_auroc),
        'min_auroc': float(min(aurocs)),
        'max_auroc': float(max(aurocs)),
        'p_value': float(p_value),
        'n_folds': n_splits,
        'fold_results': fold_results
    }
    
    return results

def plot_cv_results(results_dict, output_dir="results"):
    """
    Create publication-ready figure with error bars
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data
    genes = []
    aurocs = []
    stds = []
    
    for gene, results in results_dict.items():
        genes.append(gene)
        aurocs.append(results['mean_auroc'])
        stds.append(results['std_auroc'])
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(genes))
    bars = ax.bar(x, aurocs, yerr=stds, capsize=5, alpha=0.7,
                  color=['#2ecc71' if g == 'BRCA1' else '#3498db' for g in genes])
    
    # Reference lines
    ax.axhline(y=0.5, color='red', linestyle='--', label='Random', linewidth=2)
    ax.axhline(y=0.7, color='orange', linestyle='--', label='Clinical Threshold', alpha=0.5, linewidth=2)
    
    # Formatting
    ax.set_ylabel('AUROC', fontsize=14, fontweight='bold')
    ax.set_title('Evo2 Performance Across Genes (Cross-Validated)', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(genes, fontsize=12)
    ax.set_ylim(0.4, 0.9)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, axis='y')
    
    # Add significance stars
    for i, (gene, auroc) in enumerate(zip(genes, aurocs)):
        p_val = results_dict[gene]['p_value']
        if p_val < 0.001:
            stars = '***'
        elif p_val < 0.01:
            stars = '**'
        elif p_val < 0.05:
            stars = '*'
        else:
            stars = 'ns'
        
        ax.text(i, auroc + stds[i] + 0.02, stars, ha='center', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/multi_gene_cv_auroc.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Saved figure: {output_dir}/multi_gene_cv_auroc.png")

def main():
    """
    Run cross-validation for all genes
    """
    print("="*70)
    print("MULTI-GENE CROSS-VALIDATION ANALYSIS")
    print("="*70)
    
    # Define genes and their score files
    genes = {
        'BRCA1': 'results/evo2_results/brca1_evo2_scores.csv',  # Already done
        'BRCA2': 'results/brca2_evo2_scores.csv',
        'PALB2': 'results/palb2_evo2_scores.csv'
    }
    
    all_results = {}
    
    for gene, scores_file in genes.items():
        import os
        if not os.path.exists(scores_file):
            print(f"\n⚠️  {gene}: Score file not found, skipping")
            continue
        
        results = cross_validate_gene(gene, scores_file, n_splits=5)
        
        if results:
            all_results[gene] = results
            
            # Save individual gene results
            output_file = f"results/{gene.lower()}_cv_results.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"  💾 Saved: {output_file}")
    
    # Meta-analysis plot
    if len(all_results) > 1:
        plot_cv_results(all_results)
    
    # Summary table
    print("\n" + "="*70)
    print("SUMMARY TABLE (For Manuscript)")
    print("="*70)
    print(f"{'Gene':<8} {'AUROC':<15} {'p-value':<12} {'Data Source':<15}")
    print("-" * 70)
    
    for gene, results in all_results.items():
        auroc_str = f"{results['mean_auroc']:.3f} ± {results['std_auroc']:.3f}"
        p_str = f"{results['p_value']:.2e}"
        source = "Findlay" if gene == "BRCA1" else "ClinVar"
        print(f"{gene:<8} {auroc_str:<15} {p_str:<12} {source:<15}")
    
    print("="*70)
    print("\n✅ Cross-validation complete!")
    print("   Ready for manuscript Table 1")

if __name__ == "__main__":
    main()
