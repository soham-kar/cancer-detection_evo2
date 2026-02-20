"""
Script 04: Indel Analysis
Analyze indel drivers - Evo2's unique advantage over AlphaMissense
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import *

def analyze_indel_distribution(df):
    """Compare SNV vs indel score distributions"""
    df['is_indel'] = df['variant_type'] == 'INDEL'
    
    snvs = df[~df['is_indel']]
    indels = df[df['is_indel']]
    
    print("=" * 60)
    print("INDEL DISTRIBUTION ANALYSIS")
    print("=" * 60)
    print(f"Total variants: {len(df)}")
    print(f"SNVs: {len(snvs)} ({len(snvs)/len(df)*100:.1f}%)")
    print(f"Indels: {len(indels)} ({len(indels)/len(df)*100:.1f}%)")
    
    print(f"\nSNV scores:")
    print(f"  Mean: {snvs['delta_score'].mean():.4f}")
    print(f"  Median: {snvs['delta_score'].median():.4f}")
    print(f"  Std: {snvs['delta_score'].std():.4f}")
    
    print(f"\nIndel scores:")
    print(f"  Mean: {indels['delta_score'].mean():.4f}")
    print(f"  Median: {indels['delta_score'].median():.4f}")
    print(f"  Std: {indels['delta_score'].std():.4f}")
    
    # Statistical test
    stat, p_value = stats.mannwhitneyu(snvs['delta_score'], indels['delta_score'])
    print(f"\nMann-Whitney U test: p={p_value:.2e}")
    
    return snvs, indels

def identify_high_impact_indels(indels, threshold_percentile=1):
    """Identify high-impact indels (most deleterious)"""
    threshold = indels['delta_score'].quantile(threshold_percentile / 100)
    high_impact = indels[indels['delta_score'] < threshold]
    
    print("\n" + "=" * 60)
    print(f"HIGH-IMPACT INDELS (Top {threshold_percentile}%)")
    print("=" * 60)
    print(f"Threshold: {threshold:.4f}")
    print(f"Count: {len(high_impact)}")
    
    # Top genes
    print(f"\nTop genes with high-impact indels:")
    gene_counts = high_impact['gene'].value_counts().head(10)
    for gene, count in gene_counts.items():
        pct = count / len(high_impact) * 100
        print(f"  {gene}: {count} ({pct:.1f}%)")
    
    return high_impact

def analyze_fat1_indels(df):
    """
    Deep dive into FAT1 indels - the key Indian-specific finding
    FAT1 is mutated in 28% of Indian OSCC vs 0-12% in Western HNSCC
    """
    fat1 = df[df['gene'] == 'FAT1']
    fat1_indels = fat1[fat1['variant_type'] == 'INDEL']
    
    print("\n" + "=" * 60)
    print("FAT1 INDEL ANALYSIS (Indian-Specific Driver)")
    print("=" * 60)
    print(f"Total FAT1 variants: {len(fat1)}")
    print(f"FAT1 indels: {len(fat1_indels)}")
    
    if len(fat1_indels) > 0:
        print(f"\nFAT1 indel scores:")
        print(fat1_indels[['patient_id', 'pos', 'ref', 'alt', 'delta_score']].to_string())
        
        print(f"\nScore statistics:")
        print(f"  Mean: {fat1_indels['delta_score'].mean():.4f}")
        print(f"  Median: {fat1_indels['delta_score'].median():.4f}")
        print(f"  Min (most deleterious): {fat1_indels['delta_score'].min():.4f}")
        
        # Compare to synonymous (neutral) variants
        synonymous = df[df['consequence'].str.contains('synonymous', na=False)]
        if len(synonymous) > 0:
            stat, p_value = stats.mannwhitneyu(
                fat1_indels['delta_score'], 
                synonymous['delta_score'],
                alternative='less'
            )
            print(f"\nFAT1 indels vs synonymous: p={p_value:.2e}")
            print(f"  (Significant if p < 0.05: FAT1 indels are more deleterious)")
    
    return fat1_indels

def create_indel_figures(snvs, indels, high_impact_indels, output_dir):
    """Generate publication-quality figures"""
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = FIGURE_CONFIG['font_size']
    
    fig, axes = plt.subplots(2, 2, figsize=FIGURE_CONFIG['figure_size'])
    
    # Panel A: Score distributions
    ax = axes[0, 0]
    sns.histplot(snvs['delta_score'], bins=50, alpha=0.6, label='SNVs', 
                 ax=ax, color='blue', stat='density')
    sns.histplot(indels['delta_score'], bins=50, alpha=0.6, label='Indels', 
                 ax=ax, color='red', stat='density')
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Evo2 Score (ΔLL)')
    ax.set_ylabel('Density')
    ax.set_title('A. Evo2 Scores by Variant Type')
    ax.legend()
    
    # Panel B: High-impact indel genes
    ax = axes[0, 1]
    gene_counts = high_impact_indels['gene'].value_counts().head(10)
    gene_counts.plot(kind='barh', ax=ax, color='coral')
    ax.set_xlabel('Number of High-Impact Indels')
    ax.set_ylabel('Gene')
    ax.set_title('B. Top Genes with Deleterious Indels')
    ax.invert_yaxis()
    
    # Panel C: Violin plot comparison
    ax = axes[1, 0]
    data_for_violin = pd.concat([
        snvs[['delta_score']].assign(type='SNV'),
        indels[['delta_score']].assign(type='Indel')
    ])
    sns.violinplot(data=data_for_violin, x='type', y='delta_score', ax=ax)
    ax.set_ylabel('Evo2 Score (ΔLL)')
    ax.set_xlabel('Variant Type')
    ax.set_title('C. Score Distribution Comparison')
    
    # Panel D: Cumulative distribution
    ax = axes[1, 1]
    snv_sorted = np.sort(snvs['delta_score'])
    indel_sorted = np.sort(indels['delta_score'])
    ax.plot(snv_sorted, np.linspace(0, 1, len(snv_sorted)), 
            label='SNVs', color='blue', linewidth=2)
    ax.plot(indel_sorted, np.linspace(0, 1, len(indel_sorted)), 
            label='Indels', color='red', linewidth=2)
    ax.set_xlabel('Evo2 Score (ΔLL)')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('D. Cumulative Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save in multiple formats
    for fmt in FIGURE_CONFIG['format']:
        output_path = output_dir / f"figure_indel_analysis.{fmt}"
        plt.savefig(output_path, dpi=FIGURE_CONFIG['dpi'], bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Analyze indel drivers")
    parser.add_argument("--input", type=str, required=True, help="Evo2 scores CSV")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("PHASE 3: INDEL ANALYSIS")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv(args.input)
    print(f"\nLoaded {len(df)} scored variants")
    
    # Analyze distributions
    snvs, indels = analyze_indel_distribution(df)
    
    # Identify high-impact indels
    high_impact = identify_high_impact_indels(indels, threshold_percentile=1)
    
    # FAT1 analysis (Indian-specific)
    fat1_indels = analyze_fat1_indels(df)
    
    # Save results
    output_dir = Path(args.output_dir) if args.output_dir else RESULTS_DIR
    
    high_impact_path = output_dir / "high_impact_indels.csv"
    high_impact.to_csv(high_impact_path, index=False)
    print(f"\nSaved high-impact indels: {high_impact_path}")
    
    if len(fat1_indels) > 0:
        fat1_path = output_dir / "fat1_indels.csv"
        fat1_indels.to_csv(fat1_path, index=False)
        print(f"Saved FAT1 indels: {fat1_path}")
    
    # Generate figures
    create_indel_figures(snvs, indels, high_impact, FIGURES_DIR)
    
    print("\n" + "=" * 60)
    print("KEY FINDINGS:")
    print("=" * 60)
    print(f"1. Evo2 scored {len(indels)} indels (AlphaMissense cannot score these)")
    print(f"2. {len(high_impact)} high-impact indels identified")
    print(f"3. FAT1 has {len(fat1_indels)} indels (Indian-specific driver)")
    print("\nThis demonstrates Evo2's advantage over protein-based predictors!")
    print("=" * 60)

if __name__ == "__main__":
    main()
