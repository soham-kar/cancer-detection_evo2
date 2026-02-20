"""
Script 06: Gender-Stratified Analysis
Identify sex-specific driver mutations (e.g., CASP8 in females)
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

def load_clinical_data():
    """Load patient clinical data with gender information"""
    clinical_path = PROCESSED_DATA_DIR / "gse213862_clinical.csv"
    if not clinical_path.exists():
        print(f"Warning: Clinical data not found at {clinical_path}")
        return None
    return pd.read_csv(clinical_path)

def merge_with_clinical(variants_df, clinical_df):
    """Merge variant data with clinical metadata"""
    merged = variants_df.merge(
        clinical_df[['patient_id', 'sex', 'age', 'tobacco', 'survival_status']],
        on='patient_id',
        how='left'
    )
    
    print(f"Merged data:")
    print(f"  Total variants: {len(merged)}")
    print(f"  Patients with gender data: {merged['sex'].notna().sum()}")
    print(f"  Male: {(merged['sex'] == 'M').sum()}")
    print(f"  Female: {(merged['sex'] == 'F').sum()}")
    
    return merged

def compare_driver_frequencies(df, genes_to_test=None):
    """Compare driver gene frequencies between males and females"""
    if genes_to_test is None:
        genes_to_test = ANALYSIS_CONFIG["drivers"]["known_genes"]
    
    male = df[df['sex'] == 'M']
    female = df[df['sex'] == 'F']
    
    male_patients = male['patient_id'].nunique()
    female_patients = female['patient_id'].nunique()
    
    print("\n" + "=" * 60)
    print("GENDER-STRATIFIED DRIVER ANALYSIS")
    print("=" * 60)
    print(f"Male patients: {male_patients}")
    print(f"Female patients: {female_patients}")
    
    results = []
    
    for gene in genes_to_test:
        # Count patients with mutations in this gene
        male_with_gene = male[male['gene'] == gene]['patient_id'].nunique()
        female_with_gene = female[female['gene'] == gene]['patient_id'].nunique()
        
        male_freq = male_with_gene / male_patients if male_patients > 0 else 0
        female_freq = female_with_gene / female_patients if female_patients > 0 else 0
        
        # Fisher's exact test
        table = [
            [male_with_gene, male_patients - male_with_gene],
            [female_with_gene, female_patients - female_with_gene]
        ]
        
        if male_with_gene + female_with_gene > 0:
            odds_ratio, p_value = stats.fisher_exact(table)
        else:
            odds_ratio, p_value = np.nan, np.nan
        
        results.append({
            'gene': gene,
            'male_count': male_with_gene,
            'male_freq': male_freq,
            'female_count': female_with_gene,
            'female_freq': female_freq,
            'odds_ratio': odds_ratio,
            'p_value': p_value,
            'significant': p_value < 0.05 if not np.isnan(p_value) else False
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('p_value')
    
    print("\nDriver gene comparison:")
    print(results_df.to_string(index=False))
    
    return results_df

def analyze_casp8_enrichment(df):
    """
    Analyze CASP8 enrichment in females
    Expected from literature (Kolar et al.)
    """
    casp8 = df[df['gene'] == 'CASP8']
    
    print("\n" + "=" * 60)
    print("CASP8 ANALYSIS (Female-Enriched Driver)")
    print("=" * 60)
    
    if len(casp8) == 0:
        print("No CASP8 variants found in dataset")
        return None
    
    male_casp8 = casp8[casp8['sex'] == 'M']
    female_casp8 = casp8[casp8['sex'] == 'F']
    
    print(f"Total CASP8 variants: {len(casp8)}")
    print(f"  Male: {len(male_casp8)}")
    print(f"  Female: {len(female_casp8)}")
    
    # Evo2 scores by gender
    if len(male_casp8) > 0 and len(female_casp8) > 0:
        print(f"\nEvo2 scores:")
        print(f"  Male mean: {male_casp8['delta_score'].mean():.4f}")
        print(f"  Female mean: {female_casp8['delta_score'].mean():.4f}")
        
        stat, p_value = stats.mannwhitneyu(
            male_casp8['delta_score'],
            female_casp8['delta_score']
        )
        print(f"  Mann-Whitney U: p={p_value:.3f}")
    
    return casp8

def create_gender_figures(comparison_df, df, output_dir):
    """Generate gender comparison figures"""
    sns.set_style("whitegrid")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Panel A: Odds ratio forest plot
    ax = axes[0, 0]
    sig_genes = comparison_df[comparison_df['significant']]
    colors = ['red' if p < 0.05 else 'gray' for p in comparison_df['p_value']]
    
    y_pos = range(len(comparison_df))
    ax.scatter(comparison_df['odds_ratio'], y_pos, c=colors, s=100, alpha=0.7)
    ax.axvline(x=1, color='black', linestyle='--', alpha=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(comparison_df['gene'])
    ax.set_xlabel('Odds Ratio (Male vs Female)')
    ax.set_xscale('log')
    ax.set_title('A. Sex-Specific Driver Enrichment')
    ax.grid(True, alpha=0.3)
    
    # Add significance stars
    for i, row in comparison_df.iterrows():
        if row['p_value'] < 0.001:
            ax.text(row['odds_ratio'], i, ' ***', ha='left', va='center')
        elif row['p_value'] < 0.01:
            ax.text(row['odds_ratio'], i, ' **', ha='left', va='center')
        elif row['p_value'] < 0.05:
            ax.text(row['odds_ratio'], i, ' *', ha='left', va='center')
    
    # Panel B: Frequency comparison
    ax = axes[0, 1]
    x = np.arange(len(comparison_df))
    width = 0.35
    ax.bar(x - width/2, comparison_df['male_freq'], width, label='Male', alpha=0.8)
    ax.bar(x + width/2, comparison_df['female_freq'], width, label='Female', alpha=0.8)
    ax.set_xlabel('Gene')
    ax.set_ylabel('Mutation Frequency')
    ax.set_title('B. Driver Mutation Frequency by Sex')
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df['gene'], rotation=45, ha='right')
    ax.legend()
    
    # Panel C: CASP8 score distribution by gender
    ax = axes[1, 0]
    casp8 = df[df['gene'] == 'CASP8']
    if len(casp8) > 0:
        male_casp8 = casp8[casp8['sex'] == 'M']['delta_score']
        female_casp8 = casp8[casp8['sex'] == 'F']['delta_score']
        
        if len(male_casp8) > 0 and len(female_casp8) > 0:
            ax.hist(male_casp8, bins=20, alpha=0.6, label='Male', color='blue')
            ax.hist(female_casp8, bins=20, alpha=0.6, label='Female', color='pink')
            ax.set_xlabel('Evo2 Score (ΔLL)')
            ax.set_ylabel('Count')
            ax.set_title('C. CASP8 Variant Scores by Sex')
            ax.legend()
    
    # Panel D: Top genes by gender
    ax = axes[1, 1]
    male_top = df[df['sex'] == 'M']['gene'].value_counts().head(10)
    female_top = df[df['sex'] == 'F']['gene'].value_counts().head(10)
    
    # Combine and plot
    all_genes = list(set(male_top.index) | set(female_top.index))
    male_counts = [male_top.get(g, 0) for g in all_genes]
    female_counts = [female_top.get(g, 0) for g in all_genes]
    
    x = np.arange(len(all_genes))
    width = 0.35
    ax.barh(x - width/2, male_counts, width, label='Male', alpha=0.8)
    ax.barh(x + width/2, female_counts, width, label='Female', alpha=0.8)
    ax.set_yticks(x)
    ax.set_yticklabels(all_genes)
    ax.set_xlabel('Number of Variants')
    ax.set_title('D. Top Mutated Genes by Sex')
    ax.legend()
    ax.invert_yaxis()
    
    plt.tight_layout()
    
    # Save
    for fmt in FIGURE_CONFIG['format']:
        output_path = output_dir / f"figure_gender_analysis.{fmt}"
        plt.savefig(output_path, dpi=FIGURE_CONFIG['dpi'], bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Gender-stratified driver analysis")
    parser.add_argument("--input", type=str, required=True, help="Evo2 scores CSV")
    parser.add_argument("--clinical", type=str, default=None, help="Clinical data CSV")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("PHASE 4: GENDER-STRATIFIED ANALYSIS")
    print("=" * 60)
    
    # Load data
    variants = pd.read_csv(args.input)
    print(f"\nLoaded {len(variants)} variants")
    
    # Load clinical data
    if args.clinical:
        clinical = pd.read_csv(args.clinical)
    else:
        clinical = load_clinical_data()
    
    if clinical is None:
        print("Error: Clinical data required for gender analysis")
        return
    
    # Merge
    df = merge_with_clinical(variants, clinical)
    
    # Remove variants without gender data
    df = df[df['sex'].notna()]
    
    # Compare driver frequencies
    comparison = compare_driver_frequencies(df)
    
    # CASP8 analysis
    casp8 = analyze_casp8_enrichment(df)
    
    # Save results
    comparison_path = RESULTS_DIR / "gender_driver_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    print(f"\nSaved comparison: {comparison_path}")
    
    # Generate figures
    create_gender_figures(comparison, df, FIGURES_DIR)
    
    print("\n" + "=" * 60)
    print("KEY FINDINGS:")
    print("=" * 60)
    
    # Highlight significant findings
    sig_genes = comparison[comparison['significant']]
    if len(sig_genes) > 0:
        print("Significantly different genes:")
        for _, row in sig_genes.iterrows():
            enriched_in = "males" if row['odds_ratio'] > 1 else "females"
            print(f"  {row['gene']}: {row['odds_ratio']:.2f}x enriched in {enriched_in} (p={row['p_value']:.3f})")
    else:
        print("No significant sex-specific enrichment found")
        print("(May need larger sample size)")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
