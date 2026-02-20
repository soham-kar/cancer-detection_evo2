"""
Generate Publication-Ready Comparison Tables

Creates formatted tables for manuscript submission showing:
1. Multi-gene performance comparison
2. Calibration impact summary
3. Hero variant case studies
"""

import pandas as pd
import numpy as np

def create_performance_table():
    """Table 1: Multi-Gene Performance Comparison"""
    
    # Load multi-gene metrics
    df_metrics = pd.read_csv("results/multigene_metrics.csv")
    
    print("="*80)
    print("TABLE 1: Multi-Gene Evo2 Performance Comparison")
    print("="*80)
    
    # Format for publication
    table = df_metrics.copy()
    table = table.rename(columns={
        'Gene': 'Gene',
        'N': 'Variants (n)',
        'AUC': 'AUROC',
        'Sensitivity': 'Sensitivity',
        'Specificity': 'Specificity', 
        'Threshold': 'Optimal Threshold'
    })
    
    # Round to appropriate precision
    table['AUROC'] = table['AUROC'].apply(lambda x: f"{x:.3f}")
    table['Sensitivity'] = table['Sensitivity'].apply(lambda x: f"{x:.1%}")
    table['Specificity'] = table['Specificity'].apply(lambda x: f"{x:.1%}")
    table['Optimal Threshold'] = table['Optimal Threshold'].apply(lambda x: f"{x:.6f}")
    
    print("\n" + table.to_string(index=False))
    
    # Save as CSV
    table.to_csv("results/table1_multigene_performance.csv", index=False)
    
    # Save as LaTeX
    latex = table.to_latex(index=False, escape=False, 
                           caption="Multi-gene performance comparison of Evo2 variant pathogenicity prediction",
                           label="tab:multigene")
    with open("results/table1_multigene_performance.tex", 'w') as f:
        f.write(latex)
    
    print("\n✅ Saved to:")
    print("   - table1_multigene_performance.csv")
    print("   - table1_multigene_performance.tex")

def create_calibration_summary_table():
    """Table 2: Population-Aware Calibration Impact"""
    
    df = pd.read_csv("results/brca1_evo2_calibrated.csv")
    df_with_af = df[df['pop_bias'] > 0]
    
    print("\n\n" + "="*80)
    print("TABLE 2: Population-Aware Calibration Impact Summary")
    print("="*80)
    
    # Calculate statistics
    data = {
        'Metric': [
            'Total Variants Analyzed',
            'Variants with gnomAD Data',
            'Coverage (%)',
            'High Bias Variants (>0.5)',
            'Extreme Bias Variants (>0.7)',
            'Mean Population Bias Score',
            'Mean AFR Calibration Δ',
            'Mean EUR Calibration Δ',
            'Max AFR Calibration Δ',
            'Max EUR Calibration Δ',
            'Variants Adjusted (AFR, Δ>0.001)',
            'Variants Adjusted (EUR, Δ>0.001)'
        ],
        'Value': [
            len(df),
            len(df_with_af),
            f"{len(df_with_af)/len(df)*100:.1f}%",
            (df_with_af['pop_bias'] > 0.5).sum(),
            (df_with_af['pop_bias'] > 0.7).sum(),
            f"{df_with_af['pop_bias'].mean():.3f}",
            f"{(df_with_af['calibrated_afr'] - df_with_af['evo2_score']).abs().mean():.6f}",
            f"{(df_with_af['calibrated_eur'] - df_with_af['evo2_score']).abs().mean():.6f}",
            f"{(df_with_af['calibrated_afr'] - df_with_af['evo2_score']).abs().max():.6f}",
            f"{(df_with_af['calibrated_eur'] - df_with_af['evo2_score']).abs().max():.6f}",
            ((df_with_af['calibrated_afr'] - df_with_af['evo2_score']).abs() > 0.001).sum(),
            ((df_with_af['calibrated_eur'] - df_with_af['evo2_score']).abs() > 0.001).sum()
        ]
    }
    
    table = pd.DataFrame(data)
    print("\n" + table.to_string(index=False))
    
    # Save
    table.to_csv("results/table2_calibration_summary.csv", index=False)
    
    latex = table.to_latex(index=False, escape=False,
                           caption="Summary of population-aware calibration impact on BRCA1 variants",
                           label="tab:calibration")
    with open("results/table2_calibration_summary.tex", 'w') as f:
        f.write(latex)
    
    print("\n✅ Saved to:")
    print("   - table2_calibration_summary.csv")
    print("   - table2_calibration_summary.tex")

def create_hero_variants_table():
    """Table 3: Hero Variant Case Studies"""
    
    df_cases = pd.read_csv("results/calibration_case_studies.csv")
    
    print("\n\n" + "="*80)
    print("TABLE 3: Representative Variants Demonstrating Calibration Impact")
    print("="*80)
    
    # Select top variants
    table = df_cases[['variant_id', 'func_class', 'evo2_score', 
                      'calibrated_afr', 'calibrated_eur', 
                      'pop_bias', 'af_afr', 'af_eur']].head(6)
    
    table = table.rename(columns={
        'variant_id': 'Variant',
        'func_class': 'Class',
        'evo2_score': 'Raw Score',
        'calibrated_afr': 'Cal. AFR',
        'calibrated_eur': 'Cal. EUR',
        'pop_bias': 'Bias',
        'af_afr': 'AF_AFR',
        'af_eur': 'AF_EUR'
    })
    
    # Format
    for col in ['Raw Score', 'Cal. AFR', 'Cal. EUR']:
        table[col] = table[col].apply(lambda x: f"{x:.6f}" if pd.notna(x) else "N/A")
    table['Bias'] = table['Bias'].apply(lambda x: f"{x:.3f}")
    table['AF_AFR'] = table['AF_AFR'].apply(lambda x: f"{x:.6f}" if pd.notna(x) and x > 0 else "0.000000")
    table['AF_EUR'] = table['AF_EUR'].apply(lambda x: f"{x:.6f}" if pd.notna(x) and x > 0 else "0.000000")
    
    print("\n" + table.to_string(index=False))
    
    # Save
    table.to_csv("results/table3_hero_variants.csv", index=False)
    
    latex = table.to_latex(index=False, escape=False,
                           caption="Representative variants demonstrating significant calibration effects",
                           label="tab:heroes")
    with open("results/table3_hero_variants.tex", 'w') as f:
        f.write(latex)
    
    print("\n✅ Saved to:")
    print("   - table3_hero_variants.csv")
    print("   - table3_hero_variants.tex")

def main():
    print("\n" + "="*80)
    print("GENERATING PUBLICATION TABLES")
    print("="*80 + "\n")
    
    create_performance_table()
    create_calibration_summary_table()
    create_hero_variants_table()
    
    print("\n\n" + "="*80)
    print("✅ ALL TABLES GENERATED!")
    print("="*80)
    print("\n📊 Tables ready for manuscript:")
    print("   Table 1: Multi-gene performance (CSV + LaTeX)")
    print("   Table 2: Calibration impact summary (CSV + LaTeX)")
    print("   Table 3: Hero variant case studies (CSV + LaTeX)")
    print("\n💡 LaTeX tables can be directly copied into manuscript!")

if __name__ == "__main__":
    main()
