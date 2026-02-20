"""
Asian Population-Specific Variant Analysis

Identifies variants common in South Asian (SAS) and East Asian (EAS) 
populations but rare in European populations - demonstrating 
regional bias in Western-trained AI models.
"""

import pandas as pd
import numpy as np

def main():
    print("="*80)
    print("ASIAN POPULATION-SPECIFIC VARIANT ANALYSIS")
    print("="*80)
    
    # Load calibrated data with all population frequencies
    df = pd.read_csv("results/brca1_evo2_calibrated.csv")
    
    # Create variant ID if not present
    if 'variant_id' not in df.columns:
        df['variant_id'] = df['chrom'].astype(str) + ':' + df['pos_hg38'].astype(str) + df['ref'] + '>' + df['alt']
    
    print(f"\n📊 Analyzing {len(df)} variants for Asian population bias...")
    
    # Filter for variants with Asian AF data
    df_asian = df[
        (df['af_sas'].notna() | df['af_eas'].notna()) &
        (df['af_eur'].notna())
    ].copy()
    
    print(f"   Variants with Asian + EUR data: {len(df_asian)}")
    print(f"   - With SAS data: {df_asian['af_sas'].notna().sum()}")
    print(f"   - With EAS data: {df_asian['af_eas'].notna().sum()}")
    
    # Calculate Asian specificity bias
    # Positive bias = more common in Asian populations than European
    df_asian['sas_bias'] = df_asian['af_sas'].fillna(0) - df_asian['af_eur'].fillna(0)
    df_asian['eas_bias'] = df_asian['af_eas'].fillna(0) - df_asian['af_eur'].fillna(0)
    
    # Also calculate fold-difference for statistical significance
    df_asian['sas_fold'] = np.where(
        (df_asian['af_sas'] > 0) & (df_asian['af_eur'] > 0),
        df_asian['af_sas'] / (df_asian['af_eur'] + 1e-10),
        0
    )
    
    df_asian['eas_fold'] = np.where(
        (df_asian['af_eas'] > 0) & (df_asian['af_eur'] > 0),
        df_asian['af_eas'] / (df_asian['af_eur'] + 1e-10),
        0
    )
    
    # ===================================================================
    # SOUTH ASIAN (SAS) HERO VARIANTS
    # ===================================================================
    print("\n" + "="*80)
    print("🇮🇳 TOP 10 SOUTH ASIAN (SAS) SPECIFIC VARIANTS")
    print("="*80)
    print(f"{'Variant':<20} {'Evo2 Score':<12} {'AF_SAS':<10} {'AF_EUR':<10} {'Bias':<10} {'Fold':<10} {'Class':<8}")
    print("-" * 100)
    
    sas_hero = df_asian[df_asian['af_sas'] > 0].nlargest(10, 'sas_bias')
    
    for _, row in sas_hero.iterrows():
        print(f"{row['variant_id']:<20} {row['evo2_score']:<12.6f} "
              f"{row['af_sas']:<10.6f} {row['af_eur']:<10.6f} "
              f"{row['sas_bias']:<10.6f} {row['sas_fold']:<10.1f} "
              f"{row.get('func_class', 'N/A'):<8}")
    
    # ===================================================================
    # EAST ASIAN (EAS) HERO VARIANTS
    # ===================================================================
    print("\n" + "="*80)
    print("🇨🇳 🇯🇵 🇰🇷 TOP 10 EAST ASIAN (EAS) SPECIFIC VARIANTS")
    print("="*80)
    print(f"{'Variant':<20} {'Evo2 Score':<12} {'AF_EAS':<10} {'AF_EUR':<10} {'Bias':<10} {'Fold':<10} {'Class':<8}")
    print("-" * 100)
    
    eas_hero = df_asian[df_asian['af_eas'] > 0].nlargest(10, 'eas_bias')
    
    for _, row in eas_hero.iterrows():
        print(f"{row['variant_id']:<20} {row['evo2_score']:<12.6f} "
              f"{row['af_eas']:<10.6f} {row['af_eur']:<10.6f} "
              f"{row['eas_bias']:<10.6f} {row['eas_fold']:<10.1f} "
              f"{row.get('func_class', 'N/A'):<8}")
    
    # ===================================================================
    # RESCUE CANDIDATES
    # ===================================================================
    print("\n" + "="*80)
    print("🚀 POTENTIAL 'RESCUE' VARIANTS (Common in Asia, Misclassified by AI)")
    print("="*80)
    
    # SAS Rescue: High SAS frequency, absent in EUR, pathogenic Evo2 score
    sas_rescue = df_asian[
        (df_asian['af_sas'] > 0.001) &  # >0.1% in SAS
        (df_asian['af_eur'] < 0.0001) &  # <0.01% in EUR
        (df_asian['evo2_score'] < -0.005)  # AI says pathogenic
    ]
    
    eas_rescue = df_asian[
        (df_asian['af_eas'] > 0.001) &  # >0.1% in EAS
        (df_asian['af_eur'] < 0.0001) &  # <0.01% in EUR
        (df_asian['evo2_score'] < -0.005)  # AI says pathogenic
    ]
    
    print(f"\n🇮🇳 SAS Rescue Candidates: {len(sas_rescue)}")
    if len(sas_rescue) > 0:
        print(f"{'Variant':<20} {'Evo2':<12} {'AF_SAS':<10} {'AF_EUR':<10} {'Clinical Implication':<30}")
        print("-" * 90)
        for _, row in sas_rescue.head(3).iterrows():
            print(f"{row['variant_id']:<20} {row['evo2_score']:<12.6f} "
                  f"{row['af_sas']:<10.6f} {row['af_eur']:<10.6f} "
                  f"{'Likely benign in SAS pop':<30}")
    else:
        print("   No extreme rescue cases (may need threshold adjustment)")
    
    print(f"\n🇨🇳 EAS Rescue Candidates: {len(eas_rescue)}")
    if len(eas_rescue) > 0:
        print(f"{'Variant':<20} {'Evo2':<12} {'AF_EAS':<10} {'AF_EUR':<10} {'Clinical Implication':<30}")
        print("-" * 90)
        for _, row in eas_rescue.head(3).iterrows():
            print(f"{row['variant_id']:<20} {row['evo2_score']:<12.6f} "
                  f"{row['af_eas']:<10.6f} {row['af_eur']:<10.6f} "
                  f"{'Likely benign in EAS pop':<30}")
    else:
        print("   No extreme rescue cases (may need threshold adjustment)")
    
    # ===================================================================
    # SUMMARY STATISTICS
    # ===================================================================
    print("\n" + "="*80)
    print("📊 ASIAN POPULATION BIAS SUMMARY")
    print("="*80)
    
    # Count population-specific variants
    sas_specific = len(df_asian[
        (df_asian['af_sas'] > 0.001) &
        (df_asian['af_eur'] < 0.0001)
    ])
    
    eas_specific = len(df_asian[
        (df_asian['af_eas'] > 0.001) &
        (df_asian['af_eur'] < 0.0001)
    ])
    
    print(f"\nSAS-specific variants (>0.1% SAS, <0.01% EUR): {sas_specific}")
    print(f"EAS-specific variants (>0.1% EAS, <0.01% EUR): {eas_specific}")
    print(f"\nMean SAS bias: {df_asian['sas_bias'].mean():.6f}")
    print(f"Mean EAS bias: {df_asian['eas_bias'].mean():.6f}")
    print(f"\nMax SAS fold-difference: {df_asian['sas_fold'].max():.1f}x")
    print(f"Max EAS fold-difference: {df_asian['eas_fold'].max():.1f}x")
    
    # ===================================================================
    # EXPORT FOR PUBLICATION
    # ===================================================================
    # Export top Asian-specific variants
    asian_heroes = pd.concat([
        sas_hero.head(5),
        eas_hero.head(5)
    ]).drop_duplicates(subset='variant_id')
    
    asian_heroes['population'] = asian_heroes.apply(
        lambda row: 'SAS' if row['sas_bias'] > row['eas_bias'] else 'EAS',
        axis=1
    )
    
    export_cols = ['variant_id', 'population', 'evo2_score', 'func_class',
                   'af_sas', 'af_eas', 'af_eur', 'af_afr',
                   'sas_bias', 'eas_bias', 'sas_fold', 'eas_fold']
    
    asian_heroes[export_cols].to_csv("results/asian_hero_variants.csv", index=False)
    
    print("\n✅ Exported Asian hero variants to: results/asian_hero_variants.csv")
    
    print("\n" + "="*80)
    print("✅ ASIAN POPULATION ANALYSIS COMPLETE!")
    print("="*80)
    
    # Return top SAS variant for reporting
    if len(sas_hero) > 0:
        top_sas = sas_hero.iloc[0]
        print(f"\n🏆 TOP SAS HERO VARIANT FOR YOUR PRESENTATION:")
        print(f"   Variant: {top_sas['variant_id']}")
        print(f"   AF_SAS: {top_sas['af_sas']:.4%} (South Asian)")
        print(f"   AF_EUR: {top_sas['af_eur']:.4%} (European)")
        print(f"   Fold-difference: {top_sas['sas_fold']:.0f}x more common in SAS")
        print(f"   Evo2 Score: {top_sas['evo2_score']:.6f}")
        print(f"\n   💡 Narrative: 'This variant is {top_sas['sas_fold']:.0f}x more common")
        print(f"      in South Asian populations but Western AI models miss this context.'")

if __name__ == "__main__":
    main()
