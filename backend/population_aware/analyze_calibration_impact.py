"""
Analyze Calibration Impact - Find "Hero Variants"

Identifies variants where population-aware calibration makes
the biggest difference, especially "rescue" cases where raw
scores suggest pathogenicity but calibrated scores suggest
benign/VUS in specific populations.
"""

import pandas as pd
import numpy as np

def main():
    print("="*80)
    print("CALIBRATION IMPACT ANALYSIS - Finding Hero Variants")
    print("="*80)
    
    # Load calibrated results
    df = pd.read_csv("results/brca1_evo2_calibrated.csv")
    
    # Create variant ID for display
    df['variant_id'] = df['chrom'].astype(str) + ':' + df['pos_hg38'].astype(str) + df['ref'] + '>' + df['alt']
    
    # Filter for variants with calibration (pop_bias > 0)
    calibrated_df = df[df['pop_bias'] > 0].copy()
    
    print(f"\n📊 Analyzing {len(calibrated_df)} variants with population data...")
    
    # Calculate correction magnitude
    calibrated_df['delta_afr'] = (calibrated_df['evo2_score'] - calibrated_df['calibrated_afr']).abs()
    calibrated_df['delta_eur'] = (calibrated_df['evo2_score'] - calibrated_df['calibrated_eur']).abs()
    
    # === TOP CORRECTIONS (AFR) ===
    print("\n🔍 TOP 10 VARIANTS WITH LARGEST AFR CALIBRATION:")
    print("-" * 100)
    print(f"{'Variant ID':<25} {'Raw Score':<12} {'Cal AFR':<12} {'Delta':<10} {'Pop Bias':<10} {'Class':<12}")
    print("-" * 100)
    
    top_afr = calibrated_df.nlargest(10, 'delta_afr')
    for _, row in top_afr.iterrows():
        print(f"{row['variant_id']:<25} {row['evo2_score']:<12.6f} {row['calibrated_afr']:<12.6f} "
              f"{row['delta_afr']:<10.6f} {row['pop_bias']:<10.3f} {row.get('func_class', 'N/A'):<12}")
    
    # === TOP CORRECTIONS (EUR) ===
    print("\n\n🔍 TOP 10 VARIANTS WITH LARGEST EUR CALIBRATION:")
    print("-" * 100)
    print(f"{'Variant ID':<25} {'Raw Score':<12} {'Cal EUR':<12} {'Delta':<10} {'Pop Bias':<10} {'Class':<12}")
    print("-" * 100)
    
    top_eur = calibrated_df.nlargest(10, 'delta_eur')
    for _, row in top_eur.iterrows():
        print(f"{row['variant_id']:<25} {row['evo2_score']:<12.6f} {row['calibrated_eur']:<12.6f} "
              f"{row['delta_eur']:<10.6f} {row['pop_bias']:<10.3f} {row.get('func_class', 'N/A'):<12}")
    
    # === RESCUE CASES ===
    # Note: Evo2 scores are negative for pathogenic, so we need to adjust the logic
    # Lower (more negative) = more pathogenic
    # Higher (less negative or positive) = more benign
    
    print("\n\n🚀 SEARCHING FOR 'RESCUE' VARIANTS...")
    print("   (Raw score suggests pathogenic, calibrated suggests benign/VUS)")
    print("-" * 100)
    
    # Variants where raw score is very negative (pathogenic)
    # but calibrated score is less negative (less pathogenic)
    rescued_afr = calibrated_df[
        (calibrated_df['evo2_score'] < -0.01) &  # Raw score suggests pathogenic
        (calibrated_df['calibrated_afr'] > calibrated_df['evo2_score'] + 0.005)  # AFR calibration reduces severity
    ].sort_values('delta_afr', ascending=False)
    
    rescued_eur = calibrated_df[
        (calibrated_df['evo2_score'] < -0.01) &  # Raw score suggests pathogenic
        (calibrated_df['calibrated_eur'] > calibrated_df['evo2_score'] + 0.005)  # EUR calibration reduces severity
    ].sort_values('delta_eur', ascending=False)
    
    if len(rescued_afr) > 0:
        print(f"\n✅ FOUND {len(rescued_afr)} AFR 'RESCUED' VARIANTS:")
        print(f"{'Variant ID':<25} {'Raw':<12} {'Cal AFR':<12} {'Improvement':<12} {'AF_AFR':<10} {'AF_EUR':<10}")
        print("-" * 100)
        for _, row in rescued_afr.head(5).iterrows():
            improvement = row['calibrated_afr'] - row['evo2_score']
            print(f"{row['variant_id']:<25} {row['evo2_score']:<12.6f} {row['calibrated_afr']:<12.6f} "
                  f"{improvement:<12.6f} {row.get('af_afr', 0):<10.6f} {row.get('af_eur', 0):<10.6f}")
    else:
        print("\nℹ️ No AFR rescue cases found with current thresholds")
    
    if len(rescued_eur) > 0:
        print(f"\n\n✅ FOUND {len(rescued_eur)} EUR 'RESCUED' VARIANTS:")
        print(f"{'Variant ID':<25} {'Raw':<12} {'Cal EUR':<12} {'Improvement':<12} {'AF_AFR':<10} {'AF_EUR':<10}")
        print("-" * 100)
        for _, row in rescued_eur.head(5).iterrows():
            improvement = row['calibrated_eur'] - row['evo2_score']
            print(f"{row['variant_id']:<25} {row['evo2_score']:<12.6f} {row['calibrated_eur']:<12.6f} "
                  f"{improvement:<12.6f} {row.get('af_afr', 0):<10.6f} {row.get('af_eur', 0):<10.6f}")
    else:
        print("\nℹ️ No EUR rescue cases found with current thresholds")
    
    # === HIGH BIAS VARIANTS ===
    print("\n\n⚠️  TOP 5 HIGHEST POPULATION BIAS VARIANTS:")
    print("   (Large frequency differences between AFR and EUR)")
    print("-" * 100)
    print(f"{'Variant ID':<25} {'AF_AFR':<12} {'AF_EUR':<12} {'Pop Bias':<12} {'Raw Score':<12}")
    print("-" * 100)
    
    high_bias = calibrated_df.nlargest(5, 'pop_bias')
    for _, row in high_bias.iterrows():
        print(f"{row['variant_id']:<25} {row.get('af_afr', 0):<12.6f} {row.get('af_eur', 0):<12.6f} "
              f"{row['pop_bias']:<12.3f} {row['evo2_score']:<12.6f}")
    
    # === SUMMARY STATISTICS ===
    print("\n\n📊 CALIBRATION IMPACT SUMMARY:")
    print("-" * 80)
    print(f"Total variants with AF data: {len(calibrated_df)}")
    print(f"High bias variants (>0.5): {(calibrated_df['pop_bias'] > 0.5).sum()}")
    print(f"Extreme bias variants (>0.7): {(calibrated_df['pop_bias'] > 0.7).sum()}")
    print(f"\nMean correction magnitude:")
    print(f"  AFR: {calibrated_df['delta_afr'].mean():.6f}")
    print(f"  EUR: {calibrated_df['delta_eur'].mean():.6f}")
    print(f"\nMax correction magnitude:")
    print(f"  AFR: {calibrated_df['delta_afr'].max():.6f}")
    print(f"  EUR: {calibrated_df['delta_eur'].max():.6f}")
    
    # === EXPORT TOP CASES ===
    print("\n\n💾 Exporting detailed case studies...")
    
    # Combine top corrections and rescue cases
    case_studies = pd.concat([
        top_afr.head(3),
        top_eur.head(3),
        rescued_afr.head(3) if len(rescued_afr) > 0 else pd.DataFrame(),
        rescued_eur.head(3) if len(rescued_eur) > 0 else pd.DataFrame()
    ]).drop_duplicates(subset='variant_id')
    
    case_studies = case_studies[[
        'variant_id', 'chrom', 'pos_hg38', 'ref', 'alt',
        'evo2_score', 'calibrated_afr', 'calibrated_eur',
        'delta_afr', 'delta_eur', 'pop_bias',
        'af_afr', 'af_eur', 'func_class'
    ]]
    
    output_file = "results/calibration_case_studies.csv"
    case_studies.to_csv(output_file, index=False)
    print(f"✅ Saved {len(case_studies)} case study variants to {output_file}")
    
    print("\n" + "="*80)
    print("✅ CALIBRATION ANALYSIS COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()
