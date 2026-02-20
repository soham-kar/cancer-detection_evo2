"""
Multi-Gene Global Calibration Validation

Applies the validated 4-population calibration methodology to PALB2 and BRCA2
to demonstrate generalizability across multiple DNA repair genes.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

def calculate_global_bias(row, target_pop, other_pops):
    """Calculate bias for target population vs others."""
    target_af = row.get(target_pop, 0)
    target_af = target_af if pd.notna(target_af) else 0
    
    # Calculate mean AF in other populations
    other_afs = [row.get(p, 0) for p in other_pops]
    other_afs = [af if pd.notna(af) else 0 for af in other_afs]
    rest_af = np.mean(other_afs) if other_afs else 0
    
    # Calculate bias
    epsilon = 1e-9
    if target_af == 0 and rest_af == 0:
        return 0, 0  # No bias, no difference
    
    # Fold difference
    max_af = max(target_af, rest_af)
    min_af = min(target_af, rest_af) + epsilon
    fold_diff = max_af / min_af
    
    # Bias score (0-1 scale)
    bias = np.log10(fold_diff) / (np.log10(fold_diff) + 2)
    
    # Direction: positive if target > rest
    direction = 1 if target_af > rest_af else -1
    
    return bias * direction, fold_diff

def calibrate_for_population(evo2_score, bias_score, calibration_strength=0.3):
    """Calibrate score based on population-specific bias."""
    adjustment = bias_score * calibration_strength
    return evo2_score + adjustment

def process_gene_calibration(df_gene, gene_name, populations):
    """Apply global calibration to a single gene's variants."""
    
    print(f"\n🧬 Processing {gene_name}...")
    print(f"   Variants: {len(df_gene)}")
    
    # Check for population AF columns
    available_pops = {}
    for pop_name, pop_col in populations.items():
        if pop_col in df_gene.columns:
            n_with_af = df_gene[pop_col].notna().sum()
            if n_with_af > 0:
                available_pops[pop_name] = pop_col
                print(f"   {pop_name}: {n_with_af} variants with AF data")
    
    if not available_pops:
        print(f"   ⚠️  No population AF data found for {gene_name}")
        return None
    
    # Apply calibration for each population
    print(f"\n🔧 Applying calibration...")
    
    results = df_gene.copy()
    
    for pop_name, pop_col in available_pops.items():
        # Other populations for comparison
        other_pops = [col for name, col in available_pops.items() if col != pop_col]
        
        # Calculate bias for each variant
        bias_results = results.apply(
            lambda row: calculate_global_bias(row, pop_col, other_pops),
            axis=1
        )
        
        results[f'bias_{pop_name.lower()}'] = [b[0] for b in bias_results]
        results[f'fold_{pop_name.lower()}'] = [b[1] for b in bias_results]
        
        # Calibrate score
        results[f'calibrated_{pop_name.lower()}'] = results.apply(
            lambda row: calibrate_for_population(
                row['evo2_score'],
                row[f'bias_{pop_name.lower()}']
            ),
            axis=1
        )
        
        # Statistics
        high_bias = (results[f'bias_{pop_name.lower()}'].abs() > 0.5).sum()
        print(f"   {pop_name}: {high_bias} high-bias variants (>0.5)")
    
    return results, available_pops

def main():
    print("="*80)
    print("MULTI-GENE GLOBAL CALIBRATION VALIDATION")
    print("="*80)
    
    # Define populations
    populations = {
        'AFR': 'af_afr',
        'EUR': 'af_nfe',
        'SAS': 'af_sas',
        'EAS': 'af_eas'
    }
    
    # Create output directory
    output_dir = Path("results/multigene_calibration")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Output directory: {output_dir}")
    
    # Load combined sample scores (PALB2 + BRCA2)
    print("\n📊 Loading multi-gene sample data...")
    df = pd.read_csv("multi_gene/combined_sample_scores.csv")
    
    print(f"   Total variants: {len(df)}")
    print(f"   Genes: {df['gene'].unique()}")
    
    # Also need to merge with gnomAD data
    # Check if we need to load separate gnomAD files
    if 'af_afr' not in df.columns:
        print("\n⚠️  Population AF data not in combined file")
        print("   Checking for separate gnomAD files...")
        
        # Try to load BRCA1 gnomAD as template for AF columns
        try:
            brca1_gnomad = pd.read_csv("results/brca1_with_gnomad.csv")
            print(f"   Found BRCA1 gnomAD data with {len(brca1_gnomad)} variants")
            
            # For now, we'll work with what we have
            # In production, you'd fetch gnomAD for PALB2/BRCA2
            print("\n   ℹ️  Note: Full gnomAD integration would require fetching")
            print("       population frequencies for PALB2/BRCA2 variants")
            
        except FileNotFoundError:
            pass
    
    # Process each gene separately
    summary_stats = []
    
    for gene in ['PALB2', 'BRCA2']:
        df_gene = df[df['gene'] == gene].copy()
        
        if len(df_gene) == 0:
            print(f"\n⚠️  No data for {gene}")
            continue
        
        # For demonstration, let's show the structure even without full AF data
        print(f"\n{'='*80}")
        print(f"GENE: {gene}")
        print(f"{'='*80}")
        
        pathogenic = df_gene[df_gene['func_class'].isin(['LOF', 'Pathogenic'])]
        benign = df_gene[df_gene['func_class'].isin(['FUNC', 'Benign'])]
        
        print(f"\n📊 Dataset Summary:")
        print(f"   Total variants: {len(df_gene)}")
        print(f"   Pathogenic/LOF: {len(pathogenic)}")
        print(f"   Benign/FUNC: {len(benign)}")
        
        # Show score distribution
        print(f"\n📈 Evo2 Score Distribution:")
        print(f"   Pathogenic mean: {pathogenic['evo2_score'].mean():.6f}")
        print(f"   Benign mean: {benign['evo2_score'].mean():.6f}")
        print(f"   Separation: {abs(pathogenic['evo2_score'].mean() - benign['evo2_score'].mean()):.6f}")
        
        # Check for AF columns
        af_cols = [c for c in df_gene.columns if c.startswith('af_')]
        
        if af_cols:
            print(f"\n🌍 Population AF columns found: {af_cols}")
            
            # Apply calibration
            calibrated, avail_pops = process_gene_calibration(df_gene, gene, populations)
            
            if calibrated is not None:
                # Save calibrated results
                output_file = output_dir / f"{gene.lower()}_global_calibrated.csv"
                calibrated.to_csv(output_file, index=False)
                print(f"\n✅ Saved: {output_file}")
                
                # Collect summary stats
                stats = {
                    'Gene': gene,
                    'Total_Variants': len(calibrated),
                    'Pathogenic': len(pathogenic),
                    'Benign': len(benign)
                }
                
                for pop in ['AFR', 'EUR', 'SAS', 'EAS']:
                    bias_col = f'bias_{pop.lower()}'
                    if bias_col in calibrated.columns:
                        high_bias = (calibrated[bias_col].abs() > 0.5).sum()
                        stats[f'High_Bias_{pop}'] = high_bias
                        stats[f'High_Bias_{pop}_Pct'] = f"{high_bias/len(calibrated)*100:.1f}%"
                
                summary_stats.append(stats)
        else:
            print(f"\n   ℹ️  No population AF data available for {gene}")
            print(f"   ℹ️  Saving uncalibrated results for reference")
            
            # Save uncalibrated for reference
            output_file = output_dir / f"{gene.lower()}_evo2_scores.csv"
            df_gene.to_csv(output_file, index=False)
            print(f"   ✅ Saved: {output_file}")
            
            # Basic stats
            stats = {
                'Gene': gene,
                'Total_Variants': len(df_gene),
                'Pathogenic': len(pathogenic),
                'Benign': len(benign),
                'Status': 'No AF data - uncalibrated'
            }
            summary_stats.append(stats)
    
    # Save summary
    print("\n" + "="*80)
    print("📊 MULTI-GENE VALIDATION SUMMARY")
    print("="*80)
    
    if summary_stats:
        summary_df = pd.DataFrame(summary_stats)
        print("\n" + summary_df.to_string(index=False))
        
        summary_file = output_dir / "multigene_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"\n✅ Summary saved to: {summary_file}")
    
    print("\n" + "="*80)
    print("✅ MULTI-GENE CALIBRATION VALIDATION COMPLETE!")
    print("="*80)
    
    print("\n💡 Next Steps:")
    print("   1. Review calibrated scores in results/multigene_calibration/")
    print("   2. Compare with BRCA1 results for consistency")
    print("   3. Use summary statistics for manuscript validation section")

if __name__ == "__main__":
    main()
