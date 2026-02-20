
import pandas as pd
from pathlib import Path

def get_variant_type(ref, alt):
    if len(ref) == 1 and len(alt) == 1:
        return 'SNV'
    elif len(ref) > len(alt):
        return 'DEL'
    elif len(ref) < len(alt):
        return 'INS'
    else:
        return 'MNV' # Multi-nucleotide variant or complex

def main():
    # Paths
    input_path = Path(r"D:/project/biotech-evo2/backend/population_aware/data/real_variants_for_evo2.csv")
    output_path = Path("D:/project/biotech-evo2/backend/somatic_driver_analysis/data/processed/india_variants.csv")
    
    print(f"Reading from {input_path}")
    df = pd.read_csv(input_path)
    
    # 1. Rename columns
    # chromosome,position,reference,alternative,gene_symbol,source
    rename_map = {
        'chromosome': 'chrom',
        'position': 'pos',
        'reference': 'ref',
        'alternative': 'alt',
        'gene_symbol': 'gene'
    }
    df = df.rename(columns=rename_map)
    
    # 2. Add patient_id
    # We will assume each source file represents a patient or just assign a dummy one 'P_IND_001'
    # The source column has filenames like "20201028_CCDG_14151_B01_GRM_WGS_2020-08-05_chr22.recalibrated_variants.vcf"
    # We can use the first part of filename as patient ID
    df['patient_id'] = df['source'].apply(lambda x: x.split('_')[0] if isinstance(x, str) else 'Unknown')
    
    # 3. Add variant_type
    df['variant_type'] = df.apply(lambda row: get_variant_type(str(row['ref']), str(row['alt'])), axis=1)
    
    # 4. Fill missing genes
    df['gene'] = df['gene'].fillna('Unknown')
    
    # Ensure directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Select and order columns
    cols = ['patient_id', 'chrom', 'pos', 'ref', 'alt', 'gene', 'variant_type']
    df_final = df[cols]
    
    print(f"Converting {len(df)} variants...")
    print(df_final.head())
    
    df_final.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()
