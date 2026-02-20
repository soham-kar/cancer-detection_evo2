"""
Parse CHANGE-seq Supplementary Tables (v2)

Correctly handles the Excel structure with headers.
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent

def parse_change_seq():
    """Parse CHANGE-seq Table 3 with off-target sites."""
    xlsx_path = DATA_DIR / "41587_2020_555_MOESM3_ESM.xlsx"
    
    print("=" * 60)
    print("Parsing CHANGE-seq Table 3")
    print("=" * 60)
    
    # Read with row 0 as header
    df = pd.read_excel(xlsx_path, sheet_name='CHANGE-seq_Supp_Table_3', header=0)
    
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst 3 rows:")
    print(df.head(3))
    
    # --- Prepare for Evo2 ---
    output = pd.DataFrame()
    output['site_id'] = range(len(df))
    output['chrom'] = df['chrom']
    output['chromStart'] = df['chromStart']
    output['chromEnd'] = df['chromEnd'] if 'chromEnd' in df.columns else df['chromStart'] + 23
    
    # Find reads column
    reads_cols = [c for c in df.columns if 'read' in c.lower() or 'count' in c.lower()]
    print(f"\nReads columns found: {reads_cols}")
    
    if reads_cols:
        output['change_seq_reads'] = df[reads_cols[0]]
    else:
        # Try to find by position (usually column 5-7)
        print("Looking for numeric column for reads...")
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64'] and col not in ['chromStart', 'chromEnd', 'distance']:
                output['change_seq_reads'] = df[col]
                print(f"Using '{col}' as reads column")
                break
    
    # Target sequence (gRNA + PAM)
    output['target_sequence'] = df['target'] if 'target' in df.columns else df.iloc[:, -1]
    
    # Distance from on-target
    if 'distance' in df.columns:
        output['distance'] = df['distance']
    
    # Add log-transformed reads for regression
    if 'change_seq_reads' in output.columns:
        output['log_reads'] = np.log1p(output['change_seq_reads'])
        output['normalized_reads'] = output['log_reads'] / output['log_reads'].max()
    
    # Remove rows with missing chrom
    output = output[output['chrom'].notna() & (output['chrom'] != '')]
    
    print(f"\n{'='*60}")
    print(f"PREPARED DATA")
    print(f"{'='*60}")
    print(f"Total sites: {len(output)}")
    if 'change_seq_reads' in output.columns:
        print(f"Read count range: {output['change_seq_reads'].min():.0f} - {output['change_seq_reads'].max():.0f}")
    print(f"\nSample:")
    print(output.head(3))
    
    # Save
    output_path = DATA_DIR / "change_seq_evo2_input.csv"
    output.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")
    
    return output


if __name__ == "__main__":
    df = parse_change_seq()
    
    print(f"\n{'='*60}")
    print("NEXT STEPS")
    print("="*60)
    print("1. Extract 8kb genomic context for each site (pyfaidx + hg38)")
    print("2. Run Evo2 feature extraction on Modal")
    print("3. Query ATAC-seq bigWig for chromatin accessibility at each site")
    print("4. Train chromatin-aware cleavage predictor")
