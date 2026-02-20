"""
Parse GSE149363 data to create training labels.
Downloads CHANGE-seq cleavage counts and ATAC-seq chromatin accessibility signals.
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path

def parse_gse149363_data(
    change_seq_file="GSE149363_CHANGE-seq_counts.txt",
    atac_seq_file="GSE149363_ATAC-seq_signals.txt",
    output_dir="data"
):
    """
    Parse GSE149363 supplementary files.
    
    Expected format:
    - CHANGE-seq: Tab-delimited with columns: chrom, start, end, guide_id, cleavage_count
    - ATAC-seq: Tab-delimited with columns: chrom, start, end, guide_id, signal_bins (100 values)
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Parse CHANGE-seq cleavage data
    print(f"Parsing {change_seq_file}...")
    change_df = pd.read_csv(change_seq_file, sep='\t')
    
    change_labels = []
    for idx, row in change_df.iterrows():
        change_labels.append({
            'seq_id': f'seq_{idx}',
            'chrom': row['chrom'],
            'start': int(row['start']),
            'end': int(row['end']),
            'guide_id': row.get('guide_id', f'guide_{idx}'),
            'cleavage_count': float(row['cleavage_count'])
        })
    
    with open(output_dir / 'change_seq_labels.json', 'w') as f:
        json.dump(change_labels, f, indent=2)
    
    print(f"  Saved {len(change_labels)} CHANGE-seq labels")
    
    # Parse ATAC-seq chromatin accessibility
    print(f"Parsing {atac_seq_file}...")
    atac_df = pd.read_csv(atac_seq_file, sep='\t')
    
    atac_signals = []
    for idx, row in atac_df.iterrows():
        # Extract 100-bin signal (columns 4-103)
        signal = [float(row[f'bin_{i}']) for i in range(100)]
        
        atac_signals.append({
            'seq_id': f'seq_{idx}',
            'signal': signal
        })
    
    with open(output_dir / 'atac_signals.json', 'w') as f:
        json.dump(atac_signals, f, indent=2)
    
    print(f"  Saved {len(atac_signals)} ATAC-seq signals")
    print(f"\nReady for training! Run: python prepare_training_data.py")

if __name__ == "__main__":
    # TODO: Download GSE149363 supplementary files first
    # https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE149363
    
    print("NOTE: Download GSE149363 data files first:")
    print("  1. Go to https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE149363")
    print("  2. Download supplementary files")
    print("  3. Place in this directory")
    print("  4. Update filenames in this script")
    
    # Uncomment when files are ready:
    # parse_gse149363_data()
