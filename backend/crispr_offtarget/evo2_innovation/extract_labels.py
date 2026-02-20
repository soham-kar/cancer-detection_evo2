"""
Extract ATAC-seq chromatin accessibility signals and prepare training data.
Combines Evo2 embeddings + CHANGE-seq reads + ATAC-seq signals.
"""
import json
import numpy as np
import pandas as pd
import pyBigWig
from pathlib import Path

def extract_atac_signals(
    change_seq_file="../../../data/change_seq/change_seq_evo2_input.csv",
    atac_bigwig="../../../data/change_seq/raw/GSM4498611_ATAC_FE.bdg.bw",
    output_file="data/atac_signals.json",
    n_bins=100,
    window_size=8000
):
    """
    Extract ATAC-seq signals in 100 bins across 8kb windows.
    """
    
    df = pd.read_csv(change_seq_file)
    print(f"Processing {len(df)} sites...")
    
    bw = pyBigWig.open(atac_bigwig)
    
    atac_data = []
    bin_size = window_size // n_bins
    
    for idx, row in df.iterrows():
        chrom = str(row['chrom'])
        center = int(row['chromStart'])
        start = max(0, center - window_size // 2)
        end = start + window_size
        
        # Extract signal in bins
        try:
            signals = []
            for i in range(n_bins):
                bin_start = start + i * bin_size
                bin_end = bin_start + bin_size
                
                # Get mean signal in bin
                vals = bw.stats(chrom, bin_start, bin_end, type="mean")
                signal = vals[0] if vals and vals[0] is not None else 0.0
                signals.append(float(signal))
            
            atac_data.append({
                'seq_id': f'seq_{idx}',
                'site_id': int(row['site_id']),
                'signal': signals
            })
            
        except Exception as e:
            print(f"  Warning: Failed for {chrom}:{start}-{end}: {e}")
            atac_data.append({
                'seq_id': f'seq_{idx}',
                'site_id': int(row['site_id']),
                'signal': [0.0] * n_bins
            })
        
        if (idx + 1) % 10000 == 0:
            print(f"  {idx+1}/{len(df)} processed")
    
    bw.close()
    
    # Save
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(atac_data, f)
    
    print(f"\nSaved {len(atac_data)} ATAC-seq signals to {output_file}")
    return atac_data


def create_training_labels(
    change_seq_file="../../../data/change_seq/change_seq_evo2_input.csv",
    output_file="data/change_seq_labels.json"
):
    """
    Create training labels from CHANGE-seq data.
    """
    
    df = pd.read_csv(change_seq_file)
    
    labels = []
    for idx, row in df.iterrows():
        labels.append({
            'seq_id': f'seq_{idx}',
            'site_id': int(row['site_id']),
            'chrom': str(row['chrom']),
            'start': int(row['chromStart']),
            'end': int(row['chromEnd']),
            'cleavage_count': float(row['change_seq_reads']),
            'log_reads': float(row['log_reads']),
            'normalized_reads': float(row['normalized_reads'])
        })
    
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(labels, f)
    
    print(f"Saved {len(labels)} labels to {output_file}")
    return labels


if __name__ == "__main__":
    print("Step 1: Creating CHANGE-seq labels...")
    create_training_labels()
    
    print("\nStep 2: Extracting ATAC-seq signals...")
    extract_atac_signals()
    
    print("\n✓ Ready for training! Run: python prepare_training_data.py")
