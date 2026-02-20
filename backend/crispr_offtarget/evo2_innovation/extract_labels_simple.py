"""
Create training labels from CHANGE-seq data (no ATAC-seq for now).
"""
import json
import pandas as pd
from pathlib import Path

def create_training_labels(
    change_seq_file="../../../data/change_seq/change_seq_evo2_input.csv",
    output_file="data/change_seq_labels.json"
):
    """Create training labels from CHANGE-seq data."""
    
    df = pd.read_csv(change_seq_file)
    print(f"Processing {len(df)} sites...")
    
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
        
        if (idx + 1) % 50000 == 0:
            print(f"  {idx+1}/{len(df)} processed")
    
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(labels, f)
    
    print(f"\n✓ Saved {len(labels)} labels to {output_file}")
    return labels

if __name__ == "__main__":
    create_training_labels()
    print("\nNext: python prepare_training_data.py")
