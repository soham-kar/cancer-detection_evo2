"""
Modal script to extract ATAC-seq signals and prepare training data in the cloud.
Avoids Windows pyBigWig compilation issues.
"""
import modal

app = modal.App("prepare-chromatin-training-data")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy", "pandas", "pyBigWig")
)

feature_cache = modal.Volume.from_name("evo2-features-cache", create_if_missing=True)
data_vol = modal.Volume.from_name("crispr-data", create_if_missing=True)

@app.function(
    image=image,
    volumes={"/features": feature_cache, "/data": data_vol},
    timeout=7200
)
def extract_atac_and_prepare_training(n_bins=100, window_size=8000):
    """Extract ATAC-seq signals and prepare training data."""
    import json
    import numpy as np
    import pandas as pd
    import pyBigWig
    from pathlib import Path
    
    # Load CHANGE-seq data
    change_seq_file = "/data/change_seq/change_seq_evo2_input.csv"
    atac_bigwig = "/data/change_seq/raw/GSM4498611_ATAC_FE.bdg.bw"
    
    print("Loading CHANGE-seq data...")
    df = pd.read_csv(change_seq_file)
    print(f"  {len(df)} sites")
    
    # Extract ATAC-seq signals
    print("\nExtracting ATAC-seq signals...")
    bw = pyBigWig.open(atac_bigwig)
    bin_size = window_size // n_bins
    
    atac_data = []
    labels = []
    
    for idx, row in df.iterrows():
        chrom = str(row['chrom'])
        center = int(row['chromStart'])
        start = max(0, center - window_size // 2)
        end = start + window_size
        
        # Extract ATAC signal
        try:
            signals = []
            for i in range(n_bins):
                bin_start = start + i * bin_size
                bin_end = bin_start + bin_size
                vals = bw.stats(chrom, bin_start, bin_end, type="mean")
                signal = vals[0] if vals and vals[0] is not None else 0.0
                signals.append(float(signal))
        except:
            signals = [0.0] * n_bins
        
        atac_data.append({
            'seq_id': f'seq_{idx}',
            'signal': signals
        })
        
        # Create label
        labels.append({
            'seq_id': f'seq_{idx}',
            'site_id': int(row['site_id']),
            'chrom': chrom,
            'start': int(row['chromStart']),
            'cleavage_count': float(row['change_seq_reads']),
            'log_reads': float(row['log_reads']),
            'normalized_reads': float(row['normalized_reads'])
        })
        
        if (idx + 1) % 10000 == 0:
            print(f"  {idx+1}/{len(df)} processed")
    
    bw.close()
    
    # Combine with Evo2 embeddings (only 1000 samples)
    print("\nCombining with Evo2 embeddings...")
    feature_files = sorted(Path("/features").glob("seq_*.npz"))
    print(f"  Found {len(feature_files)} cached embeddings")
    
    # Only process samples that have embeddings
    labels_dict = {item['seq_id']: item for item in labels[:len(feature_files)]}
    atac_dict = {item['seq_id']: item['signal'] for item in atac_data[:len(feature_files)]}
    
    X_features = []
    y_cleavage = []
    y_log_reads = []
    y_atac = []
    seq_ids = []
    
    for feat_file in feature_files:
        seq_id = feat_file.stem
        
        if seq_id not in labels_dict or seq_id not in atac_dict:
            continue
        
        # Load Evo2 embeddings
        data = np.load(feat_file)
        global_emb = data['global_embedding']
        center_emb = data['center_embedding']
        combined = np.concatenate([global_emb, center_emb])
        
        X_features.append(combined)
        y_cleavage.append(labels_dict[seq_id]['cleavage_count'])
        y_log_reads.append(labels_dict[seq_id]['log_reads'])
        y_atac.append(atac_dict[seq_id])
        seq_ids.append(seq_id)
    
    # Convert to arrays
    X = np.array(X_features, dtype=np.float32)
    y_cleavage = np.array(y_cleavage, dtype=np.float32)
    y_log_reads = np.array(y_log_reads, dtype=np.float32)
    y_atac = np.array(y_atac, dtype=np.float32)
    
    # Save to volume
    output_path = "/data/training_data.npz"
    np.savez(
        output_path,
        X=X,
        y_cleavage=y_cleavage,
        y_log_reads=y_log_reads,
        y_atac=y_atac,
        seq_ids=seq_ids
    )
    
    data_vol.commit()
    
    print(f"\n✓ Saved training data:")
    print(f"  X shape: {X.shape}")
    print(f"  y_cleavage shape: {y_cleavage.shape}")
    print(f"  y_log_reads shape: {y_log_reads.shape}")
    print(f"  y_atac shape: {y_atac.shape}")
    print(f"\nDownload with: modal volume get crispr-data training_data.npz .")
    
    return {
        'n_samples': len(seq_ids),
        'feature_dim': X.shape[1],
        'atac_bins': y_atac.shape[1]
    }

@app.local_entrypoint()
def main():
    """Run ATAC extraction and training data preparation."""
    result = extract_atac_and_prepare_training.remote()
    print(f"\n✓ Complete: {result}")
