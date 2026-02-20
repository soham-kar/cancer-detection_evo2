"""
Parallel ATAC-seq extraction with checkpointing and resume capability.
"""
import modal

app = modal.App("prepare-chromatin-parallel")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy", "pandas", "pyBigWig")
)

feature_cache = modal.Volume.from_name("evo2-features-cache", create_if_missing=True)
data_vol = modal.Volume.from_name("crispr-data", create_if_missing=True)

@app.function(
    image=image,
    volumes={"/data": data_vol},
    timeout=600
)
def extract_atac_chunk(chunk_data, n_bins=100, window_size=8000):
    """Extract ATAC-seq for a chunk of samples."""
    import pyBigWig
    import numpy as np
    
    atac_bigwig = "/data/change_seq/raw/GSM4498611_ATAC_FE.bdg.bw"
    bw = pyBigWig.open(atac_bigwig)
    bin_size = window_size // n_bins
    
    results = []
    for item in chunk_data:
        idx, chrom, chromStart, reads, log_reads, norm_reads = item
        
        center = int(chromStart)
        start = max(0, center - window_size // 2)
        end = start + window_size
        
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
        
        results.append({
            'seq_id': f'seq_{idx}',
            'site_id': idx,
            'chrom': chrom,
            'start': int(chromStart),
            'cleavage_count': float(reads),
            'log_reads': float(log_reads),
            'normalized_reads': float(norm_reads),
            'atac_signal': signals
        })
    
    bw.close()
    return results

@app.function(
    image=image,
    volumes={"/features": feature_cache, "/data": data_vol},
    timeout=3600
)
def combine_and_save(checkpoint_file="/data/atac_checkpoint.npz"):
    """Combine ATAC results with Evo2 embeddings and save."""
    import numpy as np
    import json
    from pathlib import Path
    
    # Load checkpoint
    checkpoint = np.load(checkpoint_file, allow_pickle=True)
    all_results = checkpoint['results'].tolist()
    
    print(f"Loaded {len(all_results)} ATAC results from checkpoint")
    
    # Load Evo2 embeddings
    feature_files = sorted(Path("/features").glob("seq_*.npz"))
    print(f"Found {len(feature_files)} Evo2 embeddings")
    
    # Create lookup
    results_dict = {item['seq_id']: item for item in all_results}
    
    X_features = []
    y_cleavage = []
    y_log_reads = []
    y_atac = []
    seq_ids = []
    
    for feat_file in feature_files:
        seq_id = feat_file.stem
        
        if seq_id not in results_dict:
            continue
        
        # Load Evo2 embeddings
        data = np.load(feat_file)
        global_emb = data['global_embedding']
        center_emb = data['center_embedding']
        combined = np.concatenate([global_emb, center_emb])
        
        result = results_dict[seq_id]
        X_features.append(combined)
        y_cleavage.append(result['cleavage_count'])
        y_log_reads.append(result['log_reads'])
        y_atac.append(result['atac_signal'])
        seq_ids.append(seq_id)
    
    # Convert to arrays
    X = np.array(X_features, dtype=np.float32)
    y_cleavage = np.array(y_cleavage, dtype=np.float32)
    y_log_reads = np.array(y_log_reads, dtype=np.float32)
    y_atac = np.array(y_atac, dtype=np.float32)
    
    # Save final training data
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
    print(f"  X: {X.shape}")
    print(f"  y_cleavage: {y_cleavage.shape}")
    print(f"  y_log_reads: {y_log_reads.shape}")
    print(f"  y_atac: {y_atac.shape}")
    
    return len(seq_ids)

@app.local_entrypoint()
def main(chunk_size: int = 5000, resume: bool = False):
    """
    Run parallel ATAC extraction with checkpointing.
    
    Args:
        chunk_size: Number of samples per parallel job
        resume: Skip ATAC extraction and just combine results
    """
    import pandas as pd
    import numpy as np
    from pathlib import Path
    
    if not resume:
        # Load CHANGE-seq data
        print("Loading CHANGE-seq data...")
        csv_path = r"D:\project\biotech-evo2\backend\crispr_offtarget\data\change_seq\change_seq_evo2_input.csv"
        df = pd.read_csv(csv_path)
        print(f"  {len(df)} total sites")
        
        # Prepare chunks
        chunks = []
        for i in range(0, len(df), chunk_size):
            chunk_df = df.iloc[i:i+chunk_size]
            chunk_data = [
                (idx, row['chrom'], row['chromStart'], row['change_seq_reads'], 
                 row['log_reads'], row['normalized_reads'])
                for idx, row in chunk_df.iterrows()
            ]
            chunks.append(chunk_data)
        
        print(f"\nProcessing {len(chunks)} chunks in parallel...")
        
        # Process chunks in parallel
        results = []
        for chunk_results in extract_atac_chunk.map(chunks):
            results.extend(chunk_results)
            print(f"  Completed: {len(results)}/{len(df)}")
        
        # Save checkpoint
        print("\nSaving checkpoint...")
        checkpoint_data = np.array(results, dtype=object)
        
        # Upload checkpoint to Modal volume
        checkpoint_path = Path("atac_checkpoint.npz")
        np.savez(checkpoint_path, results=checkpoint_data)
        
        import subprocess
        subprocess.run([
            "modal", "volume", "put", "crispr-data",
            str(checkpoint_path), "atac_checkpoint.npz"
        ])
        
        print(f"✓ Checkpoint saved: {len(results)} samples")
    
    # Combine with Evo2 embeddings
    print("\nCombining with Evo2 embeddings...")
    n_samples = combine_and_save.remote()
    
    print(f"\n✓ Complete! {n_samples} samples ready")
    
    # Auto-download result
    print("\nDownloading training_data.npz...")
    import subprocess
    result = subprocess.run([
        "modal", "volume", "get", "crispr-data",
        "training_data.npz", "training_data.npz"
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✓ Downloaded: training_data.npz")
    else:
        print(f"Download failed: {result.stderr}")
        print("Manual download: modal volume get crispr-data training_data.npz .")
