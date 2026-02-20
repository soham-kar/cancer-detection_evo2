"""
Prepare training data for chromatin-aware CRISPR prediction.
Combines Evo2 embeddings with CHANGE-seq cleavage counts and ATAC-seq signals.
"""
import json
import numpy as np
from pathlib import Path

def prepare_training_data(
    features_dir="features_cache",
    change_seq_data="data/change_seq_labels.json",
    output_file="training_data.npz"
):
    """
    Combine Evo2 embeddings with CHANGE-seq cleavage counts.
    
    Expected format:
    - change_seq_labels.json: [{"seq_id": "seq_0", "cleavage_count": 42, "log_reads": 3.5}, ...]
    """
    
    features_dir = Path(features_dir)
    feature_files = sorted(features_dir.glob("seq_*.npz"))
    
    print(f"Found {len(feature_files)} feature files")
    
    # Load labels
    with open(change_seq_data) as f:
        labels = {item['seq_id']: item for item in json.load(f)}
    
    print(f"Loaded {len(labels)} labels")
    
    X_features = []
    y_cleavage = []
    y_log_reads = []
    seq_ids = []
    
    for feat_file in feature_files:
        seq_id = feat_file.stem
        
        if seq_id not in labels:
            continue
        
        # Load Evo2 embeddings
        data = np.load(feat_file)
        global_emb = data['global_embedding']  # (512,) float16
        center_emb = data['center_embedding']  # (512,) float16
        
        # Concatenate: (1024,)
        combined = np.concatenate([global_emb, center_emb])
        
        # Get targets
        cleavage_count = labels[seq_id]['cleavage_count']
        log_reads = labels[seq_id]['log_reads']
        
        X_features.append(combined)
        y_cleavage.append(cleavage_count)
        y_log_reads.append(log_reads)
        seq_ids.append(seq_id)
    
    # Convert to arrays
    X = np.array(X_features, dtype=np.float32)  # (N, 1024)
    y_cleavage = np.array(y_cleavage, dtype=np.float32)  # (N,)
    y_log_reads = np.array(y_log_reads, dtype=np.float32)  # (N,)
    
    # Save
    np.savez(
        output_file,
        X=X,
        y_cleavage=y_cleavage,
        y_log_reads=y_log_reads,
        seq_ids=seq_ids
    )
    
    print(f"\nSaved training data:")
    print(f"  X shape: {X.shape}")
    print(f"  y_cleavage shape: {y_cleavage.shape}")
    print(f"  y_log_reads shape: {y_log_reads.shape}")
    print(f"  Samples: {len(seq_ids)}")
    
    return X, y_cleavage, y_log_reads, seq_ids

if __name__ == "__main__":
    prepare_training_data()
