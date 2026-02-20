"""
Stratified Sampling for CHANGE-seq Evo2 Training
FIXED: Works with AVAILABLE embeddings, not full 202K dataset

Key insight: We only have 2,387 embeddings (site_ids 0-2386).
Must sample from THIS subset, then stratify by mismatch.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
CHANGE_SEQ_PATH = DATA_DIR / "change_seq" / "change_seq_evo2_input.csv"
FEATURES_CACHE = Path(__file__).parent / "features_cache"
OUTPUT_DIR = Path(__file__).parent


def get_available_embedding_ids(cache_dir: Path) -> set:
    """Get set of site_ids that have embeddings cached."""
    available = set()
    for f in cache_dir.glob("seq_*.npz"):
        try:
            seq_id = int(f.stem.split('_')[1])
            available.add(seq_id)
        except (ValueError, IndexError):
            continue
    return available


def create_stratified_from_available(
    df: pd.DataFrame,
    available_ids: set,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Create stratified sample from samples WITH embeddings.
    Oversample hard cases (0-3 mm) proportionally.
    """
    # Filter to samples with embeddings
    df_available = df[df['site_id'].isin(available_ids)].copy()
    logging.info(f"Samples with embeddings: {len(df_available)}")
    
    # Check mismatch distribution
    logging.info("\nMismatch distribution of available samples:")
    mm_dist = df_available['distance'].value_counts().sort_index()
    for mm, count in mm_dist.items():
        logging.info(f"  {mm} mm: {count}")
    
    # Stratified sampling: take ALL hard cases, undersample easy cases
    hard_cases = df_available[df_available['distance'] <= 3].copy()
    easy_cases = df_available[df_available['distance'] > 3].copy()
    
    logging.info(f"\nHard cases (0-3 mm): {len(hard_cases)}")
    logging.info(f"Easy cases (4+ mm): {len(easy_cases)}")
    
    # Take ALL hard cases (they're rare and valuable)
    # Sample easy cases to match OR cap at reasonable size
    n_easy_sample = min(len(easy_cases), len(hard_cases) * 3)  # 3:1 ratio max
    
    if n_easy_sample < len(easy_cases):
        # Stratify easy cases by mismatch
        easy_sampled = []
        for mm in easy_cases['distance'].unique():
            mm_subset = easy_cases[easy_cases['distance'] == mm]
            # Proportional sampling
            n_take = max(10, int(n_easy_sample * len(mm_subset) / len(easy_cases)))
            n_take = min(n_take, len(mm_subset))
            easy_sampled.append(mm_subset.sample(n=n_take, random_state=random_state))
        easy_final = pd.concat(easy_sampled, ignore_index=True)
    else:
        easy_final = easy_cases
    
    # Also stratify by activity within each mismatch bin
    def stratify_by_activity(subset, n_bins=4, random_state=42):
        """Further stratify by log_reads to ensure activity diversity."""
        if len(subset) < n_bins * 2:
            return subset
        try:
            subset = subset.copy()
            subset['activity_bin'] = pd.qcut(subset['log_reads'], q=n_bins, labels=False, duplicates='drop')
            # Sample equally from bins
            sampled = []
            for b in subset['activity_bin'].unique():
                bin_data = subset[subset['activity_bin'] == b]
                sampled.append(bin_data)  # Take all from each bin
            return pd.concat(sampled, ignore_index=True)
        except:
            return subset
    
    hard_stratified = stratify_by_activity(hard_cases, random_state=random_state)
    easy_stratified = stratify_by_activity(easy_final, random_state=random_state)
    
    # Combine
    result = pd.concat([hard_stratified, easy_stratified], ignore_index=True)
    result = result.sample(frac=1.0, random_state=random_state)  # Shuffle
    
    logging.info(f"\nFinal stratified sample: {len(result)}")
    logging.info("Distribution:")
    logging.info(result['distance'].value_counts().sort_index())
    
    return result


def create_train_test_split(
    df: pd.DataFrame,
    test_fraction: float = 0.2,
    random_state: int = 42
) -> tuple:
    """Split with stratification by mismatch."""
    train_parts = []
    test_parts = []
    
    for mm in df['distance'].unique():
        mm_data = df[df['distance'] == mm]
        if len(mm_data) < 5:
            # Too few for split, put all in train
            train_parts.append(mm_data)
            continue
        
        n_test = max(1, int(len(mm_data) * test_fraction))
        test_subset = mm_data.sample(n=n_test, random_state=random_state)
        train_subset = mm_data.drop(test_subset.index)
        
        train_parts.append(train_subset)
        test_parts.append(test_subset)
    
    train = pd.concat(train_parts, ignore_index=True).sample(frac=1.0, random_state=random_state)
    test = pd.concat(test_parts, ignore_index=True).sample(frac=1.0, random_state=random_state) if test_parts else pd.DataFrame()
    
    logging.info(f"\nTrain: {len(train)} | Test: {len(test)}")
    
    return train, test


def load_embeddings_for_samples(df: pd.DataFrame, cache_dir: Path) -> tuple:
    """Load embeddings aligned with dataframe."""
    embeddings = []
    valid_rows = []
    
    for idx, row in df.iterrows():
        site_id = int(row['site_id'])
        npz_path = cache_dir / f"seq_{site_id}.npz"
        
        if not npz_path.exists():
            continue
        
        try:
            data = np.load(npz_path, allow_pickle=True)
            parts = []
            if 'global_embedding' in data:
                parts.append(data['global_embedding'])
            if 'center_embedding' in data:
                parts.append(data['center_embedding'])
            
            if parts:
                emb = np.concatenate(parts)
                embeddings.append(emb)
                valid_rows.append(row)
        except Exception as e:
            logging.warning(f"Error loading {npz_path}: {e}")
    
    embeddings_array = np.stack(embeddings) if embeddings else None
    valid_df = pd.DataFrame(valid_rows).reset_index(drop=True)
    
    return embeddings_array, valid_df


def main():
    logging.info("=" * 60)
    logging.info("CHANGE-seq Stratified Training - Working with Available Embeddings")
    logging.info("=" * 60)
    
    # 1. Load CHANGE-seq data
    df = pd.read_csv(CHANGE_SEQ_PATH)
    logging.info(f"Loaded CHANGE-seq: {len(df)} total samples")
    
    # 2. Get available embedding IDs
    available_ids = get_available_embedding_ids(FEATURES_CACHE)
    logging.info(f"Available embeddings: {len(available_ids)}")
    
    if len(available_ids) < 100:
        logging.error("Not enough embeddings! Run Modal extraction first.")
        return
    
    # 3. Create stratified sample from available
    stratified = create_stratified_from_available(df, available_ids, random_state=42)
    
    # 4. Split
    train_df, test_df = create_train_test_split(stratified, test_fraction=0.2)
    
    # 5. Load actual embeddings
    logging.info("\nLoading embeddings...")
    X_train, train_final = load_embeddings_for_samples(train_df, FEATURES_CACHE)
    X_test, test_final = load_embeddings_for_samples(test_df, FEATURES_CACHE)
    
    logging.info(f"Train embeddings: {X_train.shape}")
    logging.info(f"Test embeddings: {X_test.shape}")
    
    # 6. Save
    # Save CSVs
    train_final.to_csv(OUTPUT_DIR / "stratified_train.csv", index=False)
    test_final.to_csv(OUTPUT_DIR / "stratified_test.csv", index=False)
    
    # Save embeddings + labels as NPZ for training
    np.savez(
        OUTPUT_DIR / "stratified_training_data.npz",
        X_train=X_train,
        y_train=train_final['log_reads'].values,
        mm_train=train_final['distance'].values,
        X_test=X_test,
        y_test=test_final['log_reads'].values,
        mm_test=test_final['distance'].values
    )
    
    logging.info(f"\n✅ Saved to {OUTPUT_DIR}")
    logging.info("   - stratified_train.csv")
    logging.info("   - stratified_test.csv")
    logging.info("   - stratified_training_data.npz")
    
    # Show final distribution
    logging.info("\n=== Final Training Distribution ===")
    logging.info("Train by mismatch:")
    logging.info(train_final['distance'].value_counts().sort_index())
    logging.info("\nTest by mismatch:")
    logging.info(test_final['distance'].value_counts().sort_index())
    
    # Key metric: how many hard cases?
    n_hard_train = (train_final['distance'] <= 3).sum()
    n_hard_test = (test_final['distance'] <= 3).sum()
    logging.info(f"\nHard cases (0-3 mm): Train={n_hard_train}, Test={n_hard_test}")


if __name__ == "__main__":
    main()
