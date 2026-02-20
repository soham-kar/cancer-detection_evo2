"""
Prepare genomic coordinates for Evo2 embedding extraction.

Takes ATAC-seq peak locations and creates 8kb windows for embedding extraction.
Splits by chromosome into train/val/test sets.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Paths
DATA_DIR = Path(__file__).parent / "data" / "atac"
OUTPUT_PATH = DATA_DIR / "atac_coordinates.csv"

# Chromosome split for train/val/test
TRAIN_CHROMS = [f"chr{i}" for i in range(1, 18)]  # chr1-17
VAL_CHROMS = ["chr18", "chr19", "chr20"]
TEST_CHROMS = ["chr21", "chr22"]

# Window parameters
WINDOW_SIZE = 8192  # 8kb windows (Evo2 optimal)


def load_peaks(cell_type: str) -> pd.DataFrame:
    """Load peaks for a cell type."""
    path = DATA_DIR / f"{cell_type.lower()}_peaks.csv"
    if not path.exists():
        raise FileNotFoundError(f"Peak file not found: {path}")
    
    df = pd.read_csv(path)
    df["cell_type"] = cell_type
    logging.info(f"Loaded {len(df)} peaks from {cell_type}")
    return df


def create_windows(peaks: pd.DataFrame, window_size: int = WINDOW_SIZE) -> pd.DataFrame:
    """Create 8kb windows centered on peaks."""
    half_window = window_size // 2
    
    peaks = peaks.copy()
    peaks["window_start"] = peaks["center"] - half_window
    peaks["window_end"] = peaks["center"] + half_window
    
    # Filter out negative coordinates
    peaks = peaks[peaks["window_start"] >= 0]
    
    logging.info(f"Created {len(peaks)} windows of size {window_size}bp")
    return peaks


def assign_splits(df: pd.DataFrame) -> pd.DataFrame:
    """Assign train/val/test splits by chromosome."""
    df = df.copy()
    
    def get_split(chrom):
        if chrom in TRAIN_CHROMS:
            return "train"
        elif chrom in VAL_CHROMS:
            return "val"
        elif chrom in TEST_CHROMS:
            return "test"
        else:
            return "exclude"
    
    df["split"] = df["chrom"].apply(get_split)
    df = df[df["split"] != "exclude"]
    
    split_counts = df["split"].value_counts()
    logging.info(f"Split distribution:\n{split_counts}")
    
    return df


def filter_overlapping(df: pd.DataFrame, min_gap: int = 1000) -> pd.DataFrame:
    """Remove windows that overlap by more than min_gap."""
    df = df.sort_values(["chrom", "window_start"]).reset_index(drop=True)
    
    keep = [True] * len(df)
    for i in range(1, len(df)):
        if df.iloc[i]["chrom"] == df.iloc[i-1]["chrom"]:
            gap = df.iloc[i]["window_start"] - df.iloc[i-1]["window_end"]
            if gap < min_gap:
                keep[i] = False
    
    original_len = len(df)
    df = df[keep]
    removed = original_len - len(df)
    logging.info(f"Removed {removed} overlapping windows ({len(df)} remaining)")
    
    return df


def sample_balanced(df: pd.DataFrame, n_per_split: dict = None) -> pd.DataFrame:
    """Sample balanced number from each split."""
    if n_per_split is None:
        n_per_split = {"train": 3000, "val": 500, "test": 500}
    
    samples = []
    for split, n in n_per_split.items():
        split_df = df[df["split"] == split]
        n_sample = min(n, len(split_df))
        samples.append(split_df.sample(n=n_sample, random_state=42))
    
    result = pd.concat(samples, ignore_index=True)
    logging.info(f"Sampled {len(result)} total windows")
    return result


def main(max_per_split: int = None):
    logging.info("=" * 60)
    logging.info("Preparing ATAC-seq coordinates for Evo2 extraction")
    logging.info("=" * 60)
    
    # Load peaks from all cell types
    all_peaks = []
    for cell_type in ["K562", "GM12878"]:
        try:
            peaks = load_peaks(cell_type)
            all_peaks.append(peaks)
        except FileNotFoundError:
            logging.warning(f"No peaks for {cell_type}, skipping")
    
    if not all_peaks:
        logging.error("No peak files found. Run 01_download_encode_atac.py first.")
        return
    
    peaks = pd.concat(all_peaks, ignore_index=True)
    logging.info(f"Total peaks: {len(peaks)}")
    
    # Create windows
    windows = create_windows(peaks)
    
    # Filter overlapping
    windows = filter_overlapping(windows)
    
    # Assign splits
    windows = assign_splits(windows)
    
    # Sample if needed
    if max_per_split:
        n_per_split = {
            "train": min(max_per_split, 3000),
            "val": min(max_per_split // 6, 500),
            "test": min(max_per_split // 6, 500)
        }
        windows = sample_balanced(windows, n_per_split)
    
    # Add index for embedding matching
    windows = windows.reset_index(drop=True)
    windows["coord_id"] = windows.index
    
    # Select output columns
    output_cols = [
        "coord_id", "chrom", "window_start", "window_end", "center",
        "signal", "cell_type", "split"
    ]
    windows = windows[output_cols]
    
    # Save
    windows.to_csv(OUTPUT_PATH, index=False)
    logging.info(f"\n✅ Saved {len(windows)} coordinates to {OUTPUT_PATH}")
    
    # Summary
    logging.info("\n=== Summary ===")
    logging.info(f"Train: {(windows['split'] == 'train').sum()}")
    logging.info(f"Val:   {(windows['split'] == 'val').sum()}")
    logging.info(f"Test:  {(windows['split'] == 'test').sum()}")
    logging.info(f"Signal range: {windows['signal'].min():.1f} - {windows['signal'].max():.1f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=None, 
                        help="Max windows per split (for testing)")
    args = parser.parse_args()
    
    main(max_per_split=args.max)
