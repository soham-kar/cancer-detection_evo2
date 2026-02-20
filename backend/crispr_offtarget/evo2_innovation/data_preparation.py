"""
Data Preparation for CHANGE-seq and Other Quantitative Datasets

Downloads and prepares datasets with continuous cleavage rates.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import subprocess
import sys

from config import DATA_DIR, SEED


def download_change_seq():
    """
    Download CHANGE-seq data.
    
    CHANGE-seq provides quantitative cleavage frequencies across thousands of sites.
    
    Data Sources:
    - Raw FASTQ: NCBI SRA BioProject PRJNA625995 (requires processing)
    - Processed: Nature paper Supplementary Tables (recommended)
    
    Reference: Lazzarotto et al. Nature Biotechnology 2020
    DOI: 10.1038/s41587-020-0555-7
    """
    output_dir = DATA_DIR / "change_seq"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("CHANGE-seq Data Download")
    print("=" * 60)
    
    print("\nOption 1: PROCESSED DATA (Recommended)")
    print("-" * 40)
    print("Go to: https://www.nature.com/articles/s41587-020-0555-7")
    print("Download: Supplementary Tables 3 & 4 (Excel with read counts)")
    print("Save to:", output_dir)
    
    print("\nOption 2: RAW SEQUENCING DATA")
    print("-" * 40)
    print("SRA BioProject: https://www.ncbi.nlm.nih.gov/bioproject/PRJNA625995")
    print("⚠️  Requires FASTQ processing pipeline")
    
    print("\nExpected output structure:")
    print("""
    change_seq/
    ├── Supplementary_Table_3.xlsx   # Off-target sites with read counts
    ├── Supplementary_Table_4.xlsx   # Specificity ratios
    └── processed/                   # Parsed CSV files
    """)
    
    return output_dir


def prepare_kleinstiver_quantitative():
    """
    Prepare Kleinstiver data with read counts (not binary labels).
    
    Uses raw read counts as continuous target instead of binary is_validated.
    """
    input_path = DATA_DIR / "benchmark/data/kleinstiver2015/Kleinstiver_5gRNA_wholeDataset.csv"
    output_dir = DATA_DIR / "kleinstiver_quantitative"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Kleinstiver Quantitative Preparation")
    print("=" * 60)
    
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} samples")
    
    # Keep only samples with > 0 reads for regression (positives)
    # For regression, we need continuous values
    positives = df[df['Read'] > 0].copy()
    print(f"Positives (Read > 0): {len(positives)}")
    
    # Log-transform read counts (standard for count data)
    positives['log_read'] = np.log1p(positives['Read'])
    
    # Normalize to 0-1 range
    positives['normalized_read'] = positives['log_read'] / positives['log_read'].max()
    
    # Clean sequences (some have lowercase for mismatches)
    positives['sgRNA_clean'] = positives['sgRNA_seq'].str[:20].str.upper()
    positives['target_clean'] = positives['off_seq'].str[:20].str.upper()
    
    # Compute mismatch count
    def count_mm(row):
        return sum(1 for a, b in zip(row['sgRNA_clean'], row['target_clean']) if a != b)
    
    positives['mismatch_count'] = positives.apply(count_mm, axis=1)
    
    print(f"\nMismatch distribution in positives:")
    print(positives['mismatch_count'].value_counts().sort_index())
    
    print(f"\nRead count stats:")
    print(positives['Read'].describe())
    
    # Save
    output_path = output_dir / "kleinstiver_quantitative.csv"
    positives.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")
    
    return output_path


def prepare_circle_seq_quantitative():
    """
    Prepare CIRCLE-seq with continuous read counts.
    
    Uses the already-scored data with read_count column.
    """
    input_path = DATA_DIR / "benchmark/circle_seq/circle_seq_balanced.csv"
    output_dir = DATA_DIR / "circle_seq_quantitative"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("CIRCLE-seq Quantitative Preparation")
    print("=" * 60)
    
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} samples")
    
    # Check for read_count column
    if 'Read Count' in df.columns:
        read_col = 'Read Count'
    elif 'read_count' in df.columns:
        read_col = 'read_count'
    else:
        print("No read count column found. Available columns:")
        print(df.columns.tolist())
        return None
    
    # Keep only samples with > 0 reads
    positives = df[df[read_col] > 0].copy()
    print(f"Positives ({read_col} > 0): {len(positives)}")
    
    # Log-transform
    positives['log_read'] = np.log1p(positives[read_col])
    positives['normalized_read'] = positives['log_read'] / positives['log_read'].max()
    
    print(f"\nRead count stats:")
    print(positives[read_col].describe())
    
    # Save
    output_path = output_dir / "circle_seq_quantitative.csv"
    positives.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")
    
    return output_path


def create_train_val_test_split(
    df: pd.DataFrame,
    grna_col: str = 'sgRNA_clean',
    test_size: float = 0.15,
    val_size: float = 0.15
):
    """
    Create train/val/test split that doesn't leak between gRNAs.
    
    Key insight: Split by gRNA, not by site.
    Otherwise, model memorizes gRNA-specific patterns.
    """
    np.random.seed(SEED)
    
    # Get unique gRNAs
    grnas = df[grna_col].unique()
    np.random.shuffle(grnas)
    
    n_test = int(len(grnas) * test_size)
    n_val = int(len(grnas) * val_size)
    
    test_grnas = set(grnas[:n_test])
    val_grnas = set(grnas[n_test:n_test + n_val])
    train_grnas = set(grnas[n_test + n_val:])
    
    train_df = df[df[grna_col].isin(train_grnas)]
    val_df = df[df[grna_col].isin(val_grnas)]
    test_df = df[df[grna_col].isin(test_grnas)]
    
    print(f"\nSplit by gRNA (no leakage):")
    print(f"  Train: {len(train_df)} samples ({len(train_grnas)} gRNAs)")
    print(f"  Val:   {len(val_df)} samples ({len(val_grnas)} gRNAs)")
    print(f"  Test:  {len(test_df)} samples ({len(test_grnas)} gRNAs)")
    
    return train_df, val_df, test_df


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("DATA PREPARATION PIPELINE")
    print("=" * 60)
    
    # Prepare Kleinstiver (we have this data)
    kleinstiver_path = prepare_kleinstiver_quantitative()
    
    # Show CHANGE-seq download instructions
    print("\n")
    download_change_seq()
    
    # Try CIRCLE-seq if available
    print("\n")
    try:
        circle_seq_path = prepare_circle_seq_quantitative()
    except FileNotFoundError:
        print("CIRCLE-seq balanced data not found, skipping...")
