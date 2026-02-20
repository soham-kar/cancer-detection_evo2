"""
Create Balanced CRISPR Off-Target Benchmark

Uses Kleinstiver 2015 dataset (Nature paper) to create a balanced
evaluation set with proper positive/negative labels.

Source: data/benchmark/data/kleinstiver2015/Kleinstiver_5gRNA_wholeDataset.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
BENCHMARK_FILE = DATA_DIR / "benchmark" / "data" / "kleinstiver2015" / "Kleinstiver_5gRNA_wholeDataset.csv"
OUTPUT_FILE = DATA_DIR / "kleinstiver_balanced.csv"

def create_balanced_benchmark(ratio: float = 1.0, seed: int = 42):
    """
    Create balanced benchmark dataset.
    
    Args:
        ratio: Negative:Positive ratio (1.0 = equal, 2.0 = 2x negatives)
        seed: Random seed for reproducibility
    """
    print("=" * 60)
    print("CREATING BALANCED CRISPR BENCHMARK")
    print("=" * 60)
    
    # Load benchmark
    print(f"\n📂 Loading: {BENCHMARK_FILE}")
    df = pd.read_csv(BENCHMARK_FILE)
    
    print(f"   Total samples: {len(df):,}")
    print(f"   Columns: {list(df.columns)}")
    
    # Separate positives and negatives
    positives = df[df['label'] == 1].copy()
    negatives = df[df['label'] == 0].copy()
    
    print(f"\n📊 Original distribution:")
    print(f"   Positives (label=1): {len(positives)}")
    print(f"   Negatives (label=0): {len(negatives):,}")
    print(f"   Imbalance ratio: 1:{len(negatives)//len(positives)}")
    
    # Sample negatives to balance
    n_positives = len(positives)
    n_negatives_sample = int(n_positives * ratio)
    
    np.random.seed(seed)
    negatives_sampled = negatives.sample(n=n_negatives_sample, random_state=seed)
    
    print(f"\n🔄 Balancing with ratio {ratio}:")
    print(f"   Positives: {n_positives}")
    print(f"   Sampled negatives: {n_negatives_sample}")
    
    # Combine
    balanced = pd.concat([positives, negatives_sampled], ignore_index=True)
    balanced = balanced.sample(frac=1, random_state=seed).reset_index(drop=True)  # Shuffle
    
    # Rename columns to our format
    balanced = balanced.rename(columns={
        'sgRNA_seq': 'grna_sequence',
        'off_seq': 'target_sequence',
        'Read': 'read_count',
        'label': 'is_validated'
    })
    
    # Clean sequences (remove PAM if present)
    balanced['grna_sequence'] = balanced['grna_sequence'].str[:20].str.upper()
    balanced['target_sequence'] = balanced['target_sequence'].str[:20].str.upper()
    
    # Calculate mismatches
    def count_mismatches(row):
        grna = str(row['grna_sequence'])
        target = str(row['target_sequence'])
        return sum(1 for a, b in zip(grna, target) if a != b and a in 'ACGT' and b in 'ACGT')
    
    def get_mismatch_positions(row):
        grna = str(row['grna_sequence'])
        target = str(row['target_sequence'])
        return [i for i, (a, b) in enumerate(zip(grna, target)) if a != b and a in 'ACGT' and b in 'ACGT']
    
    print("\n🧬 Calculating mismatches...")
    balanced['mismatches'] = balanced.apply(count_mismatches, axis=1)
    balanced['mismatch_positions'] = balanced.apply(get_mismatch_positions, axis=1)
    
    # Add placeholder columns
    balanced['grna_name'] = 'kleinstiver_' + balanced.index.astype(str)
    balanced['chromosome'] = 'chrNA'
    balanced['position'] = 0
    balanced['strand'] = '+'
    balanced['pam'] = 'NGG'
    
    # Convert is_validated to boolean
    balanced['is_validated'] = balanced['is_validated'].astype(bool)
    
    # Save
    balanced.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\n✅ Saved: {OUTPUT_FILE}")
    print(f"   Total samples: {len(balanced)}")
    print(f"   Positives: {balanced['is_validated'].sum()}")
    print(f"   Negatives: {(~balanced['is_validated']).sum()}")
    print(f"   Balance: {balanced['is_validated'].sum() / len(balanced) * 100:.1f}% positive")
    
    # Stats by mismatches
    print("\n📈 Mismatch distribution:")
    print(balanced.groupby('mismatches')['is_validated'].agg(['count', 'sum', 'mean']).rename(
        columns={'count': 'total', 'sum': 'positives', 'mean': 'positive_rate'}
    ))
    
    return balanced


if __name__ == "__main__":
    # Create 1:1 balanced dataset
    df = create_balanced_benchmark(ratio=1.0)
    
    print("\n" + "=" * 60)
    print("🔧 Next step: Score with Evo2")
    print("=" * 60)
    print("   python production/crispr_scorer.py --input kleinstiver_balanced.csv")
