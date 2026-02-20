"""
Prepare CIRCLE-seq balanced dataset for Evo2 validation.

CIRCLE-seq has:
- 7,371 validated off-targets (positives)
- 577,578 non-cleaved sites (negatives)

We'll create a balanced subset for fair AUROC evaluation.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
CIRCLE_SEQ_PATH = DATA_DIR / "benchmark" / "circle_seq" / "circle_seq_10gRNA.csv"
OUTPUT_PATH = DATA_DIR / "circle_seq_balanced.csv"

def count_mismatches(grna: str, target: str) -> tuple:
    """Count mismatches and their positions between gRNA and target."""
    # Clean sequences
    grna = grna.strip().upper().replace('-', '').replace('_', '')
    target = target.strip().upper().replace('-', '').replace('_', '')
    
    # Remove prefix if present (e.g., "G_")
    if grna.startswith('G_'):
        grna = grna[2:]
    if target.startswith('-'):
        target = target[1:]
    
    # Align to 20bp core (without PAM)
    grna_core = grna[:20] if len(grna) >= 20 else grna
    target_core = target[:20] if len(target) >= 20 else target
    
    # Pad if needed
    min_len = min(len(grna_core), len(target_core))
    
    mismatches = 0
    positions = []
    for i in range(min_len):
        if grna_core[i] != target_core[i] and grna_core[i] != 'N' and target_core[i] != 'N':
            mismatches += 1
            positions.append(i + 1)  # 1-indexed
    
    return mismatches, positions


def main():
    print("=" * 60)
    print("PREPARING CIRCLE-SEQ BALANCED DATASET")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv(CIRCLE_SEQ_PATH)
    print(f"\n📂 Loaded {len(df):,} samples from CIRCLE-seq")
    print(f"   Columns: {df.columns.tolist()}")
    
    # Separate positives and negatives
    positives = df[df['label'] == 1].copy()
    negatives = df[df['label'] == 0].copy()
    
    print(f"\n📊 Original distribution:")
    print(f"   Positives: {len(positives):,}")
    print(f"   Negatives: {len(negatives):,}")
    
    # Sample balanced dataset
    n_samples = min(len(positives), 5000)  # Cap at 5000 each for manageable size
    
    balanced_positives = positives.sample(n=n_samples, random_state=42)
    balanced_negatives = negatives.sample(n=n_samples, random_state=42)
    
    balanced = pd.concat([balanced_positives, balanced_negatives], ignore_index=True)
    balanced = balanced.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
    
    print(f"\n✅ Balanced dataset: {len(balanced):,} samples ({n_samples} pos + {n_samples} neg)")
    
    # Process sequences and extract mismatches
    print("\n🔧 Processing sequences...")
    
    processed_rows = []
    for idx, row in balanced.iterrows():
        grna = str(row['sgRNA_seq'])
        target = str(row['off_seq'])
        
        # Clean sequences
        grna_clean = grna.replace('G_', '').replace('-', '').replace('_', '')[:20]
        target_clean = target.replace('-', '').replace('_', '')[:20]
        
        if len(grna_clean) < 15 or len(target_clean) < 15:
            continue
        
        mismatches, positions = count_mismatches(grna, target)
        
        processed_rows.append({
            'grna_sequence': grna_clean,
            'target_sequence': target_clean,
            'read_count': row.get('Read', 0),
            'is_validated': int(row['label']),
            'mismatches': mismatches,
            'mismatch_positions': str(positions),
            'grna_name': row.get('sgRNA_type', 'unknown'),
            'cell_type': row.get('Cell', 'unknown'),
        })
        
        if (idx + 1) % 1000 == 0:
            print(f"   Processed {idx + 1:,}/{len(balanced):,}")
    
    result_df = pd.DataFrame(processed_rows)
    
    # Final stats
    print(f"\n📈 Final dataset stats:")
    print(f"   Total: {len(result_df):,}")
    print(f"   Positives: {result_df['is_validated'].sum():,}")
    print(f"   Negatives: {(result_df['is_validated'] == 0).sum():,}")
    print(f"\n   Mismatch distribution:")
    print(result_df['mismatches'].value_counts().sort_index().head(10))
    
    # Save
    result_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✅ Saved: {OUTPUT_PATH}")
    
    return result_df


if __name__ == "__main__":
    main()
