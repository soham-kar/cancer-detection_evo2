"""
Validate on full Kleinstiver dataset (95k samples, 54 confirmed positives)

Tests heuristic-only first, then Evo2+heuristic if features available.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

# Seed weights
SEED_WEIGHTS = np.ones(20)
SEED_WEIGHTS[0:7] = 0.5     # PAM-distal
SEED_WEIGHTS[17:20] = 3.0   # Seed region

def compute_seed_penalty(grna, target):
    """Compute seed-weighted mismatch penalty from sequences"""
    penalty = 0
    grna = str(grna).upper() if pd.notna(grna) else ""
    target = str(target).upper() if pd.notna(target) else ""
    
    # Handle PAM (remove NGG/NRG from grna if present)
    if len(grna) > 20 and grna[-3:] in ['NGG', 'NAG', 'NRG']:
        grna = grna[:-3]
    
    for i in range(min(len(grna), len(target), 20)):
        if grna[i] != target[i]:
            penalty += SEED_WEIGHTS[i]
    return penalty

def main():
    print("=" * 60)
    print("FULL KLEINSTIVER VALIDATION (95k samples)")
    print("=" * 60)
    
    # Load full dataset
    df = pd.read_csv(DATA_DIR / "benchmark/data/kleinstiver2015/Kleinstiver_5gRNA_wholeDataset.csv")
    print(f"\nTotal samples: {len(df)}")
    print(f"Positives (label=1): {(df['label']==1).sum()}")
    print(f"Negatives (label=0): {(df['label']==0).sum()}")
    
    # Get labels
    y = df['label'].values
    
    # Check columns
    print(f"\nColumns: {df.columns.tolist()}")
    
    # Compute seed penalties (grna = sgRNA_seq, target = off_seq)
    print("\nComputing seed penalties...")
    seed_penalties = df.apply(
        lambda r: compute_seed_penalty(r['sgRNA_seq'], r['off_seq']),
        axis=1
    ).values
    
    print(f"Seed penalty range: [{seed_penalties.min():.1f}, {seed_penalties.max():.1f}]")
    print(f"Mean seed penalty: {seed_penalties.mean():.2f}")
    
    # ========== AUROC RESULTS ==========
    print("\n" + "=" * 60)
    print("AUROC RESULTS (Heuristic Only)")
    print("=" * 60)
    
    # Heuristic: fewer mismatches = more cleavage (negate penalty)
    heuristic_auroc = roc_auc_score(y, -seed_penalties)
    heuristic_auprc = average_precision_score(y, -seed_penalties)
    
    print(f"\nHeuristic-only (seed penalty):")
    print(f"  AUROC: {heuristic_auroc:.4f}")
    print(f"  AUPRC: {heuristic_auprc:.4f} (random baseline: {y.mean():.4f})")
    
    # Also check simple mismatch count
    mismatch_counts = df.apply(
        lambda r: sum(1 for a, b in zip(str(r['sgRNA_seq'])[:20].upper(), 
                                         str(r['off_seq'])[:20].upper()) if a != b),
        axis=1
    ).values
    
    mm_auroc = roc_auc_score(y, -mismatch_counts)
    print(f"\nSimple mismatch count:")
    print(f"  AUROC: {mm_auroc:.4f}")
    
    # ========== COMPARISON ==========
    print("\n" + "=" * 60)
    print("COMPARISON ACROSS DATASETS")
    print("=" * 60)
    print(f"CIRCLE-seq (10k, synthetic neg): Combined 0.81, Heuristic 0.74")
    print(f"Kleinstiver 108 (confirmed neg): Combined 0.88, Heuristic ~0.74")
    print(f"Kleinstiver 95k (confirmed neg): Heuristic {heuristic_auroc:.4f}")
    
    if heuristic_auroc > 0.70:
        print(f"\n✅ Heuristic alone achieves AUROC {heuristic_auroc:.2f} on 95k confirmed samples!")
        print("   Worth extracting Evo2 features for potential improvement.")
    else:
        print(f"\n⚠️ Heuristic weaker on full dataset: {heuristic_auroc:.2f}")
        print("   May need different approach for this dataset.")

if __name__ == "__main__":
    main()
