"""
Diagnostic analysis of Kleinstiver paradox:
Why does simple counting beat biologically-informed seed weighting?

Phase 1: Learn position weights, analyze positives, compare datasets.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "diagnostics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def encode_mismatches(grna, target):
    """One-hot encode mismatch positions (20 positions)"""
    encoding = np.zeros(20)
    grna = str(grna).upper()
    target = str(target).upper()
    
    # Remove PAM if present
    if len(grna) > 20 and grna[-3:] in ['NGG', 'NAG', 'NRG']:
        grna = grna[:-3]
    
    for i in range(min(len(grna), len(target), 20)):
        if grna[i] != target[i]:
            encoding[i] = 1
    return encoding

def main():
    print("=" * 70)
    print("DIAGNOSTIC ANALYSIS: WHY SIMPLE > SEED-WEIGHTED?")
    print("=" * 70)
    
    # Load Kleinstiver data
    df = pd.read_csv(DATA_DIR / "benchmark/data/kleinstiver2015/Kleinstiver_5gRNA_wholeDataset.csv")
    print(f"\nKleinstiver: {len(df)} samples, {df['label'].sum()} positives")
    
    # ========== EXPERIMENT 1: Learn Position Weights ==========
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: LEARNED POSITION WEIGHTS")
    print("=" * 70)
    
    # Encode mismatches
    X = np.array([encode_mismatches(row['sgRNA_seq'], row['off_seq']) for _, row in df.iterrows()])
    y = df['label'].values
    
    # Fit logistic regression
    model = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    model.fit(X, y)
    
    learned_weights = model.coef_[0]
    print("\nLearned position weights (negative = higher mismatch penalty):")
    print("Position:  1    2    3    4    5    6    7    8    9   10   11   12   13   14   15   16   17   18   19   20")
    print("Weight: ", end="")
    print(" ".join(f"{w:5.2f}" for w in learned_weights))
    
    # Compare with heuristic weights
    heuristic_weights = [0.5]*7 + [1.0]*10 + [3.0]*3
    print("\nHeuristic weights (inverted for comparison):")
    print("Weight: ", end="")
    print(" ".join(f"{-w:5.2f}" for w in heuristic_weights))
    
    # Predict with learned weights
    learned_scores = model.predict_proba(X)[:, 1]
    learned_auroc = roc_auc_score(y, learned_scores)
    print(f"\nLearned weights AUROC: {learned_auroc:.4f}")
    
    # ========== EXPERIMENT 2: Analyze Positives ==========
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: MISMATCH DISTRIBUTION IN POSITIVES")
    print("=" * 70)
    
    positives = df[df['label'] == 1].copy()
    negatives = df[df['label'] == 0].copy()
    
    # Compute mismatch counts
    def count_mismatches(grna, target):
        grna = str(grna).upper()
        target = str(target).upper()
        if len(grna) > 20 and grna[-3:] in ['NGG', 'NAG', 'NRG']:
            grna = grna[:-3]
        return sum(1 for a, b in zip(grna[:20], target[:20]) if a != b)
    
    positives['mm_count'] = positives.apply(lambda r: count_mismatches(r['sgRNA_seq'], r['off_seq']), axis=1)
    negatives['mm_count'] = negatives.apply(lambda r: count_mismatches(r['sgRNA_seq'], r['off_seq']), axis=1)
    
    print("\nPositives mismatch distribution:")
    print(positives['mm_count'].value_counts().sort_index())
    print(f"Mean: {positives['mm_count'].mean():.2f}, Median: {positives['mm_count'].median():.0f}")
    
    print("\nNegatives mismatch distribution (sample):")
    print(negatives['mm_count'].value_counts().sort_index().head(10))
    print(f"Mean: {negatives['mm_count'].mean():.2f}, Median: {negatives['mm_count'].median():.0f}")
    
    # Position-specific mismatch frequency in positives
    pos_mismatches = np.zeros(20)
    for _, row in positives.iterrows():
        enc = encode_mismatches(row['sgRNA_seq'], row['off_seq'])
        pos_mismatches += enc
    pos_mismatches /= len(positives)
    
    print("\nMismatch frequency by position in POSITIVES:")
    print("Position:  1    2    3    4    5    6    7    8    9   10   11   12   13   14   15   16   17   18   19   20")
    print("Freq:   ", end="")
    print(" ".join(f"{f:5.2f}" for f in pos_mismatches))
    
    # ========== EXPERIMENT 3: Visualize ==========
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: VISUALIZATIONS")
    print("=" * 70)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Learned vs Heuristic weights
    ax1 = axes[0, 0]
    positions = np.arange(1, 21)
    ax1.bar(positions - 0.2, -learned_weights, 0.4, label='Learned (inverted)', alpha=0.7)
    ax1.bar(positions + 0.2, heuristic_weights, 0.4, label='Heuristic', alpha=0.7)
    ax1.set_xlabel('Position')
    ax1.set_ylabel('Weight (penalty)')
    ax1.set_title('Learned vs Heuristic Position Weights')
    ax1.legend()
    ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    # Plot 2: Mismatch frequency in positives
    ax2 = axes[0, 1]
    ax2.bar(positions, pos_mismatches, color='red', alpha=0.7)
    ax2.set_xlabel('Position')
    ax2.set_ylabel('Mismatch Frequency')
    ax2.set_title('Where Mismatches Occur in Positives (Cleaved Sites)')
    
    # Plot 3: Mismatch count distribution
    ax3 = axes[1, 0]
    ax3.hist(positives['mm_count'], bins=range(0, 10), alpha=0.7, label='Positives', density=True)
    ax3.hist(negatives['mm_count'].sample(1000, random_state=42), bins=range(0, 10), alpha=0.5, label='Negatives (sample)', density=True)
    ax3.set_xlabel('Mismatch Count')
    ax3.set_ylabel('Density')
    ax3.set_title('Mismatch Count Distribution')
    ax3.legend()
    
    # Plot 4: ROC comparison
    ax4 = axes[1, 1]
    # Simple count
    simple_scores = -df.apply(lambda r: count_mismatches(r['sgRNA_seq'], r['off_seq']), axis=1).values
    # Seed-weighted
    SEED_WEIGHTS = np.array([0.5]*7 + [1.0]*10 + [3.0]*3)
    seed_scores = -np.array([(encode_mismatches(row['sgRNA_seq'], row['off_seq']) * SEED_WEIGHTS).sum() 
                             for _, row in df.iterrows()])
    
    from sklearn.metrics import roc_curve
    fpr_simple, tpr_simple, _ = roc_curve(y, simple_scores)
    fpr_seed, tpr_seed, _ = roc_curve(y, seed_scores)
    fpr_learned, tpr_learned, _ = roc_curve(y, learned_scores)
    
    ax4.plot(fpr_simple, tpr_simple, label=f'Simple Count (AUC={roc_auc_score(y, simple_scores):.3f})')
    ax4.plot(fpr_seed, tpr_seed, label=f'Seed-Weighted (AUC={roc_auc_score(y, seed_scores):.3f})')
    ax4.plot(fpr_learned, tpr_learned, label=f'Learned Weights (AUC={roc_auc_score(y, learned_scores):.3f})')
    ax4.plot([0, 1], [0, 1], 'k--')
    ax4.set_xlabel('False Positive Rate')
    ax4.set_ylabel('True Positive Rate')
    ax4.set_title('ROC Comparison')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'kleinstiver_diagnostic.png', dpi=150)
    print(f"\nSaved: {OUTPUT_DIR / 'kleinstiver_diagnostic.png'}")
    
    # ========== SUMMARY ==========
    print("\n" + "=" * 70)
    print("SUMMARY: WHY SIMPLE > SEED-WEIGHTED")
    print("=" * 70)
    
    # Check if seed region has fewer mismatches in positives
    seed_mm_freq = pos_mismatches[17:20].mean()
    distal_mm_freq = pos_mismatches[0:7].mean()
    
    print(f"\nMismatch frequency in positives:")
    print(f"  PAM-distal (pos 1-7):  {distal_mm_freq:.3f}")
    print(f"  Seed region (pos 18-20): {seed_mm_freq:.3f}")
    
    if seed_mm_freq < distal_mm_freq:
        print("\n→ Positives AVOID seed mismatches (expected)")
        print("  But this doesn't explain why simple counting wins...")
    else:
        print("\n→ Positives DON'T avoid seed mismatches!")
        print("  This explains why seed-weighting doesn't help.")
    
    # Check if learned weights are flat
    weight_variance = np.var(learned_weights)
    print(f"\nLearned weight variance: {weight_variance:.4f}")
    if weight_variance < 0.5:
        print("→ Weights are relatively flat — position doesn't matter much!")
    else:
        print("→ Weights vary — position does matter, but differently than heuristic.")

if __name__ == "__main__":
    main()
