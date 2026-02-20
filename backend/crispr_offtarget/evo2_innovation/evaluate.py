"""
Evaluation and Comparison Script

Compares Evo2 regression to heuristic baselines.
"""
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import DATA_DIR, RESULTS_DIR, SEED
from quantitative_model import Evo2CleavageRegressor, HeuristicBaseline


def evaluate_heuristic_baseline(df: pd.DataFrame, target_col: str = 'normalized_read'):
    """
    Evaluate simple heuristic baselines.
    
    Returns Spearman correlation for:
    1. Simple mismatch count
    2. Seed-weighted penalty
    3. Learned position weights
    """
    results = {}
    
    y_true = df[target_col].values
    
    # Compute mismatch counts if not present
    if 'mismatch_count' not in df.columns:
        df['mismatch_count'] = df.apply(
            lambda r: sum(1 for a, b in zip(r['sgRNA_clean'], r['target_clean']) if a != b),
            axis=1
        )
    
    # 1. Simple mismatch count (negated: fewer = higher cleavage)
    simple_score = -df['mismatch_count'].values
    spearman_simple, _ = spearmanr(simple_score, y_true)
    results['simple_count'] = spearman_simple
    
    # 2. Seed-weighted penalty
    SEED_WEIGHTS = np.array([0.5]*7 + [1.0]*10 + [3.0]*3)
    
    def seed_weighted_score(row):
        penalty = 0
        grna = row['sgRNA_clean'][:20]
        target = row['target_clean'][:20]
        for i, (a, b) in enumerate(zip(grna, target)):
            if a != b:
                penalty += SEED_WEIGHTS[i]
        return -penalty  # Negate
    
    seed_scores = df.apply(seed_weighted_score, axis=1).values
    spearman_seed, _ = spearmanr(seed_scores, y_true)
    results['seed_weighted'] = spearman_seed
    
    # 3. Thermodynamic (simple proxy: GC content at mismatch)
    def gc_mismatch_score(row):
        grna = row['sgRNA_clean'][:20]
        target = row['target_clean'][:20]
        gc_penalty = 0
        for i, (a, b) in enumerate(zip(grna, target)):
            if a != b:
                if a in 'GC':  # G/C mismatch is more destabilizing
                    gc_penalty += 1.5
                else:
                    gc_penalty += 1.0
        return -gc_penalty
    
    gc_scores = df.apply(gc_mismatch_score, axis=1).values
    spearman_gc, _ = spearmanr(gc_scores, y_true)
    results['gc_weighted'] = spearman_gc
    
    return results


def compare_all_methods(
    test_df: pd.DataFrame,
    evo2_predictions: np.ndarray,
    target_col: str = 'normalized_read',
    output_dir: Path = RESULTS_DIR
):
    """
    Comprehensive comparison of all methods.
    """
    y_true = test_df[target_col].values
    
    print("=" * 60)
    print("METHOD COMPARISON (Spearman Correlation)")
    print("=" * 60)
    
    # Heuristic baselines
    heuristics = evaluate_heuristic_baseline(test_df, target_col)
    
    for name, corr in heuristics.items():
        print(f"  {name:20s}: {corr:.4f}")
    
    # Evo2 model
    spearman_evo2, _ = spearmanr(evo2_predictions, y_true)
    print(f"  {'Evo2 regression':20s}: {spearman_evo2:.4f}")
    
    # Summary
    best_heuristic = max(heuristics.values())
    improvement = spearman_evo2 - best_heuristic
    
    print("\n" + "-" * 60)
    print(f"Best heuristic: {max(heuristics, key=heuristics.get)} ({best_heuristic:.4f})")
    print(f"Evo2 improvement: {'+' if improvement > 0 else ''}{improvement:.4f}")
    
    if improvement > 0.05:
        print("\n✅ Evo2 provides meaningful improvement!")
    elif improvement > 0:
        print("\n⚠️  Marginal improvement - may not justify cost")
    else:
        print("\n❌ Heuristic performs better - Evo2 not needed for this task")
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Scatter - Simple count vs True
    ax1 = axes[0]
    simple_scores = -test_df['mismatch_count'].values
    ax1.scatter(simple_scores, y_true, alpha=0.5, s=10)
    ax1.set_xlabel('Simple Mismatch Count (negated)')
    ax1.set_ylabel('True Cleavage Rate')
    ax1.set_title(f'Simple Count (ρ={heuristics["simple_count"]:.3f})')
    
    # Plot 2: Scatter - Evo2 vs True
    ax2 = axes[1]
    ax2.scatter(evo2_predictions, y_true, alpha=0.5, s=10, color='orange')
    ax2.set_xlabel('Evo2 Predicted')
    ax2.set_ylabel('True Cleavage Rate')
    ax2.set_title(f'Evo2 Regression (ρ={spearman_evo2:.3f})')
    
    # Plot 3: Bar comparison
    ax3 = axes[2]
    methods = list(heuristics.keys()) + ['Evo2']
    corrs = list(heuristics.values()) + [spearman_evo2]
    colors = ['skyblue'] * len(heuristics) + ['orange']
    ax3.bar(methods, corrs, color=colors)
    ax3.set_ylabel('Spearman Correlation')
    ax3.set_title('Method Comparison')
    ax3.set_ylim([0, 1])
    for i, v in enumerate(corrs):
        ax3.text(i, v + 0.02, f'{v:.3f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'method_comparison.png', dpi=150)
    print(f"\nSaved: {output_dir / 'method_comparison.png'}")
    
    return {
        **heuristics,
        'evo2': spearman_evo2,
        'improvement': improvement
    }


def main():
    """Run evaluation on test set."""
    print("Loading test data...")
    
    # Check for test data
    test_path = RESULTS_DIR / "features" / "test_labels.csv"
    if not test_path.exists():
        print(f"Test data not found at {test_path}")
        print("Run data_preparation.py and train_regression.py first.")
        return
    
    test_df = pd.read_csv(test_path)
    
    # Load model predictions (or compute them)
    pred_path = RESULTS_DIR / "test_predictions.npy"
    if pred_path.exists():
        predictions = np.load(pred_path)
    else:
        print("Predictions not found. Run inference first.")
        # Could add inference code here
        return
    
    # Compare methods
    results = compare_all_methods(test_df, predictions)
    
    # Save results
    import json
    with open(RESULTS_DIR / "evaluation_results.json", 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
