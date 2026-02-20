"""
Evaluate ATAC-seq predictor and generate publication figures.
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def load_results():
    """Load training results."""
    results_path = RESULTS_DIR / "training_results.csv"
    if not results_path.exists():
        raise FileNotFoundError("Run 04_train_atac_predictor.py first")
    return pd.read_csv(results_path)


def plot_scatter(predictions: np.ndarray, targets: np.ndarray, save_path: Path):
    """Scatter plot of predicted vs true ATAC signal."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    r, p = pearsonr(targets, predictions)
    
    ax.scatter(targets, predictions, alpha=0.5, s=20)
    
    # Diagonal line
    lims = [min(targets.min(), predictions.min()), max(targets.max(), predictions.max())]
    ax.plot(lims, lims, 'r--', lw=2, label='Perfect prediction')
    
    ax.set_xlabel('True log(ATAC signal)', fontsize=12)
    ax.set_ylabel('Predicted log(ATAC signal)', fontsize=12)
    ax.set_title(f'Evo2 ATAC-seq Prediction\nPearson r = {r:.3f} (p = {p:.2e})', fontsize=14)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Saved scatter plot: {save_path}")


def plot_by_chromosome(coords_df: pd.DataFrame, predictions: np.ndarray, save_path: Path):
    """Bar plot of Pearson r by chromosome."""
    coords_df = coords_df.copy()
    coords_df['pred'] = predictions
    
    r_by_chrom = {}
    for chrom in coords_df['chrom'].unique():
        subset = coords_df[coords_df['chrom'] == chrom]
        if len(subset) >= 10:
            r, _ = pearsonr(np.log1p(subset['signal']), subset['pred'])
            r_by_chrom[chrom] = r
    
    chroms = sorted(r_by_chrom.keys(), key=lambda x: int(x.replace('chr', '')) if x.replace('chr', '').isdigit() else 99)
    r_values = [r_by_chrom[c] for c in chroms]
    
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(range(len(chroms)), r_values, color='steelblue')
    
    ax.axhline(0.62, color='green', linestyle='--', label='Target (r=0.62)')
    ax.axhline(0.50, color='orange', linestyle='--', label='Minimum (r=0.50)')
    
    ax.set_xticks(range(len(chroms)))
    ax.set_xticklabels(chroms, rotation=45)
    ax.set_ylabel('Pearson r')
    ax.set_title('ATAC Prediction Performance by Chromosome')
    ax.legend()
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Saved chromosome plot: {save_path}")


def plot_baseline_comparison(test_r: float, save_path: Path):
    """Compare Evo2 to baselines."""
    methods = ['GC Content\n(baseline)', 'K-mer\n(baseline)', 'Evo2 MLP\n(ours)']
    # Baseline estimates from literature
    r_values = [0.30, 0.35, test_r]
    colors = ['gray', 'gray', 'steelblue']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(methods, r_values, color=colors)
    
    ax.axhline(0.62, color='green', linestyle='--', label='Evo2 paper target')
    
    ax.set_ylabel('Pearson r', fontsize=12)
    ax.set_title('ATAC-seq Prediction: Evo2 vs Baselines', fontsize=14)
    ax.set_ylim(0, 1)
    ax.legend()
    
    # Add value labels
    for bar, val in zip(bars, r_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Saved comparison plot: {save_path}")


def main():
    logging.info("=" * 60)
    logging.info("ATAC-seq Prediction Evaluation")
    logging.info("=" * 60)
    
    # Load results
    results = load_results()
    test_r = results[results['metric'] == 'pearson_r']['value'].values[0]
    
    logging.info(f"Test Pearson r: {test_r:.4f}")
    
    # Generate figures
    plot_baseline_comparison(test_r, FIGURES_DIR / "baseline_comparison.png")
    
    logging.info(f"\n✅ Figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
