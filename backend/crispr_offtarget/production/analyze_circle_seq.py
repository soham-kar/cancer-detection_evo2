#!/usr/bin/env python3
"""
CIRCLE-seq Baseline Analysis

Process pre-processed CIRCLE-seq data (584,949 samples)
- 7,371 validated off-targets (positives)
- 577,578 non-cleaved sites (negatives)

Run mismatch count baseline and prepare sample for Evo2.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results" / "circle_seq_benchmark"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_circle_seq_data():
    """Load CIRCLE-seq pre-processed data."""
    filepath = DATA_DIR / "benchmark" / "circle_seq" / "circle_seq_10gRNA.csv"
    df = pd.read_csv(filepath)
    print(f"✅ Loaded {len(df):,} CIRCLE-seq samples")
    print(f"   Columns: {df.columns.tolist()}")
    print(f"\n📊 Label distribution:")
    print(f"   Positives (label=1): {(df['label']==1).sum():,}")
    print(f"   Negatives (label=0): {(df['label']==0).sum():,}")
    print(f"   Positive rate: {df['label'].mean():.4f}")
    return df


def clean_sequence(seq):
    """Clean gRNA/target sequence for comparison."""
    if pd.isna(seq):
        return ""
    seq = str(seq).upper()
    # Remove prefixes like "G_" and characters like "-" "_"
    seq = seq.replace('G_', '').replace('-', '').replace('_', '')
    # Take first 20bp (core without PAM)
    return seq[:20]


def count_mismatches(grna, target):
    """Count mismatches between gRNA and target (20bp core)."""
    grna = clean_sequence(grna)
    target = clean_sequence(target)
    
    if len(grna) < 10 or len(target) < 10:
        return -1  # Invalid
    
    min_len = min(len(grna), len(target), 20)
    mismatches = sum(
        grna[i] != target[i] and grna[i] != 'N' and target[i] != 'N'
        for i in range(min_len)
    )
    return mismatches


def seed_weighted_score(grna, target):
    """
    Compute seed-weighted mismatch score.
    Seed region (positions 1-12) is more important for binding.
    """
    grna = clean_sequence(grna)
    target = clean_sequence(target)
    
    if len(grna) < 10 or len(target) < 10:
        return 0
    
    min_len = min(len(grna), len(target), 20)
    
    score = 0
    for i in range(min_len):
        if grna[i] != target[i] and grna[i] != 'N' and target[i] != 'N':
            # Seed region (PAM-proximal, positions 1-12) weighted higher
            # Position numbering: 1 is PAM-proximal
            pos = 20 - i  # Convert to 1-indexed from PAM
            if 1 <= pos <= 8:
                score += 3.0  # High weight for seed
            elif 9 <= pos <= 12:
                score += 2.0  # Medium weight
            else:
                score += 1.0  # Low weight for PAM-distal
    
    return score


def run_baseline_analysis(df):
    """Compute mismatch count and seed-weighted baselines."""
    print("\n" + "="*60)
    print("BASELINE ANALYSIS")
    print("="*60)
    
    # Compute scores
    print("\n🔧 Computing mismatch counts...")
    df['mismatch_count'] = df.apply(
        lambda row: count_mismatches(row['sgRNA_seq'], row['off_seq']), 
        axis=1
    )
    
    # Filter invalid
    valid = df[df['mismatch_count'] >= 0].copy()
    print(f"   Valid samples: {len(valid):,} / {len(df):,}")
    
    print("🔧 Computing seed-weighted scores...")
    valid['seed_score'] = valid.apply(
        lambda row: seed_weighted_score(row['sgRNA_seq'], row['off_seq']),
        axis=1
    )
    
    # Scores: lower mismatch = higher cleavage probability
    # So we negate for AUROC calculation
    valid['mismatch_score'] = -valid['mismatch_count']
    valid['seed_score_neg'] = -valid['seed_score']
    
    # Calculate metrics
    y_true = valid['label'].values
    
    # Mismatch count
    auroc_mm = roc_auc_score(y_true, valid['mismatch_score'])
    auprc_mm = average_precision_score(y_true, valid['mismatch_score'])
    
    # Seed-weighted
    auroc_seed = roc_auc_score(y_true, valid['seed_score_neg'])
    auprc_seed = average_precision_score(y_true, valid['seed_score_neg'])
    
    print("\n📊 RESULTS:")
    print(f"   {'Method':<25} {'AUROC':>10} {'AUPRC':>10}")
    print(f"   {'-'*45}")
    print(f"   {'Mismatch Count':<25} {auroc_mm:>10.4f} {auprc_mm:>10.4f}")
    print(f"   {'Seed-Weighted Score':<25} {auroc_seed:>10.4f} {auprc_seed:>10.4f}")
    
    return valid, {
        'mismatch_auroc': auroc_mm,
        'mismatch_auprc': auprc_mm,
        'seed_auroc': auroc_seed,
        'seed_auprc': auprc_seed,
        'n_samples': len(valid),
        'n_positives': int(y_true.sum()),
        'n_negatives': int((y_true == 0).sum())
    }


def prepare_evo2_sample(df, sample_size=5000):
    """Prepare balanced sample for Evo2 scoring."""
    print(f"\n🔧 Preparing {sample_size*2} samples for Evo2 scoring...")
    
    pos = df[df['label'] == 1]
    neg = df[df['label'] == 0]
    
    n_pos = min(len(pos), sample_size)
    n_neg = min(len(neg), sample_size)
    
    pos_sample = pos.sample(n=n_pos, random_state=42)
    neg_sample = neg.sample(n=n_neg, random_state=42)
    
    sample = pd.concat([pos_sample, neg_sample]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Prepare clean sequences
    sample['grna_sequence'] = sample['sgRNA_seq'].apply(clean_sequence)
    sample['target_sequence'] = sample['off_seq'].apply(clean_sequence)
    sample['is_validated'] = sample['label'].astype(int)
    
    # Filter valid
    sample = sample[
        (sample['grna_sequence'].str.len() >= 15) & 
        (sample['target_sequence'].str.len() >= 15)
    ].copy()
    
    print(f"   Sample size: {len(sample):,}")
    print(f"   Positives: {sample['is_validated'].sum():,}")
    print(f"   Negatives: {(sample['is_validated']==0).sum():,}")
    
    return sample


def plot_results(valid, metrics):
    """Generate ROC and PR curves."""
    print("\n📈 Generating plots...")
    
    y_true = valid['label'].values
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # ROC Curve
    ax1 = axes[0]
    for score_col, name, color in [
        ('mismatch_score', 'Mismatch Count', '#2196F3'),
        ('seed_score_neg', 'Seed-Weighted', '#FF5722')
    ]:
        fpr, tpr, _ = roc_curve(y_true, valid[score_col])
        auroc = roc_auc_score(y_true, valid[score_col])
        ax1.plot(fpr, tpr, color=color, linewidth=2, label=f'{name} (AUROC={auroc:.3f})')
    
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title('CIRCLE-seq ROC Curves', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # PR Curve
    ax2 = axes[1]
    for score_col, name, color in [
        ('mismatch_score', 'Mismatch Count', '#2196F3'),
        ('seed_score_neg', 'Seed-Weighted', '#FF5722')
    ]:
        precision, recall, _ = precision_recall_curve(y_true, valid[score_col])
        auprc = average_precision_score(y_true, valid[score_col])
        ax2.plot(recall, precision, color=color, linewidth=2, label=f'{name} (AUPRC={auprc:.3f})')
    
    baseline = y_true.mean()
    ax2.axhline(y=baseline, color='k', linestyle='--', alpha=0.5, label=f'Baseline ({baseline:.3f})')
    ax2.set_xlabel('Recall', fontsize=12)
    ax2.set_ylabel('Precision', fontsize=12)
    ax2.set_title('CIRCLE-seq Precision-Recall Curves', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    fig_path = RESULTS_DIR / 'circle_seq_baseline_curves.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"   Saved: {fig_path}")
    plt.close()


def save_results(valid, sample, metrics):
    """Save analysis results."""
    import json
    
    # Save metrics
    metrics_path = RESULTS_DIR / 'baseline_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"   Saved metrics: {metrics_path}")
    
    # Save Evo2 sample
    sample_cols = ['grna_sequence', 'target_sequence', 'mismatch_count', 'is_validated', 'Cell']
    sample_path = DATA_DIR / 'circle_seq_balanced.csv'
    sample[sample_cols].to_csv(sample_path, index=False)
    print(f"   Saved Evo2 sample: {sample_path}")


def main():
    print("="*60)
    print("CIRCLE-SEQ BENCHMARK ANALYSIS")
    print("="*60)
    
    # Load data
    df = load_circle_seq_data()
    
    # Run baseline analysis
    valid, metrics = run_baseline_analysis(df)
    
    # Prepare Evo2 sample
    sample = prepare_evo2_sample(valid)
    
    # Plot results
    plot_results(valid, metrics)
    
    # Save everything
    save_results(valid, sample, metrics)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"✅ Analyzed {metrics['n_samples']:,} CIRCLE-seq samples")
    print(f"   - {metrics['n_positives']:,} validated off-targets")
    print(f"   - {metrics['n_negatives']:,} true negatives")
    print(f"\n📊 Baseline AUROC: {metrics['mismatch_auroc']:.4f}")
    print(f"   Seed-weighted AUROC: {metrics['seed_auroc']:.4f}")
    print(f"\n🎯 Next: Run Evo2 on circle_seq_balanced.csv")


if __name__ == "__main__":
    main()
