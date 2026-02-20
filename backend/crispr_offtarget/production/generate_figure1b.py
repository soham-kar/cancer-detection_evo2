
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import os
import seaborn as sns

def get_mismatches(seq, target):
    target_clean = target.upper().replace('N', '')
    t_len = len(target_clean)
    center = 4000
    if len(seq) < 8000:
        center = len(seq) // 2
    s_start = center - t_len // 2
    
    min_mm = 100
    for offset in range(-20, 21):
        idx = s_start + offset
        if idx < 0 or idx + t_len > len(seq): continue
        s_sub = seq[idx : idx + t_len].upper()
        mm = 0
        for i in range(t_len):
            if s_sub[i] != target_clean[i]:
                mm += 1
        if mm < min_mm: min_mm = mm
    return min_mm

def plot_quantitative_analysis(df):
    os.makedirs('results/nature_figures', exist_ok=True)
    
    # Use normalized_reads for analysis
    df['cleavage'] = df['normalized_reads']
    
    # Define Activity Groups relative to Max in dataset
    max_cleavage = df['cleavage'].max()
    
    # Option 4 Stratification
    # High: >50% of max
    # Medium: 10-50% of max
    # Low: 1-10% of max
    # (Note: In this 1k subset, we might not have the global max, but we use local max)
    
    df['Activity_Group'] = 'Trace (<1%)'
    df.loc[df['cleavage'] >= 0.01 * max_cleavage, 'Activity_Group'] = 'Low (1-10%)'
    df.loc[df['cleavage'] >= 0.10 * max_cleavage, 'Activity_Group'] = 'Medium (10-50%)'
    df.loc[df['cleavage'] >= 0.50 * max_cleavage, 'Activity_Group'] = 'High (>50%)'
    
    activity_order = ['High (>50%)', 'Medium (10-50%)', 'Low (1-10%)']
    
    # 1. Option 4 Plot: Scatter by Activity Group
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, group in enumerate(activity_order):
        subset = df[df['Activity_Group'] == group]
        ax = axes[i]
        
        if len(subset) > 1:
            # Evo2 Correlation
            corr_evo, _ = spearmanr(subset['evo2_score'], subset['cleavage'])
            # Heuristic Correlation
            corr_heur, _ = spearmanr(subset['pred_heuristic'], subset['cleavage'])
            
            # Scatter
            ax.scatter(subset['evo2_score'], subset['cleavage'], alpha=0.6, color='blue', label=f'Evo2 (ρ={corr_evo:.2f})')
            ax.set_title(f"{group}\nn={len(subset)}")
            ax.set_xlabel("Evo2 Score")
            ax.set_ylabel("Normalized Cleavage")
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f"Insufficient Data\n(n={len(subset)})", ha='center')
            ax.set_title(group)
            
    plt.tight_layout()
    plt.savefig('results/nature_figures/figure1_quantitative_activity.png', dpi=300)
    print("Saved Activity Stratification plot")

    # 2. Option 3/1 Plot: Correlation by Mismatch Count
    # Focus on 4-6 MM (Ambiguous) vs 7+ (Easy)
    # 0-3 might be rare
    
    df['MM_Group'] = '>6 mismatches'
    df.loc[df['mismatch_count'].between(4, 6), 'MM_Group'] = '4-6 mismatches'
    df.loc[df['mismatch_count'] <= 3, 'MM_Group'] = '0-3 mismatches'
    
    mm_order = ['0-3 mismatches', '4-6 mismatches', '>6 mismatches']
    
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, group in enumerate(mm_order):
        subset = df[df['MM_Group'] == group]
        ax = axes2[i]
        
        if len(subset) > 1:
            corr_evo, _ = spearmanr(subset['evo2_score'], subset['cleavage'])
            
            sns.regplot(data=subset, x='evo2_score', y='cleavage', ax=ax, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
            ax.set_title(f"{group}\nn={len(subset)}, ρ={corr_evo:.2f}")
            ax.set_xlabel("Evo2 Score")
            ax.set_ylabel("Cleavage")
        else:
            ax.text(0.5, 0.5, f"n={len(subset)}", ha='center')
            ax.set_title(group)
            
    plt.tight_layout()
    plt.savefig('results/nature_figures/figure1_quantitative_mismatch.png', dpi=300)
    print("Saved Mismatch Stratification plot")

    # Print Summary Stats
    print("\nSummary Statistics:")
    print(df.groupby('Activity_Group')['cleavage'].agg(['count', 'min', 'max', 'mean']))
    print("\nMismatch Group Stats:")
    print(df.groupby('MM_Group')['cleavage'].agg(['count', 'mean']))


def main():
    print("Loading data...")
    try:
        with open('data/features/evo2_feats_1k.json') as f:
            evo_data = json.load(f)
        with open('data/features/evo2_input_limit1000.json') as f:
            meta_data = json.load(f)
    except FileNotFoundError:
        print("Files not found")
        return

    n = min(len(evo_data), len(meta_data))
    rows = []
    
    for i in range(n):
        e = evo_data[i]
        m = meta_data[i]
        
        reads = m.get('reads', 0.0)
        norm_reads = m.get('normalized_reads', 0.0)
        seq = m['sequence_8kb']
        target = m['grna_target_seq']
        
        mm = get_mismatches(seq, target)
        heuristic = 20 - mm
        
        rows.append({
            'seq_id': e.get('seq_id'),
            'mismatch_count': mm,
            'reads': reads,
            'normalized_reads': norm_reads,
            'pred_heuristic': heuristic,
            'evo2_score': e.get('evo2_score', 0)
        })
        
    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} samples")
    
    plot_quantitative_analysis(df)

if __name__ == "__main__":
    main()
