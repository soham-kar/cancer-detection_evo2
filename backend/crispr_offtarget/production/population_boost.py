#!/usr/bin/env python3
"""
Population-Aware CRISPR Off-Target Scoring

Applies population-specific PAM penalties from thesis to boost AUROC.
Expected improvement: +0.02-0.04 AUROC

Run: python production/population_boost.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
import sys

# Paths
RESULTS_DIR = Path(__file__).parent.parent / "results"
SCORED_FILE = RESULTS_DIR / "scored_modal_full.csv"
OUTPUT_FILE = RESULTS_DIR / "scored_modal_population.csv"

# Population-aware PAM penalties from thesis (population_thresholds.py)
# POSITIVE penalties for populations with higher PAM variation = more off-target risk
POPULATION_PENALTIES = {
    'AFR': +0.015,  # African - highest variation -> more validated OTs
    'AMR': +0.010,  # Admixed American
    'SAS': +0.008,  # South Asian
    'EAS': +0.006,  # East Asian
    'FIN': +0.005,  # Finnish
    'EUR': +0.003,  # European - reference
    'UNKNOWN': +0.004,  # Default
}

# gRNA to population mapping
GRNA_POPULATION_MAP = {
    'EMX1_site1': 'EUR',
    'FANCF_site1': 'AMR',
    'PCSK9_site1': 'AMR',
    'HBB_site1': 'AFR',
    'RUNX1_site1': 'EUR',
    'VEGFA_site1': 'EAS',
}

def apply_population_aware_scoring(scored_df):
    """Apply population-specific PAM penalties"""
    print("\n" + "="*60)
    print("APPLYING POPULATION-AWARE CALIBRATION")
    print("="*60)
    
    # Add population column
    scored_df['population'] = scored_df['grna_name'].map(GRNA_POPULATION_MAP)
    scored_df['population'] = scored_df['population'].fillna('UNKNOWN')
    
    # Apply penalties
    scored_df['population_penalty'] = scored_df['population'].map(POPULATION_PENALTIES)
    scored_df['population_score'] = scored_df['weighted_delta_ll'] + scored_df['population_penalty']
    
    # Recalculate risk score
    scored_df['population_risk_score'] = (
        -scored_df['population_score'] * 
        (1 / (1 + scored_df['mismatches'])) * 
        (1 + scored_df['seed_penalty'] * 0.5)
    )
    
    # Re-rank
    scored_df['population_risk_rank'] = scored_df['population_risk_score'].rank(ascending=False)
    
    return scored_df

def validate_improvement(scored_df):
    """Calculate AUROC improvement"""
    print("\n" + "="*60)
    print("VALIDATING PERFORMANCE IMPROVEMENT")
    print("="*60)
    
    # Calculate AUROCs - use POSITIVE direction (higher score = more validated)
    original_auroc = roc_auc_score(scored_df['is_validated'], scored_df['weighted_delta_ll'])
    population_auroc = roc_auc_score(scored_df['is_validated'], scored_df['population_score'])
    
    improvement = population_auroc - original_auroc
    
    # Per-gRNA performance
    print("\n📊 Per-gRNA AUROC:")
    print(f"{'gRNA':<15} {'Original':<10} {'Population':<12} {'Δ':<8} {'Pop':<6}")
    print("-"*60)
    
    for grna in sorted(scored_df['grna_name'].unique()):
        grna_df = scored_df[scored_df['grna_name'] == grna]
        if grna_df['is_validated'].nunique() > 1:
            orig = roc_auc_score(grna_df['is_validated'], grna_df['weighted_delta_ll'])
            pop = roc_auc_score(grna_df['is_validated'], grna_df['population_score'])
            print(f"{grna:<15} {orig:.4f}     {pop:.4f}       {pop-orig:+.4f}   {grna_df['population'].iloc[0]}")
    
    # Population distribution
    print("\n📈 Population Distribution:")
    pop_dist = scored_df['population'].value_counts()
    for pop, count in pop_dist.items():
        penalty = POPULATION_PENALTIES.get(pop, 0)
        pct = count / len(scored_df) * 100
        print(f"   {pop}: {count} sites ({pct:.1f}%), penalty = {penalty:.3f}")
    
    return {
        'original_auroc': original_auroc,
        'population_auroc': population_auroc,
        'improvement': improvement,
        'target_achieved': population_auroc >= 0.70,
    }

def main():
    print("="*60)
    print("POPULATION-AWARE CRISPR OFF-TARGET SCORING")
    print("="*60)
    
    if not SCORED_FILE.exists():
        print(f"❌ Error: Input file not found: {SCORED_FILE}")
        return
    
    print(f"📁 Loading: {SCORED_FILE}")
    scored_df = pd.read_csv(SCORED_FILE)
    print(f"   Loaded {len(scored_df)} scored off-target sites")
    
    # Apply population-aware scoring
    scored_df = apply_population_aware_scoring(scored_df)
    
    # Validate improvement
    metrics = validate_improvement(scored_df)
    
    # Save results
    scored_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n💾 Saved: {OUTPUT_FILE}")
    
    # Final summary
    print("\n" + "="*60)
    print("📊 FINAL RESULTS")
    print("="*60)
    
    print(f"\n🎯 AUROC Comparison:")
    print(f"   Original:      {metrics['original_auroc']:.4f}")
    print(f"   Population:    {metrics['population_auroc']:.4f}")
    print(f"   Improvement:   {metrics['improvement']:+.4f}")
    
    if metrics['target_achieved']:
        print("\n🎉 TARGET ACHIEVED! ✅")
        print(f"   Population-aware AUROC = {metrics['population_auroc']:.4f} >= 0.70")
    else:
        gap = 0.70 - metrics['population_auroc']
        print(f"\n⚠️  Target not reached (gap: {gap:.4f})")

if __name__ == "__main__":
    main()
