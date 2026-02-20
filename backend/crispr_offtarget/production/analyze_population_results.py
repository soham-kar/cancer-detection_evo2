"""
Population-Aware CRISPR Analysis

Applies population penalties to scored CRISPR data and generates:
1. Per-population AUROC scores
2. Risky gRNAs by population
3. Population risk heatmap
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import json

# ============================================================
# Configuration
# ============================================================

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures" / "population"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Population-specific PAM penalties (from thesis: population_thresholds.py)
POPULATION_PENALTIES = {
    'AFR': 0.015,  # African - highest diversity, most PAM variants
    'AMR': 0.010,  # Admixed American
    'SAS': 0.008,  # South Asian
    'EAS': 0.006,  # East Asian
    'FIN': 0.005,  # Finnish (bottleneck)
    'EUR': 0.003,  # European (reference)
}

# gRNA to target population (based on clinical trial demographics)
GRNA_TARGET_POPULATION = {
    'EMX1_site1': 'EUR',
    'FANCF_site1': 'AMR',
    'PCSK9_site1': 'AMR',
    'HBB_site1': 'AFR',  # Sickle cell focus
    'RUNX1_site1': 'EUR',
    'VEGFA_site1': 'EAS',
}


def apply_population_penalties(df: pd.DataFrame) -> pd.DataFrame:
    """Apply population-specific penalties to each off-target site"""
    
    populations = list(POPULATION_PENALTIES.keys())
    
    for pop in populations:
        penalty = POPULATION_PENALTIES[pop]
        
        # Get target population for each gRNA
        df['target_pop'] = df['grna_name'].map(GRNA_TARGET_POPULATION).fillna('EUR')
        
        # Apply penalty (higher if gRNA not designed for this population)
        df[f'{pop}_penalty'] = df.apply(
            lambda row: penalty * (1.5 if row['target_pop'] != pop else 1.0),
            axis=1
        )
        
        # Calculate population-adjusted score
        df[f'{pop}_score'] = df['final_score'] - df[f'{pop}_penalty']
        
        # Calculate population-specific risk
        df[f'{pop}_risk'] = (
            -df[f'{pop}_score'] *
            (1 / (1 + df['mismatches'])) *
            (1 + df['seed_penalty'] * 2.0)
        )
    
    return df


def calculate_population_aurocs(df: pd.DataFrame) -> dict:
    """Calculate AUROC for each population"""
    
    populations = list(POPULATION_PENALTIES.keys())
    aurocs = {}
    
    for pop in populations:
        try:
            auroc = roc_auc_score(df['is_validated'], df[f'{pop}_score'])
            aurocs[pop] = round(auroc, 4)
        except:
            aurocs[pop] = None
    
    return aurocs


def identify_risky_grnas(df: pd.DataFrame, threshold: float = 0.02) -> pd.DataFrame:
    """Identify gRNAs with elevated risk in specific populations"""
    
    populations = list(POPULATION_PENALTIES.keys())
    risky = []
    
    for pop in populations:
        risk_col = f'{pop}_risk'
        high_risk = df[df[risk_col] > threshold]
        
        for _, row in high_risk.iterrows():
            # Check if this is specifically high for THIS population
            other_risks = [row[f'{p}_risk'] for p in populations if p != pop]
            avg_other = np.mean(other_risks)
            
            if row[risk_col] > avg_other * 1.2:  # 20% higher than average
                risky.append({
                    'grna_name': row['grna_name'],
                    'population': pop,
                    'risk_score': round(row[risk_col], 4),
                    'avg_other_risk': round(avg_other, 4),
                    'relative_risk': round(row[risk_col] / avg_other, 2) if avg_other > 0 else 1,
                    'recommendation': f'CAUTION in {pop}'
                })
    
    return pd.DataFrame(risky)


def create_population_heatmap(df: pd.DataFrame, output_file: Path):
    """Create heatmap of population-specific risks"""
    
    populations = list(POPULATION_PENALTIES.keys())
    
    # Aggregate by gRNA
    grna_risks = df.groupby('grna_name')[[f'{p}_risk' for p in populations]].mean()
    grna_risks.columns = populations
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(grna_risks, cmap='YlOrRd', annot=True, fmt='.3f',
                cbar_kws={'label': 'Risk Score'})
    plt.title('Population-Specific Off-Target Risk\n(Higher = More Dangerous)',
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Population', fontsize=12, fontweight='bold')
    plt.ylabel('gRNA', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def create_auroc_comparison(aurocs: dict, baseline: float, output_file: Path):
    """Create bar chart comparing AUROC across populations"""
    
    populations = list(aurocs.keys())
    values = [aurocs[p] for p in populations]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#ff6b6b' if v < baseline else '#4ecdc4' for v in values]
    bars = ax.bar(populations, values, color=colors, edgecolor='black')
    
    # Add baseline line
    ax.axhline(y=baseline, color='blue', linestyle='--', linewidth=2,
               label=f'Baseline AUROC: {baseline:.3f}')
    ax.axhline(y=0.70, color='green', linestyle=':', linewidth=1,
               label='Target: 0.70')
    
    ax.set_xlabel('Population', fontsize=12, fontweight='bold')
    ax.set_ylabel('AUROC', fontsize=12, fontweight='bold')
    ax.set_title('CRISPR Off-Target Prediction by Population\nPopulation-Aware Scoring',
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='lower right')
    ax.set_ylim(0.5, 0.85)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    print("="*60)
    print("POPULATION-AWARE CRISPR ANALYSIS")
    print("="*60)
    
    # Load scored data
    input_file = RESULTS_DIR / "crispr_offtarget_final.csv"
    if not input_file.exists():
        print(f"❌ Input file not found: {input_file}")
        print("   Run final_scorer.py first")
        return
    
    print(f"\n📂 Loading: {input_file}")
    df = pd.read_csv(input_file)
    print(f"   Loaded {len(df)} sites")
    
    # Calculate baseline AUROC
    baseline = roc_auc_score(df['is_validated'], df['final_score'])
    print(f"\n📊 Baseline AUROC: {baseline:.4f}")
    
    # Apply population penalties
    print("\n🌍 Applying population-specific penalties...")
    df = apply_population_penalties(df)
    
    # Calculate per-population AUROC
    print("\n📊 Per-Population AUROC:")
    aurocs = calculate_population_aurocs(df)
    for pop, auroc in aurocs.items():
        delta = auroc - baseline if auroc else 0
        status = "✅" if auroc and auroc > baseline else "⚠️"
        print(f"   {pop}: {auroc:.4f} ({delta:+.4f}) {status}")
    
    # Identify risky gRNAs
    print("\n🔍 Identifying population-risky gRNAs...")
    risky_df = identify_risky_grnas(df)
    print(f"   Found {len(risky_df)} population-specific risks")
    
    if len(risky_df) > 0:
        print("\n⚠️ Population-Specific Risks:")
        print(risky_df.to_string(index=False))
    
    # Save results
    output_file = RESULTS_DIR / "crispr_offtarget_population.csv"
    df.to_csv(output_file, index=False)
    print(f"\n✅ Saved: {output_file}")
    
    if len(risky_df) > 0:
        risky_file = RESULTS_DIR / "population_risky_grnas.csv"
        risky_df.to_csv(risky_file, index=False)
        print(f"✅ Saved: {risky_file}")
    
    # Save summary
    summary = {
        'baseline_auroc': baseline,
        'population_aurocs': aurocs,
        'auroc_variance': max(aurocs.values()) / min([a for a in aurocs.values() if a]),
        'risky_grnas_count': len(risky_df),
        'populations_analyzed': list(POPULATION_PENALTIES.keys())
    }
    
    summary_file = RESULTS_DIR / "population_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Saved: {summary_file}")
    
    # Generate figures
    print("\n📊 Generating figures...")
    
    heatmap_file = FIGURES_DIR / "population_risk_heatmap.png"
    create_population_heatmap(df, heatmap_file)
    print(f"   ✅ {heatmap_file}")
    
    auroc_file = FIGURES_DIR / "population_auroc_comparison.png"
    create_auroc_comparison(aurocs, baseline, auroc_file)
    print(f"   ✅ {auroc_file}")
    
    # Final summary
    print("\n" + "="*60)
    print("📊 POPULATION-AWARE ANALYSIS COMPLETE")
    print("="*60)
    print(f"\n   Baseline AUROC: {baseline:.4f}")
    print(f"   Best population: {max(aurocs, key=aurocs.get)} ({max(aurocs.values()):.4f})")
    print(f"   Population variance: {summary['auroc_variance']:.2f}x")
    print(f"   Risky gRNAs identified: {len(risky_df)}")


if __name__ == "__main__":
    main()
