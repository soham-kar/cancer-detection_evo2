import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

# Configuration
INPUT_FILE = "results/indian_chr22_evo2_scores_PARTIAL.csv" # Using PARTIAL to capture latest data
OUTPUT_DIR = "results/nature_figures"

def setup_directories():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"📂 Created output directory: {OUTPUT_DIR}")

def analyze_evo2_results():
    setup_directories()
    
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Input file not found: {INPUT_FILE}")
        return

    print(f"⏳ Loading data from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    
    print("="*60)
    print("INDIAN CHROMOSOME 22 EVO2 ANALYSIS (Nature Genetics Pipeline)")
    print("="*60)
    print(f"Total variants analyzed: {len(df):,}")
    
    # Clean data just in case
    df = df[df['status'] == 'success'].copy()
    print(f"Successful scores: {len(df):,} ({len(df)/len(pd.read_csv(INPUT_FILE))*100:.1f}%)")
    
    # 1. Distribution of Evo2 scores
    print(f"\n📊 Score Distribution:")
    print(f"Mean: {df['evo2_score'].mean():.6f}")
    print(f"Std:  {df['evo2_score'].std():.6f}")
    print(f"Min:  {df['evo2_score'].min():.6f}")
    print(f"Max:  {df['evo2_score'].max():.6f}")
    
    # 2. HIGH-IMPACT DRIVERS (Evo2 < -5.0)
    # Evo2 scores are log-likelihood ratios. Very negative = disrupts sequence patterns = Pathogenic
    drivers = df[df['evo2_score'] < -5.0].sort_values('evo2_score')
    print(f"\n🔴 HIGH-IMPACT DRIVERS (Evo2 < -5.0): {len(drivers)}")
    if not drivers.empty:
        print("Top 5 drivers:")
        print(drivers[['variant', 'af_indian', 'evo2_score']].head(5).to_string(index=False))
        drivers.to_csv(f"{OUTPUT_DIR}/high_impact_drivers.csv", index=False)
    
    # 3. RESCUE CASES (Population-aware correction)
    # Logic: Looks pathogenic (Evo2 < -2) BUT is common in India (>1%) and Rare Globally (<1%)
    # Note: User prompt suggested strict AF_Indian > 0.05, but 0.01 is also significant. 
    # Using User's logic: AF_Ind > 0.05 & AF_Global < 0.01
    rescues = df[
        (df['evo2_score'] < -2.0) &       # Predicted Pathogenic
        (df['af_indian'] > 0.05) &        # Common in India (>5%)
        (df['af_global'] < 0.01)          # Rare Globally (<1%)
    ]
    print(f"\n🟢 RESCUE CASES (Preventing False Positives): {len(rescues)}")
    if not rescues.empty:
        print("Top Rescue variants:")
        print(rescues[['variant', 'af_indian', 'af_global', 'evo2_score']].head(5).to_string(index=False))
        rescues.to_csv(f"{OUTPUT_DIR}/rescue_cases.csv", index=False)
    else:
        # Relax criteria slightly for interest if strictly empty (e.g. >1% Indian)
        print("   (No strict rescues found >5% Indian AF. Checking >1%...)")
        rescues_relaxed = df[
            (df['evo2_score'] < -2.0) & 
            (df['af_indian'] > 0.01) & 
            (df['af_global'] < 0.001)
        ]
        print(f"   (Relaxed Criteria >1% Indian / <0.1% Global): {len(rescues_relaxed)}")
    
    # 4. POPULATION-UNIQUE (Indian-specific)
    unique = df[
        (df['af_indian'] > 0.01) &    # >1% in Indians
        (df['af_global'] < 0.001)     # <0.1% globally
    ]
    print(f"\n🟡 INDIAN-ENRICHED UNIQUE: {len(unique)}")
    print(f"Enrichment rate: {len(unique)/len(df)*100:.2f}%")
    if not unique.empty:
        unique.to_csv(f"{OUTPUT_DIR}/indian_enriched.csv", index=False)
    
    # 5. Pharmacogenomic variants (CYP genes)
    pgx_genes = ['CYP2C19', 'CYP2C9', 'CYP2B6', 'CYP4F2', 'CYP2D6']
    # Filter by variant string containing gene name if encoded, OR map by position if we had annotations.
    # Our CSV 'variant' column is chr22:.... so we don't strictly have gene names unless seq_context/id has it.
    # However, user mentions checking text. We'll search if 'variant' column has gene text (unlikely for chr:pos)
    # BUT, let's assume valid chromosomal regions for these genes on Chr22.
    # Actually, CYP2D6 is on Chr22! (22q13.1) ~ pos 42,126,000 to 42,132,000
    # Let's filter by position range for CYP2D6 (Main one on Chr22)
    
    cyp2d6_start = 42126000
    cyp2d6_end = 42133000
    pgx = df[(df['pos'] >= cyp2d6_start) & (df['pos'] <= cyp2d6_end)]
    
    print(f"\n💊 CYP2D6 (Chr22 PGx) variants: {len(pgx)}")
    if not pgx.empty:
        high_pgx = pgx[pgx['evo2_score'] < -2.0]
        print(f"   Functional/Deleterious PGx variants (Evo2 < -2): {len(high_pgx)}")
        if not high_pgx.empty:
            print(high_pgx[['variant', 'af_indian', 'evo2_score']].head().to_string(index=False))
            high_pgx.to_csv(f"{OUTPUT_DIR}/cyp2d6_high_impact.csv", index=False)

    # ==========================================
    # GENERATE FIGURES
    # ==========================================
    print(f"\n🎨 Generating Figures in {OUTPUT_DIR}/...")

    # Figure 1: Score Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['evo2_score'], bins=100, kde=True, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(x=-5, color='red', linestyle='--', linewidth=2, label='High-Impact (<-5)')
    plt.axvline(x=-2, color='orange', linestyle='--', linewidth=2, label='Deleterious (<-2)')
    plt.axvline(x=0, color='gray', linestyle=':', label='Neutral')
    plt.xlabel('Evo2 Score (ΔLogLikelihood)', fontsize=12)
    plt.ylabel('Variant Count', fontsize=12)
    plt.title('Figure 1: Distribution of Variant Impact Scores (Indian Chr22)', fontsize=14)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(f"{OUTPUT_DIR}/figure1_score_distribution.png", dpi=300, bbox_inches='tight')
    print("   ✅ Generated Figure 1: Score Distribution")

    # Figure 2: Population Enrichment
    plt.figure(figsize=(10, 6))
    # Downsample for scatter plot if too huge
    plot_df = df if len(df) < 50000 else df.sample(50000, random_state=42)
    
    plt.scatter(plot_df['af_global'], plot_df['af_indian'], alpha=0.4, s=5, c=plot_df['evo2_score'], cmap='coolwarm_r')
    plt.colorbar(label='Evo2 Score (Red=Deleterious)')
    plt.axline((0, 0), slope=1, color='black', linestyle='--', linewidth=1, label='Equal Frequency')
    
    # Highlight Enriched zone
    plt.fill_between([0, 0.001], 0.01, 1.0, color='yellow', alpha=0.1, label='Indian Enriched')
    
    plt.xlabel('Global Allele Frequency', fontsize=12)
    plt.ylabel('Indian Allele Frequency', fontsize=12)
    plt.title('Figure 2: Population-Specific Enrichment', fontsize=14)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/figure2_population_enrichment.png", dpi=300)
    print("   ✅ Generated Figure 2: Population Enrichment")

    # Figure 3: Rescue Cases
    if not rescues.empty:
        plt.figure(figsize=(12, 6))
        
        # Taking top 10 rescues
        top_rescues = rescues.sort_values('af_indian', ascending=False).head(10)
        
        indices = range(len(top_rescues))
        plt.bar([x - 0.2 for x in indices], top_rescues['af_indian'], width=0.4, label='Indian AF', color='gold', edgecolor='black')
        plt.bar([x + 0.2 for x in indices], top_rescues['af_global'], width=0.4, label='Global AF', color='gray', edgecolor='black')
        
        # Annotate with Evo2 Score
        for i, (idx, row) in enumerate(top_rescues.iterrows()):
            plt.text(i, row['af_indian'] + 0.002, f"Evo2\n{row['evo2_score']:.2f}", ha='center', fontsize=9, color='red')
            
        plt.xticks(indices, top_rescues['variant'], rotation=45, ha='right')
        plt.ylabel('Allele Frequency')
        plt.title('Figure 3: Top Rescue Cases (Potential False Positives prevented)', fontsize=14)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/figure3_rescue_cases.png", dpi=300)
        print("   ✅ Generated Figure 3: Rescue Cases")
    else:
        print("   ⚠️ No strict rescue cases for Figure 3 (Skipping plot).")
        
    # Figure 4: Indian-Enriched Distribution (Requested by User)
    print("   🎨 Generating Figure 4: Indian-Enriched Distribution...")
    enriched = df[(df['af_indian'] > 0.01) & (df['af_global'] < 0.001)]
    
    if not enriched.empty:
        plt.figure(figsize=(14, 6))
        # Plot all density in background in gray
        plt.scatter(df['pos'], df['af_indian'], alpha=0.05, s=1, color='lightgray', label='All Variants')
        
        # Plot enriched
        plt.scatter(enriched['pos'], enriched['af_indian'], alpha=0.6, s=15, c='purple', label='Indian Enriched (>1%)')
        
        plt.xlabel('Genomic Position (Chr22)', fontsize=12)
        plt.ylabel('Indian Allele Frequency', fontsize=12)
        plt.title('Figure 4: Distribution of Indian-Enriched Variants (n={})'.format(len(enriched)), fontsize=14)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/figure4_indian_enriched_distribution.png", dpi=300)
        print("   ✅ Generated Figure 4: Indian-Enriched Distribution")
    else:
        print("   ⚠️ No enriched variants found for Figure 4.")

    print("\n✅ Analysis Complete. All artifacts in results/nature_figures/")

if __name__ == "__main__":
    analyze_evo2_results()
