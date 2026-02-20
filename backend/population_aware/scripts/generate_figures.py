import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from pathlib import Path

def setup_directories(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print(f"📂 Output directory: {output_dir}")

def generate_figures(input_file, dpi, fmt):
    print(f"⏳ Loading data from {input_file}...")
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        return

    df = pd.read_csv(input_file)
    df = df[df['status'] == 'success']
    print(f"   Loaded {len(df)} variants.")
    
    output_dir = "results/nature_figures_highres"
    setup_directories(output_dir)

    # Set aesthetics for high quality
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    
    # Figure 1: Score Distribution
    print("🎨 Generating Figure 1 (Darker)...")
    plt.figure(figsize=(10, 6))
    sns.histplot(df['evo2_score'], bins=100, kde=True, color='#2166ac', edgecolor='#053061', alpha=0.9)
    plt.axvline(x=-5, color='#b2182b', linestyle='--', linewidth=2, label='High-Impact (<-5)')
    plt.axvline(x=-2, color='#e66101', linestyle='--', linewidth=2, label='Deleterious (<-2)')
    plt.xlabel('Evo2 Score (ΔLogLikelihood)')
    plt.ylabel('Variant Count')
    plt.title('Distribution of Variant Impact Scores')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/figure1_score_distribution.{fmt}", dpi=dpi)
    plt.close()

    # Figure 2: Population Enrichmemnt
    print("🎨 Generating Figure 2 (Darker)...")
    plt.figure(figsize=(10, 6))
    plot_df = df if len(df) < 50000 else df.sample(50000, random_state=42)
    plt.scatter(plot_df['af_global'], plot_df['af_indian'], alpha=0.6, s=5, c=plot_df['evo2_score'], cmap='RdBu') # Darker contrast map
    plt.colorbar(label='Evo2 Score')
    plt.axline((0, 0), slope=1, color='black', linestyle='--', linewidth=1.5)
    plt.xlabel('Global Allele Frequency')
    plt.ylabel('Indian Allele Frequency')
    plt.title('Population-Specific Enrichment')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/figure2_population_enrichment.{fmt}", dpi=dpi)
    plt.close()

    # Figure 3: Indian Enriched Distribution
    print("🎨 Generating Figure 3 (Darker)...")
    enriched = df[(df['af_indian'] > 0.01) & (df['af_global'] < 0.001)]
    if not enriched.empty:
        plt.figure(figsize=(14, 6))
        plt.scatter(df['pos'], df['af_indian'], alpha=0.1, s=1, color='#7f7f7f', label='All Variants') # Darker gray background
        plt.scatter(enriched['pos'], enriched['af_indian'], alpha=0.9, s=20, c='#4b0082', label='Indian Enriched (>1%)') # Deep Indigo
        plt.xlabel('Genomic Position (Chr22)')
        plt.ylabel('Indian Allele Frequency')
        plt.title(f'Distribution of Indian-Enriched Variants (n={len(enriched)})')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/figure3_indian_enriched.{fmt}", dpi=dpi)
        plt.close()

    print(f"✅ High-Res Figures generated in {output_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate high-res figures")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for output")
    parser.add_argument("--format", type=str, default="png", help="Output format (png, pdf, svg)")
    args = parser.parse_args()
    
    # Default to PARTIAL if main not found
    input_path = "results/indian_chr22_evo2_scores_PARTIAL.csv"
    if not os.path.exists(input_path):
        input_path = "results/indian_chr22_evo2_scores.csv"
        
    generate_figures(input_path, args.dpi, args.format)
