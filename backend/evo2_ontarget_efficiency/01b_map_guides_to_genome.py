"""
01b_map_guides_to_genome.py

Map DeepHF guide sequences to genomic coordinates in hg38.
Uses exact string matching with pysam for speed.

METHOD:
1. Load 23bp sequences (20bp guide + 3bp PAM)
2. Search for exact matches in hg38 (including reverse complement)
3. Keep only uniquely-mapping guides
4. Output: chromosome, position, strand for each guide

REQUIREMENTS:
- hg38.fa (indexed with .fai)
- pysam: pip install pysam
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import pysam
from collections import defaultdict
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

DATA_DIR = Path(__file__).parent / "data"
GENOME_PATH = Path("d:/project/biotech-evo2/backend/crispr_offtarget/data/genome/hg38.fa")

# Chromosomes to search (skip alt/random contigs for speed)
CHROMOSOMES = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]


def reverse_complement(seq: str) -> str:
    """Get reverse complement of DNA sequence."""
    complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}
    return ''.join(complement.get(base, 'N') for base in reversed(seq.upper()))


def find_guide_in_chromosome(guide_20bp: str, chrom_seq: str, chrom_name: str) -> list:
    """
    Find all occurrences of a 20bp guide in a chromosome.
    Searches both strands.
    Returns: list of (chrom, position, strand)
    """
    matches = []
    guide = guide_20bp.upper()
    guide_rc = reverse_complement(guide)
    
    # Forward strand: guide followed by NGG PAM
    # Pattern: [guide][N][G][G]
    pattern_fwd = guide + "[ACGT]GG"
    for match in re.finditer(pattern_fwd, chrom_seq):
        matches.append((chrom_name, match.start(), '+'))
    
    # Reverse strand: CCN followed by reverse complement of guide
    # Pattern: CC[N][rc_guide]
    pattern_rev = "CC[ACGT]" + guide_rc
    for match in re.finditer(pattern_rev, chrom_seq):
        # Position is at the end of guide (start of PAM on - strand)
        matches.append((chrom_name, match.start() + 3, '-'))
    
    return matches


def map_guides_to_genome(deephf_path: Path, genome_path: Path, output_path: Path):
    """
    Map all DeepHF guides to hg38 genome coordinates.
    """
    logging.info("=" * 60)
    logging.info("Mapping DeepHF Guides to hg38 Genome")
    logging.info("=" * 60)
    
    # Load DeepHF data
    df = pd.read_csv(deephf_path)
    logging.info(f"Loaded {len(df)} guides from DeepHF")
    
    # Extract 20bp guide (remove PAM)
    df['guide_20bp'] = df['23-nt sequence'].str[:20].str.upper()
    
    # Remove duplicates and NaN
    df = df.dropna(subset=['23-nt sequence', 'Wt_Efficiency'])
    unique_guides = df['guide_20bp'].unique()
    logging.info(f"Unique 20bp guides: {len(unique_guides)}")
    
    # Open genome
    logging.info(f"Loading genome: {genome_path}")
    genome = pysam.FastaFile(str(genome_path))
    
    # Build guide -> matches mapping
    guide_matches = defaultdict(list)
    
    for i, chrom in enumerate(CHROMOSOMES):
        logging.info(f"Searching {chrom} ({i+1}/{len(CHROMOSOMES)})...")
        
        try:
            chrom_seq = genome.fetch(chrom).upper()
        except Exception as e:
            logging.warning(f"Could not fetch {chrom}: {e}")
            continue
        
        # Search for each unique guide
        for guide in unique_guides:
            matches = find_guide_in_chromosome(guide, chrom_seq, chrom)
            guide_matches[guide].extend(matches)
        
        # Progress
        found_so_far = sum(1 for g in unique_guides if len(guide_matches[g]) == 1)
        logging.info(f"  Unique matches so far: {found_so_far}")
    
    genome.close()
    
    # Filter to uniquely-mapping guides
    logging.info("\nFiltering to uniquely-mapping guides...")
    unique_mapped = {g: m[0] for g, m in guide_matches.items() if len(m) == 1}
    multi_mapped = {g: m for g, m in guide_matches.items() if len(m) > 1}
    no_match = [g for g in unique_guides if len(guide_matches[g]) == 0]
    
    logging.info(f"  Unique match: {len(unique_mapped)} ({100*len(unique_mapped)/len(unique_guides):.1f}%)")
    logging.info(f"  Multiple matches: {len(multi_mapped)}")
    logging.info(f"  No match: {len(no_match)}")
    
    # Create output dataframe
    mapped_data = []
    for _, row in df.iterrows():
        guide = row['guide_20bp']
        if guide in unique_mapped:
            chrom, pos, strand = unique_mapped[guide]
            mapped_data.append({
                '23-nt sequence': row['23-nt sequence'],
                'guide_20bp': guide,
                'efficiency': row['Wt_Efficiency'],
                'chromosome': chrom,
                'position': pos,
                'strand': strand,
                'window_start': pos - 4096,
                'window_end': pos + 4096,
            })
    
    mapped_df = pd.DataFrame(mapped_data)
    mapped_df['sample_id'] = range(len(mapped_df))
    
    # Save
    mapped_df.to_csv(output_path, index=False)
    logging.info(f"\n✅ Saved {len(mapped_df)} mapped guides to: {output_path}")
    
    # Summary stats
    logging.info(f"\nEfficiency distribution of mapped guides:")
    logging.info(f"  Mean: {mapped_df['efficiency'].mean():.3f}")
    logging.info(f"  Std:  {mapped_df['efficiency'].std():.3f}")
    logging.info(f"  Min:  {mapped_df['efficiency'].min():.3f}")
    logging.info(f"  Max:  {mapped_df['efficiency'].max():.3f}")
    
    return mapped_df


def create_train_test_split(mapped_df: pd.DataFrame, test_chroms: list = ['chr8', 'chr9']):
    """
    Split by chromosome for proper train/test separation.
    """
    logging.info("\nCreating train/test split by chromosome...")
    
    test_df = mapped_df[mapped_df['chromosome'].isin(test_chroms)]
    train_df = mapped_df[~mapped_df['chromosome'].isin(test_chroms)]
    
    # Re-index
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df['sample_id'] = range(len(train_df))
    test_df['sample_id'] = range(len(test_df))
    
    # Save
    train_df.to_csv(DATA_DIR / "train_coords.csv", index=False)
    test_df.to_csv(DATA_DIR / "test_coords.csv", index=False)
    
    logging.info(f"  Train: {len(train_df)} samples (all except {test_chroms})")
    logging.info(f"  Test:  {len(test_df)} samples ({test_chroms})")
    
    return train_df, test_df


def main():
    deephf_path = DATA_DIR / "DeepHF Dataset.csv"
    output_path = DATA_DIR / "deephf_mapped.csv"
    
    if not deephf_path.exists():
        logging.error(f"DeepHF data not found: {deephf_path}")
        return
    
    if not GENOME_PATH.exists():
        logging.error(f"hg38 genome not found: {GENOME_PATH}")
        return
    
    # Map guides to genome
    mapped_df = map_guides_to_genome(deephf_path, GENOME_PATH, output_path)
    
    # Create train/test split
    if len(mapped_df) > 0:
        create_train_test_split(mapped_df)
        
        logging.info("\n" + "=" * 60)
        logging.info("NEXT STEPS:")
        logging.info("  1. Upload hg38.fa to Modal: modal volume put genome-data ...")
        logging.info("  2. Run extraction: modal run 02_modal_extract.py")
        logging.info("=" * 60)


if __name__ == "__main__":
    main()
