"""
Fetch Sequence Context for Indian Variants (Chr22) - HIGH PERFORMANCE VERSION

Reads: results/indian_chr22_rare_variants.csv
Writes: results/indian_chr22_ready_for_evo2.csv

OPTIMIZATION:
Instead of querying UCSC API 600,000 times (which takes days),
this script downloads the Chr22 FASTA (~15MB compressed), loads it into memory,
and slices it instantly.

Requires: pandas, tqdm, requests
"""

import pandas as pd
import requests
import gzip
import shutil
import os
from tqdm import tqdm
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# ================= CONFIGURATION =================
INPUT_FILE = "results/indian_chr22_rare_variants.csv"
OUTPUT_FILE = "results/indian_chr22_ready_for_evo2.csv"
WINDOW_SIZE = 512  # Evo2 needs context. We fetch 512bp on each side. Total 1025bp.

# UCSC hg38 Chr22 FASTA
FASTA_URL = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr22.fa.gz"
FASTA_GZ = "data/chr22.fa.gz"
FASTA_FILE = "data/chr22.fa"

def download_file(url, local_path):
    if Path(local_path).exists():
        logger.info(f"File already exists: {local_path}")
        return

    logger.info(f"Downloading {url} to {local_path}...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    logger.info("Download complete.")

def load_chr22_sequence():
    # 1. Download if needed
    if not Path(FASTA_FILE).exists():
        download_file(FASTA_URL, FASTA_GZ)
        
        # 2. Extract
        logger.info("Extracting FASTA...")
        with gzip.open(FASTA_GZ, 'rb') as f_in:
            with open(FASTA_FILE, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    
    # 3. Load into memory
    logger.info("Loading Chr22 sequence into memory...")
    seq_parts = []
    with open(FASTA_FILE, 'r') as f:
        # Skip header (>chr22 ...)
        header = f.readline()
        if not header.startswith(">"):
            logger.warning(f"Unexpected header: {header.strip()}")
        
        # Read remaining lines (sequence)
        # They are usually 50 lines line wrapped.
        for line in f:
            seq_parts.append(line.strip().upper())
            
    full_seq = "".join(seq_parts)
    logger.info(f"Chr22 Sequence Loaded. Length: {len(full_seq):,} bp")
    return full_seq

def main():
    if not Path(INPUT_FILE).exists():
        logger.error(f"Input file not found: {INPUT_FILE}")
        return

    # Load variants
    logger.info(f"Loading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    logger.info(f"Loaded {len(df)} variants")
    
    # Load Genome
    try:
        genome_seq = load_chr22_sequence()
    except Exception as e:
        logger.error(f"Failed to load genome: {e}")
        return

    # Pre-allocate lists for speed
    seq_contexts = []
    rel_positions = []
    
    # process
    valid_count = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting sequences"):
        chrom = str(row['chrom'])
        pos = int(row['pos']) # 1-based position
        
        # Verify Chromosome (just in case filtering failed)
        if "22" not in chrom:
            seq_contexts.append(None)
            rel_positions.append(None)
            continue

        # 0-based index of mutation
        # VCF pos 1 = index 0
        mut_idx = pos - 1
        
        # Calculate window
        # We want [mut_idx - 512, mut_idx + 512 + 1]
        start_idx = mut_idx - WINDOW_SIZE
        end_idx = mut_idx + WINDOW_SIZE + 1 # +1 for python slice exclusivity
        
        # Check bounds
        if start_idx < 0 or end_idx > len(genome_seq):
            seq_contexts.append(None) # Too close to edge
            rel_positions.append(None)
            continue
            
        # Extract
        seq = genome_seq[start_idx:end_idx]
        
        # Verify Ref Allele (Sanity Check)
        # The base at mut_idx (relative to genome) should match row['ref']
        # But genome is reference.
        # Relative index of mutation in 'seq' is WINDOW_SIZE (512)
        ref_base_in_genome = seq[WINDOW_SIZE]
        
        # Optional: Print mismatch warning? (Might span output depending on ref genome version mismatch)
        # gnomAD v4 is hg38. We downloaded hg38. Should match.
        
        seq_contexts.append(seq)
        rel_positions.append(WINDOW_SIZE)
        valid_count += 1

    # Update DataFrame
    df['seq_context'] = seq_contexts
    df['rel_pos'] = rel_positions
    
    # Drop failures
    df_clean = df.dropna(subset=['seq_context'])
    
    logger.info(f"Saving {len(df_clean)} variants to {OUTPUT_FILE}...")
    df_clean.to_csv(OUTPUT_FILE, index=False)
    logger.info("Done!")

if __name__ == "__main__":
    main()
