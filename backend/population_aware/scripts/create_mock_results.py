"""
Create Mock Results for Indian Chr22 (When GPU Credits Run Out)

This script loads the REAL variants and REAL allele frequencies,
but generates SIMULATED Evo2 scores so we can test the downstream analysis
and visualization pipelines.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

INPUT_FILE = "results/indian_chr22_ready_for_evo2.csv"
OUTPUT_FILE = "results/indian_chr22_evo2_scores.csv"

def main():
    if not Path(INPUT_FILE).exists():
        logger.error(f"Input file not found: {INPUT_FILE}")
        return

    logger.info(f"Loading real variants from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    n = len(df)
    logger.info(f"Loaded {n} variants.")

    # Generate Synthetic Evo2 Scores
    # Distribute them somewhat realistically:
    # - Most variance (-1 to 1) -> Benignish
    # - Tail (-8 to -2) -> Pathogenic
    
    np.random.seed(42) # Reproducible
    
    # 80% Benign-ish (Normal dist around 0)
    scores_benign = np.random.normal(0, 1, int(n * 0.8))
    
    # 20% Pathogenic-ish (Normal dist around -5)
    scores_pathogenic = np.random.normal(-5, 2, n - int(n * 0.8))
    
    all_scores = np.concatenate([scores_benign, scores_pathogenic])
    np.random.shuffle(all_scores)
    
    df['evo2_score'] = all_scores
    
    logger.info(f"Generated {n} mock Evo2 scores.")
    
    # Save
    df.to_csv(OUTPUT_FILE, index=False)
    logger.info(f"Saved mock results to {OUTPUT_FILE}")
    logger.info("You can now run 'python scripts/analyze_vus_reduction.py'!")

if __name__ == "__main__":
    main()
