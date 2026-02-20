"""
Day 1: Add gnomAD Population Frequencies to Findlay BRCA1 Dataset

This script:
1. Loads the Findlay et al. (2018) BRCA1 dataset (hg19 coordinates)
2. Converts coordinates from hg19 → hg38 using liftover
3. Queries local gnomAD v4 VCF files for population-specific allele frequencies
4. Saves annotated dataset for downstream analysis

Expected runtime: 3-5 minutes (using local VCF files)
"""

import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm
import logging

from utils.liftover import CoordinateLiftover
from utils.gnomad_client_local import get_gnomad_af

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_findlay_dataset(excel_path: str) -> pd.DataFrame:
    """
    Load the original Findlay et al. (2018) BRCA1 dataset.
    
    Args:
        excel_path: Path to the Excel file from Arc Institute's notebook
        
    Returns:
        DataFrame with BRCA1 variants
    """
    logger.info(f"Loading Findlay dataset from {excel_path}")
    
    df = pd.read_excel(
        excel_path,
        header=2,  # Data starts at row 3
    )
    
    # Rename columns for consistency
    df = df[[
        'chromosome', 'position (hg19)', 'reference', 'alt', 
        'function.score.mean', 'func.class',
    ]].copy()
    
    df.rename(columns={
        'chromosome': 'chrom',
        'position (hg19)': 'pos_hg19',
        'reference': 'ref',
        'alt': 'alt',
        'function.score.mean': 'func_score',
        'func.class': 'func_class',
    }, inplace=True)
    
    logger.info(f"Loaded {len(df)} variants")
    logger.info(f"Class distribution: {df['func_class'].value_counts().to_dict()}")
    
    return df


def add_liftover_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert hg19 coordinates to hg38 using liftover.
    
    Args:
        df: DataFrame with 'chrom' and 'pos_hg19' columns
        
    Returns:
        DataFrame with added 'pos_hg38' and 'liftover_success' columns
    """
    logger.info("Converting coordinates: hg19 → hg38")
    
    lo = CoordinateLiftover('hg19', 'hg38')
    
    # Apply liftover to all variants
    df['pos_hg38'] = df.apply(
        lambda row: lo.convert_position(str(row['chrom']), int(row['pos_hg19'])),
        axis=1
    )
    
    # Flag successful liftovers
    df['liftover_success'] = df['pos_hg38'].notna()
    
    success_count = df['liftover_success'].sum()
    success_rate = success_count / len(df) * 100
    
    logger.info(f"Liftover success: {success_count}/{len(df)} ({success_rate:.1f}%)")
    
    if success_rate < 95:
        logger.warning(f"Low liftover success rate: {success_rate:.1f}%")
    
    return df


def annotate_with_gnomad(df: pd.DataFrame) -> pd.DataFrame:
    """
    Query local gnomAD v4 VCF files for population allele frequencies.
    
    Args:
        df: DataFrame with hg38 coordinates
        
    Returns:
        DataFrame with added AF columns (af_nfe, af_afr, etc.)
    """
    logger.info("Querying gnomAD v4 VCF files for population frequencies...")
    
    # Initialize AF columns
    for pop in ['nfe', 'afr', 'eas', 'sas', 'amr']:
        df[f'af_{pop}'] = None
    
    df['gnomad_found'] = False
    
    # Query each variant
    successful_queries = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Querying gnomAD"):
        # Skip if liftover failed
        if not row['liftover_success']:
            continue
        
        # Query local VCF
        af_data = get_gnomad_af(
            chrom=str(row['chrom']),
            pos=int(row['pos_hg38']),
            ref=row['ref'],
            alt=row['alt']
        )
        
        if af_data:
            # Populate AF columns
            for key, value in af_data.items():
                df.at[idx, key] = value
            
            df.at[idx, 'gnomad_found'] = True
            successful_queries += 1
    
    success_rate = successful_queries / len(df) * 100
    logger.info(f"gnomAD found: {successful_queries}/{len(df)} ({success_rate:.1f}%)")
    
    # Convert AF columns to float (will be None/NaN for not found)
    for pop in ['nfe', 'afr', 'eas', 'sas', 'amr']:
        df[f'af_{pop}'] = pd.to_numeric(df[f'af_{pop}'], errors='coerce')
    
    return df


def generate_summary_stats(df: pd.DataFrame):
    """Print summary statistics about the annotated dataset."""
    logger.info("\n" + "="*60)
    logger.info("ANNOTATION SUMMARY")
    logger.info("="*60)
    
    # Liftover stats
    logger.info(f"Total variants: {len(df)}")
    logger.info(f"Liftover success: {df['liftover_success'].sum()} ({df['liftover_success'].sum() / len(df) * 100:.1f}%)")
    logger.info(f"gnomAD found: {df['gnomad_found'].sum()} ({df['gnomad_found'].sum() / len(df) * 100:.1f}%)")
    
    # Population coverage
    logger.info("\nPopulation AF coverage:")
    for pop in ['nfe', 'afr', 'eas', 'sas', 'amr']:
        col = f'af_{pop}'
        non_zero = (df[col] > 0).sum()
        logger.info(f"  {pop.upper()}: {non_zero} variants ({non_zero / len(df) * 100:.1f}%)")
    
    # Variants with both EUR and AFR
    both_eur_afr = ((df['af_nfe'] > 0) & (df['af_afr'] > 0)).sum()
    logger.info(f"\nVariants with both EUR and AFR frequencies: {both_eur_afr}")
    
    logger.info("="*60 + "\n")


def main():
    """Main execution pipeline for Day 1."""
    # Define paths
    project_root = Path(__file__).parent.parent.parent
    excel_path = project_root / 'backend' / 'evo2' / 'notebooks' / 'brca1' / '41586_2018_461_MOESM3_ESM.xlsx'
    output_path = project_root / 'backend' / 'population_aware' / 'results' / 'brca1_with_gnomad.csv'
    
    # Verify Excel file exists
    if not excel_path.exists():
        logger.error(f"Excel file not found: {excel_path}")
        logger.error("Please ensure the Arc Institute BRCA1 notebook data is available")
        return
    
    # Step 1: Load Findlay dataset
    df = load_findlay_dataset(str(excel_path))
    
    # Step 2: Liftover hg19 → hg38
    df = add_liftover_coordinates(df)
    
    # Step 3: Annotate with gnomAD
    df = annotate_with_gnomad(df)
    
    # Step 4: Generate summary
    generate_summary_stats(df)
    
    # Step 5: Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"✅ Saved annotated dataset to: {output_path}")
    
    # Step 6: Save a preview
    logger.info("\nFirst 5 variants (preview):")
    preview_cols = ['chrom', 'pos_hg19', 'pos_hg38', 'ref', 'alt', 'func_class', 
                    'af_nfe', 'af_afr', 'liftover_success', 'gnomad_found']
    print(df[preview_cols].head().to_string())


if __name__ == "__main__":
    main()
