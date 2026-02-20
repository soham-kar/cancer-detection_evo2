"""
Process Extracted Indian Variants from 26GB VCF

This script takes the extracted VCF (data/indian_chr22_subset.vcf.gz) containing ONLY Indian samples,
calculates Allele Frequencies (AF) specifically for this population, and prepares a CSV for Evo2 scoring.

Input: data/indian_chr22_subset.vcf.gz
Output: results/indian_chr22_rare_variants.csv
"""

import pysam
import pandas as pd
import logging
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def process_indian_vcf(vcf_path: str, output_path: str):
    logger.info(f"Processing VCF: {vcf_path}")
    
    if not Path(vcf_path).exists():
        logger.error(f"VCF file not found: {vcf_path}")
        return

    vcf = pysam.VariantFile(vcf_path)
    
    variants = []
    
    # Iterate through all variants
    count = 0
    kept = 0
    
    for record in vcf.fetch():
        count += 1
        if count % 10000 == 0:
            logger.info(f"Processed {count} variants... (Kept: {kept})")
            
        # Calculate Indian-specific Allele Frequency
        # Info field AC/AN might be from the ORIGINAL file (Global) if we just subsetted without re-calculating INFO
        # When using 'bcftools view', INFO tags AC/AN are not automatically updated unless 'bcftools +fill-tags' is used.
        # So we must calculate from Genotypes (GT) manually to be safe, OR trust the user ran +fill-tags.
        #
        # Better approach: Calculate from genotypes for accuracy on this subset.
        
        ac_indian = 0
        an_indian = 0
        
        # Fast parsing of genotypes
        # samples.values() returns list of objects, heavy.
        # record.samples is a proxy.
        # We can just rely on the fact that the VCF ONLY contains Indian samples.
        
        # To optimize speed, we can assume the extraction was done correctly.
        # But let's verify AC/AN. If we rely on INFO, it might be the OLD global counts depending on how bcftools was run.
        # The user's command was: "bcftools view -S ... -o ..." WITHOUT +fill-tags.
        # THIS MEANS INFO FIELDS ARE WRONG (They reflect the original 26GB file).
        # WE MUST CALCULATE AF FROM GENOTYPES.
        
        # Calculate AC/AN from genotypes
        # This is slow in Python for 26GB file, but for a subset of rare variants it might be okay.
        # Or we can iterate samples.
        
        # Let's try to be efficient.
        # Check if GT is available.
        
        hom_ref = 0
        het = 0
        hom_alt = 0
        missing = 0
        
        # Loop through samples (350 samples)
        for sample in record.samples.values():
            gt = sample['GT']
            if gt == (None, None) or gt == (None,): 
                missing += 1
                continue
                
            if gt == (0, 0):
                hom_ref += 1
            elif gt == (0, 1) or gt == (1, 0) or gt == (0, 2): # Simple het
                het += 1
                # Note: Multiallelic handling is complex, assuming biallelic for speed or simple splits
                # If splitting was done...
            elif gt == (1, 1):
                hom_alt += 1
            # Ignore others for now
            
        an_indian = (hom_ref + het + hom_alt) * 2
        ac_indian = het + (hom_alt * 2)
        
        if an_indian == 0:
            continue
            
        af_indian = ac_indian / an_indian
        
        # Filter: Rare variants only (< 5%)
        # But wait, we want to find RESCUE cases.
        # Rescue = Pathogenic by AI (likely rare) but Benign by Pop (common in India).
        # So we typically want somewhat common variants in India?
        # Actually, if it's > 5%, it's definitely Benign. 
        # So we keep things up to maybe 10% to capture those "common in India, rare globally" cases.
        # Let's filter > 10% as clearly benign and not needing AI.
        # Wait, if we filter them out, we can't show them as "Rescued".
        # We should keep "Common in India" variants to prove they are rescued.
        # But we don't need "Common Globally" variants.
        
        # Filter:
        # 1. Must be variant (AC > 0)
        # 2. Skip if AF > 25% (very common)
        
        if ac_indian == 0:
            continue
        if af_indian > 0.25:
            continue
            
        # Global AF (from INFO if available)
        # Note: 'AF' in INFO is likely the Global AF from the original file
        af_global = 0.0
        if 'AF' in record.info:
            val = record.info['AF']
            if isinstance(val, (list, tuple)):
                af_global = val[0]
            else:
                af_global = val
        
        variants.append({
            'chrom': record.chrom,
            'pos': record.pos,
            'ref': record.ref,
            'alt': record.alts[0], # Take first alt
            'af_indian': af_indian,
            'ac_indian': ac_indian,
            'an_indian': an_indian,
            'af_global': af_global
        })
        
        kept += 1

    logger.info(f"Finished. Total variants kept: {kept}")
    
    df = pd.DataFrame(variants)
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved to {output_path}")

if __name__ == "__main__":
    # Adjust paths as needed
    INPUT_VCF = "data/indian_chr22_subset.vcf.gz"
    OUTPUT_CSV = "results/indian_chr22_rare_variants.csv"
    
    process_indian_vcf(INPUT_VCF, OUTPUT_CSV)
