"""
Local gnomAD client using VCF files with pysam (for Linux/WSL2).

This is the ORIGINAL version that works perfectly with indexed VCF access.
Use this in WSL2 where pysam installs without issues.
"""

import pysam
import logging
from pathlib import Path
from typing import Optional, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache opened VCF files to avoid repeated file opening
_vcf_cache: Dict[str, pysam.VariantFile] = {}


def get_gnomad_af(
    chrom: str, 
    pos: int, 
    ref: str, 
    alt: str, 
    data_dir: str = "/mnt/d/project/biotech-evo2/backend/population_aware/data"
) -> Dict[str, float]:
    """
    Query local gnomAD v4 VCF for population-specific allele frequencies.
    
    Uses pysam for FAST indexed access - works in Linux/WSL2.
    
    Args:
        chrom: Chromosome ("17" or "chr17")
        pos: Position in hg38 coordinates
        ref: Reference allele
        alt: Alternate allele
        data_dir: Directory containing gnomAD VCF files
    
    Returns:
        Dictionary mapping population codes to allele frequencies
        e.g., {'af_nfe': 0.001, 'af_afr': 0.0002, ...}
        Returns empty dict if variant not found.
    """
    # Normalize chromosome
    chrom_num = chrom.replace('chr', '')
    
    # Construct VCF path
    vcf_key = f"chr{chrom_num}"
    vcf_path = Path(data_dir) / f"gnomad.genomes.v4.0.sites.{vcf_key}.vcf.bgz"
    
    # Open VCF if not already cached
    if vcf_key not in _vcf_cache:
        if not vcf_path.exists():
            logger.error(f"VCF file not found: {vcf_path}")
            return {}
        
        try:
            _vcf_cache[vcf_key] = pysam.VariantFile(str(vcf_path), "r")
            logger.info(f"✓ Opened {vcf_path.name}")
        except Exception as e:
            logger.error(f"Failed to open VCF: {e}")
            return {}
    
    vcf = _vcf_cache[vcf_key]
    
    # Query position using index (FAST!)
    try:
        # gnomAD VCF uses "chr17" format, not "17"
        for record in vcf.fetch(vcf_key, pos-1, pos):
            if record.pos == pos and record.ref == ref:
                # Check if this alt allele matches
                if alt in record.alts:
                    # Extract population AFs
                    af_dict = {}
                    # gnomAD v4 uses lowercase: AF_nfe, AF_afr, etc.
                    for pop in ['nfe', 'afr', 'eas', 'sas', 'amr']:
                        key = f'AF_{pop}'
                        if key in record.info:
                            # gnomAD stores AF as a list (one per alt allele)
                            alt_index = record.alts.index(alt)
                            af_values = record.info[key]
                            
                            if isinstance(af_values, (list, tuple)):
                                af = af_values[alt_index] if alt_index < len(af_values) else 0.0
                            else:
                                af = af_values
                            
                            af_dict[f'af_{pop.lower()}'] = float(af) if af is not None else 0.0
                        else:
                            af_dict[f'af_{pop.lower()}'] = 0.0
                    
                    return af_dict
    
    except Exception as e:
        logger.warning(f"Error querying {chrom}:{pos} {ref}>{alt}: {e}")
    
    return {}


def test_local_gnomad():
    """Test with known BRCA1 variant."""
    logger.info("Testing local gnomAD client with pysam...")
    
    # rs28897672 (common BRCA1 variant)
    af_data = get_gnomad_af("chr17", 43044295, "G", "A")
    
    if af_data:
        logger.info("✓ Successfully queried variant!")
        logger.info(f"  Population frequencies:")
        for pop, af in af_data.items():
            if af > 0:
                logger.info(f"    {pop}: {af:.6f}")
        logger.info("✓ Test passed!")
    else:
        logger.warning("Variant not found (may be normal for some positions)")


if __name__ == "__main__":
    test_local_gnomad()
