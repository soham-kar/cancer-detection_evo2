"""
gnomAD GraphQL API Client

Fetches population frequencies from gnomAD v4 dataset using their official GraphQL API.
Used when local VCF files are not available or too large to download.
"""

import requests
import time
import logging
from typing import Optional, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GNOMAD_API_URL = "https://gnomad.broadinstitute.org/api"

def fetch_gnomad_variant_data(chrom: str, pos: int, ref: str, alt: str, dataset: str = "gnomad_r4") -> Optional[Dict[str, Any]]:
    """
    Fetch variant data from gnomAD GraphQL API.
    
    Args:
        chrom: Chromosome (e.g., "22" or "chr22")
        pos: Position (hg38)
        ref: Reference allele
        alt: Alternate allele
        dataset: gnomAD dataset ID (default: "gnomad_r4")
        
    Returns:
        Dictionary with variant data or None if not found/error.
    """
    # Normalize chrom
    chrom_clean = chrom.replace("chr", "")
    variant_id = f"{chrom_clean}-{pos}-{ref}-{alt}"
    
    query = """
    query GnomadVariant($variantId: String!, $datasetId: DatasetId!) {
        variant(variantId: $variantId, dataset: $datasetId) {
            exome {
                ac
                an
                af
                populations {
                    id
                    ac
                    an
                    freq
                }
            }
            genome {
                ac
                an
                af
                populations {
                    id
                    ac
                    an
                    freq
                }
            }
        }
    }
    """
    
    variables = {
        "variantId": variant_id,
        "datasetId": dataset
    }
    
    try:
        response = requests.post(
            GNOMAD_API_URL,
            json={"query": query, "variables": variables},
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            if "errors" in data:
                logger.warning(f"GraphQL error for {variant_id}: {data['errors'][0]['message']}")
                return None
                
            return data.get("data", {}).get("variant")
        else:
            logger.error(f"API request failed: {response.status_code}")
            return None
            
    except Exception as e:
        logger.error(f"Exception querying {variant_id}: {e}")
        return None

def extract_frequencies(variant_data: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """
    Extract population frequencies from API response.
    Prioritizes Genome data (v4 is mostly genomes), falls back to Exome.
    """
    if not variant_data:
        return {}
    
    # Prefer genome data for v4, but check exome if genome missing
    data_source = variant_data.get("genome")
    if not data_source:
        data_source = variant_data.get("exome")
        
    if not data_source:
        return {}
        
    freqs = {}
    
    # Global AF
    freqs["af_global"] = data_source.get("af", 0.0)
    
    # Population specific
    # Map gnomAD pop IDs to our standard keys
    # gnomAD IDs: afr, amr, asj, eas, fin, nfe, sas, oth
    pop_map = {
        "afr": "af_afr",
        "amr": "af_amr",
        "eas": "af_eas",
        "nfe": "af_nfe", # Non-Finnish European
        "sas": "af_sas"  # South Asian
    }
    
    populations = data_source.get("populations", [])
    for pop in populations:
        pop_id = pop.get("id", "").lower()
        if pop_id in pop_map:
            freqs[pop_map[pop_id]] = pop.get("freq", 0.0)
            
    return freqs

def test_client():
    """Test with a known variant"""
    # Example variant on chr22
    # chr22:18137350 A>G (just a random example, or use one from our list)
    # Actually let's use a real one from our CSV: chr22:10510061 A>T 
    # (Note: This specific variant might be rare/missing, but we can test the call)
    
    print("Testing gnomAD API client...")
    
    # Using a known common variant on chr22 to ensure hits
    # rs2075596 : chr22:23916624 C>T (hg38)
    # Actually wait, let's use one from our file to see if it works.
    chrom, pos, ref, alt = "22", 10510061, "A", "T"
    
    data = fetch_gnomad_variant_data(chrom, pos, ref, alt)
    freqs = extract_frequencies(data)
    
    print(f"Query: {chrom}:{pos} {ref}>{alt}")
    if freqs:
        print("Found frequencies:")
        for k, v in freqs.items():
            print(f"  {k}: {v:.6f}")
    else:
        print("Variant not found in gnomAD (or API error)")

if __name__ == "__main__":
    test_client()
