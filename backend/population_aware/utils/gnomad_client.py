"""
gnomAD API client for querying population allele frequencies.

Uses the gnomAD v4 GraphQL API to retrieve ancestry-specific allele frequencies.
"""

import requests
import time
from typing import Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GnomADClient:
    """Client for querying gnomAD v4 API."""
    
    API_URL = "https://gnomad.broadinstitute.org/api"
    
    # Population codes in gnomAD v4
    POPULATIONS = {
        'nfe': 'Non-Finnish European',
        'afr': 'African/African American',
        'eas': 'East Asian',
        'sas': 'South Asian',
        'amr': 'Latino/Admixed American',
        'asj': 'Ashkenazi Jewish',
        'fin': 'Finnish',
        'mid': 'Middle Eastern',
        'remaining': 'Remaining'
    }
    
    def __init__(self, rate_limit_delay: float = 0.2):
        """
        Initialize gnomAD client.
        
        Args:
            rate_limit_delay: Delay between requests in seconds (default: 0.2s = 5 req/s)
        """
        self.rate_limit_delay = rate_limit_delay
        self.last_request_time = 0
    
    def _rate_limit(self):
        """Enforce rate limiting."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()
    
    def query_variant(
        self, 
        chrom: str, 
        pos: int, 
        ref: str, 
        alt: str,
        dataset: str = "gnomad_r4"
    ) -> Optional[Dict[str, float]]:
        """
        Query gnomAD for population-specific allele frequencies.
        
        Args:
            chrom: Chromosome (e.g., '17' or 'chr17')
            pos: Position (hg38 coordinates for gnomAD v4!)
            ref: Reference allele
            alt: Alternate allele
            dataset: gnomAD dataset version (default: gnomad_r4)
            
        Returns:
            Dictionary mapping population codes to allele frequencies,
            e.g., {'af_nfe': 0.001, 'af_afr': 0.0002, ...}
            Returns None if variant not found or query fails.
        """
        self._rate_limit()
        
        # Remove 'chr' prefix if present
        chrom = chrom.replace('chr', '')
        
        # GraphQL query for gnomAD v4
        query = """
        query VariantQuery($dataset: DatasetId!, $chrom: String!, $pos: Int!, $ref: String!, $alt: String!) {
            variant(dataset: $dataset, reference_genome: GRCh38, chrom: $chrom, pos: $pos, ref: $ref, alt: $alt) {
                genome {
                    populations {
                        id
                        ac
                        an
                    }
                }
            }
        }
        """
        
        variables = {
            "dataset": dataset,
            "chrom": chrom,
            "pos": pos,
            "ref": ref,
            "alt": alt
        }
        
        try:
            response = requests.post(
                self.API_URL,
                json={'query': query, 'variables': variables},
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            # Check for errors
            if 'errors' in data:
                logger.warning(f"GraphQL error for {chrom}:{pos} {ref}>{alt}: {data['errors']}")
                return None
            
            # Extract variant data
            variant = data.get('data', {}).get('variant')
            if not variant:
                logger.debug(f"Variant not found in gnomAD: {chrom}:{pos} {ref}>{alt}")
                return None
            
            # Extract population frequencies
            populations = variant.get('genome', {}).get('populations', [])
            
            af_dict = {}
            for pop in populations:
                pop_id = pop['id'].lower()
                ac = pop.get('ac', 0)
                an = pop.get('an', 0)
                
                # Calculate allele frequency
                af = ac / an if an > 0 else 0.0
                af_dict[f'af_{pop_id}'] = af
            
            # Ensure all standard populations are present (fill with 0 if missing)
            for pop_code in ['nfe', 'afr', 'eas', 'sas', 'amr']:
                key = f'af_{pop_code}'
                if key not in af_dict:
                    af_dict[key] = 0.0
            
            logger.debug(f"Found AFs for {chrom}:{pos}: {af_dict}")
            return af_dict
            
        except requests.RequestException as e:
            logger.error(f"API request failed for {chrom}:{pos} {ref}>{alt}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error querying {chrom}:{pos} {ref}>{alt}: {e}")
            return None


def test_gnomad_client():
    """Test gnomAD client with known BRCA1 variant."""
    client = GnomADClient()
    
    # Test variant: BRCA1 pathogenic variant
    # rs80357906 (c.5266dupC, p.Gln1756fs)
    # hg38: chr17:43124096 C>CC
    
    result = client.query_variant(
        chrom='17',
        pos=43124096,
        ref='C',
        alt='CC'
    )
    
    if result:
        print("✅ gnomAD query successful")
        print("Population frequencies:")
        for pop, af in sorted(result.items()):
            if af > 0:
                print(f"  {pop}: {af:.6f}")
    else:
        print("❌ gnomAD query failed")
    
    return result


if __name__ == "__main__":
    test_gnomad_client()
