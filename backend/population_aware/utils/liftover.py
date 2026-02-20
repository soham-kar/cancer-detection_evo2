"""
Coordinate liftover utilities for converting between genome builds.

The Findlay (2018) dataset uses hg19/GRCh37 coordinates.
gnomAD v4 uses hg38/GRCh38 coordinates.
We need to convert hg19 → hg38 to query gnomAD properly.
"""

from pyliftover import LiftOver
from typing import Optional, Tuple


class CoordinateLiftover:
    """Handle coordinate conversion between genome builds."""
    
    def __init__(self, from_build: str = 'hg19', to_build: str = 'hg38'):
        """
        Initialize liftover chain.
        
        Args:
            from_build: Source genome build (default: hg19)
            to_build: Target genome build (default: hg38)
        """
        print(f"Loading liftover chain: {from_build} → {to_build}")
        self.lo = LiftOver(from_build, to_build)
        self.from_build = from_build
        self.to_build = to_build
    
    def convert_position(
        self, 
        chrom: str, 
        pos: int
    ) -> Optional[int]:
        """
        Convert a genomic position from source to target build.
        
        Args:
            chrom: Chromosome (e.g., '17' or 'chr17')
            pos: Position in source build
            
        Returns:
            Position in target build, or None if liftover failed
        """
        # Ensure chr prefix
        if not chrom.startswith('chr'):
            chrom = f'chr{chrom}'
        
        try:
            result = self.lo.convert_coordinate(chrom, pos)
            
            if result and len(result) > 0:
                # pyliftover returns list of tuples
                # Each tuple has at least (chrom, pos, ...)
                # We only need the position
                converted = result[0]
                if isinstance(converted, tuple) and len(converted) >= 2:
                    new_pos = converted[1]
                    return int(new_pos)
                else:
                    return None
            else:
                return None
        except Exception as e:
            print(f"Liftover failed for {chrom}:{pos} - {e}")
            return None
    
    def convert_variant(
        self, 
        chrom: str, 
        pos: int, 
        ref: str, 
        alt: str
    ) -> Optional[Tuple[int, str, str]]:
        """
        Convert a variant from source to target build.
        
        Note: This only converts position. For indels, alleles might need
        adjustment based on reference genome differences.
        
        Args:
            chrom: Chromosome
            pos: Position in source build
            ref: Reference allele
            alt: Alternate allele
            
        Returns:
            Tuple of (new_pos, ref, alt) or None if failed
        """
        new_pos = self.convert_position(chrom, pos)
        
        if new_pos is None:
            return None
        
        # For SNVs, alleles typically stay the same
        # For complex variants, might need validation
        return (new_pos, ref, alt)


def test_liftover():
    """Test liftover with known BRCA1 variant."""
    lo = CoordinateLiftover('hg19', 'hg38')
    
    # Test variant from Findlay dataset
    # BRCA1 c.5266dupC (rs80357906)
    # hg19: chr17:41276135
    # hg38: chr17:43124096 (expected)
    
    test_chrom = '17'
    test_pos_hg19 = 41276135
    
    result = lo.convert_position(test_chrom, test_pos_hg19)
    
    if result:
        print(f"✅ Liftover successful: chr{test_chrom}:{test_pos_hg19} (hg19) → chr{test_chrom}:{result} (hg38)")
        
        # Check if close to expected (allowing for small differences)
        expected_hg38 = 43124096
        if abs(result - expected_hg38) < 100:
            print(f"✅ Position matches expected: {result} ≈ {expected_hg38}")
        else:
            print(f"⚠️ Position differs from expected: {result} vs {expected_hg38}")
    else:
        print(f"❌ Liftover failed for chr{test_chrom}:{test_pos_hg19}")
    
    return result


if __name__ == "__main__":
    test_liftover()
