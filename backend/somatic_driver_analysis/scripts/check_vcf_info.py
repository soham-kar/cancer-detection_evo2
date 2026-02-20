
import gzip
from pathlib import Path

def check_vcf():
    path = Path("d:/project/biotech-evo2/backend/population_aware/data/india_project/20201028_CCDG_14151_B01_GRM_WGS_2020-08-05_chr22.recalibrated_variants.vcf.gz")
    size_gb = path.stat().st_size / (1024**3)
    print(f"File: {path.name}")
    print(f"Size: {size_gb:.2f} GB")
    
    count = 0
    max_count = 5000
    last_pos = 0
    start_pos = 0
    
    with gzip.open(path, 'rt') as f:
        for line in f:
            if line.startswith('##'):
                continue
            if line.startswith('#CHROM'):
                print(f"Header: {line.strip()[:100]}...") # Check samples
                continue
            
            # Data
            params = line.split('\t')
            pos = int(params[1])
            if count == 0:
                start_pos = pos
            last_pos = pos
            count += 1
            if count >= max_count:
                break
                
    print(f"Inspected first {max_count} variants.")
    print(f"Position range: {start_pos} - {last_pos} (Delta: {last_pos - start_pos})")
    
    # Estimate
    # Chr22 is ~50M bp.
    # If standard gVCF, it covers most bases.
    # If variant only...
    
if __name__ == "__main__":
    check_vcf()
