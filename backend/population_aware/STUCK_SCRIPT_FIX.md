# URGENT: day1_add_gnomad.py is STUCK!

## Problem
The script is stuck at 0% because reading a 25GB VCF file sequentially with gzip is **extremely slow**.

Without pysam/cyvcf2, we can't use the index files (.tbi) for fast random access.
It's scanning millions of variants linearly to find each position.

## Solution Options

### Option 1: Install pysam via conda (RECOMMENDED)

```bash
# Add required channels
conda config --add channels conda-forge
conda config --add channels bioconda

# Install pysam
conda install pysam -y
```

Then revert `gnomad_client_local.py` to use pysam (I changed it to pure Python which is too slow).

### Option 2: Extract BRCA1 region only (Quick workaround)

Create a small VCF with only BRCA1 variants:

```powershell
cd D:\project\biotech-evo2\backend\population_aware\data

# Install 7-Zip if not available
# Download from: https://www.7-zip.org/

# Extract BRCA1 region (chr17:41196312-41277500)
# This creates a ~100MB uncompressed VCF instead of 25GB
7z e -so gnomad.genomes.v4.0.sites.chr17.vcf.bgz | findstr /R "^17.*41[12][0-9][0-9][0-9][0-9][0-9]" > brca1_region.vcf
```

Then modify the script to use this smaller file.

### Option 3: Use WSL2 (Best long-term)

Install WSL2 (Linux on Windows):
```bash
wsl --install
```

Then in WSL:
```bash
pip install pysam  # Works perfectly in Linux
```

---

## Immediate Action Required

1. **Press Ctrl+C** to stop the stuck script
2. Choose one of the options above
3. I'll update the code accordingly

Which option do you want to try?
