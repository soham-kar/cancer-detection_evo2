"""
Download GSE213862 RNA-seq counts from NCBI
NCBI provides pre-computed counts for RNA-seq datasets
"""

import requests
import gzip
import pandas as pd
from pathlib import Path
from io import BytesIO

OUTPUT_DIR = Path("D:/project/biotech-evo2/data/india")

def download_rnaseq_counts():
    """Download pre-computed RNA-seq counts from NCBI GEO."""
    print("="*60)
    print("DOWNLOADING GSE213862 RNA-seq COUNTS")
    print("="*60)
    
    # URLs for RNA-seq counts (NCBI provides these for processed datasets)
    urls = [
        # NCBI RNA-seq counts format
        "https://www.ncbi.nlm.nih.gov/geo/download/?type=rnaseq_counts&acc=GSE213862&format=file&file=GSE213862_raw_counts_GRCh38.p13_NCBI.tsv.gz",
        # Alternative formats
        "https://www.ncbi.nlm.nih.gov/geo/download/?type=rnaseq_counts&acc=GSE213862&format=file&file=GSE213862_FPKM_GRCh38.p13_NCBI.tsv.gz",
        "https://www.ncbi.nlm.nih.gov/geo/download/?type=rnaseq_counts&acc=GSE213862&format=file&file=GSE213862_TPM_GRCh38.p13_NCBI.tsv.gz",
    ]
    
    for url in urls:
        filename = url.split('file=')[1] if 'file=' in url else url.split('/')[-1]
        print(f"\nTrying: {filename}...")
        
        try:
            response = requests.get(url, timeout=120, allow_redirects=True)
            print(f"  Status: {response.status_code}")
            
            if response.status_code == 200 and len(response.content) > 1000:
                # Save
                output_path = OUTPUT_DIR / filename
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                
                size_kb = len(response.content) / 1024
                print(f"  ✅ Downloaded: {size_kb:.1f} KB")
                
                # Try to read and preview
                try:
                    if filename.endswith('.gz'):
                        df = pd.read_csv(BytesIO(response.content), sep='\t', compression='gzip', nrows=10)
                    else:
                        df = pd.read_csv(BytesIO(response.content), sep='\t', nrows=10)
                    
                    print(f"  Shape preview: {df.shape}")
                    print(f"  Columns: {list(df.columns[:5])}...")
                    
                    return filename
                except Exception as e:
                    print(f"  Could not parse: {e}")
            else:
                print(f"  ❌ Not available or empty")
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    return None


def try_direct_geo_browse():
    """Try the GEO browse page for supplementary files."""
    print("\n" + "="*60)
    print("CHECKING GEO BROWSE PAGE")
    print("="*60)
    
    # Check GEO query page for supplementary links
    geo_url = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE213862"
    print(f"Checking: {geo_url}")
    
    try:
        response = requests.get(geo_url, timeout=30)
        
        # Look for supplementary file links
        import re
        suppl_files = re.findall(r'ftp://[^\s"<>]+GSE213862[^\s"<>]*', response.text)
        
        if suppl_files:
            print("Found supplementary file links:")
            for f in suppl_files[:5]:
                print(f"  - {f}")
        else:
            print("No FTP links found")
        
        # Look for any downloadable content
        download_links = re.findall(r'https://www\.ncbi\.nlm\.nih\.gov/geo/download/[^\s"<>]+', response.text)
        if download_links:
            print("\nFound download links:")
            for link in download_links[:5]:
                print(f"  - {link[:80]}...")
        
    except Exception as e:
        print(f"Error: {e}")


def check_sra_run_table():
    """Check if we can get expression from SRA run info."""
    print("\n" + "="*60)
    print("CHECKING SRA RUN TABLE")
    print("="*60)
    
    # SRA BioProject
    sra_url = "https://www.ncbi.nlm.nih.gov/Traces/study/?acc=PRJNA882884&o=acc_s%3Aa"
    print(f"BioProject: PRJNA882884")
    print(f"SRA Study URL: {sra_url}")
    
    # For RNA-seq, we'd need to run alignment to get counts
    # But many datasets have pre-processed counts available
    
    print("\nNote: If no pre-processed counts are available,")
    print("expression data needs to be generated from raw FASTQ files using:")
    print("  1. Download FASTQ from SRA (sra-tools)")
    print("  2. Align to genome (STAR/HISAT2)")
    print("  3. Count features (featureCounts/HTSeq)")
    print("\nAlternatively, use recount3 for pre-processed counts.")


def main():
    # Try 1: NCBI pre-computed counts
    counts_file = download_rnaseq_counts()
    
    if counts_file:
        print(f"\n✅ SUCCESS: Downloaded {counts_file}")
        
        # Parse and check
        df = pd.read_csv(OUTPUT_DIR / counts_file, sep='\t', compression='gzip')
        print(f"\nExpression matrix: {df.shape}")
        print(f"Genes: {df.shape[0]}")
        print(f"Samples: {df.shape[1]-1}")  # First column is gene ID
        
        # Save as CSV for easier use
        df.to_csv(OUTPUT_DIR / "GSE213862_expression.csv", index=False)
        print(f"\nSaved to: {OUTPUT_DIR / 'GSE213862_expression.csv'}")
        
    else:
        print("\n❌ No pre-computed counts available from NCBI")
        
        # Try 2: Check GEO browse page
        try_direct_geo_browse()
        
        # Try 3: SRA info
        check_sra_run_table()
        
        print("\n" + "="*60)
        print("NEXT STEPS")
        print("="*60)
        print("1. Manually download from GEO:")
        print("   https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE213862")
        print("   Look for 'Supplementary files' or 'Browse data'")
        print("\n2. Or use recount3 (R package) or refine.bio (web)")
        print("   These provide pre-processed counts for many GEO datasets")
        print("\n3. Or process from SRA (requires alignment pipeline)")


if __name__ == "__main__":
    main()
