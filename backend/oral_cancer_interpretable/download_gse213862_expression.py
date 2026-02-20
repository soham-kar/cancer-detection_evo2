"""
Download GSE213862 Expression Data
Try multiple approaches to get the RNA-seq expression matrix
"""

import requests
import gzip
import pandas as pd
import numpy as np
from pathlib import Path
from io import BytesIO
import re

OUTPUT_DIR = Path("D:/project/biotech-evo2/data/india")

def list_supplementary_files():
    """List all supplementary files for GSE213862."""
    print("="*60)
    print("LISTING SUPPLEMENTARY FILES")
    print("="*60)
    
    base_url = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE213nnn/GSE213862/suppl/"
    
    try:
        response = requests.get(base_url, timeout=30)
        print(f"Status: {response.status_code}")
        
        # Parse the directory listing
        content = response.text
        
        # Find all file links
        files = re.findall(r'<a href="([^"]+)">', content)
        files = [f for f in files if not f.startswith('?') and not f == '..']
        
        print(f"\nFound {len(files)} files:")
        for f in files:
            print(f"  - {f}")
        
        return files
    except Exception as e:
        print(f"Error listing files: {e}")
        return []


def download_file(filename):
    """Download a specific supplementary file."""
    base_url = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE213nnn/GSE213862/suppl/"
    url = base_url + filename
    
    print(f"\nDownloading: {filename}...")
    
    try:
        response = requests.get(url, timeout=120)
        if response.status_code == 200:
            # Save
            output_path = OUTPUT_DIR / filename
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            size_kb = len(response.content) / 1024
            print(f"✅ Downloaded: {size_kb:.1f} KB")
            
            # If it's a gzipped text file, peek at contents
            if filename.endswith('.gz'):
                try:
                    with gzip.open(BytesIO(response.content), 'rt') as f:
                        lines = [f.readline() for _ in range(10)]
                    print(f"\nFirst 10 lines of {filename}:")
                    for i, line in enumerate(lines):
                        truncated = line.strip()[:100]
                        print(f"  {i+1}: {truncated}...")
                except Exception as e:
                    print(f"Could not read gzip content: {e}")
            
            return output_path
        else:
            print(f"❌ HTTP {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def try_gsm_expression():
    """Try to get expression from individual GSM files (slower but more reliable)."""
    print("\n" + "="*60)
    print("TRYING INDIVIDUAL SAMPLE EXPRESSION")
    print("="*60)
    
    # Load clinical to get sample IDs
    clinical = pd.read_csv(OUTPUT_DIR / "GSE213862_clinical.csv")
    sample_ids = clinical['sample_id'].tolist()
    
    print(f"Samples: {len(sample_ids)}")
    
    # Try first sample
    gsm_id = sample_ids[0]  # e.g., GSM6595370
    
    # GEO stores supplementary files per sample
    gsm_url = f"https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6595nnn/{gsm_id}/suppl/"
    
    print(f"Checking individual sample: {gsm_url}")
    
    try:
        response = requests.get(gsm_url, timeout=30)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            files = re.findall(r'<a href="([^"]+)">', response.text)
            files = [f for f in files if not f.startswith('?') and not f == '..']
            print(f"Sample {gsm_id} has {len(files)} supplementary files:")
            for f in files:
                print(f"  - {f}")
    except Exception as e:
        print(f"Error: {e}")


def parse_series_matrix():
    """Try to parse the series matrix file for any expression data."""
    print("\n" + "="*60)
    print("PARSING SERIES MATRIX")
    print("="*60)
    
    matrix_path = OUTPUT_DIR / "GSE213862_series_matrix.txt.gz"
    
    if not matrix_path.exists():
        print(f"File not found: {matrix_path}")
        return None
    
    try:
        with gzip.open(matrix_path, 'rt') as f:
            content = f.read()
        
        lines = content.split('\n')
        print(f"Total lines: {len(lines)}")
        
        # Find data section
        for i, line in enumerate(lines):
            if line.startswith('!series_matrix_table_begin'):
                print(f"Data starts at line {i}")
                data_start = i + 1
                break
            if i < 50:  # Show first 50 lines
                print(f"  {i}: {line[:80]}...")
        
        # Check if there's actual expression data
        if '!series_matrix_table_begin' in content:
            print("\n✅ Series matrix contains expression data!")
            # Parse the matrix
            data_lines = []
            in_data = False
            for line in lines:
                if line.startswith('!series_matrix_table_begin'):
                    in_data = True
                    continue
                if line.startswith('!series_matrix_table_end'):
                    break
                if in_data and line.strip():
                    data_lines.append(line)
            
            print(f"Data rows: {len(data_lines)}")
            if data_lines:
                # First line is header (sample IDs)
                header = data_lines[0].split('\t')
                print(f"Samples in matrix: {len(header)-1}")
                print(f"Sample IDs: {header[1:6]}...")
                
                # Parse to DataFrame
                from io import StringIO
                data_text = '\n'.join(data_lines)
                df = pd.read_csv(StringIO(data_text), sep='\t', index_col=0)
                
                print(f"Expression matrix: {df.shape}")
                print(f"Genes: {list(df.index[:10])}")
                
                # Save
                df.to_csv(OUTPUT_DIR / "GSE213862_expression.csv")
                print(f"\n✅ Saved expression matrix to GSE213862_expression.csv")
                return df
        else:
            print("❌ No expression data in series matrix")
            return None
            
    except Exception as e:
        print(f"Error parsing: {e}")
        import traceback
        traceback.print_exc()
        return None


def check_rawexp():
    """Check for RAW.tar file which contains individual sample expression."""
    print("\n" + "="*60)
    print("CHECKING FOR RAW DATA ARCHIVE")
    print("="*60)
    
    tar_url = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE213nnn/GSE213862/suppl/GSE213862_RAW.tar"
    
    print(f"Checking: GSE213862_RAW.tar")
    
    try:
        # HEAD request to check if exists
        response = requests.head(tar_url, timeout=30)
        if response.status_code == 200:
            size_mb = int(response.headers.get('content-length', 0)) / (1024*1024)
            print(f"✅ Found RAW.tar ({size_mb:.1f} MB)")
            print("This file contains all sample expression data")
            print("Would you like to download it? (It may be large)")
            return True
        else:
            print(f"❌ Not found: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    # Step 1: List supplementary files
    files = list_supplementary_files()
    
    # Step 2: Download any expression-like files
    expr_files = [f for f in files if any(kw in f.lower() for kw in ['count', 'tpm', 'expression', 'matrix', 'fpkm'])]
    
    if expr_files:
        print(f"\nFound expression files: {expr_files}")
        for f in expr_files:
            download_file(f)
    else:
        print("\nNo obvious expression files in supplementary list")
    
    # Step 3: Try parsing series matrix
    expr_df = parse_series_matrix()
    
    if expr_df is None:
        # Step 4: Check for RAW.tar
        has_raw = check_rawexp()
        
        if has_raw:
            print("\n📌 Next step: Download GSE213862_RAW.tar")
            print("   This contains individual sample CEL/txt files")
        else:
            # Step 5: Try individual sample files  
            try_gsm_expression()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Output directory: {OUTPUT_DIR}")
    print("\nFiles:")
    for f in OUTPUT_DIR.iterdir():
        print(f"  - {f.name} ({f.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
