"""
Download GSE213862: Indian Oral Cancer Dataset
Source: National Institute of Biomedical Genomics, Kalyani, India
46 patients with gingivo-buccal oral squamous cell carcinoma (OSCC-GB)
RNA-sequencing data with survival endpoints
"""

import os
import requests
import gzip
import pandas as pd
import numpy as np
from pathlib import Path
from io import StringIO

# Output directory
OUTPUT_DIR = Path("D:/project/biotech-evo2/data/india")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE213nnn/GSE213862"


def download_geo_metadata():
    """Download and parse GEO metadata for GSE213862."""
    print("="*60)
    print("DOWNLOADING GSE213862 (Indian Oral Cancer)")
    print("="*60)
    
    # Fetch the SOFT file (contains all metadata)
    soft_url = f"{BASE_URL}/soft/GSE213862_family.soft.gz"
    print(f"\nFetching metadata from: {soft_url}")
    
    try:
        response = requests.get(soft_url, timeout=60)
        response.raise_for_status()
        
        # Decompress
        import gzip
        soft_text = gzip.decompress(response.content).decode('utf-8')
        
        # Save raw SOFT file
        with open(OUTPUT_DIR / "GSE213862_family.soft", "w", encoding='utf-8') as f:
            f.write(soft_text)
        print(f"✅ Saved SOFT file")
        
        return soft_text
    except Exception as e:
        print(f"❌ Error downloading SOFT: {e}")
        return None


def parse_soft_file(soft_text):
    """Parse SOFT file to extract sample information."""
    print("\nParsing sample metadata...")
    
    samples = []
    current_sample = {}
    
    for line in soft_text.split('\n'):
        if line.startswith('^SAMPLE'):
            if current_sample:
                samples.append(current_sample)
            current_sample = {'gsm_id': line.split('=')[1].strip()}
        
        elif line.startswith('!Sample_'):
            key = line.split('=')[0].replace('!Sample_', '').strip()
            value = line.split('=')[1].strip() if '=' in line else ''
            
            if key in current_sample:
                if isinstance(current_sample[key], list):
                    current_sample[key].append(value)
                else:
                    current_sample[key] = [current_sample[key], value]
            else:
                current_sample[key] = value
    
    if current_sample:
        samples.append(current_sample)
    
    print(f"Found {len(samples)} samples")
    
    # Convert to DataFrame
    df = pd.DataFrame(samples)
    
    return df


def extract_clinical_data(samples_df):
    """Extract clinical variables from sample characteristics."""
    print("\nExtracting clinical data...")
    
    clinical_rows = []
    
    for _, row in samples_df.iterrows():
        gsm_id = row.get('gsm_id', '')
        title = row.get('title', '')
        source = row.get('source_name_ch1', '')
        
        # Parse characteristics
        chars = row.get('characteristics_ch1', [])
        if isinstance(chars, str):
            chars = [chars]
        
        clinical = {
            'sample_id': gsm_id,
            'title': title,
            'source': source
        }
        
        # Parse "key: value" format from characteristics
        for char in chars:
            if isinstance(char, str) and ':' in char:
                key, value = char.split(':', 1)
                key = key.strip().lower().replace(' ', '_')
                value = value.strip()
                clinical[key] = value
        
        clinical_rows.append(clinical)
    
    clinical_df = pd.DataFrame(clinical_rows)
    
    print(f"Clinical columns: {list(clinical_df.columns)}")
    print(f"\nFirst few rows:")
    print(clinical_df.head())
    
    return clinical_df


def download_expression_data():
    """Download supplementary expression matrix."""
    print("\nDownloading expression data...")
    
    # Check supplementary files
    suppl_url = f"{BASE_URL}/suppl/"
    print(f"Checking: {suppl_url}")
    
    try:
        response = requests.get(suppl_url, timeout=30)
        print(f"Supplementary page status: {response.status_code}")
        
        # Try common file patterns for RNA-seq
        possible_files = [
            "GSE213862_raw_counts.txt.gz",
            "GSE213862_counts.txt.gz",
            "GSE213862_TPM.txt.gz",
            "GSE213862_normalized_counts.txt.gz",
            "GSE213862_expression.txt.gz",
            "GSE213862_gene_expression.txt.gz",
            "GSE213862_processed_data.txt.gz"
        ]
        
        for filename in possible_files:
            file_url = f"{suppl_url}{filename}"
            print(f"  Trying: {filename}...", end=" ")
            
            try:
                resp = requests.head(file_url, timeout=10)
                if resp.status_code == 200:
                    print("✅ Found!")
                    
                    # Download the file
                    print(f"  Downloading {filename}...")
                    file_resp = requests.get(file_url, timeout=120)
                    
                    # Save
                    with open(OUTPUT_DIR / filename, 'wb') as f:
                        f.write(file_resp.content)
                    
                    print(f"  ✅ Saved to {OUTPUT_DIR / filename}")
                    return filename
                else:
                    print("❌")
            except:
                print("❌")
        
        print("\nNo standard expression file found. Checking via GEO2R approach...")
        
        # Alternative: Try to fetch processed data from GEO2R endpoint
        # Or look at the series_matrix file
        matrix_url = f"{BASE_URL}/matrix/GSE213862_series_matrix.txt.gz"
        print(f"\nTrying series matrix: {matrix_url}")
        
        try:
            resp = requests.get(matrix_url, timeout=60)
            if resp.status_code == 200:
                with open(OUTPUT_DIR / "GSE213862_series_matrix.txt.gz", 'wb') as f:
                    f.write(resp.content)
                print("✅ Downloaded series matrix")
                return "GSE213862_series_matrix.txt.gz"
        except Exception as e:
            print(f"❌ Could not download series matrix: {e}")
        
        return None
        
    except Exception as e:
        print(f"❌ Error checking supplementary files: {e}")
        return None


def check_for_survival_data(clinical_df):
    """Check if survival endpoints are available."""
    print("\n" + "="*60)
    print("CHECKING FOR SURVIVAL DATA")
    print("="*60)
    
    # Look for survival-related columns
    survival_keywords = ['survival', 'os', 'pfs', 'dfs', 'time', 'status', 'event', 
                         'death', 'recurrence', 'alive', 'dead', 'follow']
    
    survival_cols = []
    for col in clinical_df.columns:
        if any(kw in col.lower() for kw in survival_keywords):
            survival_cols.append(col)
    
    if survival_cols:
        print(f"✅ Found survival-related columns: {survival_cols}")
        for col in survival_cols:
            print(f"\n  {col}:")
            print(f"    Values: {clinical_df[col].value_counts().to_dict()}")
    else:
        print("⚠️ No obvious survival columns found")
        print("\nAll columns:")
        for col in clinical_df.columns:
            print(f"  - {col}: {clinical_df[col].nunique()} unique values")
    
    return survival_cols


def main():
    # Step 1: Download metadata
    soft_text = download_geo_metadata()
    if not soft_text:
        print("Failed to download metadata. Exiting.")
        return
    
    # Step 2: Parse samples
    samples_df = parse_soft_file(soft_text)
    
    # Step 3: Extract clinical data
    clinical_df = extract_clinical_data(samples_df)
    
    # Save clinical data
    clinical_df.to_csv(OUTPUT_DIR / "GSE213862_clinical.csv", index=False)
    print(f"\n✅ Saved clinical data to {OUTPUT_DIR / 'GSE213862_clinical.csv'}")
    
    # Step 4: Check for survival data
    survival_cols = check_for_survival_data(clinical_df)
    
    # Step 5: Download expression data
    expr_file = download_expression_data()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Samples: {len(clinical_df)}")
    print(f"Clinical file: {OUTPUT_DIR / 'GSE213862_clinical.csv'}")
    if expr_file:
        print(f"Expression file: {OUTPUT_DIR / expr_file}")
    else:
        print("Expression file: Not auto-downloaded (may need manual download from GEO)")
    
    if survival_cols:
        print(f"Survival data: ✅ Available ({', '.join(survival_cols)})")
    else:
        print("Survival data: ⚠️ Not found in metadata (check manually)")
    
    print("\n📌 Next steps:")
    print("1. Check if expression data was downloaded")
    print("2. If not, manually download from GEO: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE213862")
    print("3. Parse survival endpoints from clinical columns")
    print("4. Align genes with TCGA and train India-specific model")


if __name__ == "__main__":
    main()
