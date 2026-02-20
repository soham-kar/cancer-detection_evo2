"""
Download real GUIDE-seq/CIRCLE-seq data from DeepCRISPR repository.
Source: https://github.com/zztsail/DeepCRISPR
"""
import modal
import pandas as pd
import requests
from io import StringIO
import io

app = modal.App("setup-deepcrispr-data")
volume = modal.Volume.from_name("oral-cancer-model")

# Use survival image (has pandas/requests)
image = modal.Image.debian_slim(python_version="3.11").pip_install("pandas", "requests")

@app.function(image=image, gpu="T4", volumes={"/model": volume}, timeout=600)
def download_deepcrispr_data():
    """Download real GUIDE-seq data from DeepCRISPR repository"""
    print("Downloading DeepCRISPR validation data...")
    
    # Working URLs from DeepCRISPR GitHub
    urls = [
        "https://raw.githubusercontent.com/zztsail/DeepCRISPR/master/data/offtarget/GuideSeq.csv",
        "https://raw.githubusercontent.com/zztsail/DeepCRISPR/master/data/offtarget/CIRCLESeq.csv",
    ]
    
    all_data = []
    for url in urls:
        try:
            print(f"Trying: {url}")
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            
            # Parse CSV
            # DeepCRISPR CSVs are usually standard comma-separated
            content = response.content.decode('utf-8')
            df = pd.read_csv(io.StringIO(content))
            print(f"✓ Downloaded {len(df)} rows")
            print(f"Columns: {list(df.columns)}")
            
            # Standardize column names
            # Likely columns: gRNA, otSeq, chrom, chromStart, mismatchCount, guideSeqHits, label
            column_map = {
                'grna': 'guide_seq',
                'gRNA': 'guide_seq',
                'otSeq': 'offtarget_seq',
                'otseq': 'offtarget_seq',
                'chrom': 'chromosome',
                'chromStart': 'position',
                'mismatchCount': 'mismatches',
                'guideSeqHits': 'read_count',
                'label': 'is_offtarget',
                'Label': 'is_offtarget'
            }
            
            df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})
            
            # Ensure is_offtarget exists
            if 'is_offtarget' not in df.columns:
                if 'read_count' in df.columns:
                    df['is_offtarget'] = (df['read_count'] > 0).astype(int)
                else:
                    print("Warning: No label or read_count found. Assuming all are positives?")
                    # Inspect columns
                    pass
            
            all_data.append(df)
            
        except Exception as e:
            print(f"✗ Failed: {e}")
    
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        # Filter for required columns
        req_cols = ['guide_seq', 'offtarget_seq', 'is_offtarget']
        if all(c in combined.columns for c in req_cols):
             output_path = "/model/guide_seq_real_deepcrispr.csv"
             combined.to_csv(output_path, index=False)
             
             print(f"\n✓ Saved {len(combined)} total sites to {output_path}")
             print(f"Positive off-targets: {combined['is_offtarget'].sum()}")
             print(f"Negative sites: {len(combined) - combined['is_offtarget'].sum()}")
             
             # Print mismatch distribution
             if 'mismatches' in combined.columns:
                 print("\nMismatch distribution:")
                 print(combined['mismatches'].value_counts().sort_index())
             
             return output_path
        else:
            print(f"Missing columns. Found: {combined.columns.tolist()}")
            return None
    else:
        print("All downloads failed")
        return None

@app.local_entrypoint()
def main():
    download_deepcrispr_data.remote()
