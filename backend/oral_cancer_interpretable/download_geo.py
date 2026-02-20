"""
Download Tsai et al. 2015 GUIDE-seq data directly from GEO.
"""
import modal
import requests
import gzip
import pandas as pd
import io
from io import StringIO

app = modal.App("download-geo-data")
volume = modal.Volume.from_name("oral-cancer-model")

# Use survival image (has pandas/requests)
# Note: standard debian_slim might need pandas install
image = modal.Image.debian_slim(python_version="3.11").pip_install("pandas", "requests")

@app.function(image=image, gpu="T4", volumes={"/model": volume}, timeout=600)
def download_from_geo():
    """Download GUIDE-seq data directly from GEO"""
    
    # Tsai et al. 2015 - Nature (original GUIDE-seq paper)
    # Supplementary Table 4 contains the off-target data
    geo_urls = [
        "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE65769&format=file&file=GSE65769%5FGuideSeq%5Fofftargets%2Etxt%2Egz",
        # FTP backup sometimes works better
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE65nnn/GSE65769/suppl/GSE65769_GuideSeq_offtargets.txt.gz",
    ]
    
    for url in geo_urls:
        try:
            print(f"Trying: {url[:80]}...")
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            
            # Decompress
            # Check if content is gzip
            try:
                data_bytes = gzip.decompress(response.content)
            except gzip.BadGzipFile:
                # Maybe not gzipped? Or unexpected format.
                print("Warning: Gzip decompression failed. Trying plain text.")
                data_bytes = response.content
            
            # Decode
            data_str = data_bytes.decode('utf-8', errors='replace')
            
            # Read CSV
            # Try sniffing separator
            try:
                df = pd.read_csv(StringIO(data_str), sep='\t')
            except:
                df = pd.read_csv(StringIO(data_str))
            
            print(f"✓ Success! Downloaded {len(df)} rows")
            print(f"Columns: {list(df.columns)}")
            
            # Standardize columns based on inspection or user info
            # User map: gRNA -> guide_seq, offtarget_sequence -> offtarget_seq, ...
            # We must be flexible
            column_map = {
                'gRNA': 'guide_seq',
                'offtarget_sequence': 'offtarget_seq',
                'chromosome': 'chrom',
                'GUIDEseq_reads': 'read_count',
                'GuideSeq_reads': 'read_count',
                'reads': 'read_count',
                'mismatch': 'mismatches'
            }
            
            df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})
            
            # Check requirements
            if 'guide_seq' in df.columns and 'offtarget_seq' in df.columns:
                if 'read_count' in df.columns:
                     df['is_offtarget'] = (df['read_count'] > 0).astype(int)
                else:
                     df['is_offtarget'] = 1
                
                # Save
                output = "/model/guide_seq_tsai2015.csv"
                df.to_csv(output, index=False)
                
                print(f"✓ Formatted data saved to {output}")
                return {
                    "status": "success",
                    "source": "Tsai2015_GEO",
                    "rows": len(df),
                    "positives": int(df['is_offtarget'].sum()),
                    "file": output
                }
            else:
                print(f"✗ Failed to map columns. Found: {df.columns.tolist()}")

        except Exception as e:
            print(f"✗ Failed: {e}")
    
    return {"status": "error", "message": "All GEO downloads failed"}

@app.local_entrypoint()
def main():
    result = download_from_geo.remote()
    print(result)
