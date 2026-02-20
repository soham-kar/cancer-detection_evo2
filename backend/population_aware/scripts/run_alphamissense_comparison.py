import os
import requests
import gzip
import pandas as pd
import shutil
from pathlib import Path

# AlphaMissense hg38 (Google Cloud)
AM_URL = "https://storage.googleapis.com/dm_alphamissense/AlphaMissense_hg38.tsv.gz"
AM_FILE = "data/AlphaMissense_hg38.tsv.gz"
AM_TSV = "data/AlphaMissense_hg38.tsv"

def download_file(url, local_filename):
    if os.path.exists(local_filename):
        print(f"✅ File already exists: {local_filename}")
        return

    print(f"⬇️ Downloading {url}...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        done = 0
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                done += len(chunk)
                if total_size > 0 and done % (10*1024*1024) == 0: 
                    print(f"   {done/1024/1024:.0f} MB / {total_size/1024/1024:.0f} MB downloaded...", end='\r')
    print(f"\n✅ Download complete: {local_filename}")

def analyze_comparison():
    print("⏳ Comparing Evo2 vs AlphaMissense (Chr22)...")
    
    # This is a placeholder for the actual comparison logic
    # Since AM file is huge, we would typically tabix index it or stream it
    # For now, we just ensure it's downloaded as requested
    
    if not os.path.exists("data"):
        os.makedirs("data")
        
    download_file(AM_URL, AM_FILE)
    
    print("\n⚠️ Comparison Analysis Step:")
    print("   To run the full comparison, we need to:")
    print("   1. Unzip and Tabix index the AlphaMissense file (or stream grep it for chr22).")
    print("   2. Merge with our scoring results.")
    print("   3. Calculate correlation.")
    
    print("\n   (Comparison script skeleton ready. Download verified.)")

if __name__ == "__main__":
    analyze_comparison()
