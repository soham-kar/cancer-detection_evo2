"""
Decompress hg19.fa.gz to hg19.fa
"""
import gzip
import shutil
from pathlib import Path
import time

INPUT_FILE = Path('data/hg19.fa.gz')
OUTPUT_FILE = Path('data/hg19.fa')

def main():
    if not INPUT_FILE.exists():
        print(f"Error: {INPUT_FILE} does not exist yet.")
        return

    print(f"Decompressing {INPUT_FILE} to {OUTPUT_FILE}...")
    start_time = time.time()
    
    with gzip.open(INPUT_FILE, 'rb') as f_in:
        with open(OUTPUT_FILE, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
            
    elapsed = time.time() - start_time
    print(f"Done! Decompression took {elapsed:.1f} seconds.")
    print(f"Output size: {OUTPUT_FILE.stat().st_size / (1024**3):.2f} GB")

if __name__ == "__main__":
    main()
