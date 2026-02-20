"""
Crop 8kb input to smaller window (e.g. 1kb) to test for model truncation.
"""
import pandas as pd
from pathlib import Path

DATA_DIR = Path('data/benchmark/circle_seq')
INPUT_FILE = DATA_DIR / 'input_8kb.csv'

def crop_center(seq, length=1000):
    if len(seq) <= length:
        return seq
    center = len(seq) // 2
    start = center - (length // 2)
    return seq[start : start + length]

def main():
    print(f"Loading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    
    # Define new window size
    # 1024 bp is safe for almost all models
    NEW_LEN = 1000 
    
    print(f"Cropping sequences to center {NEW_LEN} bp...")
    
    # We need to crop 'target_full' and 'offtarget_full'
    # Since site is at center (index 4000), cropping center 1000 keeps the site.
    
    df['target_full'] = df['target_full'].apply(lambda x: crop_center(str(x), NEW_LEN))
    df['offtarget_full'] = df['offtarget_full'].apply(lambda x: crop_center(str(x), NEW_LEN))
    
    OUTPUT_FILE = DATA_DIR / f'input_{NEW_LEN}bp.csv'
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
