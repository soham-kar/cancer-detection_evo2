import pandas as pd
import requests
import time
from tqdm import tqdm

# ================= CONFIGURATION =================
INPUT_FILE = "results/brca1_with_gnomad.csv"
OUTPUT_FILE = "results/brca1_ready_for_evo2.csv"
WINDOW_SIZE = 512  # Evo2 needs context. We fetch 512bp on each side.

# UCSC Genome Browser API (hg38)
UCSC_API = "https://api.genome.ucsc.edu/getData/sequence"

def fetch_sequence(chrom, start, end):
    """Fetches DNA sequence from UCSC API (hg38)."""
    try:
        # Ensure chrom has 'chr' prefix
        if not str(chrom).startswith("chr"):
            chrom = f"chr{chrom}"
            
        params = {
            'genome': 'hg38',
            'chrom': chrom,
            'start': start,
            'end': end
        }
        response = requests.get(UCSC_API, params=params, timeout=10)
        if response.status_code == 200:
            return response.json()['dna'].upper()
    except Exception as e:
        pass
    return None

def main():
    print(f"Loading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} variants")
    
    print(f"Fetching sequences for {len(df)} variants (Window: +/- {WINDOW_SIZE}bp)...")
    
    # Create columns for Evo2
    df['seq_context'] = ""
    df['rel_pos'] = 0  # Where the mutation happens in the string
    
    # Rate limiting is crucial for public APIs
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Fetching sequences"):
        # Use hg38 coordinates (we lifted over earlier)
        center_pos = int(row['pos_hg38'])
        
        # UCSC is 0-based, half-open. VCF is 1-based.
        start = center_pos - WINDOW_SIZE - 1
        end = center_pos + WINDOW_SIZE
        
        seq = fetch_sequence(row['chrom'], start, end)
        
        if seq:
            df.at[i, 'seq_context'] = seq
            # The mutation is exactly in the middle
            df.at[i, 'rel_pos'] = WINDOW_SIZE 
        
        # Be nice to the API (rate limiting)
        if i % 10 == 0:
            time.sleep(0.1)

    # Filter out failed fetches
    success_count = len(df[df['seq_context'] != ""])
    print(f"\nSuccessfully fetched {success_count}/{len(df)} sequences ({success_count/len(df)*100:.1f}%)")
    
    # Save
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Ready for Evo2! Saved to {OUTPUT_FILE}")
    print(f"\nFirst 3 variants with sequences:")
    print(df[df['seq_context'] != ""][['chrom', 'pos_hg38', 'ref', 'alt', 'seq_context']].head(3))

if __name__ == "__main__":
    main()
