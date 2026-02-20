"""
Extract 8kb Genomic Windows for Evo2

Reads CHANGE-seq off-target sites and extracts 8kb sequence
centered on the cleavage site from hg38 genome.
"""
import pandas as pd
from pyfaidx import Fasta
from pathlib import Path
import json

DATA_DIR = Path(__file__).parent.parent / "data"
GENOME_PATH = DATA_DIR / "genome" / "hg38.fa"
INPUT_CSV = DATA_DIR / "change_seq" / "change_seq_evo2_input.csv"
OUTPUT_JSON = DATA_DIR / "change_seq" / "change_seq_8kb_sequences.json"

CONTEXT_SIZE = 8000  # 8kb window
FLANK = CONTEXT_SIZE // 2

def extract_sequences():
    print(f"Loading data from {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    
    print(f"Loading genome from {GENOME_PATH}")
    if not GENOME_PATH.exists():
        print("❌ Genome file not found! Please wait for download to complete.")
        return
    
    genome = Fasta(str(GENOME_PATH))
    print("Genome loaded.")
    
    results = []
    errors = 0
    
    print(f"Extracting {CONTEXT_SIZE}bp windows for {len(df)} sites...")
    
    for i, row in df.iterrows():
        try:
            chrom = str(row['chrom'])
            # Center on the site. CHANGE-seq gives 23bp site.
            # chromStart is 0-based.
            # Center = chromStart + 11
            center = row['chromStart'] + 11
            start = max(0, center - FLANK)
            end = center + FLANK
            
            # Extract sequence
            # pyfaidx handles bounds checks usually
            seq = genome[chrom][start:end].seq.upper()
            
            if len(seq) != CONTEXT_SIZE:
                # Pad if at edge (rare)
                if start == 0:
                    seq = 'N' * (CONTEXT_SIZE - len(seq)) + seq
                else:
                    seq = seq + 'N' * (CONTEXT_SIZE - len(seq))
            
            results.append({
                'site_id': int(row['site_id']),
                'chrom': chrom,
                'center': int(center),
                'reads': float(row['change_seq_reads']) if 'change_seq_reads' in row else 0.0,
                'normalized_reads': float(row['normalized_reads']) if 'normalized_reads' in row else 0.0,
                'sequence_8kb': seq,
                'grna_target_seq': row['target_sequence']
            })
            
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"Error extracting {chrom}:{row['chromStart']}: {e}")
                
        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{len(df)}")
            
    print(f"\nExtraction complete.")
    print(f"Success: {len(results)}")
    print(f"Errors: {errors}")
    
    print(f"Saving to {OUTPUT_JSON}")
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(results, f)

if __name__ == "__main__":
    extract_sequences()
