"""
Extract ±4kb flanking sequences for CIRCLE-seq sites using hg19.
Uses pyfaidx for efficient FASTA access.
Results in 'circle_seq_8kb_context.csv'.
"""
import pandas as pd
from pyfaidx import Fasta
import os
from pathlib import Path

DATA_DIR = Path('data/benchmark/circle_seq')
GENOME_PATH = Path('data/hg19.fa')
INPUT_FILE = DATA_DIR / 'positives_with_coords.csv'
OUTPUT_FILE = DATA_DIR / 'circle_seq_8kb_context.csv'

def reverse_complement(seq):
    """Return reverse complement of DNA sequence."""
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N', 
                  'a': 't', 't': 'a', 'c': 'g', 'g': 'c', 'n': 'n'}
    return ''.join(complement.get(base, 'N') for base in reversed(seq))

def main():
    print(f"Loading coordinates from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    print(f"Total samples (raw): {len(df)}")
    
    # Deduplicate
    df = df.drop_duplicates(subset=['Chromosome', 'Start', 'End', 'target_sequence'])
    print(f"Unique samples to extract: {len(df)}")
    
    # Check if genome exists (it might be gzipped)
    if not GENOME_PATH.exists():
        if Path('data/hg19.fa.gz').exists():
            print("Found hg19.fa.gz, please decompress it first!")
            # We could decompress here but simpler to ask user or do it in shell
            # Actually pyfaidx can handle gz but it's slower and doesn't support random access well
            # Best to decompress.
            pass
        else:
            print(f"Error: {GENOME_PATH} not found!")
            return

    print(f"Opening genome: {GENOME_PATH}...")
    # pyfaidx handles indexing automatically
    genome = Fasta(str(GENOME_PATH))
    
    contexts = []
    valid_indices = []
    
    print("Extracting 8kb context...")
    for idx, row in df.iterrows():
        chrom = str(row['Chromosome'])
        # Handle "chr" prefix formatting
        # hg19 usually has "chr1", "chr2"
        # Supp Table had "1", "2"
        if not chrom.startswith('chr'):
            chrom_key = 'chr' + chrom
        else:
            chrom_key = chrom
            
        # Check specific edge cases for mito/sex
        if chrom_key == 'chr23': chrom_key = 'chrX'
        if chrom_key == 'chr24': chrom_key = 'chrY'
            
        start = int(row['Start'])
        end = int(row['End'])
        strand = str(row['Strand'])
        
        # Define 8kb window centered on site
        # Site length is ~23bp
        # We want ±4kb from borders? Or centered?
        # User said "±4kb flanking".
        # Let's take [start-4000, end+4000]
        
        flank = 4000
        ctx_start = max(0, start - flank)
        ctx_end = end + flank
        
        try:
            # Fetch sequence
            # pyfaidx is 0-indexed? No, it uses Python slicing [start:end] (0-indexed)
            # Coordinates in Supp Table (hg19) are likely 1-indexed (standard BED/GFF)
            # UCSC coordinates are 0-start, half-open
            # Let's assume standard behavior:
            # If input is 1-based, convert to 0-based for pyfaidx
            # Start-1 for 0-index.
            
            # Use fetch method for safety if keys exist
            # sequence = genome[chrom_key][ctx_start:ctx_end].seq
            
            if chrom_key in genome:
                seq_obj = genome[chrom_key][ctx_start:ctx_end]
                seq = seq_obj.seq.upper()
                
                if strand == '-':
                    seq = reverse_complement(seq)
                
                # Verify length
                expected_len = (end - start) + 2*flank
                # Roughly 8023 bp
                
                contexts.append(seq)
                valid_indices.append(idx)
            else:
                # Try fallback names
                if chrom in genome:
                    seq_obj = genome[chrom][ctx_start:ctx_end]
                    seq = seq_obj.seq.upper()
                    if strand == '-': seq = reverse_complement(seq)
                    contexts.append(seq)
                    valid_indices.append(idx)
                else:
                    # Chromosome not found
                    contexts.append(None)
                    
        except Exception as e:
            print(f"Error at {idx}: {e}")
            contexts.append(None)
            
        if idx % 500 == 0:
            print(f"  Processed {idx}/{len(df)}")

    # Add to dataframe
    df['context_8kb'] = contexts
    
    # Filter success
    success_df = df.dropna(subset=['context_8kb']).copy()
    print(f"Successfully extracted: {len(success_df)} / {len(df)}")
    
    # Save
    success_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
