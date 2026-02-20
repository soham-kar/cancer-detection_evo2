"""
Prepare Evo2 input with 8kb genomic context.
Constructs 'target_full' and 'offtarget_full' columns for scoring.
"""
import pandas as pd
from pathlib import Path

DATA_DIR = Path('data/benchmark/circle_seq')
INPUT_FILE = DATA_DIR / 'circle_seq_8kb_context.csv'
OUTPUT_FILE = DATA_DIR / 'circle_seq_8kb_prepared.csv'

def clean_seq(seq):
    return str(seq).upper()

def main():
    print(f"Loading {INPUT_FILE}...")
    if not INPUT_FILE.exists():
        print(f"Error: {INPUT_FILE} missing.")
        return

    df = pd.read_csv(INPUT_FILE)
    print(f"Samples: {len(df)}")
    
    prepared_rows = []
    found_count = 0
    default_count = 0
    
    print("Constructing full sequences...")
    for idx, row in df.iterrows():
        context = clean_seq(row['context_8kb'])
        grna = clean_seq(row['grna_sequence'])
        target = clean_seq(row['target_sequence']) # Off-target site (20bp)
        
        center_idx = 4000
        search_window = 100 # Increased window
        
        # Search for offtarget sequence in context
        found_idx = -1
        
        # Try exact match near center
        sub_context = context[center_idx-search_window : center_idx+search_window + 50]
        match = sub_context.find(target)
        
        if match != -1:
            found_idx = (center_idx - search_window) + match
            found_count += 1
        else:
            found_idx = 4000
            default_count += 1

        # Construct sequences
        offtarget_full = context
        # target_full: context with 20bp off-target replaced by gRNA
        target_full = context[:found_idx] + grna + context[found_idx+len(target):]
        
        # --- Validation & Metadata ---
        # 1. Verify lengths match
        if len(target_full) != len(offtarget_full):
            print(f"Warning: Length mismatch at {idx}. T:{len(target_full)} vs O:{len(offtarget_full)}")
            # Skip or fix? If bulges, this happens. But balanced dataset should be cleaned?
            # For now, just track it.
        
        # 2. Verify only 20bp region differs (or length of target)
        # Actually diff count should be exactly n_mismatches
        # mismatch_positions relative to gRNA (0-19)
        mismatches = []
        for i in range(min(len(grna), len(target))):
            if grna[i] != target[i]:
                mismatches.append(i)
                
        # 3. Add to row
        row_dict = row.to_dict()
        row_dict['target_full'] = target_full
        row_dict['offtarget_full'] = offtarget_full
        row_dict['mismatch_positions'] = str(mismatches) # Store as string representation of list
        row_dict['n_mismatches'] = len(mismatches)
        
        prepared_rows.append(row_dict)
        
    print(f"Match Stats: Found exact site={found_count}, Defaulted to center={default_count}")
        
    # Save 8kb version
    df_8kb = pd.DataFrame(prepared_rows)
    output_8kb = DATA_DIR / 'input_8kb.csv'
    df_8kb.to_csv(output_8kb, index=False)
    print(f"Saved {len(df_8kb)} samples to {output_8kb} (8kb context)")
    
    # Save 20bp version (for fair comparison on same subset)
    # We strip the '_full' columns so scorer uses defaults (20bp)
    df_20bp = df_8kb.drop(columns=['target_full', 'offtarget_full'])
    output_20bp = DATA_DIR / 'input_20bp.csv'
    df_20bp.to_csv(output_20bp, index=False)
    print(f"Saved {len(df_20bp)} samples to {output_20bp} (20bp baseline)")

if __name__ == "__main__":
    main()
