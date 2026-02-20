"""
Map coordinates from Supplementary Table 2 to CIRCLE-seq positives.
 Robustly identifies columns by content.
"""
import pandas as pd
import numpy as np
from pathlib import Path

# Paths
DATA_DIR = Path('data/benchmark/circle_seq')
SUPP_FILE = DATA_DIR / 'Supplementary_Table_2.xlsx'
BALANCED_FILE = DATA_DIR.parent.parent / 'circle_seq_balanced.csv'
OUTPUT_FILE = DATA_DIR / 'positives_with_coords.csv'

def clean_seq(seq):
    return str(seq).upper().strip().replace('-', '').replace('_', '').split(' ')[0][:20]

def find_matching_column(df_source, target_set, col_candidates):
    """Find a column in df_source that has overlap with target_set."""
    best_col = None
    best_overlap = 0
    
    for col in df_source.columns:
        # Check if column name looks relevant
        if not any(cand in str(col).lower() for cand in col_candidates):
            continue
            
        # Check content overlap
        try:
            col_values = df_source[col].apply(clean_seq)
            overlap = len(set(col_values).intersection(target_set))
            if overlap > best_overlap:
                best_overlap = overlap
                best_col = col
        except:
            continue
            
    return best_col, best_overlap

def main():
    print("Loading data...")
    # Load Supp Table 2
    supp_df = pd.read_excel(SUPP_FILE)
    print(f"Supp Table 2 rows: {len(supp_df)}")
    print(f"Supp Columns: {supp_df.columns.tolist()}")
    
    # Load our balanced sample
    balanced_df = pd.read_csv(BALANCED_FILE)
    positives_df = balanced_df[balanced_df['is_validated'] == 1].copy()
    print(f"Balanced positives: {len(positives_df)}")
    
    # Prepare sets for matching
    bal_offtargets = set(positives_df['target_sequence'].apply(clean_seq))
    bal_guides = set(positives_df['grna_sequence'].apply(clean_seq))
    
    # Check if 'grna_name' exists
    if 'grna_name' in positives_df.columns:
        bal_names = set(positives_df['grna_name'].astype(str))
    else:
        bal_names = set()
    
    print("\n--- Identifying Columns ---")
    
    # Identify Off-Target Column
    ot_col, ot_overlap = find_matching_column(supp_df, bal_offtargets, ['seq', 'target', 'off'])
    print(f"Best Off-Target Column: {ot_col} (Overlap: {ot_overlap})")
    
    # Identify gRNA Column (Sequence)
    grna_col, grna_overlap = find_matching_column(supp_df, bal_guides, ['target', 'grna', 'guide', 'seq'])
    print(f"Best gRNA Sequence Column: {grna_col} (Overlap: {grna_overlap})")
    
    # Identify Name Column
    if len(bal_names) > 0:
        name_col, name_overlap = find_matching_column(supp_df, bal_names, ['name', 'cell', 'type', 'id', 'target', 'site'])
        print(f"Best Name Column: {name_col} (Overlap: {name_overlap})")
    else:
        print("Skipping Name matching (column missing in balanced df)")
    
    if ot_overlap == 0 and grna_overlap == 0:
        print("❌ CRITICAL FAILURE: Could not match any sequences!")
        return

    # Merge Strategy
    match_col_supp = ot_col if ot_overlap > 0 else grna_col
    match_col_bal = 'target_sequence' if ot_overlap > 0 else 'grna_sequence'
    
    print(f"\nMerging on {match_col_supp} (Supp) == {match_col_bal} (Balanced)...")
    
    supp_df['match_key'] = supp_df[match_col_supp].apply(clean_seq)
    positives_df['match_key'] = positives_df[match_col_bal].apply(clean_seq)
    
    # Select columns to keep
    potential_cols = ['Chromosome', 'Start', 'End', 'Strand', 'Read', 'TotalReads']
    keep_cols = [c for c in potential_cols if c in supp_df.columns]
    
    # Add whatever other interesting columns
    if 'Mismatches' in supp_df.columns: keep_cols.append('Mismatches')
    
    merged = pd.merge(
        positives_df,
        supp_df[keep_cols + ['match_key']],
        on='match_key',
        how='inner' # Keep only matches
    )
    
    print(f"\n✅ Successfully merged {len(merged)} / {len(positives_df)} positives")
    
    if len(merged) > 0:
        merged.to_csv(OUTPUT_FILE, index=False)
        print(f"Saved to {OUTPUT_FILE}")
    else:
        print("Merged dataframe is empty!")

if __name__ == "__main__":
    main()
