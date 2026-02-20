import pandas as pd
from pathlib import Path

def inspect_input():
    print("--- Inspecting Input (1000bp) ---")
    df = pd.read_csv('data/benchmark/circle_seq/input_1000bp.csv')
    
    diff_count = 0
    total = len(df)
    
    for idx, row in df.iterrows():
        t = row['target_full']
        o = row['offtarget_full']
        
        if t != o:
            diff_count += 1
        
        if idx == 0:
            print(f"Sample 0 lengths: T={len(t)}, O={len(o)}")
            # Find differences
            diffs = [i for i in range(min(len(t), len(o))) if t[i] != o[i]]
            print(f"Differences at indices: {diffs}")
            print(f"Number of diffs: {len(diffs)}")
            
    print(f"Rows where target != offtarget: {diff_count}/{total}")

def inspect_results():
    print("\n--- Inspecting Results (Pilot 1kb) ---")
    try:
        df = pd.read_csv('results/modal_evo2/pilot_1kb.csv')
        print(df.columns.tolist())
        print(df[['mismatches', 'weighted_delta_ll']].head())
        print(f"Mean Delta: {df['weighted_delta_ll'].mean()}")
    except Exception as e:
        print(f"Could not read results: {e}")

if __name__ == "__main__":
    inspect_input()
    inspect_results()
