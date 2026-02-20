import pandas as pd
import argparse
import os

def extract_rescue_cases(output_file):
    input_path = "results/indian_chr22_evo2_scores_PARTIAL.csv"
    if not os.path.exists(input_path):
        input_path = "results/indian_chr22_evo2_scores.csv"
    
    if not os.path.exists(input_path):
        print(f"❌ Input file not found.")
        return

    print(f"⏳ Loading {input_path}...")
    df = pd.read_csv(input_path)
    
    # Rescue Logic: 
    # Evo2 < -2.0 (Pathogenic prediction)
    # AF_Indian > 0.01 (Common locally)
    # AF_Global < 0.01 (Rare globally)
    
    print("🔍 Filtering for Rescue Cases...")
    rescues = df[
        (df['evo2_score'] < -2.0) & 
        (df['af_indian'] > 0.01) & 
        (df['af_global'] < 0.01)
    ]
    
    print(f"✅ Found {len(rescues)} rescue cases.")
    
    if not rescues.empty:
        # Sort by Indian AF desc
        rescues = rescues.sort_values('af_indian', ascending=False)
        
        # Select columns
        cols = ['variant', 'chrom', 'pos', 'ref', 'alt', 'af_indian', 'af_global', 'evo2_score']
        rescues = rescues[cols]
        
        rescues.to_csv(output_file, index=False)
        print(f"💾 Saved to {output_file}")
        
        print("\nTop 5 Examples:")
        print(rescues.head(5).to_string(index=False))
    else:
        print("⚠️ No rescue cases found matching criteria.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="supplementary_table_2.csv", help="Output CSV path")
    args = parser.parse_args()
    
    extract_rescue_cases(args.output)
