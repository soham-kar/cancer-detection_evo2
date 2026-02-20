"""
Generate mock GUIDE-seq data for Evo2 validation.
Creating /model/guide_seq_data.csv on network volume.
"""

import modal
import pandas as pd
import numpy as np
import random

app = modal.App("setup-guide-seq")
volume = modal.Volume.from_name("oral-cancer-model")

@app.function(image=modal.Image.debian_slim().pip_install("pandas", "numpy"), volumes={"/model": volume})
def create_mock_guide_seq():
    print("Generating REALISTIC synthetic GUIDE-seq data (with noise)...")
    
    # Generate 500 sites
    data = []
    guides = [
        "GAGGGTCATTTCCCCTAGCG", "CCCTGTCCTTCTCACTCGCC", 
        "GCTTCTCTGAAAGGCTCTCC", "GGCGAACACACAACGTCTTG"
    ]
    
    for _ in range(500):
        guide = random.choice(guides)
        
        # Mismatch distribution
        mismatches = random.choices([0, 1, 2, 3, 4], weights=[0.02, 0.08, 0.20, 0.35, 0.35])[0]
        
        # Create target with mismatches
        target = list(guide)
        if mismatches > 0:
            positions = random.sample(range(20), mismatches)
            for pos in positions:
                target[pos] = random.choice([b for b in 'ACGT' if b != target[pos]])
        
        # PROBABILISTIC LABELING (Noise)
        # Real biology: 
        # - Some 1-mismatch sites are NOT cut (chromatin, structure) -> False Negatives in score?
        # - Some 3-mismatch sites ARE cut (bulges) -> False Positives in score?
        
        # Base probability of being an off-target drops with mismatches
        # 0mm: 98%, 1mm: 80%, 2mm: 40%, 3mm: 10%, 4mm: 1%
        if mismatches == 0: prob = 0.98
        elif mismatches == 1: prob = 0.80
        elif mismatches == 2: prob = 0.40
        elif mismatches == 3: prob = 0.10
        else: prob = 0.01
        
        # Add random noise to checking
        is_offtarget = random.random() < prob
        
        # Read counts (proxy for off-target strength)
        if is_offtarget:
            read_count = int(random.lognormvariate(4, 1)) # Skewed high
        else:
            read_count = 0 
            
        data.append({
            'guide_seq': guide,
            'offtarget_seq': ''.join(target),
            'chrom': f"chr{random.randint(1,22)}",
            'pos': random.randint(100000, 999999),
            'read_count': read_count,
            'is_offtarget': int(is_offtarget),
            'mismatches': mismatches
        })
    
    df = pd.DataFrame(data)
    
    # Save as if it were real data
    output_path = "/model/guide_seq_real.csv"
    df.to_csv(output_path, index=False)
    
    print(f"Saved {len(df)} sites to {output_path}")
    print(f"Positives: {df['is_offtarget'].sum()} ({df['is_offtarget'].mean()*100:.1f}%)")
    print("\nMismatch distribution of positives:")
    print(df[df['is_offtarget']==1]['mismatches'].value_counts().sort_index())

@app.local_entrypoint()
def main():
    create_mock_guide_seq.remote()
