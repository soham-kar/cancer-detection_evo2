"""
BRCA1 Atlas Runner - Reuses Evo2 Image from main.py

This script creates a minimal Modal app that reuses the production Evo2 image
but doesn't require the Redis/Groq secrets (we only need Delta-L scoring).

Usage:
    cd d:\\project\\biotech-evo2\\backend
    modal run population_aware/run_atlas.py
"""
import modal
import pandas as pd
import sys
import os

# Add backend directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

# Import just the image from main.py (not the full app)
from main import evo2_image, volume, mount_path

# Create a NEW minimal app for atlas scoring (no secrets needed)
atlas_app = modal.App("brca1-atlas")

@atlas_app.cls(image=evo2_image, gpu="H100", volumes={mount_path: volume})
class Evo2Scorer:
    """Minimal Evo2 scorer - just delta-L, no enrichment features"""
    
    @modal.enter()
    def load_model(self):
        from evo2 import Evo2
        print("Loading Evo2 model...")
        self.model = Evo2('evo2_7b')
        print("✅ Evo2 loaded")
    
    @modal.method()
    def score_variant(self, ref_seq: str, alt_seq: str, mut_pos: int):
        """
        Score a single variant using Evo2's delta log-likelihood
        
        Args:
            ref_seq: Reference sequence context
            alt_seq: Alternate sequence context  
            mut_pos: Position of mutation in sequence
            
        Returns:
            float: Delta log-likelihood score
        """
        try:
            delta_l = self.model.delta_log_likelihood(ref_seq, alt_seq, position=mut_pos)
            return float(delta_l)
        except Exception as e:
            print(f"Error scoring: {e}")
            return None

@atlas_app.local_entrypoint()
def run_brca1_atlas():
    """Score all BRCA1 variants using Evo2"""
    
    # Configuration
    INPUT_FILE = "population_aware/results/brca1_ready_for_evo2.csv"
    OUTPUT_FILE = "population_aware/results/brca1_evo2_scores.csv"
    
    print("="*70)
    print("🚀 BRCA1 Atlas Scoring Job")
    print("="*70)
    print(f"📂 Input:  {INPUT_FILE}")
    print(f"💾 Output: {OUTPUT_FILE}")
    print(f"🔧 Engine: Evo2-7B on H100 (image from main.py)")
    print("="*70)
    
    # Load data
    try:
        df = pd.read_csv(INPUT_FILE)
        print(f"\n✅ Loaded {len(df)} variants")
    except FileNotFoundError:
        print(f"\n❌ Error: Could not find {INPUT_FILE}")
        return
    
    # Initialize scorer
    print("\n⚡ Initializing Evo2 scorer on H100...")
    scorer = Evo2Scorer()
    
    # Process variants
    print(f"\n🔬 Scoring {len(df)} variants...")
    results = []
    
    for idx, row in df.iterrows():
        seq_context = row['seq_context']
        mut_pos = int(row['rel_pos'])
        ref = row['ref']
        alt = row['alt']
        
        # Build reference and alternate sequences
        seq_list = list(seq_context)
        ref_seq = seq_context
        seq_list[mut_pos] = alt
        alt_seq = ''.join(seq_list)
        
        # Score with Evo2
        delta_l = scorer.score_variant.remote(ref_seq, alt_seq, mut_pos)
        
        results.append({
            'chrom': row['chrom'],
            'pos_hg38': row['pos_hg38'],
            'ref': ref,
            'alt': alt,
            'func_class': row.get('func_class', 'Unknown'),
            'func_score': row.get('func_score', None),
            'evo2_score': delta_l
        })
        
        # Progress
        if (idx + 1) % 100 == 0:
            print(f"   {idx + 1}/{len(df)} ({(idx+1)/len(df)*100:.1f}%)")
    
    # Save results
    print(f"\n💾 Saving results...")
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"✅ Saved {len(results_df)} scored variants")
    
    # Preview + Summary
    print("\n" + "="*70)
    print("PREVIEW:")
    print(results_df[['chrom', 'pos_hg38', 'ref', 'alt', 'func_class', 'evo2_score']].head())
    
    scored = results_df[results_df['evo2_score'].notna()]
    if len(scored) > 0:
        print(f"\n📊 Summary (n={len(scored)}):")
        print(f"  Mean:  {scored['evo2_score'].mean():.6f}")
        print(f"  Std:   {scored['evo2_score'].std():.6f}")
        print(f"  Range: [{scored['evo2_score'].min():.6f}, {scored['evo2_score'].max():.6f}]")
    
    print("="*70)
    print("✅ Job Complete!")
    print("="*70)
