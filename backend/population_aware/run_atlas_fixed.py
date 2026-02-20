"""
BRCA1 Atlas Scorer - Fixed to use correct Evo2 API

Uses score_sequences() method (the correct API) instead of delta_log_likelihood()
Reuses the working image from modal_evo2_production.py
"""
import modal
import pandas as pd

# Reuse the working image!
from modal_evo2_production import evo2_image, volume, mount_path

app = modal.App("brca1-atlas-fixed")

@app.cls(image=evo2_image, gpu="H100", volumes={mount_path: volume})
class Evo2FixedScorer:
    """Uses the CORRECT score_sequences() method"""
    
    @modal.enter()
    def load_model(self):
        from evo2 import Evo2
        import torch
        print("Loading Evo2-7B...")
        self.model = Evo2('evo2_7b')
        print("✅ Evo2 loaded")
    
    @modal.method()
    def score_batch(self, batch_data):
        """
        Score a batch of variants using score_sequences() - the CORRECT method
        
        Args:
            batch_data: List of dicts with keys: ref_seq, alt_seq, chrom, pos, ref, alt, func_class, func_score
        
        Returns:
            List of results with evo2 delta scores
        """
        import torch
        
        # Prepare sequences (ref and alt for each variant)
        sequences = []
        for var in batch_data:
            sequences.append(var['ref_seq'])  # Reference
            sequences.append(var['alt_seq'])  # Alternate
        
        # Score with bfloat16 precision (faster, less memory)
        try:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                scores = self.model.score_sequences(sequences)  # ✅ CORRECT METHOD!
            
            # Calculate delta scores (alt - ref)
            results = []
            for i, var in enumerate(batch_data):
                ref_score = scores[2 * i]
                alt_score = scores[2 * i + 1]
                delta_score = alt_score - ref_score
                
                results.append({
                    'chrom': var['chrom'],
                    'pos_hg38': var['pos'],
                    'ref': var['ref'],
                    'alt': var['alt'],
                    'func_class': var.get('func_class', 'Unknown'),
                    'func_score': var.get('func_score', None),
                    'evo2_score': float(delta_score)
                })
            
            return results
            
        except Exception as e:
            print(f"❌ Batch scoring error: {e}")
            # Return empty scores on error
            return [{
                'chrom': var['chrom'],
                'pos_hg38': var['pos'],
                'ref': var['ref'],
                'alt': var['alt'],
                'func_class': var.get('func_class', 'Unknown'),
                'func_score': var.get('func_score', None),
                'evo2_score': None
            } for var in batch_data]

@app.local_entrypoint()
def main():
    """Score all BRCA1 variants with CORRECT API"""
    
    INPUT_FILE = "population_aware/results/brca1_ready_for_evo2.csv"
    OUTPUT_FILE = "population_aware/results/brca1_evo2_scores_REAL.csv"
    BATCH_SIZE = 16  # 16 variants = 32 sequences
    
    print("="*70)
    print("🚀 BRCA1 Evo2 Scoring (FIXED - Using score_sequences)")
    print("="*70)
    print(f"📂 Input:  {INPUT_FILE}")
    print(f"💾 Output: {OUTPUT_FILE}")
    print(f"📦 Batch size: {BATCH_SIZE} variants")
    print("="*70)
    
    # Load data
    df = pd.read_csv(INPUT_FILE)
    print(f"\n✅ Loaded {len(df)} variants")
    
    # Prepare batches
    print(f"\n⚙️  Preparing batches...")
    batches = []
    
    for i in range(0, len(df), BATCH_SIZE):
        batch_df = df.iloc[i:i+BATCH_SIZE]
        batch_vars = []
        
        for _, row in batch_df.iterrows():
            seq_context = row['seq_context']
            mut_pos = int(row['rel_pos'])
            ref = row['ref']
            alt = row['alt']
            
            # Build sequences
            # Reference sequence (original)
            ref_seq = seq_context
            
            # Alternate sequence (with mutation)
            seq_list = list(seq_context)
            seq_list[mut_pos] = alt
            alt_seq = ''.join(seq_list)
            
            batch_vars.append({
                'ref_seq': ref_seq,
                'alt_seq': alt_seq,
                'chrom': row['chrom'],
                'pos': row['pos_hg38'],
                'ref': ref,
                'alt': alt,
                'func_class': row.get('func_class', 'Unknown'),
                'func_score': row.get('func_score', None)
            })
        
        batches.append(batch_vars)
    
    print(f"   Created {len(batches)} batches")
    
    # Score batches
    print(f"\n⚡ Scoring on H100 GPUs...")
    scorer = Evo2FixedScorer()
    
    all_results = []
    for i, batch_results in enumerate(scorer.score_batch.map(batches)):
        all_results.extend(batch_results)
        progress = ((i + 1) / len(batches)) * 100
        print(f"   Batch {i+1}/{len(batches)} ({progress:.1f}%)", end="\r")
    
    print(f"\n✅ Scored {len(all_results)} variants")
    
    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\n💾 Saved to {OUTPUT_FILE}")
    
    # Summary
    print("\n" + "="*70)
    print("📊 SUMMARY")
    print("="*70)
    scored = results_df[results_df['evo2_score'].notna()]
    print(f"Total variants:    {len(results_df)}")
    print(f"Successfully scored: {len(scored)} ({len(scored)/len(results_df)*100:.1f}%)")
    
    if len(scored) > 0:
        print(f"\n🎯 Evo2 Score Statistics:")
        print(f"  Mean:   {scored['evo2_score'].mean():.6f}")
        print(f"  Median: {scored['evo2_score'].median():.6f}")
        print(f"  Std:    {scored['evo2_score'].std():.6f}")
        print(f"  Range:  [{scored['evo2_score'].min():.6f}, {scored['evo2_score'].max():.6f}]")
        
        print(f"\n📋 By Functional Class:")
        class_stats = scored.groupby('func_class')['evo2_score'].agg(['count', 'mean', 'std'])
        print(class_stats.to_string())
    
    print("\n" + "="*70)
    print("✅ REAL EVO2 SCORING COMPLETE!")
    print("="*70)
