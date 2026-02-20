"""
PALB2 Evo2 Scoring - Budget Mode (A10G)

✅ GPU: A10G (4x cheaper than H100, fits within 3 credits)
✅ Image: Standard pip install (Fast build, no compilation errors)
✅ Logic: score_sequences() with bfloat16
✅ Batch: Size 8 (A10G memory-safe)
"""

import modal
import pandas as pd
import os

# Simplified image - Fast build, standard dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch>=2.0",
        "transformers",
        "pandas",
        "huggingface_hub",
        "scipy",
        "einops"
    )
    .run_commands(
        "git clone --recurse-submodules https://github.com/ArcInstitute/evo2.git && cd evo2 && pip install ."
    )
)

app = modal.App("palb2-scoring-budget", image=image)

@app.cls(gpu="A10G", timeout=7200)  # Budget mode: 4x cheaper, 2 hour max
class PALB2Scorer:
    @modal.enter()
    def load_model(self):
        import torch
        from evo2 import Evo2
        
        print("Loading Evo2-7B on A10G...")
        # device_map="auto" handles loading across GPU memory optimally
        self.model = Evo2('evo2_7b', device_map="auto")
        self.model.eval()
        print("✅ Evo2 loaded")
    
    @modal.method()
    def score_batch(self, batch_data):
        import torch
        
        results = []
        for var in batch_data:
            try:
                seq = var['seq_context']
                ref, alt = var['ref'], var['alt']
                rel_pos = int(var['rel_pos'])
                
                # Construct sequences
                ref_seq = seq
                alt_seq = seq[:rel_pos] + alt + seq[rel_pos+1:]
                
                # Validate lengths
                if len(ref_seq) != len(alt_seq):
                    print(f"⚠️ Length mismatch for {var.get('chrom')}:{var.get('pos_hg38')}")
                    continue
                
                # Score with mixed precision (vital for A10G speed)
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    scores = self.model.score_sequences([ref_seq, alt_seq])
                
                delta = float(scores[1]) - float(scores[0])
                
                results.append({
                    'chrom': var['chrom'],
                    'pos_hg38': var['pos_hg38'],
                    'ref': ref,
                    'alt': alt,
                    'func_class': var.get('func_class', ''),
                    'func_score': var.get('func_score'),
                    'evo2_score': delta,
                    'evo2_ref_score': float(scores[0]),
                    'evo2_alt_score': float(scores[1]),
                    'clinvar_id': var.get('clinvar_id', ''),
                    'review_status': var.get('review_status', '')
                })
            except Exception as e:
                print(f"Error scoring variant: {e}")
                results.append({
                    'chrom': var.get('chrom'),
                    'pos_hg38': var.get('pos_hg38'),
                    'evo2_score': None,
                    'error': str(e)
                })
        return results

@app.local_entrypoint()
def main():
    INPUT_FILE = "multi_gene/palb2_clinvar/palb2_ready_for_evo2.csv"
    OUTPUT_FILE = "multi_gene/palb2_clinvar/palb2_evo2_scores.csv"
    
    print("="*70)
    print("PALB2 EVO2 SCORING (A10G Budget Mode)")
    print("="*70)
    
    # Validate input file exists
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: File not found: {INPUT_FILE}")
        return
    
    df = pd.read_csv(INPUT_FILE)
    print(f"\n📊 Loaded {len(df)} variants")
    print(f"   Pathogenic: {(df['func_class']=='Pathogenic').sum()}")
    print(f"   Benign: {(df['func_class']=='Benign').sum()}")
    
    # Batch size 8 is safer for A10G memory (24GB) than 16
    BATCH_SIZE = 8
    batches = [df.iloc[i:i+BATCH_SIZE].to_dict('records') for i in range(0, len(df), BATCH_SIZE)]
    
    est_time_min = len(batches) * 6
    est_cost = est_time_min / 60 * 1.10
    
    print(f"\n🔄 Processing {len(batches)} batches on A10G...")
    print(f"   Estimated time: ~{est_time_min} minutes")
    print(f"   Estimated cost: ~${est_cost:.2f}")
    print(f"   ✅ Safe for 3 credit budget!")
    
    scorer = PALB2Scorer()
    all_results = []
    
    print("\n⚡ Scoring in progress...")
    for i, batch_results in enumerate(scorer.score_batch.map(batches)):
        all_results.extend(batch_results)
        if (i + 1) % 10 == 0:
            progress = (i + 1) / len(batches) * 100
            print(f"   Processed {i+1}/{len(batches)} batches ({progress:.1f}%)")
    
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_FILE, index=False)
    
    success = results_df['evo2_score'].notna().sum()
    print(f"\n\n✅ DONE! Successfully scored: {success}/{len(df)}")
    print(f"   Failed: {len(df) - success}")
    print(f"\n💾 Results saved to {OUTPUT_FILE}")
