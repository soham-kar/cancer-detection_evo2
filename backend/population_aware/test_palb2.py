"""
PALB2 Test - Using Production Image

✅ Uses working modal_evo2_production image (with transformer-engine)
✅ Tests 50 variants on H100
✅ Cost: ~$0.15
"""

import modal
import pandas as pd
import os

# Import the WORKING production image
from modal_evo2_production import evo2_image

app = modal.App("palb2-test-production")

@app.cls(image=evo2_image, gpu="H100", timeout=1800)
class PALB2TestScorer:
    @modal.enter()
    def load_model(self):
        import torch
        from evo2 import Evo2
        
        print("Loading Evo2-7B on H100...")
        self.model = Evo2('evo2_7b')
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
                
                ref_seq = seq
                alt_seq = seq[:rel_pos] + alt + seq[rel_pos+1:]
                
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    scores = self.model.score_sequences([ref_seq, alt_seq])
                
                delta = float(scores[1]) - float(scores[0])
                
                results.append({
                    'chrom': var['chrom'],
                    'pos_hg38': var['pos_hg38'],
                    'ref': ref,
                    'alt': alt,
                    'func_class': var.get('func_class', ''),
                    'evo2_score': delta
                })
            except Exception as e:
                print(f"Error: {e}")
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
    OUTPUT_FILE = "multi_gene/palb2_clinvar/palb2_test_scores.csv"
    
    print("="*70)
    print("🧪 PALB2 TEST - 50 Variants (Production Image)")
    print("="*70)
    
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: {INPUT_FILE} not found")
        return
    
    df = pd.read_csv(INPUT_FILE)
    df_test = df.head(50)
    
    print(f"\n📊 Test: {len(df_test)} variants (from {len(df)} total)")
    
    BATCH_SIZE = 32
    batches = [df_test.iloc[i:i+BATCH_SIZE].to_dict('records') for i in range(0, len(df_test), BATCH_SIZE)]
    
    print(f"\n🔄 {len(batches)} batches on H100")
    print(f"   Cost: ~$0.15")
    
    scorer = PALB2TestScorer()
    all_results = []
    
    for i, batch_results in enumerate( scorer.score_batch.map(batches)):
        all_results.extend(batch_results)
        print(f"   Batch {i+1}/{len(batches)} ✅")
    
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_FILE, index=False)
    
    success = results_df['evo2_score'].notna().sum()
    print(f"\n✅ TEST COMPLETE: {success}/{len(df_test)} scored")
    print(f"💾 Saved to {OUTPUT_FILE}")
