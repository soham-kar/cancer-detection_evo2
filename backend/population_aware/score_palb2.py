"""
Score PALB2 variants using existing production image
Run from: d:/project/biotech-evo2/backend/population_aware/
"""

import modal
import pandas as pd
from modal_evo2_production import evo2_image  # Import works here!

app = modal.App("palb2-scoring")

@app.cls(image=evo2_image, gpu="H100", timeout=3600)
class PALB2Scorer:
    @modal.enter()
    def load_model(self):
        import torch
        from evo2 import Evo2
        
        print("Loading Evo2-7B...")
        self.model = Evo2('evo2_7b')
        print("✅ Model loaded")
    
    @modal.method()
    def score_batch(self, batch_data):
        import torch
        
        results = []
        for var in batch_data:
            try:
                seq = var['seq_context']
                ref, alt = var['ref'], var['alt']
                rel_pos = var['rel_pos']
                
                ref_seq = seq
                alt_seq = seq[:rel_pos] + alt + seq[rel_pos+1:]
                
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    scores = self.model.score_sequences([ref_seq, alt_seq])
                
                results.append({
                    'chrom': var['chrom'],
                    'pos_hg38': var['pos_hg38'],
                    'ref': ref,
                    'alt': alt,
                    'func_class': var.get('func_class', ''),
                    'func_score': var.get('func_score'),
                    'evo2_score': float(scores[1]) - float(scores[0]),
                    'clinvar_id': var.get('clinvar_id', ''),
                    'review_status': var.get('review_status', '')
                })
            except Exception as e:
                results.append({
                    'chrom': var.get('chrom'),
                    'pos_hg38': var.get('pos_hg38'),
                    'evo2_score': None,
                    'error': str(e)
                })
        return results

@app.local_entrypoint()
def main():
    print("="*70)
    print("PALB2 EVO2 SCORING")
    print("="*70)
    
    df = pd.read_csv("multi_gene/palb2_clinvar/palb2_ready_for_evo2.csv")
    
    print(f"\n📊 Variants: {len(df)}")
    print(f"   Pathogenic: {(df['func_class']=='Pathogenic').sum()}")
    print(f"   Benign: {(df['func_class']=='Benign').sum()}")
    
    batches = [df.iloc[i:i+16].to_dict('records') for i in range(0, len(df), 16)]
    
    print(f"\n🔄 Processing {len(batches)} batches on H100")
    
    scorer = PALB2Scorer()
    all_results = []
    
    for i, batch_results in enumerate(scorer.score_batch.map(batches)):
        all_results.extend(batch_results)
        if (i + 1) % 5 == 0:
            print(f"   Progress: {i+1}/{len(batches)} ({(i+1)/len(batches)*100:.1f}%)")
    
    results_df = pd.DataFrame(all_results)
    results_df.to_csv("multi_gene/palb2_clinvar/palb2_evo2_scores.csv", index=False)
    
    print(f"\n✅ Scored: {results_df['evo2_score'].notna().sum()}/{len(df)}")
    print(f"\n💾 Saved to multi_gene/palb2_clinvar/palb2_evo2_scores.csv")
