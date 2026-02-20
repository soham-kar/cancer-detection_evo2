"""
Multi-Gene Evo2 Scoring - Uses Working BRCA1 Image

Reuses the proven modal_evo2_production image (same as BRCA1)
No need to rebuild - just import the working image!
"""

import modal
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath('../..'))

# Import the working image from modal_evo2_production
try:
    from modal_evo2_production import evo2_image as image
    print("✅ Using proven production Evo2 image (same as BRCA1)")
except ImportError:
    print("❌ Could not import modal_evo2_production - run 'modal deploy modal_evo2_production.py' first")
    raise

app = modal.App("multi-gene-evo2-scoring")

@app.cls(image=image, gpu="H100", timeout=7200)
class MultiGeneScorer:
    @modal.enter()
    def load_model(self):
        import torch
        from evo2 import Evo2
        
        print("Loading Evo2-7B on H100...")
        self.model = Evo2('evo2_7b')  # Same model ID as BRCA1
        print(f"✅ Evo2 loaded")
    
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
                    'evo2_ref_score': float(scores[0]),
                    'evo2_alt_score': float(scores[1]),
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
def main(gene: str = "palb2"):
    import pandas as pd
    
    gene = gene.lower()
    if gene not in ["brca2", "palb2"]:
        raise ValueError("Gene must be 'brca2' or 'palb2'")
    
    print("="*70)
    print(f"{gene.upper()} EVO2 SCORING (Using BRCA1 Image)")
    print("="*70)
    
    input_file = f"{gene}_clinvar/{gene}_ready_for_evo2.csv"
    output_file = f"{gene}_clinvar/{gene}_evo2_scores.csv"
    
    df = pd.read_csv(input_file)
    
    print(f"\n📊 Variants: {len(df)}")
    print(f"   Pathogenic: {(df['func_class']=='Pathogenic').sum()}")
    print(f"   Benign: {(df['func_class']=='Benign').sum()}")
    
    BATCH_SIZE = 16
    batches = [df.iloc[i:i+BATCH_SIZE].to_dict('records') for i in range(0, len(df), BATCH_SIZE)]
    
    print(f"\n🔄 Processing {len(batches)} batches on H100")
    est_time = len(batches) * 2
    print(f"   Estimated time: ~{est_time} minutes")
    
    scorer = MultiGeneScorer()
    all_results = []
    
    for i, batch_results in enumerate(scorer.score_batch.map(batches)):
        all_results.extend(batch_results)
        if (i + 1) % 10 == 0 or gene == "palb2" and (i + 1) % 5 == 0:
            progress = (i + 1) / len(batches) * 100
            print(f"   Progress: {i+1}/{len(batches)} ({progress:.1f}%)")
    
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_file, index=False)
    
    success = results_df['evo2_score'].notna().sum()
    print(f"\n📊 Summary:")
    print(f"   Successfully scored: {success}/{len(df)}")
    print(f"   Errors: {len(df) - success}")
    print(f"\n💾 Saved to {output_file}")
    print(f"\n✅ {gene.upper()} complete!")
