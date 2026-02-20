"""
BRCA2 Evo2 Scoring - Simplified for Fresh Modal Accounts

✅ No flash-attn compilation
✅ Works reliably on new accounts
⚠️ Takes ~50 min (vs ~30 min with flash-attn)
"""

import modal
import pandas as pd

# Simplified image - reuses cached layers from PALB2 if run after it
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch>=2.0",
        "transformers",
        "pandas",
        "huggingface_hub"
    )
    .run_commands(
        "git clone --recurse-submodules https://github.com/ArcInstitute/evo2.git && cd evo2 && pip install ."
    )
)

app = modal.App("brca2-scoring-simple", image=image)

@app.cls(gpu="A10G", timeout=14400)  # Budget mode: 4x cheaper
class BRCA2Scorer:
    @modal.enter()
    def load_model(self):
        import torch
        from evo2 import Evo2
        
        print("Loading Evo2-7B on H100...")
        self.model = Evo2('evo2_7b')
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
    print("="*70)
    print("BRCA2 EVO2 SCORING (Simplified)")
    print("="*70)
    
    df = pd.read_csv("multi_gene/brca2_clinvar/brca2_ready_for_evo2.csv")
    
    print(f"\n📊 Loaded {len(df)} variants")
    print(f"   Pathogenic: {(df['func_class']=='Pathogenic').sum()}")
    print(f"   Benign: {(df['func_class']=='Benign').sum()}")
    
    BATCH_SIZE = 16
    batches = [df.iloc[i:i+BATCH_SIZE].to_dict('records') for i in range(0, len(df), BATCH_SIZE)]
    
    print(f"\n🔄 Processing {len(batches)} batches on A10G (Budget Mode)")
    print(f"   Estimated time: ~{len(batches) * 6} minutes")
    print(f"   Cost: ~${len(batches) * 6 / 60 * 1.10:.2f}")
    print(f"   (Image already cached if you ran PALB2 first)")
    
    scorer = BRCA2Scorer()
    all_results = []
    
    print("\n⚡ Scoring in progress...")
    for i, batch_results in enumerate(scorer.score_batch.map(batches)):
        all_results.extend(batch_results)
        if (i + 1) % 10 == 0:
            progress = (i + 1) / len(batches) * 100
            print(f"   {i+1}/{len(batches)} batches ({progress:.1f}%)")
    
    results_df = pd.DataFrame(all_results)
    output_file = "multi_gene/brca2_clinvar/brca2_evo2_scores.csv"
    results_df.to_csv(output_file, index=False)
    
    success = results_df['evo2_score'].notna().sum()
    print(f"\n📊 Results:")
    print(f"   Successfully scored: {success}/{len(df)}")
    print(f"   Failed: {len(df) - success}")
    print(f"\n💾 Saved to {output_file}")
    print(f"\n✅ BRCA2 scoring complete!")
