"""
Multi-Gene Evo2 Scoring - Working Version

Uses Modal mounts to access parent directory's production image
Run from multi_gene/ directory
"""

import modal

# Mount parent directory so we can import modal_evo2_production
parent_dir_mount = modal.Mount.from_local_dir(
    "../",
    remote_path="/root/parent"
)

# Create a stub to import the production image
stub_image = modal.Image.debian_slim(python_version="3.11")

app = modal.App("multi-gene-scorer", mounts=[parent_dir_mount])

# Import the production image by executing in the mounted directory
with stub_image.imports():
    import sys
    sys.path.insert(0, "/root/parent")
    from modal_evo2_production import evo2_image as image

@app.cls(image=image, gpu="H100", timeout=7200)
class MultiGeneScorer:
    @modal.enter()
    def load_model(self):
        import torch
        from evo2 import Evo2
        
        print("Loading Evo2-7B on H100...")
        self.model = Evo2('evo2_7b')
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
    print(f"{gene.upper()} EVO2 SCORING")
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
    print(f"   Estimated: ~{len(batches) * 2} minutes")
    
    scorer = MultiGeneScorer()
    all_results = []
    
    for i, batch_results in enumerate(scorer.score_batch.map(batches)):
        all_results.extend(batch_results)
        if (i + 1) % 10 == 0 or gene == "palb2" and (i + 1) % 5 == 0:
            print(f"   Progress: {i+1}/{len(batches)} ({(i+1)/len(batches)*100:.1f}%)")
    
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_file, index=False)
    
    success = results_df['evo2_score'].notna().sum()
    print(f"\n📊 Successfully scored: {success}/{len(df)}")
    print(f"\n💾 Saved to {output_file}")
    print(f"\n✅ {gene.upper()} complete!")
