"""
BRCA2 Evo2 Scoring - Self-Contained Modal Script

No external imports needed - everything embedded here
Uses proven Evo2 setup from main.py
"""

import modal
import os

# Build the Evo2 image (copy of working setup from main.py)
def build_cuda_kernels():
    """Compile GPU libraries"""
    import subprocess
    import sys
    import os
    os.environ["MAX_JOBS"] = "6"
    for pkg in ("flash-attn==2.8.0.post2", "transformer_engine[pytorch]==2.8.0"):
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "--no-build-isolation"])

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "build-essential")
    .env({"CUDA_HOME": "/usr/local/cuda"})
    .run_function(build_cuda_kernels, gpu="H100")
    .pip_install("torch", "vtx")
    .pip_install("git+https://github.com/evo-design/evo2.git")
    .pip_install("huggingface_hub", "pandas")
)

app = modal.App("brca2-evo2-scoring", image=image)

@app.cls(gpu="H100", timeout=7200)
class BRCA2Scorer:
    @modal.enter()
    def load_model(self):
        """Load Evo2 model"""
        import torch
        from evo2 import Evo2
        
        print("Loading Evo2-7B on H100...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Evo2("togethercomputer/evo2-7b-base")
        self.model.to(self.device)
        self.model.eval()
        print(f"✅ Model loaded on {self.device}")
    
    @modal.method()
    def score_batch(self, batch_data):
        """Score batch of variants"""
        import torch
        
        results = []
        for var in batch_data:
            try:
                seq = var['seq_context']
                ref = var['ref']
                alt = var['alt']
                rel_pos = var['rel_pos']
                
                ref_seq = seq
                alt_seq = seq[:rel_pos] + alt + seq[rel_pos+1:]
                sequences = [ref_seq, alt_seq]
                
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    scores = self.model.score_sequences(sequences)
                
                ref_score = float(scores[0])
                alt_score = float(scores[1])
                delta_score = alt_score - ref_score
                
                results.append({
                    'chrom': var['chrom'],
                    'pos_hg38': var['pos_hg38'],
                    'ref': ref,
                    'alt': alt,
                    'func_class': var.get('func_class', ''),
                    'func_score': var.get('func_score', None),
                    'evo2_score': delta_score,
                    'evo2_ref_score': ref_score,
                    'evo2_alt_score': alt_score,
                    'clinvar_id': var.get('clinvar_id', ''),
                    'review_status': var.get('review_status', '')
                })
            except Exception as e:
                print(f"Error: {e}")
                results.append({
                    'chrom': var.get('chrom'),
                    'pos_hg38': var.get('pos_hg38'),
                    'ref': var.get('ref'),
                    'alt': var.get('alt'),
                    'evo2_score': None,
                    'error': str(e)
                })
        return results

@app.local_entrypoint()
def main():
    """Main scoring entry point"""
    import pandas as pd  # Import here, not at module level
    
    print("="*70)
    print("BRCA2 EVO2 SCORING")
    print("="*70)
    
    input_file = "brca2_clinvar/brca2_ready_for_evo2.csv"
    df = pd.read_csv(input_file)
    
    print(f"\n📊 Variants: {len(df)}")
    print(f"   Pathogenic: {(df['func_class']=='Pathogenic').sum()}")
    print(f"   Benign: {(df['func_class']=='Benign').sum()}")
    
    BATCH_SIZE = 16
    batches = []
    for i in range(0, len(df), BATCH_SIZE):
        batches.append(df.iloc[i:i+BATCH_SIZE].to_dict('records'))
    
    print(f"\n🔄 Processing {len(batches)} batches on H100")
    print(f"   Estimated time: ~{len(batches) * 2} minutes")
    
    scorer = BRCA2Scorer()
    all_results = []
    
    for i, batch_results in enumerate(scorer.score_batch.map(batches)):
        all_results.extend(batch_results)
        if (i + 1) % 10 == 0:
            print(f"   Progress: {i+1}/{len(batches)} ({(i+1)/len(batches)*100:.1f}%)")
    
    output_file = "brca2_clinvar/brca2_evo2_scores.csv"
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_file, index=False)
    
    print(f"\n📊 Summary:")
    print(f"   Successfully scored: {results_df['evo2_score'].notna().sum()}/{len(df)}")
    print(f"\n💾 Saved to {output_file}")
    print(f"\n✅ BRCA2 complete!")
