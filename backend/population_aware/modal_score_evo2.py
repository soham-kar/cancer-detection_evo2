import modal
import subprocess
import sys
import os
import time

# Build function for CUDA kernels (runs on GPU during image build)
def build_cuda_kernels():
    """Compile flash-attn and transformer-engine on a GPU machine."""
    os.environ["MAX_JOBS"] = "6"
    for pkg in ("flash-attn==2.8.0.post2",
                "transformer_engine[pytorch]==2.8.0"):
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", pkg, "--no-build-isolation"
        ])

# Modal configuration
app = modal.App("brca1-scoring-fresh")
volume = modal.Volume.from_name("evo2-cache", create_if_missing=True)

# Define image with Evo2 (exact copy from main.py)
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12"
    )
    .apt_install(
        "build-essential", "cmake", "ninja-build", "libcudnn8",
        "libcudnn8-dev", "git", "gcc", "g++"
    )
    .env({"CC": "/usr/bin/gcc", "CXX": "/usr/bin/g++"})
    .pip_install("packaging", "wheel", "setuptools", "ninja")
    .run_commands(
        "git clone --recurse-submodules https://github.com/ArcInstitute/evo2.git  && "
        "cd evo2 && pip install ."
    )
    .run_function(
        build_cuda_kernels,
        gpu="L40S",
        memory=32768,
        cpu=8,
        timeout=3600
    )
    .pip_install(
        "biopython", 
        "huggingface_hub", 
        "torch", 
        "vtx>=0.0.8", 
        "fastapi[standard]", 
        "requests", 
        "scikit-learn", 
        "redis",
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "openpyxl",
        "groq>=0.4.0",
    )
    .env({"PYTHONPATH": "/root"})
)

@app.cls(
    image=image,
    gpu="H100",  # Using H100
    memory=32000,
    timeout=3600,
    volumes={"/cache": volume},
    retries=2,
    concurrency_limit=10, # Max 10 containers running at once (Safety cap)
)
class Evo2Scorer:
    @modal.enter()
    def load_model(self):
        """Load the model ONCE when container starts"""
        import torch
        from evo2 import Evo2
        
        print("🔄 Initializing Container: Loading Evo2-7B model...")
        if torch.cuda.is_available():
            print(f"🚀 Running on GPU: {torch.cuda.get_device_name(0)}")
        
        try:
            self.model = Evo2('evo2_7b')
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise e

    @modal.method()
    def score_batch(self, df_dict):
        """
        Score a batch of variants using the pre-loaded model.
        """
        import pandas as pd
        import numpy as np
        
        df = pd.DataFrame.from_dict(df_dict)
        print(f"📋 Processing batch of {len(df)} variants...")
        
        results = []
        error_count = 0
        
        for idx, row in df.iterrows():
            try:
                chrom = str(row['chrom'])
                pos = int(row['pos'])
                ref = row['ref']
                alt = row['alt']
                seq_context = row['seq_context']
                mut_pos = int(row['rel_pos'])
                
                if pd.isna(seq_context) or len(seq_context) < mut_pos + 1:
                    error_count += 1
                    continue

                ref_seq = seq_context[:mut_pos] + ref + seq_context[mut_pos+1:]
                alt_seq = seq_context[:mut_pos] + alt + seq_context[mut_pos+1:]
                
                scores = self.model.score_sequences([ref_seq, alt_seq])
                delta_l = scores[1] - scores[0]
                
                results.append({
                    'variant': f"{chrom}:{pos}{ref}>{alt}",
                    'chrom': chrom,
                    'pos': pos,
                    'ref': ref,
                    'alt': alt,
                    'af_indian': row.get('af_indian', 0.0),
                    'ac_indian': row.get('ac_indian', 0),
                    'an_indian': row.get('an_indian', 0),
                    'af_global': row.get('af_global', 0.0),
                    'evo2_score': delta_l,
                    'status': 'success'
                })
                
            except Exception as e:
                error_count += 1
                results.append({
                    'variant': f"{str(row.get('chrom', 'NA'))}:{str(row.get('pos', 'NA'))}{str(row.get('ref', 'NA'))}>{str(row.get('alt', 'NA'))}",
                    'chrom': str(row.get('chrom', 'NA')),
                    'pos': row.get('pos', 0),
                    'ref': str(row.get('ref', 'NA')),
                    'alt': str(row.get('alt', 'NA')),
                    'af_indian': row.get('af_indian', 0.0),
                    'evo2_score': None,
                    'status': 'error',
                    'error_msg': str(e)
                })
            
            if len(df) > 1000 and idx > 0 and idx % 1000 == 0:
                print(f"   Score progress: {idx}/{len(df)}...")
        
        print(f"✅ Batch complete! Success: {len([r for r in results if r['status']=='success'])}, Errors: {error_count}")
        return results if results else []

@app.local_entrypoint()
def main():
    import pandas as pd
    import time
    
    print("Starting PARALLEL Indian Chr22 scoring (FIXED Class Pattern)...")
    start_time = time.time()
    
    input_path = "results/indian_chr22_ready_for_evo2.csv"
    partial_output_path = "results/indian_chr22_evo2_scores_PARTIAL.csv"
    final_output_path = "results/indian_chr22_evo2_scores.csv"
    
    # Ensure output directory exists
    os.makedirs("results", exist_ok=True)
    
    # Resume capability
    all_results = []
    
    if os.path.exists(partial_output_path):
        try:
            partial_df = pd.read_csv(partial_output_path)
            all_results = partial_df.to_dict('records')
            print(f"📂 Found partial file with {len(all_results)} variants.")
        except Exception as e:
            print(f"Could not resume: {e}")
            all_results = []
    
    if not os.path.exists(input_path):
        print(f"❌ Input file not found: {input_path}")
        return
    
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} variants")
    
    # TEST MODE TOGGLE
    TEST_MODE = False # Set to False for real run!

    if TEST_MODE:
        print("\n🧪 RUNNING IN TEST MODE (First 100 variants) 🧪")
        df = df.head(100)
    
    df = df.where(pd.notnull(df), None)
    
    # Batch size logic
    if TEST_MODE:
        BATCH_SIZE = 10 
    else:
        BATCH_SIZE = 6000  # Efficient for 10 concurrent containers
    
    # Create chunks
    chunks = [df.iloc[i:i+BATCH_SIZE].to_dict('records') for i in range(0, len(df), BATCH_SIZE)]
    print(f"Total batches to process: {len(chunks)}")
    
    # Figure out start index
    start_batch = 0
    if all_results:
        # Align reuse to batch boundaries
        start_batch = len(all_results) // BATCH_SIZE
        # Safe truncation to prevent duplicates
        safe_count = start_batch * BATCH_SIZE
        if len(all_results) > safe_count:
             all_results = all_results[:safe_count]
        print(f"🔄 Resuming from batch index {start_batch}")

    remaining_chunks = chunks[start_batch:]
    
    if not remaining_chunks:
        print("✅ All batches appear complete!")
    else:
        print(f"🔥 Processing {len(remaining_chunks)} remaining batches in parallel...")
        if not TEST_MODE:
             print(f"Estimated time: ~1.5-2 hours")
             print(f"Estimated cost: ~${len(chunks) * 0.13:.2f}")

        # Instantiate Scorer Class
        scorer = Evo2Scorer()
        
        processed_count = 0
        
        # PARALLEL EXECUTION: .map() launches multiple containers
        for result_batch in scorer.score_batch.map(remaining_chunks):
            
            # Cost Safety removed (User manual control)

            # As each batch finishes (in any order), we append and save
            all_results.extend(result_batch)
            processed_count += 1
            
            # Save checkpoint
            checkpoint_df = pd.DataFrame(all_results)
            checkpoint_df.to_csv(partial_output_path, index=False)
            
            current_total_batches = start_batch + processed_count
            print(f"💾 Checkpoint. Total variants: {len(all_results)}. ({current_total_batches}/{len(chunks)} batches)")
    
    # Final save
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(final_output_path, index=False)
        print(f"✅ FINAL: {len(results_df)} variants saved to {final_output_path}")
        
        if 'status' in results_df.columns:
            successes = len(results_df[results_df['status'] == 'success'])
            print(f"✅ Successfully scored: {successes} ({successes/len(results_df)*100:.1f}%)")
