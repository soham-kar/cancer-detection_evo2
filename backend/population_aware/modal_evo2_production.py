"""
BRCA1 Evo2 Scorer - Production Modal Deployment

This is a self-contained version that copies the EXACT working image
configuration from main.py (which successfully deploys).

Usage:
    cd d:\project\biotech-evo2\backend
    modal run population_aware/modal_evo2_production.py
"""
import modal
import subprocess
import sys
import os

# ===================================================================
# EXACT COPY OF BUILD FUNCTION FROM MAIN.PY
# ===================================================================
def build_cuda_kernels():
    """Compile flash-attn and transformer-engine on a GPU machine."""
    os.environ["MAX_JOBS"] = "6"
    for pkg in ("flash-attn==2.8.0.post2",
                "transformer_engine[pytorch]==2.8.0"):
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", pkg, "--no-build-isolation"
        ])

# ===================================================================
# EXACT COPY OF IMAGE DEFINITION FROM MAIN.PY
# ===================================================================
evo2_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12"
    )
    .apt_install(
        "build-essential", "cmake", "ninja-build", "libcudnn8",
        "libcudnn8-dev", "git", "gcc", "g++"
    )
    .env({"CC": "/usr/bin/gcc", "CXX": "/usr/bin/g++", "BUILD_ID": "prod-dec27-2232"})  # Force new hash
    .pip_install("packaging", "wheel", "setuptools", "ninja")
    .run_commands(
        "git clone --recurse-submodules https://github.com/ArcInstitute/evo2.git && "
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
        "torch", 
        "vtx>=0.0.8",  # CRITICAL: Vortex engine
        "pandas",
        "numpy"
    )
)

# Create app
app = modal.App("brca1-evo2-production")
volume = modal.Volume.from_name("hf_cache", create_if_missing=True)
mount_path = "/root/.cache/huggingface"

# ===================================================================
# MINIMAL SCORER CLASS (NO REDIS/GROQ DEPENDENCIES)
# ===================================================================
@app.cls(image=evo2_image, gpu="H100", volumes={mount_path: volume})
class Evo2Scorer:
    """Minimal Evo2 scorer using production image"""
    
    @modal.enter()
    def load_model(self):
        from evo2 import Evo2
        print("Loading Evo2-7B model...")
        self.model = Evo2('evo2_7b')
        print("✅ Evo2-7B loaded successfully")
    
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
            print(f"⚠️  Error scoring variant: {e}")
            return None

# ===================================================================
# BATCH SCORING FOR EFFICIENCY
# ===================================================================
@app.cls(image=evo2_image, gpu="H100", volumes={mount_path: volume})
class Evo2BatchScorer:
    """Batch scorer for higher throughput"""
    
    @modal.enter()
    def load_model(self):
        from evo2 import Evo2
        print("Loading Evo2-7B model...")
        self.model = Evo2('evo2_7b')
        print("✅ Evo2-7B loaded successfully")
    
    @modal.method()
    def score_batch(self, variants_batch: list):
        """
        Score a batch of variants
        
        Args:
            variants_batch: List of dicts with keys: ref_seq, alt_seq, mut_pos, chrom, pos, ref, alt
            
        Returns:
            List of dicts with scores
        """
        results = []
        
        for var in variants_batch:
            try:
                delta_l = self.model.delta_log_likelihood(
                    var['ref_seq'], 
                    var['alt_seq'], 
                    position=var['mut_pos']
                )
                results.append({
                    'chrom': var['chrom'],
                    'pos_hg38': var['pos'],
                    'ref': var['ref'],
                    'alt': var['alt'],
                    'evo2_score': float(delta_l)
                })
            except Exception as e:
                print(f"⚠️  Error on {var['chrom']}:{var['pos']}: {e}")
                results.append({
                    'chrom': var['chrom'],
                    'pos_hg38': var['pos'],
                    'ref': var['ref'],
                    'alt': var['alt'],
                    'evo2_score': None
                })
        
        return results

# ===================================================================
# LOCAL ENTRYPOINT
# ===================================================================
@app.local_entrypoint()
def main():
    """Score all BRCA1 variants with batch processing"""
    import pandas as pd
    
    INPUT_FILE = "population_aware/results/brca1_ready_for_evo2.csv"
    OUTPUT_FILE = "population_aware/results/brca1_evo2_scores_REAL.csv"
    BATCH_SIZE = 50  # Process 50 variants at a time
    
    print("="*70)
    print("🚀 BRCA1 Evo2 Scoring - Production Run")
    print("="*70)
    print(f"📂 Input:  {INPUT_FILE}")
    print(f"💾 Output: {OUTPUT_FILE}")
    print(f"🔧 Engine: Evo2-7B on H100 (production image)")
    print(f"📦 Batch size: {BATCH_SIZE}")
    print("="*70)
    
    # Load data
    df = pd.read_csv(INPUT_FILE)
    print(f"\n✅ Loaded {len(df)} variants")
    
    # Prepare batches
    print(f"\n⚙️  Preparing {len(df) // BATCH_SIZE + 1} batches...")
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
            seq_list = list(seq_context)
            ref_seq = seq_context
            seq_list[mut_pos] = alt
            alt_seq = ''.join(seq_list)
            
            batch_vars.append({
                'ref_seq': ref_seq,
                'alt_seq': alt_seq,
                'mut_pos': mut_pos,
                'chrom': row['chrom'],
                'pos': row['pos_hg38'],
                'ref': ref,
                'alt': alt
            })
        
        batches.append(batch_vars)
    
    # Initialize scorer
    print(f"\n⚡ Initializing Evo2 batch scorer on H100...")
    scorer = Evo2BatchScorer()
    
    # Score batches in parallel
    print(f"\n🔬 Scoring {len(df)} variants in {len(batches)} batches...")
    all_results = []
    
    for i, batch_results in enumerate(scorer.score_batch.map(batches)):
        all_results.extend(batch_results)
        progress = ((i + 1) / len(batches)) * 100
        print(f"   Batch {i+1}/{len(batches)} ({progress:.1f}%)")
    
    # Create results dataframe
    results_df = pd.DataFrame(all_results)
    
    # Merge with original data (to keep func_class, func_score)
    final_df = df[['chrom', 'pos_hg38', 'ref', 'alt', 'func_class', 'func_score']].merge(
        results_df,
        on=['chrom', 'pos_hg38', 'ref', 'alt'],
        how='left'
    )
    
    # Save
    final_df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\n💾 Saved {len(final_df)} scored variants")
    
    # Summary
    print("\n" + "="*70)
    print("📊 SCORING SUMMARY")
    print("="*70)
    scored = final_df[final_df['evo2_score'].notna()]
    print(f"Total variants:      {len(final_df)}")
    print(f"Successfully scored: {len(scored)} ({len(scored)/len(final_df)*100:.1f}%)")
    
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
    print(f"\n📝 Next steps:")
    print(f"  1. Copy {OUTPUT_FILE} to brca1_evo2_scores.csv")
    print(f"  2. Run: python day2_validate_accuracy.py")
    print(f"  3. Run: python day3_create_atlas.py")
    print(f"  4. Compare real vs mock performance")
    print("="*70)
