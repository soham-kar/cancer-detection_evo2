"""
Modal Job for Evo2 Feature Extraction - Chromatin-Aware CRISPR Prediction
Fixed: tokenizer.encode() and download paths
"""
import subprocess
import sys
import os
import modal
from modal import Image, App, gpu, Volume

app = modal.App("evo2-chromatin-innovation-fixed")

# ============================================================================
# IMAGE BUILD - Matches working main.py exactly
# ============================================================================
def build_cuda_kernels():
    os.environ["MAX_JOBS"] = "6"
    for pkg in ("flash-attn==2.8.0.post2", "transformer_engine[pytorch]==2.8.0"):
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", pkg, "--no-build-isolation"
        ])

evo2_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install("build-essential", "cmake", "ninja-build", "libcudnn8", "libcudnn8-dev", "git", "gcc", "g++")
    .env({"CC": "/usr/bin/gcc", "CXX": "/usr/bin/g++"})
    .pip_install("packaging", "wheel", "setuptools", "ninja")
    .run_commands("git clone --recurse-submodules https://github.com/ArcInstitute/evo2.git && cd evo2 && pip install .")
    .run_function(build_cuda_kernels, gpu="L40S", memory=32768, cpu=8, timeout=3600)
    .pip_install("biopython", "huggingface_hub", "torch", "numpy", "pandas", "zstandard", "pyBigWig")
    .env({"PYTHONPATH": "/root"})
)

hf_cache = modal.Volume.from_name("hf_cache", create_if_missing=True)
evo_cache = modal.Volume.from_name("evo2-cache", create_if_missing=True)
data_vol = modal.Volume.from_name("crispr-data", create_if_missing=True)
feature_cache = modal.Volume.from_name("evo2-features-cache", create_if_missing=True)

@app.cls(
    image=evo2_image,
    gpu="H100",
    timeout=3600,
    volumes={
        "/cache/huggingface": hf_cache,
        "/cache/evo2": evo_cache,
        "/data": data_vol,
        "/features": feature_cache,
    },
    scaledown_window=120,
)
class Evo2ChromatinExtractor:
    """Extract 8kb chromatin features"""
    
    @modal.enter()
    def load_model(self):
        from evo2 import Evo2
        import torch
        
        print("Loading Evo2 7B...")
        self.model = Evo2("evo2_7b")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loaded on {self.device}")
        
        # Check tokenizer type
        if hasattr(self.model, 'tokenizer'):
            print(f"Tokenizer type: {type(self.model.tokenizer)}")
        else:
            print("WARNING: No tokenizer attribute found")
    
    def _extract_single(self, sequence: str, seq_id: str = None) -> dict:
        """Extract features for one 8kb sequence"""
        import torch
        import numpy as np
        
        seq = sequence.upper().ljust(8000, 'N')[:8000]
        
        try:
            # Get score (proven working from main.py)
            score = self.model.score_sequences([seq])[0]
            
            # Get embeddings - use score_sequences internal path
            with torch.no_grad():
                # Tokenize using CharLevelTokenizer.encode()
                if hasattr(self.model, 'tokenizer') and hasattr(self.model.tokenizer, 'encode'):
                    tokens = self.model.tokenizer.encode(seq)
                    input_ids = torch.tensor([tokens], dtype=torch.long).to(self.device)
                else:
                    # Fallback
                    tokens = [ord(c) for c in seq[:8192]]
                    input_ids = torch.tensor([tokens], dtype=torch.long).to(self.device)
                
                # Get hidden states - StripedHyena doesn't support output_hidden_states
                model_core = self.model.model if hasattr(self.model, 'model') else self.model
                hidden = model_core(input_ids)  # Returns hidden states directly
                
                # Handle different output formats
                if isinstance(hidden, tuple):
                    hidden = hidden[0]
                if len(hidden.shape) == 3:
                    hidden = hidden.squeeze(0)
                
                # Extract features (BFloat16-safe)
                global_emb = hidden.mean(dim=0).cpu().to(torch.float16).numpy()
                center_start = max(0, hidden.shape[0] // 2 - 500)
                center_end = min(hidden.shape[0], hidden.shape[0] // 2 + 500)
                center_emb = hidden[center_start:center_end].mean(dim=0).cpu().to(torch.float16).numpy()
                
            return {
                'seq_id': seq_id,
                'evo2_score': float(score),
                'seq_length': hidden.shape[0],
                'success': True
            }
            
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return {'seq_id': seq_id, 'success': False, 'error': 'GPU_OOM'}
        except Exception as e:
            import traceback
            return {
                'seq_id': seq_id,
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    @modal.method()
    def extract_with_cache(self, seq_id: str, sequence: str) -> dict:
        """Extract with caching"""
        from pathlib import Path
        import numpy as np
        
        cache_path = Path(f"/features/{seq_id}.npz")
        
        if cache_path.exists():
            data = np.load(cache_path)
            return {
                'seq_id': seq_id,
                'source': 'cache',
                'evo2_score': float(data['evo2_score']),
                'success': True
            }
        
        result = self._extract_single(sequence, seq_id)
        
        if result['success']:
            # Re-extract embeddings for caching
            import torch
            seq = sequence.upper().ljust(8000, 'N')[:8000]
            with torch.no_grad():
                if hasattr(self.model, 'tokenizer') and hasattr(self.model.tokenizer, 'encode'):
                    tokens = self.model.tokenizer.encode(seq)
                    input_ids = torch.tensor([tokens], dtype=torch.long).to(self.device)
                else:
                    tokens = [ord(c) for c in seq[:8192]]
                    input_ids = torch.tensor([tokens], dtype=torch.long).to(self.device)
                
                model_core = self.model.model if hasattr(self.model, 'model') else self.model
                hidden = model_core(input_ids)
                if isinstance(hidden, tuple):
                    hidden = hidden[0]
                if len(hidden.shape) == 3:
                    hidden = hidden.squeeze(0)
                
                global_emb = hidden.mean(dim=0).cpu().to(torch.float16).numpy()
                center_start = max(0, hidden.shape[0] // 2 - 500)
                center_end = min(hidden.shape[0], hidden.shape[0] // 2 + 500)
                center_emb = hidden[center_start:center_end].mean(dim=0).cpu().to(torch.float16).numpy()
            
            np.savez_compressed(
                cache_path,
                evo2_score=result['evo2_score'],
                global_embedding=global_emb,
                center_embedding=center_emb,
            )
            feature_cache.commit()
            result['source'] = 'fresh'
        
        return result

@app.function(
    image=evo2_image,
    gpu="H100",
    timeout=7200,
    volumes={"/data": data_vol, "/features": feature_cache},
)
def process_dataset(input_path: str, output_path: str, limit: int = None, checkpoint_every: int = 1000):
    import json
    from pathlib import Path
    
    if not Path(input_path).exists():
        print(f"ERROR: {input_path} not found")
        for f in Path("/data").rglob("*.json"):
            print(f"  Available: {f}")
        return 0
    
    with open(input_path) as f:
        data = json.load(f)
    
    if limit:
        data = data[:limit]
        print(f"Limited to {limit} samples")
    
    print(f"Processing {len(data)} items from {input_path}")
    
    extractor = Evo2ChromatinExtractor()
    results = []
    
    try:
        for i, item in enumerate(data):
            seq_id = item.get('seq_id', f'seq_{i}')
            result = extractor.extract_with_cache.remote(seq_id, item['sequence_8kb'])
            results.append(result)
            
            # Checkpoint every N samples
            if (i + 1) % checkpoint_every == 0:
                checkpoint_path = f"/data/checkpoint_{i+1}.json"
                with open(checkpoint_path, 'w') as f:
                    json.dump(results, f)
                data_vol.commit()
                print(f"  Checkpoint saved: {i+1}/{len(data)} done")
            elif (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(data)} done")
    
    except Exception as e:
        print(f"ERROR at sample {i}: {e}")
        # Save progress before failing
        emergency_path = f"/data/emergency_checkpoint_{i}.json"
        with open(emergency_path, 'w') as f:
            json.dump(results, f)
        data_vol.commit()
        print(f"Emergency checkpoint saved: {emergency_path}")
        raise
    
    # Final save
    with open(output_path, 'w') as f:
        json.dump(results, f)
    
    feature_cache.commit()
    data_vol.commit()
    print(f"Saved to {output_path}")
    return len(results)

@app.local_entrypoint()
def main():
    """Run extraction for all 202K samples"""
    import subprocess
    import json
    from pathlib import Path
    
    print("=" * 60)
    print("EVO2 FULL 202K EXTRACTION (H100)")
    print("=" * 60)
    
    input_path = "/data/change_seq_8kb_sequences.json"
    output_path = "/data/full_202k_results.json"
    
    print(f"\nInput: {input_path}")
    print(f"Output: {output_path}")
    print("\nStarting FULL extraction (no limit)...")
    print("Estimated time: ~60 hours")
    print("Estimated cost: ~$240 on H100")
    
    n_processed = process_dataset.remote(input_path, output_path, limit=None)
    
    print(f"\n✅ Processed {n_processed} sequences")
    print(f"Results saved to {output_path}")
    print("\nDownload: modal volume get crispr-data full_202k_results.json .")
    print("=" * 60)
