"""
Feature Extraction for Evo2 (Linear Probe Strategy)

Extracts 4096-dim hidden states from Evo2 model at the target site position.
Based on crispr_scorer.py infrastructure.
"""
import modal
import torch
import numpy as np
import pandas as pd
from typing import List, Dict
from pathlib import Path
import ast

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"

# --- MODAL CONFIGURATION ---
# Robust Evo2 Image with CUDA kernels compiled (Mirrors main.py config)
evo2_image = (
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
        "git clone --recurse-submodules https://github.com/ArcInstitute/evo2.git && "
        "cd evo2 && pip install ."
    )
    .run_commands(
        "MAX_JOBS=6 pip install flash-attn==2.8.0.post2 --no-build-isolation",
        "MAX_JOBS=6 pip install transformer_engine[pytorch]==2.8.0 --no-build-isolation",
        gpu="L40S"
    )
    .pip_install(
        "biopython", 
        "huggingface_hub", 
        "torch", 
        "vtx>=0.0.8", 
        "fastapi[standard]", 
        "requests", 
        "scikit-learn", 
        "pandas",
        "einops", 
        "accelerate",
        # Extra utils from requirements.txt to match main.py env
        "matplotlib",
        "seaborn",
        "openpyxl",
        "redis>=5.0.0",
        "groq>=0.4.0"
    )
    .env({"PYTHONPATH": "/root", "BUILD_TIMESTAMP": "20260128_2243"})
)

app = modal.App("evonator-features", image=evo2_image)

# Use shared cache volumes (existing from previous runs)
hf_volume = modal.Volume.from_name("hf_cache", create_if_missing=True)
evo2_volume = modal.Volume.from_name("evo2-cache", create_if_missing=True)
hf_mount = "/root/.cache/huggingface"
evo2_mount = "/root/.cache/evo2"

@app.cls(
    image=evo2_image,
    gpu="H100",
    timeout=3600,
    volumes={
        hf_mount: hf_volume,
        evo2_mount: evo2_volume
    },
    scaledown_window=1200,
    max_containers=10
)
class Evo2FeatureExtractor:
    @modal.enter()
    def load_model(self):
        print("Loading Evo2 model...")
        from evo2 import Evo2
        # Initialize wrapper
        self.wrapper = Evo2("evo2_7b")
        
        # Access underlying model and tokenizer
        # Evo2 wrapper has: self.model (the actual model), self.tokenizer
        self.model = self.wrapper.model
        self.tokenizer = self.wrapper.tokenizer
        
        # Determine hidden size by inspecting model config if available
        # Default for evo2_7b is 4096
        self.hidden_size = getattr(self.model.config, 'hidden_size', 4096) if hasattr(self.model, 'config') else 4096
        print(f"Evo2 model loaded. Hidden size: {self.hidden_size}")

    @modal.method()
    def extract_batch(self, rows: List[Dict]) -> List[np.ndarray]:
        """Extract features for a batch of sequences"""
        features_list = []
        
        for row in rows:
            seq = row.get('offtarget_full')
            if not seq:
                print("Missing sequence!")
                features_list.append(np.zeros(self.hidden_size))
                continue
            
            try:
                # Evo2's CharLevelTokenizer has a .tokenize() method
                # It returns token IDs as a list
                tokens = self.tokenizer.tokenize(seq)
                input_ids = torch.tensor([tokens], dtype=torch.long).to("cuda")
                
                with torch.no_grad():
                    # Forward pass - get hidden states
                    # The model architecture determines how to get hidden states
                    # Try common patterns
                    try:
                        # Pattern 1: HuggingFace style
                        outputs = self.model(input_ids, output_hidden_states=True)
                        if hasattr(outputs, 'hidden_states'):
                            hidden = outputs.hidden_states[-1]  # [1, seq_len, hidden]
                        elif hasattr(outputs, 'last_hidden_state'):
                            hidden = outputs.last_hidden_state
                        else:
                            # Fallback: outputs might be a tuple
                            hidden = outputs[0] if isinstance(outputs, tuple) else outputs
                    except TypeError:
                        # Pattern 2: Simple forward without kwargs
                        # Model might not support output_hidden_states
                        # In this case, output IS the hidden states
                        outputs = self.model(input_ids)
                        hidden = outputs[0] if isinstance(outputs, tuple) else outputs
                
                # Extract at center (target site location)
                n_tokens = hidden.shape[1]
                center = n_tokens // 2
                window = 10  # ±10 tokens
                
                start = max(0, center - window)
                end = min(n_tokens, center + window)
                
                # Mean pool over the site and convert to float32 (numpy doesn't support bfloat16)
                site_vector = hidden[0, start:end, :].mean(dim=0).float().cpu().numpy()
                features_list.append(site_vector)
                
            except Exception as e:
                print(f"Error extracting features: {e}")
                features_list.append(np.zeros(self.hidden_size))
            
        return features_list

@app.local_entrypoint()
def extract_main(
    input_file: str = "benchmark/circle_seq/input_8kb.csv",
    output_file: str = "evo2_features_8kb.npy",
    sample: int = 0,
    batch_size: int = 50
):
    print(f"Loading {input_file}...")
    input_path = DATA_DIR / input_file
    df = pd.read_csv(input_path)
    
    if sample > 0:
        df = df.sample(sample, random_state=42)
        
    print(f"Extracting features for {len(df)} samples...")
    
    extractor = Evo2FeatureExtractor()
    
    all_rows = df.to_dict('records')
    batches = [all_rows[i:i+batch_size] for i in range(0, len(all_rows), batch_size)]
    
    all_features = []
    
    for i, batch_features in enumerate(extractor.extract_batch.map(batches, order_outputs=True)):
        all_features.extend(batch_features)
        print(f"Batch {i+1}/{len(batches)} done. Total: {len(all_features)}")
        
    # Save as NPY
    output_path = DATA_DIR / output_file
    np.save(output_path, np.array(all_features))
    print(f"Saved features to {output_path} shape={np.array(all_features).shape}")
    
    # Save labels corresponding to these features (important!)
    # We should ensure order is preserved (it is with order_outputs=True)
    labels = df['is_validated'].values
    labels_path = DATA_DIR / output_file.replace('.npy', '_labels.npy')
    np.save(labels_path, labels)
    print(f"Saved labels to {labels_path}")
