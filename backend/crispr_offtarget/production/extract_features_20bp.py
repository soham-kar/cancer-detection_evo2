"""
Feature Extraction for Evo2 (20bp Sequences)

Extracts hidden state features from 20bp target sequences.
Uses balanced dataset (5k positives + 5k negatives).
"""
import modal
import torch
import numpy as np
import pandas as pd
from typing import List, Dict
from pathlib import Path

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"

# --- MODAL CONFIGURATION ---
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
        "biopython", "huggingface_hub", "torch", "vtx>=0.0.8", 
        "fastapi[standard]", "requests", "scikit-learn", "pandas",
        "einops", "accelerate", "matplotlib", "seaborn", "openpyxl"
    )
    .env({"PYTHONPATH": "/root", "BUILD_TIMESTAMP": "20260203"})
)

app = modal.App("evonator-features-20bp", image=evo2_image)

# Cache volumes
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
        self.wrapper = Evo2("evo2_7b")
        self.model = self.wrapper.model
        self.tokenizer = self.wrapper.tokenizer
        
        self.hidden_size = getattr(self.model.config, 'hidden_size', 4096) if hasattr(self.model, 'config') else 4096
        print(f"Evo2 model loaded. Hidden size: {self.hidden_size}")

    @modal.method()
    def extract_batch(self, rows: List[Dict]) -> List[np.ndarray]:
        """Extract features for a batch of 20bp sequences"""
        features_list = []
        
        for row in rows:
            # Use target_sequence (20bp off-target site)
            seq = row.get('target_sequence', '')
            if not seq or len(seq) < 15:
                print(f"Invalid sequence: {seq}")
                features_list.append(np.zeros(self.hidden_size))
                continue
            
            try:
                # Tokenize
                tokens = self.tokenizer.tokenize(seq)
                input_ids = torch.tensor([tokens], dtype=torch.long).to("cuda")
                
                with torch.no_grad():
                    try:
                        outputs = self.model(input_ids, output_hidden_states=True)
                        if hasattr(outputs, 'hidden_states'):
                            hidden = outputs.hidden_states[-1]
                        elif hasattr(outputs, 'last_hidden_state'):
                            hidden = outputs.last_hidden_state
                        else:
                            hidden = outputs[0] if isinstance(outputs, tuple) else outputs
                    except TypeError:
                        outputs = self.model(input_ids)
                        hidden = outputs[0] if isinstance(outputs, tuple) else outputs
                
                # Mean pool over entire sequence (it's only 20bp)
                seq_vector = hidden[0, :, :].mean(dim=0).float().cpu().numpy()
                features_list.append(seq_vector)
                
            except Exception as e:
                print(f"Error extracting features: {e}")
                features_list.append(np.zeros(self.hidden_size))
            
        return features_list

@app.local_entrypoint()
def extract_main(
    input_file: str = "circle_seq_balanced.csv",
    output_file: str = "evo2_features_balanced.npy",
    sample: int = 0,
    batch_size: int = 100
):
    print(f"Loading {input_file}...")
    input_path = DATA_DIR / input_file
    df = pd.read_csv(input_path)
    
    if sample > 0:
        df = df.sample(sample, random_state=42)
        
    print(f"Total samples: {len(df)}")
    print(f"Label distribution: {df['is_validated'].value_counts().to_dict()}")
    
    extractor = Evo2FeatureExtractor()
    
    all_rows = df.to_dict('records')
    batches = [all_rows[i:i+batch_size] for i in range(0, len(all_rows), batch_size)]
    
    all_features = []
    
    for i, batch_features in enumerate(extractor.extract_batch.map(batches, order_outputs=True)):
        all_features.extend(batch_features)
        print(f"Batch {i+1}/{len(batches)} done. Total: {len(all_features)}")
        
    # Save features
    output_path = DATA_DIR / output_file
    np.save(output_path, np.array(all_features))
    print(f"Saved features to {output_path} shape={np.array(all_features).shape}")
    
    # Save labels
    labels = df['is_validated'].values
    labels_path = DATA_DIR / output_file.replace('.npy', '_labels.npy')
    np.save(labels_path, labels)
    print(f"Saved labels to {labels_path}")
    
    # Save mismatch positions for Option B
    mismatch_path = DATA_DIR / output_file.replace('.npy', '_mismatches.csv')
    df[['grna_sequence', 'target_sequence', 'mismatch_count', 'is_validated']].to_csv(mismatch_path, index=False)
    print(f"Saved mismatch info to {mismatch_path}")
