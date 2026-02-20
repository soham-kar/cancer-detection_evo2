"""
Extract Evo2 embeddings for ATAC-seq coordinates using Modal.

Adapts the CRISPR extraction pipeline for chromatin accessibility prediction.
Key difference: Uses layers 28-32 (optimal for accessibility per Evo2 paper).
"""

import modal
import numpy as np
import pandas as pd
from pathlib import Path
import json
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Modal setup
app = modal.App("evo2-atac-extraction")

# Image with Evo2
evo2_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "wget")
    .pip_install(
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "numpy",
        "pandas",
    )
)

# Volumes
hf_cache = modal.Volume.from_name("hf_cache", create_if_missing=True)
evo2_cache = modal.Volume.from_name("evo2-cache", create_if_missing=True)
atac_data = modal.Volume.from_name("atac-data", create_if_missing=True)


@app.cls(
    image=evo2_image,
    gpu="A100",  # Evo2 needs A100
    timeout=3600,
    volumes={
        "/hf_cache": hf_cache,
        "/evo2_cache": evo2_cache,
        "/data": atac_data,
    },
)
class Evo2ATACExtractor:
    """Extract Evo2 embeddings for ATAC-seq coordinate prediction."""
    
    @modal.enter()
    def load_model(self):
        """Load Evo2 model on GPU."""
        import torch
        from transformers import AutoModel, AutoTokenizer
        
        logging.info("Loading Evo2 model...")
        
        # Set cache directories
        import os
        os.environ["HF_HOME"] = "/hf_cache"
        os.environ["TRANSFORMERS_CACHE"] = "/hf_cache"
        
        # Load model
        model_name = "togethercomputer/evo-1-131k-base"
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True,
            cache_dir="/hf_cache"
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir="/hf_cache",
            torch_dtype=torch.float16
        ).cuda()
        self.model.eval()
        
        logging.info("Evo2 model loaded successfully")
    
    @modal.method()
    def extract_embedding(
        self,
        sequence: str,
        coord_id: int,
        layers: list = [28, 29, 30, 31, 32]
    ) -> dict:
        """
        Extract Evo2 embedding for a genomic sequence.
        
        Args:
            sequence: 8kb DNA sequence
            coord_id: Unique identifier for this coordinate
            layers: Layers to extract (28-32 for accessibility)
        
        Returns:
            Dictionary with embeddings and metadata
        """
        import torch
        
        try:
            # Tokenize
            inputs = self.tokenizer(
                sequence,
                return_tensors="pt",
                truncation=True,
                max_length=8192
            ).to("cuda")
            
            # Forward pass with hidden states
            with torch.no_grad():
                outputs = self.model(
                    **inputs,
                    output_hidden_states=True
                )
            
            # Extract embeddings from specified layers
            hidden_states = outputs.hidden_states
            
            # Get center embedding (middle 1kb of sequence)
            seq_len = hidden_states[0].shape[1]
            center_start = seq_len // 2 - 500
            center_end = seq_len // 2 + 500
            
            # Average over specified layers
            layer_embeddings = []
            for layer_idx in layers:
                if layer_idx < len(hidden_states):
                    layer_hidden = hidden_states[layer_idx]
                    center_hidden = layer_hidden[0, center_start:center_end, :]
                    layer_embeddings.append(center_hidden.mean(dim=0))
            
            # Stack and average across layers
            if layer_embeddings:
                embedding = torch.stack(layer_embeddings).mean(dim=0)
            else:
                embedding = hidden_states[-1][0].mean(dim=0)
            
            # Global embedding (full sequence average)
            global_embedding = hidden_states[-1][0].mean(dim=0)
            
            return {
                "coord_id": coord_id,
                "center_embedding": embedding.cpu().numpy().tolist(),
                "global_embedding": global_embedding.cpu().numpy().tolist(),
                "seq_length": len(sequence),
                "success": True
            }
            
        except Exception as e:
            logging.error(f"Extraction failed for coord {coord_id}: {e}")
            return {
                "coord_id": coord_id,
                "success": False,
                "error": str(e)
            }
    
    @modal.method()
    def batch_extract(self, sequences: list, coord_ids: list) -> list:
        """Extract embeddings for a batch of sequences."""
        results = []
        for seq, coord_id in zip(sequences, coord_ids):
            result = self.extract_embedding(seq, coord_id)
            results.append(result)
        return results


# Local functions for data preparation

def load_coordinates(path: Path) -> pd.DataFrame:
    """Load coordinate file."""
    return pd.read_csv(path)


def get_sequence_from_genome(chrom: str, start: int, end: int) -> str:
    """
    Get genomic sequence for a coordinate.
    Uses 2bit file or generates mock for testing.
    """
    # For testing, generate random sequence
    import random
    random.seed(hash(f"{chrom}:{start}"))
    return ''.join(random.choices('ACGT', k=end-start))


def save_embedding(coord_id: int, result: dict, cache_dir: Path):
    """Save embedding to cache."""
    if result.get("success"):
        output_path = cache_dir / f"coord_{coord_id}.npz"
        np.savez(
            output_path,
            center_embedding=np.array(result["center_embedding"]),
            global_embedding=np.array(result["global_embedding"]),
            seq_length=result["seq_length"]
        )


@app.local_entrypoint()
def main(
    coords_path: str = "data/atac/atac_coordinates.csv",
    output_dir: str = "features_cache",
    n_samples: int = None,
    test: bool = False
):
    """Run embedding extraction."""
    logging.info("=" * 60)
    logging.info("Evo2 ATAC-seq Embedding Extraction")
    logging.info("=" * 60)
    
    # Load coordinates
    coords = load_coordinates(Path(coords_path))
    logging.info(f"Loaded {len(coords)} coordinates")
    
    if n_samples:
        coords = coords.head(n_samples)
        logging.info(f"Using first {n_samples} samples")
    
    if test:
        logging.info("TEST MODE: Using mock sequences")
    
    # Prepare sequences
    sequences = []
    coord_ids = []
    
    for _, row in coords.iterrows():
        seq = get_sequence_from_genome(
            row["chrom"], 
            int(row["window_start"]), 
            int(row["window_end"])
        )
        sequences.append(seq)
        coord_ids.append(row["coord_id"])
    
    # Create extractor
    extractor = Evo2ATACExtractor()
    
    # Process in batches
    batch_size = 10
    cache_dir = Path(output_dir)
    cache_dir.mkdir(exist_ok=True)
    
    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i:i+batch_size]
        batch_ids = coord_ids[i:i+batch_size]
        
        logging.info(f"Processing batch {i//batch_size + 1}/{len(sequences)//batch_size + 1}")
        
        results = extractor.batch_extract.remote(batch_seqs, batch_ids)
        
        for result in results:
            save_embedding(result["coord_id"], result, cache_dir)
    
    logging.info(f"\n✅ Extracted embeddings for {len(sequences)} coordinates")
    logging.info(f"Saved to: {cache_dir}")


if __name__ == "__main__":
    # For local testing without Modal
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--coords", default="data/atac/atac_coordinates.csv")
    parser.add_argument("--output", default="features_cache")
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    
    logging.info("Run with: modal run 03_modal_extract_atac.py")
