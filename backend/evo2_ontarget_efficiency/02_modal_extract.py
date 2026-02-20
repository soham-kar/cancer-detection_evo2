"""
02_modal_extract.py

Extract Evo2 embeddings for on-target efficiency prediction.
Uses 8kb genomic windows centered on guide target sites.

ALIGNED WITH main.py:
- Uses REAL Evo2 from Arc Institute (evo2_7b)
- Same CUDA image with flash-attn
- pysam for fast indexed FASTA access
- Batch size 50 for cost efficiency
"""

import subprocess
import sys
import os
import modal
import numpy as np
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Modal setup
app = modal.App("evo2-ontarget-extraction")


# Build function for CUDA kernels (same as main.py)
def build_cuda_kernels():
    """Compile flash-attn and transformer-engine on a big GPU machine."""
    os.environ["MAX_JOBS"] = "6"
    for pkg in ("flash-attn==2.8.0.post2",
                "transformer_engine[pytorch]==2.8.0"):
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", pkg, "--no-build-isolation"
        ])


# Image with REAL Evo2 (same as main.py)
evo2_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12"
    )
    .apt_install(
        "build-essential", "cmake", "ninja-build", "libcudnn8",
        "libcudnn8-dev", "git", "gcc", "g++", "samtools"  # samtools for pysam
    )
    .env({"CC": "/usr/bin/gcc", "CXX": "/usr/bin/g++"})
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
        "biopython",
        "huggingface_hub",
        "torch",
        "numpy",
        "pandas",
        "pysam",  # Fast indexed FASTA access
    )
)

# Volumes (same as main.py)
hf_cache = modal.Volume.from_name("hf_cache", create_if_missing=True)
genome_vol = modal.Volume.from_name("genome-data", create_if_missing=True)
results_vol = modal.Volume.from_name("ontarget-results", create_if_missing=True)


@app.cls(
    image=evo2_image,
    gpu="H100",  # Same as main.py
    timeout=3600,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/genome": genome_vol,
        "/results": results_vol,
    },
)
class Evo2OnTargetExtractor:
    """Extract Evo2 embeddings with integrated genome access."""
    
    @modal.enter()
    def load_model(self):
        """Load REAL Evo2 model and genome on GPU server."""
        from evo2 import Evo2
        import pysam
        
        # Load Evo2 model (same as main.py)
        logging.info("Loading Evo2 7B model...")
        self.model = Evo2('evo2_7b')
        logging.info("Evo2 model loaded successfully")
        
        # Load genome with pysam
        logging.info("Loading hg38 genome...")
        self.genome = pysam.FastaFile("/genome/hg38.fa")
        logging.info(f"Genome loaded: {self.genome.nreferences} chromosomes")
    
    def get_sequence(self, chrom: str, start: int, end: int) -> str:
        """Fetch sequence from hg38."""
        try:
            if chrom in self.genome.references:
                seq = self.genome.fetch(chrom, start, end)
            elif chrom.replace('chr', '') in self.genome.references:
                seq = self.genome.fetch(chrom.replace('chr', ''), start, end)
            elif f"chr{chrom}" in self.genome.references:
                seq = self.genome.fetch(f"chr{chrom}", start, end)
            else:
                logging.error(f"Chromosome {chrom} not found in genome")
                return 'N' * (end - start)
            
            return seq.upper()
        except Exception as e:
            logging.error(f"Failed to fetch {chrom}:{start}-{end}: {e}")
            return 'N' * (end - start)
    
    @modal.method()
    def test_genome(self):
        """Quick sanity check for genome access."""
        test_seq = self.get_sequence("chr1", 1000000, 1001000)
        gc = (test_seq.count('G') + test_seq.count('C')) / len(test_seq) if test_seq else 0
        return {
            "length": len(test_seq),
            "gc_content": round(gc, 3),
            "preview": test_seq[:50],
            "valid": len(test_seq) == 1000 and 'N' not in test_seq[:100]
        }
    
    @modal.method()
    def extract_embedding(
        self,
        chrom: str,
        start: int,
        end: int,
        sample_id: int,
    ) -> dict:
        """
        Extract Evo2 embedding for a genomic region.
        
        Uses Evo2's internal representation for embedding extraction.
        """
        import torch
        
        try:
            # 1. Get genomic sequence
            sequence = self.get_sequence(chrom, start, end)
            
            if sequence.count('N') > len(sequence) * 0.5:
                return {
                    "sample_id": sample_id,
                    "success": False,
                    "error": "Too many Ns in sequence"
                }
            
            # 2. Get Evo2 embedding using the real model
            # Evo2 provides embeddings via its internal forward pass
            with torch.no_grad():
                # Get sequence embedding from Evo2
                # The model.embed_sequences method returns embeddings
                embedding_result = self.model.embed_sequences([sequence])
                
                # embedding_result is a tuple: (embeddings, lengths)
                if isinstance(embedding_result, tuple):
                    embeddings = embedding_result[0]
                else:
                    embeddings = embedding_result
                
                # Convert to numpy
                if isinstance(embeddings, torch.Tensor):
                    embedding_np = embeddings[0].cpu().numpy()
                else:
                    embedding_np = np.array(embeddings[0])
                
                # Get center region (middle of sequence for target site)
                seq_len = embedding_np.shape[0] if len(embedding_np.shape) > 1 else 1
                if len(embedding_np.shape) > 1:
                    center_start = max(0, seq_len // 2 - 500)
                    center_end = min(seq_len, seq_len // 2 + 500)
                    center_embedding = embedding_np[center_start:center_end].mean(axis=0)
                    global_embedding = embedding_np.mean(axis=0)
                else:
                    center_embedding = embedding_np
                    global_embedding = embedding_np
            
            return {
                "sample_id": sample_id,
                "center_embedding": center_embedding.tolist(),
                "global_embedding": global_embedding.tolist(),
                "seq_length": len(sequence),
                "success": True
            }
            
        except Exception as e:
            logging.error(f"Extraction failed for sample {sample_id}: {e}")
            return {
                "sample_id": sample_id,
                "success": False,
                "error": str(e)
            }
    
    @modal.method()
    def batch_extract(self, coords: list) -> list:
        """Extract embeddings for a batch of coordinates."""
        results = []
        for coord in coords:
            result = self.extract_embedding(
                coord["chrom"],
                coord["start"],
                coord["end"],
                coord["sample_id"]
            )
            results.append(result)
        return results
    
    @modal.method()
    def get_delta_score(self, chrom: str, position: int, ref: str, alt: str) -> dict:
        """
        Calculate delta score (same as main.py for consistency).
        This is the standard Evo2 pathogenicity scoring approach.
        """
        try:
            # Get 8kb window
            window_size = 8192
            start = max(0, position - window_size // 2)
            end = position + window_size // 2
            
            ref_seq = self.get_sequence(chrom, start, end)
            relative_pos = position - start
            
            # Create variant sequence
            var_seq = ref_seq[:relative_pos] + alt + ref_seq[relative_pos + len(ref):]
            
            # Score both sequences
            ref_score = self.model.score_sequences([ref_seq])[0]
            var_score = self.model.score_sequences([var_seq])[0]
            delta_score = var_score - ref_score
            
            return {
                "delta_score": float(delta_score),
                "ref_score": float(ref_score),
                "var_score": float(var_score),
                "success": True
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


# ============= LOCAL FUNCTIONS =============

def save_embedding(sample_id: int, result: dict, cache_dir: Path):
    """Save embedding to NPZ file."""
    if result.get("success"):
        output_path = cache_dir / f"sample_{sample_id}.npz"
        np.savez(
            output_path,
            center_embedding=np.array(result["center_embedding"]),
            global_embedding=np.array(result["global_embedding"]),
            seq_length=result["seq_length"]
        )


def get_processed_ids(cache_dir: Path) -> set:
    """Get already processed sample IDs for checkpointing."""
    processed = set()
    for f in cache_dir.glob("sample_*.npz"):
        try:
            sample_id = int(f.stem.replace("sample_", ""))
            processed.add(sample_id)
        except ValueError:
            continue
    return processed


@app.local_entrypoint()
def main(
    input_csv: str = "data/train_coords.csv",
    output_dir: str = "features_train",
    n_samples: int = None,
    test_genome: bool = False
):
    """Run embedding extraction."""
    logging.info("=" * 60)
    logging.info("Evo2 On-Target Efficiency Embedding Extraction")
    logging.info("Using REAL Evo2 7B model (same as main.py)")
    logging.info("=" * 60)
    
    extractor = Evo2OnTargetExtractor()
    
    # Test genome access first
    if test_genome:
        logging.info("Testing genome access...")
        result = extractor.test_genome.remote()
        logging.info(f"Genome test: {result}")
        if not result.get("valid"):
            logging.error("Genome access failed! Upload hg38.fa to genome-data volume.")
            return
        logging.info("✅ Genome access verified")
        return
    
    # Load coordinates
    coords = pd.read_csv(input_csv)
    logging.info(f"Loaded {len(coords)} coordinates from {input_csv}")
    
    if n_samples:
        coords = coords.head(n_samples)
        logging.info(f"Using first {n_samples} samples")
    
    # Checkpointing
    cache_dir = Path(output_dir)
    cache_dir.mkdir(exist_ok=True)
    processed_ids = get_processed_ids(cache_dir)
    
    if processed_ids:
        logging.info(f"Resuming: {len(processed_ids)} already processed")
        coords = coords[~coords['sample_id'].isin(processed_ids)]
        logging.info(f"Remaining: {len(coords)} to process")
    
    if len(coords) == 0:
        logging.info("All samples already processed!")
        return
    
    # Prepare coordinate batches
    coord_list = []
    for _, row in coords.iterrows():
        coord_list.append({
            "chrom": row["chromosome"],
            "start": int(row["window_start"]),
            "end": int(row["window_end"]),
            "sample_id": int(row["sample_id"])
        })
    
    # Process in batches
    batch_size = 50
    total_batches = (len(coord_list) + batch_size - 1) // batch_size
    
    success_count = 0
    for i in range(0, len(coord_list), batch_size):
        batch = coord_list[i:i+batch_size]
        batch_num = i // batch_size + 1
        
        logging.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} samples)")
        
        try:
            results = extractor.batch_extract.remote(batch)
            
            for result in results:
                if result.get("success"):
                    save_embedding(result["sample_id"], result, cache_dir)
                    success_count += 1
                else:
                    logging.warning(f"Failed: sample {result.get('sample_id')}: {result.get('error')}")
                    
        except Exception as e:
            logging.error(f"Batch {batch_num} failed: {e}")
            continue
    
    logging.info(f"\n✅ Extracted {success_count}/{len(coord_list)} embeddings")
    logging.info(f"Saved to: {cache_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/train_coords.csv")
    parser.add_argument("--output", default="features_train")
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--test_genome", action="store_true")
    args = parser.parse_args()
    
    logging.info("Usage:")
    logging.info("  1. Upload genome: modal volume put genome-data /path/to/hg38.fa /")
    logging.info("  2. Test genome:   modal run 02_modal_extract.py --test_genome")
    logging.info("  3. Run extraction: modal run 02_modal_extract.py --input data/train_coords.csv")
