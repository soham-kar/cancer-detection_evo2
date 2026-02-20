"""
Benchmark-Optimized Modal Backend for Evo2 Variant Analysis

This is a STRIPPED-DOWN version of main.py for benchmarking purposes:
- KEEPS: Evo2 scoring, Redis caching, UCSC sequence fetching
- REMOVES: gnomAD, PubMed RAG, ACMG mapping (not needed for accuracy metrics)

Deploy:
    modal deploy main_benchmark.py

Original main.py is UNTOUCHED.
"""

import subprocess, sys, os
import modal
from pydantic import BaseModel, validator
from typing import List, Optional
import json
import time
import logging
from datetime import datetime

# ===========================================================================
# SIMPLE LOGGER (Reduced from original for speed)
# ===========================================================================
class SimpleLogger:
    def info(self, msg, **kwargs): 
        print(f"[INFO] {msg} {kwargs if kwargs else ''}")
    def warning(self, msg, **kwargs): 
        print(f"[WARN] {msg} {kwargs if kwargs else ''}")
    def error(self, msg, **kwargs): 
        print(f"[ERROR] {msg} {kwargs if kwargs else ''}")
    def set_context(self, **kwargs): pass
    def clear_context(self): pass

logger = SimpleLogger()

# ===========================================================================
# DATA MODELS (Same as original)
# ===========================================================================
class VariantRequest(BaseModel):
    variant_position: int
    alternative: str
    genome: str
    chromosome: str
    reference: str = None
    gene_symbol: str = None
    
    @validator('genome')
    def validate_genome(cls, v):
        valid_genomes = ['hg19', 'hg38', 'mm10', 'mm39']
        if v not in valid_genomes:
            raise ValueError(f'Invalid genome build. Must be one of: {valid_genomes}')
        return v
    
    @validator('chromosome')
    def validate_chromosome(cls, v):
        valid_chroms = [f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY', 'chrM']
        if v not in valid_chroms:
            raise ValueError(f'Invalid chromosome. Must be chr1-chr22, chrX, chrY, or chrM')
        return v

class BatchRequest(BaseModel):
    requests: List[VariantRequest]

# ===========================================================================
# IMAGE BUILD (Same as original, but without clinical_enrichment.py)
# ===========================================================================
def build_cuda_kernels():
    """Compile flash-attn and transformer-engine on GPU machine."""
    os.environ["MAX_JOBS"] = "6"
    for pkg in ("flash-attn==2.8.0.post2", "transformer_engine[pytorch]==2.8.0"):
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", pkg, "--no-build-isolation"
        ])

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
        "redis",
        "pandas",
    )
    .env({"PYTHONPATH": "/root"})
)

# Different app name to not conflict with original
app = modal.App(
    "variant-analysis-benchmark", 
    image=evo2_image,
    secrets=[
        modal.Secret.from_name("redis-credentials"),
    ]
)

volume = modal.Volume.from_name("hf_cache", create_if_missing=True)
mount_path = "/root/.cache/huggingface"

# ===========================================================================
# HELPER: Get Genome Sequence (WITH Redis Caching)
# ===========================================================================
def get_genome_sequence(position, genome: str, chromosome: str, window_size=8192, redis_client=None):
    import requests
    
    half_window = window_size // 2
    start = max(0, position - 1 - half_window)
    end = position - 1 + half_window + 1
    
    cache_key = f"genome:{genome}:{chromosome}:{start}-{end}"
    
    # Try Redis Cache
    if redis_client:
        try:
            cached_data = redis_client.get(cache_key)
            if cached_data:
                logger.info(f"Cache HIT: {cache_key}")
                if isinstance(cached_data, bytes): 
                    data = json.loads(cached_data.decode("utf-8"))
                elif isinstance(cached_data, str): 
                    data = json.loads(cached_data)
                else: 
                    data = cached_data
                return data["sequence"], data["start"]
        except Exception as e:
            logger.warning(f"Redis read error: {e}")
            
    # Cache Miss - Fetch from UCSC
    logger.info(f"Cache MISS: Fetching from UCSC API - {chromosome}:{start}-{end}")
    api_url = f"https://api.genome.ucsc.edu/getData/sequence?genome={genome};chrom={chromosome};start={start};end={end}"
    response = requests.get(api_url, timeout=30)
    
    if response.status_code != 200:
        raise Exception(f"Failed to fetch genome sequence: {response.status_code}")
    
    genome_data = response.json()
    if "dna" not in genome_data:
        raise Exception(f"UCSC API error: {genome_data.get('error', 'Unknown error')}")
        
    sequence = genome_data.get("dna", "").upper()
    if len(sequence) == 0:
        raise ValueError(f"UCSC API returned empty sequence for {chromosome}:{start}-{end}")
    
    # Cache to Redis (30 day TTL)
    if redis_client:
        try:
            cache_data = json.dumps({"sequence": sequence, "start": start})
            redis_client.set(cache_key, cache_data, ex=2592000)
            logger.info(f"Cached to Redis: {cache_key}")
        except Exception as e:
            logger.warning(f"Redis write error: {e}")
    
    return sequence, start

# ===========================================================================
# MAIN EVO2 MODEL CLASS (Simplified - No gnomAD, No PubMed, No ACMG)
# ===========================================================================
@app.cls(gpu="H100", volumes={mount_path: volume}, max_containers=3, retries=2, scaledown_window=120)
class Evo2Model:
    
    @modal.enter()
    def load_resources(self):
        from evo2 import Evo2
        import redis
        
        # 1. Load AI Model
        print("Loading evo2 model...")
        self.model = Evo2('evo2_7b')
        print("Evo2 model loaded")
        
        # 2. Connect to Redis
        try:
            redis_url = os.getenv("REDIS_URL")
            if redis_url:
                self.redis = redis.from_url(redis_url, decode_responses=True)
                logger.info("Redis connected successfully")
            else:
                self.redis = None
                logger.warning("Redis URL not found - caching disabled")
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            self.redis = None
            
        # 3. Classification Parameters (Tuned threshold)
        self.default_params = {
            "threshold": 0.00002,  # TUNED: Optimized for 93% sensitivity
            "lof_std": 0.0015140239,
            "func_std": 0.0009016589
        }
    
    @modal.fastapi_endpoint(method="POST")
    def analyze_batch(self, batch: BatchRequest):
        """
        Benchmark-optimized batch analysis:
        - Chunked GPU processing (prevents OOM)
        - Mixed precision (bfloat16)
        - NO gnomAD, NO PubMed RAG, NO ACMG (for speed)
        """
        import torch
        
        num_variants = len(batch.requests)
        logger.info(f"Received batch of {num_variants} variants")
        
        # CONFIGURATION
        WINDOW_SIZE = 8192
        CHUNK_SIZE = 16  # 16 variants = 32 sequences per GPU batch
        
        # STEP 1: Prepare all sequences
        sequences_to_score = []
        meta_data = []
        batch_genome_cache = {}
        
        logger.info("Preparing sequences...")
        for idx, req in enumerate(batch.requests):
            try:
                cache_key = f"{req.genome}_{req.chromosome}_{req.variant_position}"
                
                # Get genome sequence (with Redis caching)
                if cache_key in batch_genome_cache:
                    window_seq, seq_start = batch_genome_cache[cache_key]
                else:
                    window_seq, seq_start = get_genome_sequence(
                        req.variant_position, req.genome, req.chromosome, WINDOW_SIZE,
                        redis_client=self.redis
                    )
                    batch_genome_cache[cache_key] = (window_seq, seq_start)
                
                # Calculate positions
                relative_pos = req.variant_position - 1 - seq_start
                ref_len = len(req.reference) if req.reference else 1
                reference = window_seq[relative_pos : relative_pos + ref_len]
                var_seq = window_seq[:relative_pos] + req.alternative + window_seq[relative_pos + ref_len:]
                
                # Add both ref and variant sequences
                sequences_to_score.append(window_seq)
                sequences_to_score.append(var_seq)
                
                meta_data.append({
                    "req": req,
                    "reference": reference,
                    "variant_idx": idx,
                    "error": None
                })
            except Exception as e:
                logger.warning(f"Failed to prepare variant {idx}: {e}")
                meta_data.append({
                    "req": req,
                    "reference": None,
                    "variant_idx": idx,
                    "error": str(e)
                })
        
        # STEP 2: Chunked GPU scoring
        all_scores = []
        total_sequences = len(sequences_to_score)
        num_chunks = (total_sequences + (CHUNK_SIZE * 2) - 1) // (CHUNK_SIZE * 2)
        
        logger.info(f"Scoring {total_sequences} sequences in {num_chunks} chunks...")
        
        for chunk_idx in range(0, total_sequences, CHUNK_SIZE * 2):
            chunk_end = min(chunk_idx + CHUNK_SIZE * 2, total_sequences)
            chunk = sequences_to_score[chunk_idx:chunk_end]
            
            # Score with mixed precision
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                chunk_scores = self.model.score_sequences(chunk)
            
            all_scores.extend(chunk_scores)
            
            progress = min(100, int((chunk_end / total_sequences) * 100))
            logger.info(f"Chunk {chunk_idx // (CHUNK_SIZE * 2) + 1}/{num_chunks} complete ({progress}%)")
        
        logger.info("GPU scoring complete")
        
        # STEP 3: Calculate predictions (NO clinical enrichment for benchmark)
        results = []
        params = self.default_params
        score_idx = 0
        
        for i, meta in enumerate(meta_data):
            req = meta["req"]
            
            # Handle errors
            if meta["error"]:
                results.append({
                    "position": req.variant_position,
                    "reference": None,
                    "alternative": req.alternative,
                    "delta_score": None,
                    "prediction": "ERROR",
                    "classification_confidence": None,
                    "error": meta["error"]
                })
                continue
            
            ref_score = all_scores[score_idx]
            var_score = all_scores[score_idx + 1]
            score_idx += 2
            
            delta_score = var_score - ref_score
            
            # Classification using TUNED threshold
            if delta_score < params['threshold']:
                prediction = "Likely pathogenic"
                confidence = min(1.0, abs(delta_score - params['threshold']) / params['lof_std'])
            else:
                prediction = "Likely benign"
                confidence = min(1.0, abs(delta_score - params['threshold']) / params['func_std'])
            
            results.append({
                "position": req.variant_position,
                "reference": meta["reference"],
                "alternative": req.alternative,
                "delta_score": float(delta_score),
                "prediction": prediction,
                "classification_confidence": float(confidence),
                "error": None
            })
        
        logger.info(f"Batch complete: {num_variants} variants analyzed")
        
        return {
            "batch_size": num_variants,
            "chunks_processed": num_chunks,
            "precision": "bfloat16",
            "batch_results": results
        }


# ===========================================================================
# LOCAL TEST ENTRYPOINT
# ===========================================================================
@app.local_entrypoint()
def test():
    """Quick test of the benchmark model."""
    model = Evo2Model()
    
    test_batch = BatchRequest(requests=[
        VariantRequest(
            chromosome="chr17",
            variant_position=43045629,
            alternative="T",
            genome="hg38",
            reference="C"
        )
    ])
    
    result = model.analyze_batch.remote(test_batch)
    print(result)
