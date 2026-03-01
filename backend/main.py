import subprocess, sys, os
import modal
from pydantic import BaseModel, validator
from typing import List, Optional
import numpy as np
import time
import json
import logging
from datetime import datetime

# NOTE: clinical_enrichment is imported inside Evo2Model.load_resources()
# because it's added to the image and only available at container runtime

# ===========================================================================
# STRUCTURED LOGGING CONFIGURATION
# ===========================================================================
class StructuredLogger:
    """
    Production-grade structured JSON logger for observability.
    Compatible with DataDog, Grafana Loki, CloudWatch, etc.
    """
    
    def __init__(self, name: str, level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Remove default handlers
        self.logger.handlers = []
        
        # Add JSON handler
        handler = logging.StreamHandler()
        handler.setFormatter(self._JsonFormatter())
        self.logger.addHandler(handler)
        
        self.context = {}  # Request-level context
    
    class _JsonFormatter(logging.Formatter):
        def format(self, record):
            log_entry = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "level": record.levelname,
                "message": record.getMessage(),
                "logger": record.name,
            }
            
            # Add extra fields if present
            if hasattr(record, "extra_fields"):
                log_entry.update(record.extra_fields)
            
            return json.dumps(log_entry)
    
    def _log(self, level: int, message: str, **kwargs):
        """Log with structured fields"""
        extra = {"extra_fields": {**self.context, **kwargs}}
        self.logger.log(level, message, extra=extra)
    
    def info(self, message: str, **kwargs):
        self._log(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        self._log(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        self._log(logging.ERROR, message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        self._log(logging.DEBUG, message, **kwargs)
    
    def set_context(self, **kwargs):
        """Set request-level context (e.g., request_id, variant_id)"""
        self.context.update(kwargs)
    
    def clear_context(self):
        """Clear request-level context"""
        self.context = {}
    
    def timed(self, operation: str):
        """Context manager for timing operations"""
        return TimedOperation(self, operation)


class TimedOperation:
    """Context manager for timing and logging operations"""
    
    def __init__(self, logger: StructuredLogger, operation: str):
        self.logger = logger
        self.operation = operation
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.perf_counter() - self.start_time) * 1000
        
        if exc_type is None:
            self.logger.info(
                f"{self.operation} completed",
                operation=self.operation,
                duration_ms=round(duration_ms, 2),
                status="success"
            )
        else:
            self.logger.error(
                f"{self.operation} failed",
                operation=self.operation,
                duration_ms=round(duration_ms, 2),
                status="error",
                error_type=exc_type.__name__,
                error_message=str(exc_val)
            )
        return False


# Initialize structured logger
logger = StructuredLogger(__name__)
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
    
    @validator('variant_position')
    def validate_position(cls, v):
        if v < 1 or v > 300000000:  # Max human chromosome size
            raise ValueError('Position must be between 1 and 300000000')
        return v
    
class BatchRequest(BaseModel):
    requests: List[VariantRequest]
# --- 1. BUILD IMAGE & DEPENDENCIES ---
def build_cuda_kernels():
    """Compile flash-attn and transformer-engine on a big GPU machine."""
    os.environ["MAX_JOBS"] = "6"
    for pkg in ("flash-attn==2.8.0.post2",
                "transformer_engine[pytorch]==2.8.0"):
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
        "scikit-learn", 
        "redis",
        "pandas",
        "matplotlib",
        "seaborn",
        "openpyxl",
        "groq>=0.4.0",  # Official Groq SDK for LLM
    )
    .pip_install_from_requirements("requirements.txt")
    .env({"PYTHONPATH": "/root"})
    # Add clinical enrichment module to the image (MUST BE LAST)
    .add_local_file("clinical_enrichment.py", remote_path="/root/clinical_enrichment.py")
)

app = modal.App(
    "variant-analysis", 
    image=evo2_image,
    secrets=[
        # modal.Secret.from_name("redis-credentials"),  # DISABLED: Redis not needed for basic scoring
        modal.Secret.from_name("entrez-config"),
        modal.Secret.from_name("groq-config"),  # For LLM-powered literature summaries
    ]
)
volume = modal.Volume.from_name("hf_cache", create_if_missing=True)
mount_path = "/root/.cache/huggingface"
# --- 2. HELPER FUNCTIONS ---
def get_genome_sequence(position, genome: str, chromosome: str, window_size=8192):
    import requests
    
    half_window = window_size // 2
    start = max(0, position - 1 - half_window)
    end = position - 1 + half_window + 1
    
    # Fetch from UCSC API
    logger.info(f"Fetching from UCSC API - {chromosome}:{start}-{end} ({genome})")
    api_url = f"https://api.genome.ucsc.edu/getData/sequence?genome={genome};chrom={chromosome};start={start};end={end}"
    response = requests.get(api_url, timeout=30)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch genome sequence: {response.status_code}")
    
    genome_data = response.json()
    if "dna" not in genome_data:
        raise Exception(f"UCSC API error: {genome_data.get('error', 'Unknown error')}")
        
    sequence = genome_data.get("dna", "").upper()
    if len(sequence) == 0:
        raise ValueError(f"CRITICAL: UCSC API returned empty sequence for {chromosome}:{start}-{end}")
        
    expected_length = end - start
    if len(sequence) != expected_length:
        logger.warning(f"Sequence length mismatch: received {len(sequence)}, expected {expected_length}")
        
    logger.info(f"Loaded genome sequence: {len(sequence)} bases")
    
    return sequence, start
def fetch_clinvar_variants(gene_symbol: str, max_variants=1000):
    from Bio import Entrez
    Entrez.email = os.getenv("ENTREZ_EMAIL", "biotech-evo2@example.com")
    
    print(f"Fetching ClinVar data for {gene_symbol}...")
    try:
        term = f"{gene_symbol}[Gene Name] AND (clinsig_pathogenic[Filter] OR clinsig_benign[Filter]) AND single_gene_variant[Prop]"
        handle = Entrez.esearch(db="clinvar", term=term, retmax=max_variants)
        record = Entrez.read(handle)
        handle.close()
        id_list = record["IdList"]
        
        if not id_list:
            return []
        handle = Entrez.esummary(db="clinvar", id=",".join(id_list))
        summaries = Entrez.read(handle)
        handle.close()
        
        labeled_variants = []
        for item in summaries['DocumentSummarySet']['DocumentSummary']:
            try:
                if 'variation_set' in item and item['variation_set'][0]['variation_type'] == 'Single nucleotide variant':
                    sig = item['clinical_significance']['description']
                    label = "LOF" if "Pathogenic" in sig else "FUNC"
                    labeled_variants.append({"label": label, "id": item['uid']})
            except: continue
        return labeled_variants
    except Exception as e:
        print(f"Error fetching ClinVar: {e}")
        return []
# --- 3. MAIN EVO2 MODEL CLASS ---
@app.cls(
    gpu="H100", 
    volumes={mount_path: volume}, 
    scaledown_window=60,  # Shutdown after 60 seconds of inactivity (SAVES CREDITS!)
    retries=2
)
class Evo2Model: 
    @modal.enter()
    def load_resources(self):
        from evo2 import Evo2
        # Import clinical enrichment (available in container via add_local_file)
        import sys
        sys.path.insert(0, "/root")
        from clinical_enrichment import ClinicalEnricher
        
        # 1. Load AI Model
        print("Loading evo2 model...")
        self.model = Evo2('evo2_7b')
        print("Evo2 model loaded")
        
        # 2. Initialize Thresholds
        self.gene_thresholds = {}
        self.default_params = {
            "threshold": -0.0009178519,
            "lof_std": 0.0015140239,
            "func_std": 0.0009016589
        }
        
        # 3. Initialize Clinical Enricher (gnomAD + ACMG + PubMed RAG)
        # DISABLED: RAG/LLM features commented out for faster scoring
        # llm_endpoint = os.getenv("LLM_ENDPOINT")  # Groq API endpoint
        # logger.info(f"LLM_ENDPOINT configured: {llm_endpoint is not None}")
        # if llm_endpoint:
        #     logger.info(f"Using LLM endpoint: {llm_endpoint[:50]}...")
        self.clinical_enricher = ClinicalEnricher(
            redis_client=None,
            llm_endpoint=None  # Disabled for faster scoring
        )
        logger.info("Clinical enricher initialized (gnomAD + ACMG only, RAG disabled)")
        
    @modal.method()
    def calibrate_gene(self, gene_symbol: str):
        """
        Calibrate gene-specific thresholds with memory caching.
        """
        # Check Local Memory
        if gene_symbol in self.gene_thresholds: 
            return self.gene_thresholds[gene_symbol]
        
        # Fetch from ClinVar API
        logger.info(
            "Fetching ClinVar calibration data",
            gene=gene_symbol
        )
        variants = fetch_clinvar_variants(gene_symbol)
        
        # Calculate params (simplified to defaults for now)
        if len(variants) < 10:
            logger.info(
                "Insufficient ClinVar data, using defaults",
                gene=gene_symbol,
                variant_count=len(variants)
            )
            result_params = self.default_params
        else:
            # Future: Calculate custom thresholds from variants
            result_params = self.default_params 

        # Save to Memory
        self.gene_thresholds[gene_symbol] = result_params

        return result_params
        
    @modal.method()
    def run_analysis_logic(self, variant_position: int, alternative: str, genome: str, chromosome: str, provided_reference: str = None, gene_symbol: str = None):
        from fastapi import HTTPException
        import time
        request_start = time.perf_counter()
        
        # Set request context for all subsequent logs
        variant_id = f"{chromosome}-{variant_position}-{provided_reference or '?'}-{alternative}"
        logger.set_context(
            variant_id=variant_id,
            gene=gene_symbol,
            genome=genome
        )
        
        logger.info(
            "Variant analysis started",
            chromosome=chromosome,
            position=variant_position,
            ref=provided_reference,
            alt=alternative
        )
        
        WINDOW_SIZE = 8192        
        
        # =====================================================================
        # STEP 0: gnomAD Pre-Filter (check population frequency BEFORE AI inference)
        # =====================================================================
        gnomad_result = None
        if provided_reference:  # Need ref allele for gnomAD query
            gnomad_result = self.clinical_enricher.gnomad.get_allele_frequency(
                chromosome=chromosome,
                position=variant_position,
                ref=provided_reference,
                alt=alternative
            )
            
            # Auto-classify common variants without running Evo2 (saves GPU cost)
            if gnomad_result.auto_classification:
                logger.info(f"gnomAD auto-classification: {gnomad_result.auto_classification} (AF={gnomad_result.allele_frequency})")
                
                # Get literature context even for common variants
                lit_context = self.clinical_enricher.pubmed.get_literature_context(gene_symbol) if gene_symbol else None
                
                return {
                    "reference": provided_reference,
                    "alternative": alternative,
                    "delta_score": None,  # No AI scoring for common variants
                    "prediction": gnomad_result.auto_classification,
                    "classification_confidence": 1.0,  # High confidence from population data
                    "position": variant_position,
                    "classification_source": "gnomAD_population_frequency",
                    "population_frequency": {
                        "gnomad_af": gnomad_result.allele_frequency,
                        "gnomad_max_pop_af": gnomad_result.population_max_af,
                        "source": gnomad_result.source,
                        "is_common_variant": gnomad_result.is_common,
                        "acmg_frequency_rule": "BA1" if gnomad_result.allele_frequency >= 0.05 else "BS1"
                    },
                    "acmg_evidence": {
                        "code": "BA1" if gnomad_result.allele_frequency >= 0.05 else "BS1",
                        "strength": "Standalone" if gnomad_result.allele_frequency >= 0.05 else "Strong",
                        "description": "Allele frequency is too high in population for pathogenic variant",
                        "clinical_note": f"This variant is present in {gnomad_result.allele_frequency*100:.2f}% of the population, exceeding pathogenicity thresholds."
                    },
                    "literature_context": {
                        "summary": lit_context.summary if lit_context else None,
                        "pubmed_ids": lit_context.pubmed_ids if lit_context else [],
                        "gene_function": lit_context.gene_function if lit_context else None,
                        "articles_found": lit_context.num_articles_found if lit_context else 0
                    } if gene_symbol else None
                }
        
        # =====================================================================
        # STEP 1: Fetch Genome Sequence
        # =====================================================================
        window_seq, seq_start = get_genome_sequence(
            position=variant_position,
            genome=genome,
            chromosome=chromosome,
            window_size=WINDOW_SIZE
        )
            
        relative_pos = variant_position - 1 - seq_start
        if relative_pos < 0 or relative_pos >= len(window_seq):
            raise ValueError(f"Position outside window")
        if provided_reference:
            reference = provided_reference
            actual_ref = window_seq[relative_pos : relative_pos + len(reference)]
            if actual_ref != reference:
                raise HTTPException(status_code=400, detail=f"Reference Mismatch! Expected '{actual_ref}', got '{reference}'")
        else:
            reference = window_seq[relative_pos]
        
        # Query gnomAD if we didn't have ref allele before
        if gnomad_result is None:
            gnomad_result = self.clinical_enricher.gnomad.get_allele_frequency(
                chromosome=chromosome,
                position=variant_position,
                ref=reference,
                alt=alternative
            )
        
        # =====================================================================
        # STEP 2: Gene-Specific Thresholds
        # =====================================================================
        params = self.default_params
        if gene_symbol:
            if gene_symbol in self.gene_thresholds:
                params = self.gene_thresholds[gene_symbol]
            else:
                params = self.calibrate_gene.local(gene_symbol)
        
        # =====================================================================
        # STEP 3: Evo2 AI Scoring
        # =====================================================================
        var_seq = window_seq[:relative_pos] + alternative + window_seq[relative_pos + len(reference):]
        
        ref_score = self.model.score_sequences([window_seq])[0]
        var_score = self.model.score_sequences([var_seq])[0]
        delta_score = var_score - ref_score
        
        if delta_score < params['threshold']:
            prediction = "Likely pathogenic"
            confidence = min(1.0, abs(delta_score - params['threshold']) / params['lof_std'])
        else:
            prediction = "Likely benign"
            confidence = min(1.0, abs(delta_score - params['threshold']) / params['func_std'])
        
        # =====================================================================
        # STEP 4: ACMG Evidence Mapping (PP3/BP4)
        # =====================================================================
        acmg_evidence = self.clinical_enricher.acmg.map_score_to_evidence(
            delta_score=delta_score,
            confidence=confidence,
            model_name="Evo2-7B"
        )
        
        # =====================================================================
        # STEP 5: Literature Context (PubMed direct + optional Tri-Modal RAG)
        # =====================================================================
        lit_context = None
        rag_level = None
        rag_sources = None
        if gene_symbol:
            # Always run direct PubMed search first (reliable, no deployment needed)
            lit_context = self.clinical_enricher.pubmed.get_literature_context(gene_symbol)
            logger.info(f"PubMed search: {lit_context.num_articles_found} articles for {gene_symbol}")

            # Optionally enhance with Tri-Modal RAG (PubMed + ClinVar + UniProt) if deployed
            try:
                rag_function = modal.Function.lookup("multimodal-rag", "MultiModalRAG.search_and_synthesize")
                variant_str = f"{provided_reference or reference}>{alternative}" if provided_reference or reference else alternative
                rag_result = rag_function.remote(gene_symbol, variant_str)
                if rag_result.get("found"):
                    # Override with richer tri-modal result
                    lit_context = type('LitContext', (), {
                        'summary': rag_result.get("summary"),
                        'pubmed_ids': rag_result.get("pmids", []),
                        'gene_function': rag_result.get("protein_function"),
                        'num_articles_found': len(rag_result.get("pmids", []))
                    })()
                    rag_level = rag_result.get("sources", {}).get("pubmed", {}).get("level")
                    rag_sources = rag_result.get("sources")
                    logger.info(f"Tri-Modal RAG enhanced: {lit_context.num_articles_found} papers")
            except Exception as e:
                logger.info(f"Tri-Modal RAG not available ({type(e).__name__}), using PubMed results")
        
        # =====================================================================
        # STEP 6: Build Enriched Response
        # =====================================================================
        result = {
            # Core AI prediction
            "reference": reference,
            "alternative": alternative,
            "delta_score": float(delta_score),
            "prediction": prediction,
            "classification_confidence": float(confidence),
            "position": variant_position,
            "classification_source": "Evo2_AI",
            
            # NEW: Population frequency (gnomAD)
            "population_frequency": {
                "gnomad_af": gnomad_result.allele_frequency if gnomad_result else None,
                "gnomad_max_pop_af": gnomad_result.population_max_af if gnomad_result else None,
                "source": gnomad_result.source if gnomad_result else "gnomAD v4.1",
                "is_common_variant": gnomad_result.is_common if gnomad_result else False
            },
            
            # NEW: ACMG Evidence Code (PP3/BP4)
            "acmg_evidence": {
                "code": acmg_evidence.code,
                "strength": acmg_evidence.strength.value,
                "description": acmg_evidence.description,
                "clinical_note": acmg_evidence.clinical_note
            },
            
            # NEW: Literature Context (Tri-Modal RAG: PubMed + ClinVar + UniProt)
            "literature_context": {
                "summary": lit_context.summary if lit_context else None,
                "pubmed_ids": lit_context.pubmed_ids if lit_context else [],
                "gene_function": lit_context.gene_function if lit_context else None,
                "articles_found": lit_context.num_articles_found if lit_context else 0,
                "evidence_level": rag_level,  # L1_Exact_Variant or L2_Gene_Context
                "sources": rag_sources  # ClinVar, UniProt, PubMed metadata
            } if gene_symbol else None
        }
        
        # Log completion with timing
        total_duration_ms = (time.perf_counter() - request_start) * 1000
        logger.info(
            "Variant analysis completed",
            duration_ms=round(total_duration_ms, 2),
            prediction=prediction,
            delta_score=round(float(delta_score), 6),
            gnomad_hit=gnomad_result is not None,
            acmg_code=acmg_evidence.code if acmg_evidence else None
        )
        logger.clear_context()
        
        return result
    
    @modal.fastapi_endpoint(method="POST")
    def analyze_single_variant(self, request: VariantRequest):
        return self.run_analysis_logic.local(
            variant_position=request.variant_position,
            alternative=request.alternative,
            genome=request.genome,
            chromosome=request.chromosome,
            provided_reference=request.reference,
            gene_symbol=request.gene_symbol
        )   
        
    @modal.fastapi_endpoint(method="POST")
    def analyze_batch(self, batch: BatchRequest):
        """
        Production-ready batch analysis with:
        - Chunked GPU processing (prevents OOM)
        - Mixed precision (bfloat16)
        - Clinical enrichment (gnomAD, ACMG, PubMed RAG)
        - Progress logging
        """
        import torch
        
        num_variants = len(batch.requests)
        logger.info(f"🚀 Received batch of {num_variants} variants")
        
        # CONFIGURATION
        WINDOW_SIZE = 8192
        CHUNK_SIZE = 16  # 16 variants = 32 sequences per GPU batch (safe for H100)
        
        # STEP 1: Prepare all sequences and metadata
        sequences_to_score = []
        meta_data = []
        batch_genome_cache = {}
        
        logger.info("📋 Preparing sequences...")
        for idx, req in enumerate(batch.requests):
            cache_key = f"{req.genome}_{req.chromosome}_{req.variant_position}"
            
            # Get genome sequence (with caching)
            if cache_key in batch_genome_cache:
                window_seq, seq_start = batch_genome_cache[cache_key]
            else:
                window_seq, seq_start = get_genome_sequence(
                    req.variant_position, req.genome, req.chromosome, WINDOW_SIZE
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
                "variant_idx": idx
            })
        
        total_sequences = len(sequences_to_score)
        logger.info(f"📊 Total sequences to score: {total_sequences}")
        
        # STEP 2: Chunked GPU scoring with mixed precision
        all_scores = []
        num_chunks = (total_sequences + (CHUNK_SIZE * 2) - 1) // (CHUNK_SIZE * 2)
        
        logger.info(f"⚡ Scoring in {num_chunks} chunks (bfloat16 precision)...")
        
        for chunk_idx in range(0, total_sequences, CHUNK_SIZE * 2):
            chunk_end = min(chunk_idx + CHUNK_SIZE * 2, total_sequences)
            chunk = sequences_to_score[chunk_idx:chunk_end]
            
            # Score with mixed precision (bfloat16)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                chunk_scores = self.model.score_sequences(chunk)
            
            all_scores.extend(chunk_scores)
            
            # Progress logging
            progress = min(100, int((chunk_end / total_sequences) * 100))
            logger.info(f"   Chunk {chunk_idx // (CHUNK_SIZE * 2) + 1}/{num_chunks} complete ({progress}%)")
        
        logger.info(f"✅ GPU scoring complete")
        
        # STEP 3: Calculate predictions and add clinical enrichment
        results = []
        params = self.default_params
        
        logger.info("🔬 Adding clinical enrichment...")
        
        for i, meta in enumerate(meta_data):
            req = meta["req"]
            ref_score = all_scores[2 * i]
            var_score = all_scores[2 * i + 1]
            delta_score = var_score - ref_score
            
            # Classification
            if delta_score < params['threshold']:
                prediction = "Likely pathogenic"
                confidence = min(1.0, abs(delta_score - params['threshold']) / params['lof_std'])
            else:
                prediction = "Likely benign"
                confidence = min(1.0, abs(delta_score - params['threshold']) / params['func_std'])
            
            # Clinical enrichment (gnomAD + ACMG + Tri-Modal RAG)
            enrichment = self.clinical_enricher.enrich_variant(
                chromosome=req.chromosome,
                position=req.variant_position,
                ref=meta["reference"],
                alt=req.alternative,
                delta_score=delta_score,
                confidence=confidence,
                gene_symbol=req.gene_symbol
            )
            
            # Build enriched result
            results.append({
                "position": req.variant_position,
                "reference": meta["reference"],
                "alternative": req.alternative,
                "gene_symbol": req.gene_symbol,
                "delta_score": float(delta_score),
                "prediction": prediction,
                "classification_confidence": float(confidence),
                "classification_source": "Evo2_AI",
                # Clinical enrichment
                "population_frequency": enrichment["population_frequency"],
                "acmg_evidence": enrichment["acmg_evidence"],
                "literature_context": enrichment.get("literature_context"),  # From Tri-Modal RAG
            })
        
        logger.info(f"🎉 Batch complete: {num_variants} variants analyzed")
        
        return {
            "batch_size": num_variants,
            "chunks_processed": num_chunks,
            "precision": "bfloat16",
            "batch_results": results
        }
# --- 4. BRCA1 VALIDATION JOB ---
@app.function(gpu="H100", volumes={mount_path: volume}, timeout=1000)
def run_brca1_analysis():
    # Import heavy dependencies only when needed
    import pandas as pd
    import numpy as np
    from Bio import SeqIO
    import gzip
    import base64
    from io import BytesIO
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import roc_auc_score, roc_curve
    from evo2 import Evo2
    WINDOW_SIZE = 8192
    print("Loading evo2 model...")
    model = Evo2('evo2_7b')
    print("Evo2 model loaded")
    
    brca1_df = pd.read_excel('/evo2/notebooks/brca1/41586_2018_461_MOESM3_ESM.xlsx',header=2,)
    brca1_df = brca1_df[['chromosome', 'position (hg19)', 'reference', 'alt', 'function.score.mean', 'func.class',]]
    brca1_df.rename(columns={
        'chromosome': 'chrom',
        'position (hg19)': 'pos',
        'reference': 'ref',
        'alt': 'alt',
        'function.score.mean': 'score',
        'func.class': 'class',
    }, inplace=True)
    brca1_df['class'] = brca1_df['class'].replace(['FUNC', 'INT'], 'FUNC/INT')
    with gzip.open('/evo2/notebooks/brca1/GRCh37.p13_chr17.fna.gz', "rt") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            seq_chr17 = str(record.seq)
            break
    ref_seqs = []
    ref_seq_to_index = {}
    ref_seq_indexes = []
    var_seqs = []
    brca1_subset = brca1_df.iloc[:500].copy()
    for _, row in brca1_subset.iterrows():
        p = row["pos"] - 1 
        full_seq = seq_chr17
        ref_seq_start = max(0, p - WINDOW_SIZE//2)
        ref_seq_end = min(len(full_seq), p + WINDOW_SIZE//2)
        ref_seq = seq_chr17[ref_seq_start:ref_seq_end]
        snv_pos_in_ref = min(WINDOW_SIZE//2, p)
        var_seq = ref_seq[:snv_pos_in_ref] + row["alt"] + ref_seq[snv_pos_in_ref+1:]
        if ref_seq not in ref_seq_to_index:
            ref_seq_to_index[ref_seq] = len(ref_seqs)
            ref_seqs.append(ref_seq)
        ref_seq_indexes.append(ref_seq_to_index[ref_seq])
        var_seqs.append(var_seq)
    ref_seq_indexes = np.array(ref_seq_indexes)
    print(f'Scoring likelihoods of {len(ref_seqs)} reference sequences with Evo 2...')
    ref_scores = model.score_sequences(ref_seqs)
    print(f'Scoring likelihoods of {len(var_seqs)} variant sequences with Evo 2...')
    var_scores = model.score_sequences(var_seqs)
    delta_scores = np.array(var_scores) - np.array(ref_scores)[ref_seq_indexes]
    brca1_subset[f'evo2_delta_score'] = delta_scores
    y_true = (brca1_subset['class'] == 'LOF')
    auroc = roc_auc_score(y_true, -brca1_subset['evo2_delta_score'])
    y_true = (brca1_subset["class"] == "LOF")
    fpr, tpr, thresholds = roc_curve(y_true, -brca1_subset["evo2_delta_score"])
    optimal_idx = (tpr - fpr).argmax()
    optimal_threshold = -thresholds[optimal_idx]
    lof_scores = brca1_subset.loc[brca1_subset["class"]== "LOF", "evo2_delta_score"]
    func_scores = brca1_subset.loc[brca1_subset["class"]== "FUNC/INT", "evo2_delta_score"]
    confidence_params = {
        "threshold": optimal_threshold,
        "lof_std": lof_scores.std(),
        "func_std": func_scores.std()
    }
    print("Confidence params:", confidence_params)
    plt.figure(figsize=(4, 2))
    p = sns.stripplot(data=brca1_subset, x='evo2_delta_score', y='class', hue='class', order=['FUNC/INT', 'LOF'], palette=['#777777', 'C3'], size=2, jitter=0.3)
    
    sns.boxplot(showmeans=True,
                meanline=True,
                meanprops={'visible': False},
                medianprops={'color': 'k', 'ls': '-', 'lw': 2},
                whiskerprops={'visible': False},
                zorder=10,
                x="evo2_delta_score",
                y="class",
                data=brca1_subset,
                showfliers=False,
                showbox=False,
                showcaps=False,
                ax=p)
    
    plt.xlabel('Delta likelihood score, Evo 2')
    plt.ylabel('BRCA1 SNV class')
    plt.tight_layout()
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return {'variants': brca1_subset.to_dict(orient="records"), "plot": plot_data, "auroc": auroc}

@app.local_entrypoint()
def main():
    evo2Model = Evo2Model()
    result = evo2Model.run_analysis_logic.remote(
        variant_position=43119628,
        alternative="G",
        genome="hg38",
        chromosome="chr17"
    )
    print(result)
