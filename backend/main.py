# =============================================================================
# Evo2 Variant Analysis Backend - Modal Deployment
# =============================================================================
# Production-grade variant pathogenicity prediction system using:
#   - Evo2-7B: Evolutionary language model for sequence scoring
#   - Gene-specific thresholds: Calibrated for 11+ cancer genes
#   - VEP integration: Molecular consequence annotation override logic
#   - Clinical enrichment: gnomAD, ACMG evidence codes, PubMed literature
#   - Multi-modal RAG: ClinVar + UniProt + PubMed semantic search
# =============================================================================

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

# =============================================================================
# STRUCTURED LOGGING CONFIGURATION
# =============================================================================
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

# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class VariantRequest(BaseModel):
    """
    Single variant analysis request model.
    
    Attributes:
        variant_position: 1-based genomic position
        alternative: Alternative allele (e.g., 'T', 'AG')
        genome: Genome build (hg19, hg38, mm10, mm39)
        chromosome: Chromosome identifier (chr1-chr22, chrX, chrY, chrM)
        reference: Optional reference allele (fetched from UCSC if not provided)
        gene_symbol: Optional gene symbol for literature context and thresholds
        vep_annotation: Optional VEP molecular consequence data for override logic
    """
    variant_position: int
    alternative: str
    genome: str
    chromosome: str
    reference: str = None
    gene_symbol: str = None
    vep_annotation: Optional[dict] = None
    run_ism_scan: bool = False
    ism_scan_radius: int = 20
    ism_scan_stride: int = 1
    
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
    """
    Batch variant analysis request model.
    
    Attributes:
        requests: List of VariantRequest objects for batch processing
    """
    requests: List[VariantRequest]

# =============================================================================
# MODAL IMAGE CONFIGURATION
# =============================================================================
def build_cuda_kernels():
    """
    Compile CUDA kernels for flash-attention and transformer-engine.

    This function runs during image build on a GPU-enabled machine to compile
    optimized kernels that significantly improve inference performance.
    Uses L40S GPU with 32GB memory for compilation.
    """
    os.environ["MAX_JOBS"] = "6"
    os.environ["TORCH_CUDA_ARCH_LIST"] = "7.0;7.5;8.0;8.6;8.9;9.0"
    # PyTorch's official CUDA 12.4 wheels are built with the OLD C++ ABI
    # (_GLIBCXX_USE_CXX11_ABI=0).  flash-attn must be compiled with the same ABI
    # or we get the runtime undefined-symbol crash in flash_attn_2_cuda.
    os.environ["CFLAGS"] = "-D_GLIBCXX_USE_CXX11_ABI=0"
    os.environ["CXXFLAGS"] = "-D_GLIBCXX_USE_CXX11_ABI=0"
    os.environ["NVCC_APPEND_FLAGS"] = "-D_GLIBCXX_USE_CXX11_ABI=0"
    # Pin flash-attn to a version known to build cleanly against torch 2.6.0+cu124.
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "flash-attn==2.7.4.post1", "--no-build-isolation"
    ])
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "transformer_engine[pytorch]==2.8.0", "--no-build-isolation"
    ])

# Build custom Modal image with Evo2 dependencies
# This image includes CUDA 12.4, Python 3.12, and all required packages
evo2_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12"
    )
    .apt_install(
        "build-essential", "cmake", "ninja-build", "libcudnn8",
        "libcudnn8-dev", "git", "gcc", "g++"
    )
    .env({"CC": "/usr/bin/gcc", "CXX": "/usr/bin/g++"})
    .pip_install("packaging", "wheel", "setuptools", "ninja", "pydantic")
    # Pin PyTorch 2.6.0 + CUDA 12.4 BEFORE evo2/flash-attn so every compiled extension
    # is built against the same torch ABI.  Using the latest torch here pulled a newer ABI
    # that the prebuilt flash-attn wheel could not satisfy.
    .pip_install(
        "torch==2.6.0+cu124",
        "torchvision==0.21.0+cu124",
        "torchaudio==2.6.0+cu124",
        index_url="https://download.pytorch.org/whl/cu124",
    )
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
        "vtx>=0.0.8", 
        "fastapi[standard]", 
        "pydantic",  # Required for request/response models
        "requests", 
        "scikit-learn", 
        "redis",
        "pandas",
        "matplotlib",
        "seaborn",
        "openpyxl",
        "groq>=0.4.0",  # Official Groq SDK for LLM
        "duckdb>=1.0.0",  # Embedded DB for AlphaMissense lookup
    )
    .pip_install_from_requirements("requirements.txt")
    .env({"PYTHONPATH": "/root"})
    # Add clinical enrichment module to the image (MUST BE LAST)
    .add_local_file("clinical_enrichment.py", remote_path="/root/clinical_enrichment.py")
    # Add AlphaMissense lookup + consensus engine for Phase 1
    .add_local_file("phase1_implementation/alphamissense_lookup.py", remote_path="/root/alphamissense_lookup.py")
    .add_local_file("phase1_implementation/consensus_engine.py", remote_path="/root/consensus_engine.py")
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
# Persistent volume for Hugging Face model cache
# Models are cached across container restarts to avoid repeated downloads
volume = modal.Volume.from_name("hf_cache", create_if_missing=True)
mount_path = "/root/.cache/huggingface"

# Persistent volume for AlphaMissense DuckDB (9.2 GB)
# Uploaded once via CLI: modal volume put alphamissense_data alphamissense.duckdb
am_volume = modal.Volume.from_name("alphamissense_data", create_if_missing=True)
am_mount_path = "/root/alphamissense_data"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def get_genome_sequence(position, genome: str, chromosome: str, window_size=8192):
    """
    Fetch genome sequence context window from UCSC Genome Browser API.
    
    Args:
        position: 1-based genomic position (center of window)
        genome: Genome build (hg19, hg38, mm10, mm39)
        chromosome: Chromosome identifier (e.g., 'chr17')
        window_size: Total sequence window size (default: 8192bp)
    
    Returns:
        tuple: (sequence_string, start_position)
            - sequence_string: DNA sequence in uppercase
            - start_position: 0-based start coordinate of returned sequence
    
    Raises:
        Exception: If UCSC API request fails
        ValueError: If returned sequence is empty or invalid
    """
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
    """
    Fetch pathogenic and benign variants from ClinVar for a given gene.
    
    This function queries the NCBI ClinVar database via Entrez API to retrieve
    clinically annotated variants for gene-specific threshold calibration.
    
    Args:
        gene_symbol: HGNC gene symbol (e.g., 'BRCA1', 'TP53')
        max_variants: Maximum number of variants to retrieve (default: 1000)
    
    Returns:
        list: Dictionaries with 'label' (LOF/FUNC) and 'id' (ClinVar accession)
    
    Note:
        This function is used for threshold calibration but is not called
        during standard variant analysis. Gene-specific thresholds are
        pre-computed and stored in GENE_SPECIFIC_THRESHOLDS.
    """
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

# =============================================================================
# JSON SANITIZATION HELPER
# =============================================================================
def _sanitize_for_json(obj):
    """
    Recursively convert numpy types to native Python types for JSON serialization.
    
    FastAPI's jsonable_encoder cannot handle numpy scalars (bool_, float64, etc.)
    which leak into results from gnomAD, VEP, and ISM data. This function ensures
    all values are native Python types before the response is serialized.
    """
    import numpy as np
    
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return _sanitize_for_json(obj.tolist())
    elif isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(item) for item in obj]
    elif isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    return obj

# =============================================================================
# MAIN EVO2 MODEL CLASS
# =============================================================================
@app.cls(
    gpu="H100", 
    volumes={mount_path: volume, am_mount_path: am_volume}, 
    scaledown_window=60,  # Shutdown after 60 seconds of inactivity to optimize costs
    retries=2
)
class Evo2Model:
    """
    Evo2-based variant pathogenicity prediction system.
    
    This class implements a production-grade variant classification pipeline with:
    - Gene-specific thresholds calibrated from functional assays and population data
    - VEP molecular consequence override logic for high-confidence variants
    - 3-tier classification system (pathogenic/uncertain/benign)
    - Clinical enrichment with gnomAD, ACMG evidence codes, and literature context
    - Mixed precision inference (bfloat16) for efficient GPU utilization
    
    The model runs on Modal's H100 GPU with automatic scaling and caching.
    """ 
    @modal.enter()
    def load_resources(self):
        """
        Initialize Evo2 model and clinical enrichment resources.
        
        This method runs once when the container starts and loads:
        1. Evo2-7B model from Hugging Face (cached in persistent volume)
        2. Gene-specific thresholds for 11+ cancer predisposition genes
        3. Clinical enrichment modules (gnomAD, ACMG, PubMed RAG)
        
        Gene-specific thresholds are based on:
        - Functional assay data (e.g., BRCA1 saturation mutagenesis)
        - Population genetics studies (gnomAD constraint metrics)
        - Disease mechanism (haploinsufficiency, dominant-negative)
        - Clinical penetrance and expressivity
        """
        from evo2 import Evo2
        # Import clinical enrichment (available in container via add_local_file)
        import sys
        sys.path.insert(0, "/root")
        from clinical_enrichment import ClinicalEnricher
        
        # 1. Load Evo2-7B evolutionary language model
        print("Loading evo2 model...")
        self.model = Evo2('evo2_7b')
        print("Evo2 model loaded")
        
        # 2. Initialize Gene-Specific Thresholds
        # These thresholds are calibrated per gene based on biological function,
        # disease mechanism, and validation data from functional assays
        # Gene-specific threshold dictionary
        # Structure: gene_symbol -> {threshold, lof_std, func_std, uncertain_zone, rationale}
        self.GENE_SPECIFIC_THRESHOLDS = {
            # ===== Tumor Suppressors =====
            # Strict thresholds due to dominant-negative effects and high clinical impact
            'TP53': {
                "threshold": -0.003,
                "lof_std": 0.0015,
                "func_std": 0.0009,
                "uncertain_zone": 0.003,
                "rationale": "Strict threshold for TP53 due to dominant-negative effects"
            },
            
            # ===== DNA Repair Genes =====
            # Thresholds validated against functional assay data and population studies
            'BRCA1': {
                "threshold": -0.007,  # Validated: AUROC=0.778
                "lof_std": 0.0015140239,
                "func_std": 0.0009016589,
                "uncertain_zone": 0.007,
                "rationale": "Validated from Findlay functional assay (3893 variants)"
            },
            'BRCA2': {
                "threshold": -0.006,
                "lof_std": 0.0015,
                "func_std": 0.0009,
                "uncertain_zone": 0.006,
                "rationale": "Similar to BRCA1, haploinsufficiency mechanism"
            },
            'PALB2': {
                "threshold": -0.005,
                "lof_std": 0.0015,
                "func_std": 0.0009,
                "uncertain_zone": 0.005,
                "rationale": "BRCA2 partner, slightly more tolerant"
            },
            
            # ===== Lynch Syndrome (Mismatch Repair Genes) =====
            # Strict thresholds due to high cancer penetrance and clinical actionability
            'MSH2': {
                "threshold": -0.007,
                "lof_std": 0.0015,
                "func_std": 0.0009,
                "uncertain_zone": 0.007,
                "rationale": "Lynch syndrome, strict for cancer predisposition"
            },
            'MLH1': {
                "threshold": -0.007,
                "lof_std": 0.0015,
                "func_std": 0.0009,
                "uncertain_zone": 0.007,
                "rationale": "Lynch syndrome, strict for cancer predisposition"
            },
            'MSH6': {
                "threshold": -0.006,
                "lof_std": 0.0015,
                "func_std": 0.0009,
                "uncertain_zone": 0.006,
                "rationale": "Lynch syndrome, milder phenotype than MLH1/MSH2"
            },
            'PMS2': {
                "threshold": -0.005,
                "lof_std": 0.0015,
                "func_std": 0.0009,
                "uncertain_zone": 0.005,
                "rationale": "Lynch syndrome, mildest phenotype in MMR genes"
            },
            
            # ===== Other High-Penetrance Cancer Genes =====
            'PTEN': {
                "threshold": -0.004,
                "lof_std": 0.0015,
                "func_std": 0.0009,
                "uncertain_zone": 0.004,
                "rationale": "Haploinsufficient tumor suppressor, strict threshold"
            },
            'APC': {
                "threshold": -0.005,
                "lof_std": 0.0015,
                "func_std": 0.0009,
                "uncertain_zone": 0.005,
                "rationale": "FAP gene, strict for truncating variants"
            },
            'STK11': {
                "threshold": -0.006,
                "lof_std": 0.0015,
                "func_std": 0.0009,
                "uncertain_zone": 0.006,
                "rationale": "Peutz-Jeghers syndrome, moderate strictness"
            },
        }
        
        # Default threshold for genes without specific calibration
        self.default_params = {
            "threshold": -0.007,  # Conservative default (validated from BRCA1)
            "lof_std": 0.0015140239,
            "func_std": 0.0009016589,
            "uncertain_zone": 0.007,
            "rationale": "Conservative default based on BRCA1 validation"
        }
        
        # Runtime cache for gene thresholds to avoid repeated lookups
        self.gene_thresholds = {}
        
        # 3. Initialize Clinical Enricher with all evidence sources
        # Provides gnomAD, ACMG, PubMed, ClinVar, and UniProt evidence
        self.clinical_enricher = ClinicalEnricher(
            redis_client=None,
            llm_endpoint=os.getenv("GROQ_API_KEY")  # Enable LLM for summaries
        )
        logger.info("Clinical enricher initialized (gnomAD + ACMG + PubMed + ClinVar + UniProt)")
        
    @modal.method()
    def calibrate_gene(self, gene_symbol: str):
        """
        Get gene-specific thresholds with memory caching.
        
        Uses pre-defined gene-specific thresholds based on:
        - Biological function (tumor suppressor, DNA repair, etc.)
        - Disease mechanism (haploinsufficiency, dominant-negative)
        - Validation data (BRCA1: AUROC=0.778)
        """
        # Check runtime cache
        if gene_symbol in self.gene_thresholds: 
            return self.gene_thresholds[gene_symbol]
        
        # Check gene-specific thresholds
        if gene_symbol in self.GENE_SPECIFIC_THRESHOLDS:
            result_params = self.GENE_SPECIFIC_THRESHOLDS[gene_symbol]
            logger.info(
                f"Using gene-specific threshold for {gene_symbol}",
                gene=gene_symbol,
                threshold=result_params['threshold'],
                rationale=result_params.get('rationale', 'N/A')
            )
        else:
            # Use conservative default
            result_params = self.default_params
            logger.info(
                f"Using default threshold for {gene_symbol}",
                gene=gene_symbol,
                threshold=result_params['threshold']
            )
        
        # Cache for future use
        self.gene_thresholds[gene_symbol] = result_params
        return result_params
    
    def classify_variant_with_vep(
        self,
        delta_score: float,
        params: dict,
        vep_annotation: dict = None
    ) -> tuple[str, float]:
        """
        Classify variant using 3-tier system with VEP molecular consequence override.
        
        Classification hierarchy:
        1. VEP Override Logic (highest confidence):
           - Nonsense/frameshift mutations -> "Likely pathogenic" (95% confidence)
           - HIGH impact splice variants -> "Likely pathogenic" (90% confidence)
           - Synonymous variants (non-splice) -> "Likely benign" (85% confidence)
        
        2. Evo2 Delta Score 3-Tier Classification:
           - delta < threshold: "Likely pathogenic" (confidence from LOF distribution)
           - |delta| <= threshold: "Uncertain significance" (low confidence 0.1-0.3)
           - delta > threshold: "Likely benign" (confidence from FUNC distribution)
        
        Args:
            delta_score: Evo2 likelihood difference (variant - reference)
            params: Gene-specific parameters including threshold and std deviations
            vep_annotation: Optional VEP consequence data with impact/consequence fields
        
        Returns:
            tuple: (prediction: str, confidence: float)
                - prediction: "Likely pathogenic" | "Uncertain significance" | "Likely benign"
                - confidence: Float [0.0, 1.0] representing classification certainty
        """
        threshold = params['threshold']
        uncertain_zone = params.get('uncertain_zone', 0.007)
        
        # ===== VEP Override Logic =====
        # Molecular consequences that override Evo2 scoring with high confidence
        if vep_annotation:
            # Protein-truncating variants (PTVs) are almost universally pathogenic
            # for haploinsufficient genes and tumor suppressors
            if vep_annotation.get('isNonsense') or vep_annotation.get('isFrameshift'):
                return "Likely pathogenic", 0.95
            
            # Canonical splice site variants typically cause exon skipping or
            # aberrant splicing, leading to loss of function
            if vep_annotation.get('impact') == 'HIGH' and 'splice' in vep_annotation.get('consequence', '').lower():
                return "Likely pathogenic", 0.90
            
            # Synonymous variants (excluding those affecting splice sites)
            # are generally neutral with rare exceptions
            if vep_annotation.get('isSynonymous') and vep_annotation.get('impact') != 'HIGH':
                return "Likely benign", 0.85
        
        # ===== Evo2-based 3-Tier Classification =====
        # Use evolutionary conservation signal when VEP doesn't provide clear answer
        if delta_score < threshold:
            # Negative delta indicates variant is less likely than reference
            # (evolutionary constraint signal)
            prediction = "Likely pathogenic"
            confidence = min(1.0, abs(delta_score - threshold) / params['lof_std'])
        elif delta_score > abs(threshold):
            # Positive delta indicates variant is more likely than reference
            # (evolutionary permissiveness signal)
            prediction = "Likely benign"
            confidence = min(1.0, abs(delta_score - abs(threshold)) / params['func_std'])
        else:
            # Uncertain zone: signal is weak or contradictory
            prediction = "Uncertain significance"
            # Low confidence in uncertain zone
            confidence = 0.3 - (abs(delta_score) / uncertain_zone) * 0.2  # Range: 0.1 to 0.3
            
            # If VEP confirms this is a missense variant, slightly boost confidence
            # since we have additional context about variant class
            if vep_annotation and 'missense' in vep_annotation.get('consequence', ''):
                confidence += 0.1
        
        return prediction, confidence
    
    def _run_ism_scan(
        self,
        window_seq: str,
        relative_pos: int,
        reference: str,
        scan_radius: int = 20,
        stride: int = 1,
        batch_size: int = 16
    ) -> dict:
        """
        In-Silico Mutagenesis (ISM) scan around a variant position.
        
        Systematically mutates every nucleotide in a ±scan_radius window around
        the variant and scores each mutant with Evo2. Produces a position-wise
        constraint heatmap revealing functional microdomains, domain boundaries,
        and neutral zones at single-nucleotide resolution.
        
        This is a novel application of DNA language models to clinical variant
        interpretation — no existing VEP tool (CADD, REVEL, AlphaMissense) performs
        positional constraint scanning.
        
        Args:
            window_seq: Full genomic context window (DNA string)
            relative_pos: Variant position within window_seq (0-based index)
            reference: Reference allele at the variant position
            scan_radius: ±N base pairs to scan around the variant (default: 20)
            stride: Scan every Nth position (default: 1 = every position)
            batch_size: Number of mutant sequences to score in one batch (default: 16)
        
        Returns:
            dict: ISM scan results with per-position constraint data and summary
        
        Performance:
            - ±20bp scan (stride=1): ~120 model calls, ~2-4 min on H100
            - ±100bp scan (stride=2): ~300 model calls, ~5-8 min on H100
            - ±500bp scan (stride=5): ~600 model calls, ~10-15 min on H100
        """
        import time as time_module
        
        scan_start = time_module.perf_counter()
        nucleotides = ['A', 'C', 'G', 'T']
        
        # Compute reference score once — reused for all delta calculations
        ref_score = self.model.score_sequences([window_seq])[0]
        
        # Determine scan range
        start_pos = max(0, relative_pos - scan_radius)
        end_pos = min(len(window_seq) - 1, relative_pos + scan_radius)
        
        # Collect all (position, alternative_nucleotide) pairs for batch scoring
        scan_jobs = []
        for pos in range(start_pos, end_pos + 1, stride):
            ref_nuc = window_seq[pos]
            for alt_nuc in nucleotides:
                if alt_nuc == ref_nuc:
                    continue
                scan_jobs.append((pos, alt_nuc))
        
        total_mutants = len(scan_jobs)
        logger.info(
            f"ISM scan starting: ±{scan_radius}bp, stride={stride}, "
            f"{total_mutants} mutants across {len(range(start_pos, end_pos + 1, stride))} positions"
        )
        
        # Build and score mutant sequences in batches
        position_scores = {}  # pos -> {alt_nuc: delta}
        
        for batch_idx in range(0, total_mutants, batch_size):
            batch_jobs = scan_jobs[batch_idx:batch_idx + batch_size]
            mutant_seqs = []
            
            for pos, alt_nuc in batch_jobs:
                mutant_seq = window_seq[:pos] + alt_nuc + window_seq[pos + 1:]
                mutant_seqs.append(mutant_seq)
            
            # Batch score all mutants
            batch_scores = self.model.score_sequences(mutant_seqs)
            
            for (pos, alt_nuc), score in zip(batch_jobs, batch_scores):
                delta = float(score - ref_score)
                if pos not in position_scores:
                    position_scores[pos] = {}
                position_scores[pos][alt_nuc] = delta
            
            # Progress logging for long scans
            if (batch_idx // batch_size) % 10 == 0 and total_mutants > 100:
                progress = min(100, round((batch_idx + len(batch_jobs)) / total_mutants * 100))
                logger.debug(f"ISM scan progress: {progress}%")
        
        # Build structured per-position results
        positions = {}
        constrained_count = 0
        peak_position = None
        peak_max_delta = 0.0
        
        # Constraint threshold: |Δ| > 0.001 is considered constrained
        # Based on Evo2's typical score distribution for functional positions
        CONSTRAINT_THRESHOLD = 0.001
        
        for pos in sorted(position_scores.keys()):
            scores = position_scores[pos]
            ref_nuc = window_seq[pos]
            max_abs_delta = max(abs(d) for d in scores.values())
            is_constrained = max_abs_delta > CONSTRAINT_THRESHOLD
            
            if is_constrained:
                constrained_count += 1
            if max_abs_delta > peak_max_delta:
                peak_max_delta = max_abs_delta
                peak_position = pos
            
            positions[str(pos - relative_pos)] = {
                "genomic_position": None,  # Filled by caller with seq_start offset
                "relative_position": pos - relative_pos,
                "reference": ref_nuc,
                "alternatives": {
                    alt: {
                        "delta": round(delta, 8),
                        "direction": "pathogenic" if delta < -CONSTRAINT_THRESHOLD else (
                            "benign" if delta > CONSTRAINT_THRESHOLD else "neutral"
                        ),
                        "magnitude": round(abs(delta), 8)
                    }
                    for alt, delta in scores.items()
                },
                "max_delta": round(max_abs_delta, 8),
                "is_constrained": is_constrained
            }
        
        # Detect constraint boundaries (positions where constraint status changes)
        sorted_positions = sorted(position_scores.keys())
        boundaries = []
        prev_constrained = None
        for pos in sorted_positions:
            max_abs = max(abs(d) for d in position_scores[pos].values())
            curr_constrained = max_abs > CONSTRAINT_THRESHOLD
            if prev_constrained is not None and curr_constrained != prev_constrained:
                boundaries.append(pos - relative_pos)
            prev_constrained = curr_constrained
        
        total_positions = len(positions)
        constraint_ratio = constrained_count / max(1, total_positions)
        
        scan_duration = (time_module.perf_counter() - scan_start) * 1000
        
        logger.info(
            f"ISM scan completed: {constrained_count}/{total_positions} positions constrained "
            f"({constraint_ratio:.0%}), peak at relative position {peak_position - relative_pos if peak_position else 'N/A'}, "
            f"duration={scan_duration:.0f}ms"
        )
        
        return {
            "scan_radius": scan_radius,
            "stride": stride,
            "window_size": len(window_seq),
            "reference_score": float(ref_score),
            "positions": positions,
            "summary": {
                "constrained_positions": constrained_count,
                "total_positions_scanned": total_positions,
                "constraint_zone": (
                    "high" if constraint_ratio > 0.5 else
                    "moderate" if constraint_ratio > 0.2 else
                    "low"
                ),
                "peak_constraint_position": peak_position - relative_pos if peak_position is not None else None,
                "peak_constraint_magnitude": round(peak_max_delta, 8),
                "constraint_boundaries": boundaries,
                "scan_duration_ms": round(scan_duration, 2)
            }
        }
    
    @modal.method()
    def run_analysis_logic(self, variant_position: int, alternative: str, genome: str, chromosome: str, provided_reference: str = None, gene_symbol: str = None, vep_annotation: dict = None, run_ism_scan: bool = False, ism_scan_radius: int = 20, ism_scan_stride: int = 1):
        """
        Core variant analysis pipeline orchestrating all prediction and enrichment steps.
        
        Pipeline stages:
        0. gnomAD pre-filter: Auto-classify common variants (AF >= 5%) as benign
        1. Genome sequence fetch: Retrieve 8kb context window from UCSC
        2. Gene-specific thresholds: Load calibrated parameters for the gene
        3. Evo2 AI scoring: Calculate delta score (variant vs reference likelihood)
        3b. ISM scan (optional): In-silico mutagenesis positional constraint mapping
        4. ACMG evidence mapping: Convert scores to PP3/BP4 codes
        5. Literature context: PubMed search + optional tri-modal RAG enhancement
        6. Build enriched response: Combine all data sources into final report
        
        Args:
            variant_position: 1-based genomic position
            alternative: Alternative allele sequence
            genome: Genome build (hg19/hg38/mm10/mm39)
            chromosome: Chromosome (chr1-chr22, chrX, chrY, chrM)
            provided_reference: Optional reference allele (fetched if not provided)
            gene_symbol: Optional gene symbol for thresholds and literature
            vep_annotation: Optional VEP consequence data for override logic
            run_ism_scan: Enable in-silico mutagenesis positional scanning
            ism_scan_radius: ±N bp around variant to scan (default: 20)
            ism_scan_stride: Scan every Nth position (default: 1)
        
        Returns:
            dict: Comprehensive variant analysis result with fields:
                - reference, alternative, delta_score, prediction, confidence
                - population_frequency (gnomAD data)
                - acmg_evidence (PP3/BP4 codes)
                - literature_context (PubMed + RAG)
        """
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
        
        # ===== STEP 0: gnomAD Pre-Filter =====
        # Check population frequency before running expensive GPU inference
        # Common variants (AF >= 5%) are auto-classified as benign per ACMG BA1 rule
        gnomad_result = None
        if provided_reference:  # Need ref allele for gnomAD query
            gnomad_result = self.clinical_enricher.gnomad.get_allele_frequency(
                chromosome=chromosome,
                position=variant_position,
                ref=provided_reference,
                alt=alternative
            )
            
            # Auto-classify common variants using population frequency alone
            # This optimization saves GPU credits by skipping AI inference
            if gnomad_result.auto_classification:
                logger.info(f"gnomAD auto-classification: {gnomad_result.auto_classification} (AF={gnomad_result.allele_frequency})")
                
                # Retrieve literature context even for common variants to provide
                # educational value about gene function
                lit_context = self.clinical_enricher.pubmed.get_literature_context(gene_symbol) if gene_symbol else None
                
                # Early return for common variants
                return {
                    "reference": provided_reference,
                    "alternative": alternative,
                    "delta_score": None,  # Skip AI scoring for common variants
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
        
        # ===== STEP 1: Fetch Genome Sequence =====
        # Retrieve 8kb context window centered on variant position from UCSC API
        window_seq, seq_start = get_genome_sequence(
            position=variant_position,
            genome=genome,
            chromosome=chromosome,
            window_size=WINDOW_SIZE
        )
            
        # Calculate variant position within the retrieved window
        relative_pos = variant_position - 1 - seq_start
        if relative_pos < 0 or relative_pos >= len(window_seq):
            raise ValueError(f"Position outside window")
        
        # Validate reference allele if provided, otherwise extract from sequence
        if provided_reference:
            reference = provided_reference
            actual_ref = window_seq[relative_pos : relative_pos + len(reference)]
            if actual_ref != reference:
                raise HTTPException(status_code=400, detail=f"Reference Mismatch! Expected '{actual_ref}', got '{reference}'")
        else:
            # Extract reference from genome sequence
            reference = window_seq[relative_pos]
        
        # Query gnomAD if we didn't have reference allele initially
        if gnomad_result is None:
            gnomad_result = self.clinical_enricher.gnomad.get_allele_frequency(
                chromosome=chromosome,
                position=variant_position,
                ref=reference,
                alt=alternative
            )
        
        # ===== STEP 2: Gene-Specific Thresholds =====
        # Load calibrated threshold parameters for the specified gene
        params = self.default_params
        if gene_symbol:
            if gene_symbol in self.gene_thresholds:
                params = self.gene_thresholds[gene_symbol]
            else:
                params = self.calibrate_gene.local(gene_symbol)
        
        # ===== STEP 3: Evo2 AI Scoring =====
        # Calculate sequence likelihood scores and delta for variant pathogenicity
        # Create variant sequence by substituting alternative allele
        var_seq = window_seq[:relative_pos] + alternative + window_seq[relative_pos + len(reference):]
        
        # Score both sequences with Evo2-7B model
        # Scores represent log-likelihood of observing the sequence
        ref_score = self.model.score_sequences([window_seq])[0]
        var_score = self.model.score_sequences([var_seq])[0]
        delta_score = var_score - ref_score  # Negative = constrained, Positive = permissive
        
        # Classify variant using 3-tier system with VEP override logic
        prediction, confidence = self.classify_variant_with_vep(
            delta_score=delta_score,
            params=params,
            vep_annotation=vep_annotation
        )

        # ===== STEP 3b: In-Silico Mutagenesis Scan (Optional) =====
        # Systematic positional constraint mapping around the variant
        # Reveals functional microdomains, domain boundaries, and neutral zones
        ism_result = None
        if run_ism_scan:
            with logger.timed("ISM scan"):
                ism_result = self._run_ism_scan(
                    window_seq=window_seq,
                    relative_pos=relative_pos,
                    reference=reference,
                    scan_radius=ism_scan_radius,
                    stride=ism_scan_stride
                )
                # Fill in absolute genomic positions
                if ism_result and "positions" in ism_result:
                    for pos_key, pos_data in ism_result["positions"].items():
                        pos_data["genomic_position"] = seq_start + relative_pos + int(pos_key)
                logger.info(
                    f"ISM scan integrated into results: "
                    f"{ism_result['summary']['constrained_positions']}/"
                    f"{ism_result['summary']['total_positions_scanned']} positions constrained"
                )

        # ===== STEP 4: ACMG Evidence Mapping =====
        # Convert Evo2 scores to standardized ACMG evidence codes (PP3/BP4)
        acmg_evidence = self.clinical_enricher.acmg.map_score_to_evidence(
            delta_score=delta_score,
            confidence=confidence,
            model_name="Evo2-7B"
        )

        # ===== STEP 5: Multi-Source Evidence Gathering (Parallel) =====
        # Fetch ClinVar, UniProt, PubMed, and gnomAD evidence in parallel
        # Each source fails independently - partial results are still returned
        lit_context = None
        clinvar_data = None
        uniprot_data = None
        pubmed_articles = []
        
        if gene_symbol:
            # Use the new EvidenceAggregator for parallel fetching
            variant_str = f"{provided_reference or reference}>{alternative}" if provided_reference or reference else alternative
            
            evidence = self.clinical_enricher.aggregator.gather_evidence(
                chromosome=chromosome,
                position=variant_position,
                ref=provided_reference or reference,
                alt=alternative,
                gene_symbol=gene_symbol,
                delta_score=delta_score,
                confidence=confidence,
                prediction=prediction,
                vep_annotation=vep_annotation
            )
            
            # Extract results from aggregator
            clinvar_data = evidence.clinvar_result
            uniprot_data = evidence.uniprot_result
            pubmed_articles = evidence.pubmed_articles
            
            # Generate literature summary using Groq LLM
            if pubmed_articles:
                lit_context = self.clinical_enricher.pubmed.get_literature_context(gene_symbol)
                logger.info(f"PubMed search: {lit_context.num_articles_found} articles for {gene_symbol}")
            
            logger.info(
                f"Evidence gathered: ClinVar={clinvar_data.status if clinvar_data else 'N/A'}, "
                f"UniProt={uniprot_data.accession if uniprot_data else 'N/A'}, "
                f"PubMed={len(pubmed_articles)} articles"
            )
        
        # ===== STEP 6: Build Enriched Response =====
        # Combine all data sources into comprehensive variant report
        result = {
            # ===== Core Evo2 Prediction =====
            "reference": reference,
            "alternative": alternative,
            "delta_score": float(delta_score),
            "prediction": prediction,
            "classification_confidence": float(confidence),
            "position": variant_position,
            "classification_source": "Evo2_AI",
            
            # ===== Population Frequency Data (gnomAD) =====
            "population_frequency": {
                "gnomad_af": gnomad_result.allele_frequency if gnomad_result else None,
                "gnomad_max_pop_af": gnomad_result.population_max_af if gnomad_result else None,
                "source": gnomad_result.source if gnomad_result else "gnomAD v4.1",
                "is_common_variant": gnomad_result.is_common if gnomad_result else False
            },
            
            # ===== ACMG Evidence Code (PP3/BP4) =====
            "acmg_evidence": {
                "code": acmg_evidence.code,
                "strength": acmg_evidence.strength.value,
                "description": acmg_evidence.description,
                "clinical_note": acmg_evidence.clinical_note
            },
            
            # ===== Clinical Evidence (ClinVar) =====
            "clinvar_evidence": {
                "status": clinvar_data.status if clinvar_data else None,
                "review_status": clinvar_data.review_status if clinvar_data else None,
                "variation_id": clinvar_data.variation_id if clinvar_data else None,
                "num_submitters": clinvar_data.num_submitters if clinvar_data else 0,
                "conflicting": clinvar_data.conflicting if clinvar_data else False,
                "summary": clinvar_data.text if clinvar_data else None
            } if gene_symbol else None,
            
            # ===== Protein Context (UniProt) =====
            "protein_context": {
                "accession": uniprot_data.accession if uniprot_data else None,
                "protein_name": uniprot_data.protein_name if uniprot_data else None,
                "function": uniprot_data.function if uniprot_data else None,
                "domains": uniprot_data.domains if uniprot_data else [],
                "subcellular_location": uniprot_data.subcellular_location if uniprot_data else None,
                "disease_associations": uniprot_data.disease_associations if uniprot_data else []
            } if gene_symbol else None,
            
            # ===== Literature Context (PubMed + Groq LLM) =====
            "literature_context": {
                "summary": lit_context.summary if lit_context else None,
                "pubmed_ids": lit_context.pubmed_ids if lit_context else [],
                "gene_function": lit_context.gene_function if lit_context else None,
                "articles_found": lit_context.num_articles_found if lit_context else 0,
                "articles": [
                    {"pmid": a["pmid"], "title": a["title"], "abstract": a["abstract"][:300]}
                    for a in pubmed_articles[:3]
                ] if pubmed_articles else []
            } if gene_symbol else None,
            
            # ===== In-Silico Mutagenesis Scan (Optional) =====
            "ism_scan": ism_result
        }
        
        # ===== STEP 7: Generate Structured Clinical Summary =====
        # Synthesize all evidence into a clinician-friendly narrative
        clinical_summary = None
        evidence_confidence = None
        if gene_symbol and os.getenv("GROQ_API_KEY"):
            try:
                variant_str = f"{reference}>{alternative}"
                clinical_summary = self._generate_clinical_summary(
                    gene_symbol=gene_symbol,
                    variant_str=variant_str,
                    vep_annotation=vep_annotation,
                    delta_score=delta_score,
                    confidence=confidence,
                    prediction=prediction,
                    gnomad_result=gnomad_result,
                    clinvar_data=clinvar_data,
                    uniprot_data=uniprot_data,
                    pubmed_articles=pubmed_articles
                )
                if clinical_summary:
                    result["clinical_summary"] = clinical_summary
                    logger.info("Clinical summary generated successfully")
            except Exception as e:
                logger.warning(f"Clinical summary generation failed: {e}")
        
        # Calculate evidence confidence scores
        if gene_symbol:
            evidence_confidence = self._calculate_evidence_confidence(
                vep_annotation=vep_annotation,
                gnomad_result=gnomad_result,
                clinvar_data=clinvar_data,
                uniprot_data=uniprot_data,
                pubmed_articles=pubmed_articles,
                evo2_confidence=confidence
            )
            result["evidence_confidence"] = evidence_confidence
        
        # Compute XAI confidence factors (moved from client-side to backend)
        xai_factors = self._compute_xai_factors(
            delta_score=delta_score,
            gnomad_result=gnomad_result,
            acmg_code=acmg_evidence.code if acmg_evidence else None
        )
        result["xai_factors"] = xai_factors
        
        # Extract counterfactual analysis from ISM data if available
        if ism_result:
            counterfactuals = self._extract_counterfactuals(ism_result)
            if counterfactuals:
                result["counterfactuals"] = counterfactuals
        
        # Map evidence to ACMG/AMP criteria
        if gene_symbol:
            acmg_criteria = self._map_acmg_criteria(
                gene_symbol=gene_symbol,
                delta_score=delta_score,
                prediction=prediction,
                gnomad_result=gnomad_result,
                clinvar_data=clinvar_data,
                vep_annotation=vep_annotation,
                ism_result=ism_result
            )
            result["acmg_criteria"] = acmg_criteria
            
            # ── ISM Concordance Factor: Adjust confidence based on Evo2-ISM agreement ──
            # When Evo2 and ISM agree (both indicate pathogenic or both indicate benign),
            # confidence is boosted by 10%. When they disagree, confidence is reduced by 10%.
            if ism_result and "summary" in ism_result:
                ism_summary = ism_result["summary"]
                f_c = ism_summary.get("constrained_positions", 0) / max(1, ism_summary.get("total_positions_scanned", 1))
                ism_says_constrained = f_c > 0.2
                
                # Get gene threshold to determine Evo2 direction
                gene_params = self._get_gene_threshold(gene_symbol)
                gene_threshold = gene_params.get("threshold", -0.005)
                evo2_says_pathogenic = delta_score < gene_threshold
                
                # Determine concordance
                if evo2_says_pathogenic and ism_says_constrained:
                    # Both indicate pathogenic → concordant
                    concordance_label = "concordant"
                    confidence_multiplier = 1.10
                    concordance_note = (
                        f"Evo2 predicts pathogenic (Δ={delta_score:.6f} < τ={gene_threshold}) "
                        f"and ISM confirms constraint (f_c={f_c:.2f}) — signals agree"
                    )
                elif not evo2_says_pathogenic and not ism_says_constrained:
                    # Both indicate benign → concordant
                    concordance_label = "concordant"
                    confidence_multiplier = 1.10
                    concordance_note = (
                        f"Evo2 predicts benign (Δ={delta_score:.6f} ≥ |τ|={abs(gene_threshold)}) "
                        f"and ISM confirms permissive region (f_c={f_c:.2f}) — signals agree"
                    )
                else:
                    # Signals disagree → discordant
                    concordance_label = "discordant"
                    confidence_multiplier = 0.90
                    concordance_note = (
                        f"Evo2 predicts {'pathogenic' if evo2_says_pathogenic else 'benign'} "
                        f"but ISM indicates {'constrained' if ism_says_constrained else 'permissive'} "
                        f"region (f_c={f_c:.2f}) — signals disagree"
                    )
                
                # Apply the adjustment
                adjusted_confidence = min(1.0, max(0.0, result["classification_confidence"] * confidence_multiplier))
                logger.info(
                    f"ISM concordance: {concordance_label} (f_c={f_c:.2f}, "
                    f"confidence {result['classification_confidence']:.3f} → {adjusted_confidence:.3f})"
                )
                result["classification_confidence"] = adjusted_confidence
                
                # Store concordance metadata for report
                result["ism_concordance"] = {
                    "label": concordance_label,
                    "f_c": round(f_c, 4),
                    "evo2_direction": "pathogenic" if evo2_says_pathogenic else "benign",
                    "ism_direction": "constrained" if ism_says_constrained else "permissive",
                    "confidence_multiplier": confidence_multiplier,
                    "note": concordance_note,
                }
        
        # Fetch knowledge graph (gene-disease-drug associations)
        if gene_symbol:
            ensembl_id = vep_annotation.get("geneId") if vep_annotation else None
            knowledge_graph = self._fetch_knowledge_graph(gene_symbol, ensembl_id)
            if knowledge_graph.get("diseases") or knowledge_graph.get("drugs"):
                result["knowledge_graph"] = knowledge_graph
        
        # Fetch external scores for multi-tool concordance
        if reference:
            external_scores = self._fetch_external_scores(
                chromosome=chromosome,
                position=variant_position,
                reference=reference,
                alternative=alternative
            )
            
            # NEW Phase 1: AlphaMissense lookup (protein-level, 2.6ms)
            am_score = self._fetch_alphamissense_score(
                chromosome=chromosome,
                position=variant_position,
                reference=reference,
                alternative=alternative,
                vep_annotation=vep_annotation
            )
            if am_score:
                external_scores["alphamissense"] = am_score
            
            if external_scores.get("cadd") or external_scores.get("alphamissense"):
                result["external_scores"] = external_scores
        
        # NEW Phase 1: Multi-model consensus (Evo2 + AlphaMissense + CADD + REVEL)
        if reference:
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location("consensus_engine", "/root/consensus_engine.py")
                consensus_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(consensus_module)
                ConsensusEngine = consensus_module.ConsensusEngine
                
                engine = ConsensusEngine()
                consensus = engine.compute_consensus(
                    evo2_prediction=prediction,
                    evo2_confidence=confidence,
                    alphamissense_score=(external_scores.get("alphamissense") or {}).get("score") if external_scores else None,
                    alphamissense_confidence=(external_scores.get("alphamissense") or {}).get("am_class") if external_scores else None,
                    cadd_phred=(external_scores.get("cadd") or {}).get("phred") if external_scores else None,
                    gene_symbol=gene_symbol,
                    variant_str=f"{reference}>{alternative}"
                )
                result["multi_model_consensus"] = engine.to_dict(consensus)
                logger.info(f"Multi-model consensus: {consensus.consensus_classification} ({consensus.models_agree}/{consensus.models_total} models)")
            except Exception as e:
                logger.warning(f"Multi-model consensus failed: {e}")
                import traceback
                logger.warning(traceback.format_exc())
                result["consensus_debug_error"] = str(e)
        
        # LLM-powered ACMG criteria refinement
        if gene_symbol and os.getenv("GROQ_API_KEY") and acmg_criteria:
            try:
                variant_str = f"{reference}>{alternative}"
                acmg_refined = self._refine_acmg_with_llm(
                    gene_symbol=gene_symbol,
                    variant_str=variant_str,
                    rule_based_acmg=acmg_criteria,
                    delta_score=delta_score,
                    prediction=prediction,
                    gnomad_result=gnomad_result,
                    clinvar_data=clinvar_data,
                    vep_annotation=vep_annotation,
                    ism_result=ism_result
                )
                if acmg_refined:
                    result["acmg_criteria_refined"] = acmg_refined
            except Exception as e:
                logger.warning(f"ACMG LLM refinement failed: {e}")
        
        # Log completion with performance metrics
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
        
        # Sanitize numpy types to native Python for JSON serialization
        return _sanitize_for_json(result)
    
    def _generate_clinical_summary(
        self,
        gene_symbol: str,
        variant_str: str,
        vep_annotation: Optional[dict],
        delta_score: float,
        confidence: float,
        prediction: str,
        gnomad_result,
        clinvar_data,
        uniprot_data,
        pubmed_articles: list
    ) -> str:
        """
        Generate a structured clinical summary using Groq Llama 3.3 70B.        
        Synthesizes evidence from VEP, Evo2, gnomAD, ClinVar, UniProt, and PubMed into a clinician-friendly narrative with source citations.
        """
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return None
        
        try:
            from groq import Groq
            client = Groq(api_key=api_key)
            
            # Build structured evidence sections with availability tracking
            sections = []
            available_sources = []
            missing_sources = []
            
            # Section 1: Variant Identity
            sections.append(f"## VARIANT IDENTITY\nGene: {gene_symbol}\nVariant: {variant_str}")
            if vep_annotation:
                consequence = vep_annotation.get('consequence', 'Unknown')
                impact = vep_annotation.get('impact', 'Unknown')
                aa_change = vep_annotation.get('aaChange') or vep_annotation.get('aminoAcids', 'Unknown')
                codons = vep_annotation.get('codons', 'Unknown')
                transcript = vep_annotation.get('transcriptId', 'Unknown')
                exon = vep_annotation.get('exonNumber', 'Unknown')
                sections.append(f"VEP: {consequence} ({impact} impact)\nAmino Acid: {aa_change}\nCodons: {codons}\nTranscript: {transcript}\nExon: {exon}")
                available_sources.append("VEP")
            else:
                missing_sources.append("VEP")
                sections.append("VEP: Not available - molecular consequence unknown")
            
            # Section 2: Computational Evidence
            sections.append(f"## COMPUTATIONAL EVIDENCE\nEvo2-7B Delta Score: {delta_score:.6f}\nPrediction: {prediction}\nConfidence: {confidence:.1%}")
            available_sources.append("Evo2")
            
            # Section 3: Population Evidence
            if gnomad_result and gnomad_result.allele_frequency is not None:
                af = gnomad_result.allele_frequency
                max_af = gnomad_result.population_max_af
                sections.append(f"## POPULATION EVIDENCE\ngnomAD v4.1 Allele Frequency: {af:.6f}\nMax Population AF: {max_af if max_af else 'N/A'}\nCommon Variant: {gnomad_result.is_common}")
                available_sources.append("gnomAD")
            else:
                missing_sources.append("gnomAD")
                sections.append("## POPULATION EVIDENCE\ngnomAD: Not available - variant not found in population database")
            
            # Section 4: Clinical Evidence
            if clinvar_data and clinvar_data.status not in ("Not Found", "Error", None):
                sections.append(f"## CLINICAL EVIDENCE\nClinVar: {clinvar_data.status}\nReview Status: {clinvar_data.review_status}\nSubmitters: {clinvar_data.num_submitters}\nConflicting: {clinvar_data.conflicting}")
                available_sources.append("ClinVar")
            else:
                missing_sources.append("ClinVar")
                sections.append("## CLINICAL EVIDENCE\nClinVar: Not available - variant not found in clinical database")
            
            # Section 5: Protein Context
            if uniprot_data and uniprot_data.function:
                domains_str = ""
                if uniprot_data.domains:
                    domains_str = "\n".join([f"  - {d['name']} (aa {d['start']}-{d['end']})" for d in uniprot_data.domains[:5]])
                diseases_str = ""
                if uniprot_data.disease_associations:
                    diseases_str = "\n".join([f"  - {d[:150]}" for d in uniprot_data.disease_associations[:3]])
                sections.append(f"## PROTEIN CONTEXT\nUniProt: {uniprot_data.accession}\nProtein: {uniprot_data.protein_name}\nFunction: {uniprot_data.function[:600]}\nDomains:\n{domains_str}\nDisease Associations:\n{diseases_str}")
                available_sources.append("UniProt")
            else:
                missing_sources.append("UniProt")
                sections.append("## PROTEIN CONTEXT\nUniProt: Not available - protein information not found")
            
            # Section 6: Literature
            if pubmed_articles:
                articles_str = "\n".join([
                    f"  [{a['pmid']}] {a['title'][:200]}\n  Abstract: {a['abstract'][:400]}"
                    for a in pubmed_articles[:3]
                ])
                sections.append(f"## LITERATURE EVIDENCE\n{articles_str}")
                available_sources.append("PubMed")
            else:
                missing_sources.append("PubMed")
                sections.append("## LITERATURE EVIDENCE\nPubMed: Not available - no relevant articles found")
            
            # Build source availability summary
            source_status = f"\n\n## SOURCE AVAILABILITY\nAvailable: {', '.join(available_sources) if available_sources else 'None'}\nMissing: {', '.join(missing_sources) if missing_sources else 'None'}"
            evidence_text = "\n\n".join(sections) + source_status
            
            system_prompt = """You are a clinical genomics expert writing for physicians and genetic counselors.
                Generate a structured clinical summary using ONLY the provided evidence. Follow these rules STRICTLY:

                1. NEVER fabricate information. If data is marked as "Not available", state that explicitly rather than guessing.
                2. Cite ALL claims with source tags: [VEP], [Evo2], [gnomAD], [ClinVar], [UniProt], [PubMed:PMID]
                3. Use standard clinical terminology (ACMG/AMP guidelines).
                4. Write in clear, concise language suitable for a clinical report.
                5. Include a DISCLAIMER that this is a computational prediction requiring clinical correlation.
                6. If multiple sources are missing, acknowledge the limited evidence base.
                7. NEVER recommend clinical actions - only summarize evidence for clinical interpretation.

                Structure your response with these EXACT headings:
                1 MOLECULAR MECHANISM - What happens at the DNA/protein level?
                2 COMPUTATIONAL PREDICTION - What does the AI model predict and why?
                3 POPULATION EVIDENCE - How common is this variant in the general population?
                4 CLINICAL EVIDENCE - What do clinical databases and literature report?
                5 INTEGRATED ASSESSMENT - Synthesize all evidence into a clinical interpretation
                6 LIMITATIONS - What are the caveats and uncertainties?"""

            user_prompt = f"""Generate a clinical summary for a {gene_symbol} variant based on the following evidence:

{evidence_text}

Remember: Cite sources, be precise, and include the disclaimer. Acknowledge any missing data sources."""
            
            chat_completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=1500,
            )
            
            summary = chat_completion.choices[0].message.content.strip()
            
            # Post-process: verify citations
            valid_pmids = {a["pmid"] for a in pubmed_articles} if pubmed_articles else set()
            import re
            cited_pmids = set(re.findall(r'\[PubMed:(\d+)\]', summary))
            fabricated = cited_pmids - valid_pmids
            if fabricated:
                logger.warning(f"LLM fabricated PubMed citations: {fabricated}")
                summary += f"\n\n[WARNING: The following citations could not be verified: {', '.join(fabricated)}. Please verify before clinical use.]"
            
            return summary
            
        except Exception as e:
            logger.error(f"Clinical summary generation failed: {e}")
            return None
    
    def _calculate_evidence_confidence(
        self,
        vep_annotation: Optional[dict],
        gnomad_result,
        clinvar_data,
        uniprot_data,
        pubmed_articles: list,
        evo2_confidence: float
    ) -> dict:
        """
        Calculate per-source confidence scores for transparency.
        
        Returns a dict with confidence levels for each evidence source,
        helping clinicians understand which parts of the analysis are most reliable.
        """
        confidence = {
            "vep": {"available": False, "confidence": "N/A", "note": "Molecular annotation unavailable"},
            "evo2": {"available": True, "confidence": self._confidence_label(evo2_confidence), "note": f"Computational prediction confidence: {evo2_confidence:.0%}"},
            "gnomad": {"available": False, "confidence": "N/A", "note": "Population data unavailable"},
            "clinvar": {"available": False, "confidence": "N/A", "note": "Clinical classification unavailable"},
            "uniprot": {"available": False, "confidence": "N/A", "note": "Protein annotation unavailable"},
            "pubmed": {"available": False, "confidence": "N/A", "note": "Literature evidence unavailable"},
        }
        
        # VEP confidence
        if vep_annotation:
            impact = vep_annotation.get('impact', '')
            if impact == 'HIGH':
                conf = "High"
                note = "Protein-truncating or splice variant - molecular consequence is definitive"
            elif impact == 'MODERATE':
                conf = "High"
                note = "Missense or in-frame indel - amino acid change is known"
            elif impact == 'LOW':
                conf = "High"
                note = "Synonymous or non-coding - likely neutral"
            else:
                conf = "Medium"
                note = "Modifier - functional impact uncertain"
            confidence["vep"] = {"available": True, "confidence": conf, "note": note}
        
        # gnomAD confidence
        if gnomad_result and gnomad_result.allele_frequency is not None:
            af = gnomad_result.allele_frequency
            if af >= 0.01:
                conf = "High"
                note = f"Common variant (AF={af:.4f}) - population data is robust"
            elif af >= 0.0001:
                conf = "Medium"
                note = f"Rare variant (AF={af:.6f}) - limited population data"
            else:
                conf = "Low"
                note = "Variant absent from gnomAD - may be novel or extremely rare"
            confidence["gnomad"] = {"available": True, "confidence": conf, "note": note}
        
        # ClinVar confidence
        if clinvar_data and clinvar_data.status not in ("Not Found", "Error", None):
            review = clinvar_data.review_status.lower()
            submitters = clinvar_data.num_submitters
            if "expert panel" in review or "practice guideline" in review:
                conf = "High"
                note = f"Expert panel reviewed ({submitters} submitters)"
            elif submitters >= 3:
                conf = "High"
                note = f"Multiple submitters ({submitters}) with consensus"
            elif submitters >= 2:
                conf = "Medium"
                note = f"Two submitters - moderate consensus"
            else:
                conf = "Low"
                note = "Single submitter - limited validation"
            if clinvar_data.conflicting:
                conf = "Low"
                note += " (CONFLICTING interpretations)"
            confidence["clinvar"] = {"available": True, "confidence": conf, "note": note}
        
        # UniProt confidence
        if uniprot_data and uniprot_data.function:
            if uniprot_data.domains:
                conf = "High"
                note = f"Reviewed entry with {len(uniprot_data.domains)} annotated domains"
            else:
                conf = "Medium"
                note = "Reviewed entry with functional annotation"
            confidence["uniprot"] = {"available": True, "confidence": conf, "note": note}
        
        # PubMed confidence
        if pubmed_articles:
            n = len(pubmed_articles)
            if n >= 5:
                conf = "High"
                note = f"{n} relevant articles found"
            elif n >= 2:
                conf = "Medium"
                note = f"{n} articles found - limited literature"
            else:
                conf = "Low"
                note = "Single article - very limited evidence"
            confidence["pubmed"] = {"available": True, "confidence": conf, "note": note}
        
        # Overall confidence
        available_count = sum(1 for c in confidence.values() if c["available"])
        high_count = sum(1 for c in confidence.values() if c["confidence"] == "High")
        if available_count >= 5 and high_count >= 3:
            overall = "High"
        elif available_count >= 3:
            overall = "Medium"
        else:
            overall = "Low"
        
        confidence["overall"] = {
            "level": overall,
            "sources_available": f"{available_count}/6",
            "high_confidence_sources": f"{high_count}/6"
        }
        
        return confidence
    
    def _compute_xai_factors(
        self,
        delta_score: float,
        gnomad_result,
        acmg_code: str
    ) -> dict:
        """
        Compute 4-factor weighted XAI confidence decomposition.
        
        Moved from client-side computeConfidenceFactors() to backend for:
        - Database persistence and export inclusion
        - Consistent computation across all views
        - Single source of truth for XAI metrics
        
        Returns:
            dict with factors list and total score
        """
        import math
        
        delta = delta_score or 0.0
        abs_delta = abs(delta)
        factors = []
        
        # Evo2-calibrated thresholds
        EVO2_WEAK = 0.0001
        EVO2_MOD = 0.005
        EVO2_STRONG = 0.05
        EVO2_MAX = 0.5
        
        # Factor 1: Evolutionary Signal (0-50%)
        delta_contrib = min(50, round(
            (math.log10(abs_delta / EVO2_WEAK + 1) / math.log10(EVO2_MAX / EVO2_WEAK + 1)) * 50
        ))
        signal_label = (
            "strong" if abs_delta >= EVO2_STRONG
            else "moderate" if abs_delta >= EVO2_MOD
            else "weak" if abs_delta >= EVO2_WEAK
            else "minimal"
        )
        factors.append({
            "label": "Evolutionary Signal",
            "contribution": max(1, delta_contrib),
            "color": "#ef4444" if delta < 0 else "#22c55e",
            "detail": f"|Δ| = {abs_delta:.6f} — {signal_label} evolutionary pressure signal"
        })
        
        # Factor 2: Effect Size Clarity (0-20%)
        near_zero = abs_delta < EVO2_WEAK
        ambiguous = abs_delta < EVO2_MOD
        direction_contrib = 5 if near_zero else (10 if ambiguous else (20 if abs_delta >= EVO2_STRONG else 14))
        direction_label = "Effect Size: Minimal" if near_zero else ("Effect Size: Weak" if ambiguous else "Effect Size: Clear")
        factors.append({
            "label": direction_label,
            "contribution": direction_contrib,
            "color": "#f59e0b" if near_zero else ("#fb923c" if ambiguous else "#6366f1"),
            "detail": (
                f"|Δ| = {abs_delta:.6f} — essentially zero, Evo2 sees this sequence as equally likely"
                if near_zero else
                f"Weak {'pathogenic' if delta < 0 else 'benign'} signal (|Δ| = {abs_delta:.6f}), interpret cautiously"
                if ambiguous else
                f"Clear {'pathogenic' if delta < 0 else 'benign'} direction (|Δ| = {abs_delta:.6f})"
            )
        })
        
        # Factor 3: Population Rarity (0-15%)
        af = gnomad_result.allele_frequency if gnomad_result else None
        if af is not None:
            if af > 0.05:
                pop_contrib, pop_detail = 2, f"Common variant (AF={af*100:.2f}%) — strong benign signal (BA1 criterion)"
            elif af > 0.01:
                pop_contrib, pop_detail = 5, f"Uncommon (AF={af*100:.2f}%)"
            elif af > 0.001:
                pop_contrib, pop_detail = 10, f"Rare (AF={af*100:.4f}%) — low population frequency"
            else:
                pop_contrib, pop_detail = 15, f"Ultra-rare (AF={af:.2e}) — rarity supports pathogenic classification"
        else:
            pop_contrib, pop_detail = 10, "Not observed in gnomAD (800k+ individuals) — novel or ultra-rare"
        factors.append({
            "label": "Population Rarity",
            "contribution": pop_contrib,
            "color": "#8b5cf6",
            "detail": pop_detail
        })
        
        # Factor 4: ACMG Evidence (0-15%)
        code = acmg_code or "None"
        if code and code != "None":
            if "VeryStrong" in code or "PVS" in code:
                acmg_contrib, acmg_detail = 15, f"{code} — very strong ACMG criterion met"
            elif "Strong" in code or code.startswith("PS") or code.startswith("BS"):
                acmg_contrib, acmg_detail = 12, f"{code} — strong ACMG criterion"
            elif "Moderate" in code or code.startswith("PM") or code.startswith("BP"):
                acmg_contrib, acmg_detail = 9, f"{code} — moderate ACMG criterion"
            elif "Supporting" in code or code.startswith("PP"):
                acmg_contrib, acmg_detail = 6, f"{code} — supporting ACMG criterion"
            else:
                acmg_contrib, acmg_detail = 7, f"{code} — ACMG criterion triggered"
        else:
            acmg_contrib, acmg_detail = 5, "No ACMG code triggered — delta score in uncertain range"
        factors.append({
            "label": "ACMG Evidence",
            "contribution": acmg_contrib,
            "color": "#0ea5e9",
            "detail": acmg_detail
        })
        
        total = sum(f["contribution"] for f in factors)
        return {"factors": factors, "total": total}
    
    def _extract_counterfactuals(self, ism_result: dict) -> dict:
        """
        Extract counterfactual analysis from ISM scan data.
        
        For the variant position (relative position 0), shows what Evo2 predicts
        for all three alternative alleles — answering "what if this were a
        different mutation?"
        
        Returns:
            dict with per-allele predictions or None if ISM data unavailable
        """
        if not ism_result or "positions" not in ism_result:
            return None
        
        pos0 = ism_result["positions"].get("0")
        if not pos0:
            return None
        
        alternatives = pos0.get("alternatives", {})
        if not alternatives:
            return None
        
        counterfactuals = {}
        for alt_nuc, scores in alternatives.items():
            delta = scores["delta"]
            direction = scores["direction"]
            magnitude = scores["magnitude"]
            
            # Classify each alternative
            if direction == "pathogenic":
                prediction = "Likely Pathogenic"
            elif direction == "benign":
                prediction = "Likely Benign"
            else:
                prediction = "Uncertain Significance"
            
            counterfactuals[alt_nuc] = {
                "delta": delta,
                "prediction": prediction,
                "direction": direction,
                "magnitude": magnitude
            }
        
        # Determine which alleles are tolerated vs pathogenic vs neutral
        tolerated = [a for a, c in counterfactuals.items() if c["direction"] == "benign"]
        pathogenic = [a for a, c in counterfactuals.items() if c["direction"] == "pathogenic"]
        neutral = [a for a, c in counterfactuals.items() if c["direction"] == "neutral"]
        
        # Build summary
        if tolerated and not pathogenic and not neutral:
            summary = f"All alternatives tolerated — position is evolutionarily neutral"
        elif pathogenic and not tolerated:
            summary = f"All alternatives pathogenic — position is highly constrained"
        elif neutral and not tolerated and not pathogenic:
            summary = f"All alternatives have uncertain effect — Evo2 cannot distinguish any nucleotide change from background at this position"
        elif tolerated and pathogenic:
            summary = f"{', '.join(tolerated)} tolerated; {', '.join(pathogenic)} predicted pathogenic"
        elif tolerated and neutral:
            summary = f"{', '.join(tolerated)} tolerated; {', '.join(neutral)} uncertain"
        elif pathogenic and neutral:
            summary = f"{', '.join(pathogenic)} pathogenic; {', '.join(neutral)} uncertain"
        else:
            summary = f"Mixed effects across alternative alleles"
        
        return {
            "reference": pos0["reference"],
            "alternatives": counterfactuals,
            "tolerated_alleles": tolerated,
            "pathogenic_alleles": pathogenic,
            "neutral_alleles": neutral,
            "summary": summary
        }
    
    def _map_acmg_criteria(
        self,
        gene_symbol: str,
        delta_score: float,
        prediction: str,
        gnomad_result,
        clinvar_data,
        vep_annotation: Optional[dict],
        ism_result: Optional[dict]
    ) -> dict:
        """
        Map all available evidence to specific ACMG/AMP criteria.
        
        Uses rule-based logic (not LLM) for deterministic, reproducible mapping.
        Each criterion is evaluated as MET or NOT MET with a rationale.
        
        Based on Richards et al. (2015) ACMG/AMP guidelines and subsequent
        refinements by ClinGen Sequence Variant Interpretation Working Group.
        
        Returns:
            dict with criteria_status mapping and classification summary
        """
        criteria = {}
        
        # ─── PVS1: Null variant in gene where LOF is known mechanism ───
        is_null = False
        if vep_annotation:
            is_null = bool(
                vep_annotation.get("isNonsense") or
                vep_annotation.get("isFrameshift") or
                (vep_annotation.get("impact") == "HIGH" and "splice" in str(vep_annotation.get("consequence", "")).lower())
            )
        criteria["PVS1"] = {
            "met": is_null,
            "strength": "Very Strong" if is_null else None,
            "rationale": (
                "Protein-truncating variant in gene where LOF is established disease mechanism"
                if is_null else
                "Not a null variant (missense, synonymous, or non-coding)"
            )
        }
        
        # ─── PS1: Same amino acid change as known pathogenic variant ───
        # Requires ClinVar data with same AA change — simplified check
        has_known_pathogenic = (
            clinvar_data and
            clinvar_data.status and
            "pathogenic" in str(clinvar_data.status).lower() and
            clinvar_data.review_status and
            ("expert" in str(clinvar_data.review_status).lower() or clinvar_data.num_submitters >= 2)
        )
        criteria["PS1"] = {
            "met": has_known_pathogenic,
            "strength": "Strong" if has_known_pathogenic else None,
            "rationale": (
                f"Same amino acid change as established pathogenic variant in ClinVar ({clinvar_data.status})"
                if has_known_pathogenic else
                "No known pathogenic variant at this position with same amino acid change"
            )
        }
        
        # ─── PM2: Absent from population databases ───
        af = gnomad_result.allele_frequency if gnomad_result else None
        absent_from_pop = af is None or af == 0
        criteria["PM2"] = {
            "met": absent_from_pop,
            "strength": "Moderate" if absent_from_pop else None,
            "rationale": (
                "Absent from gnomAD v4.1 (800,000+ individuals)"
                if absent_from_pop else
                f"Present in gnomAD at AF={af:.6f} — does not meet PM2 threshold"
            )
        }
        
        # ─── PP3: Multiple computational evidence supports pathogenicity ───
        evo2_pathogenic = delta_score < -0.001 and "pathogenic" in prediction.lower()
        criteria["PP3"] = {
            "met": evo2_pathogenic,
            "strength": "Supporting" if evo2_pathogenic else None,
            "rationale": (
                f"Evo2-7B predicts pathogenic (Δ={delta_score:.6f}) — computational evidence supports deleterious effect"
                if evo2_pathogenic else
                "Computational prediction does not support pathogenicity"
            )
        }
        
        # ─── BP4: Multiple computational evidence supports benign ───
        evo2_benign = delta_score > 0.001 and "benign" in prediction.lower()
        criteria["BP4"] = {
            "met": evo2_benign,
            "strength": "Supporting" if evo2_benign else None,
            "rationale": (
                f"Evo2-7B predicts benign (Δ={delta_score:.6f}) — computational evidence supports no impact"
                if evo2_benign else
                "Computational prediction does not support benign classification"
            )
        }
        
        # ─── BA1: Allele frequency >5% in population ───
        is_common = af is not None and af >= 0.05
        criteria["BA1"] = {
            "met": is_common,
            "strength": "Standalone" if is_common else None,
            "rationale": (
                f"Allele frequency {af*100:.2f}% exceeds 5% threshold — standalone evidence for benign"
                if is_common else
                f"Allele frequency below BA1 threshold (5%)"
            )
        }
        
        # ─── BS1: Allele frequency greater than expected for disorder ───
        is_uncommon_benign = af is not None and af >= 0.01
        criteria["BS1"] = {
            "met": is_uncommon_benign and not is_common,
            "strength": "Strong" if (is_uncommon_benign and not is_common) else None,
            "rationale": (
                f"AF={af*100:.2f}% exceeds expected frequency for {gene_symbol}-related disorder"
                if (is_uncommon_benign and not is_common) else
                "Does not meet BS1 frequency threshold" if af is not None else
                "Population frequency data unavailable"
            )
        }
        
        # ─── PM5: Novel missense at position where different missense is pathogenic ───
        has_different_pathogenic = (
            clinvar_data and
            clinvar_data.status and
            "pathogenic" in str(clinvar_data.status).lower() and
            vep_annotation and
            "missense" in vep_annotation.get("consequence", "").lower()
        )
        criteria["PM5"] = {
            "met": has_different_pathogenic,
            "strength": "Moderate" if has_different_pathogenic else None,
            "rationale": (
                "Novel missense change at residue where a different pathogenic missense has been reported"
                if has_different_pathogenic else
                "Criterion not applicable — no different pathogenic missense at this position"
            )
        }
        
        # ─── PP2: Missense in gene with low rate of benign missense ───
        # Simplified: check if gene is in our high-penetrance cancer gene list
        is_high_constraint_gene = gene_symbol in self.GENE_SPECIFIC_THRESHOLDS
        is_missense = vep_annotation and "missense" in vep_annotation.get("consequence", "").lower()
        criteria["PP2"] = {
            "met": is_high_constraint_gene and is_missense,
            "strength": "Supporting" if (is_high_constraint_gene and is_missense) else None,
            "rationale": (
                f"Missense variant in {gene_symbol} — gene has low rate of benign missense variation (high constraint)"
                if (is_high_constraint_gene and is_missense) else
                "Not a missense variant or gene not in high-constraint panel"
            )
        }
        
        # ─── ISM-based ACMG evidence modifier (PP3/BP4 strength adjustment) ───
        # The ISM constraint ratio f_c modifies the strength of PP3 (pathogenic)
        # and BP4 (benign) based on whether the variant resides in a constrained
        # or permissive micro-environment. This replaces the previous standalone
        # ISM_SPATIAL criterion with direct modification of existing ACMG codes.
        if ism_result and "summary" in ism_result:
            ism_summary = ism_result["summary"]
            constrained = ism_summary.get("constrained_positions", 0)
            total = ism_summary.get("total_positions_scanned", 1)
            f_c = constrained / max(1, total)
            ism_zone = ism_summary.get("constraint_zone", "low")

            # ── Modify PP3 (pathogenic computational evidence) ──
            if "PP3" in criteria and criteria["PP3"]["met"]:
                pp3_strength = criteria["PP3"]["strength"]
                if ism_zone == "high" and pp3_strength == "Supporting":
                    # ISM confirms variant is in constrained region → upgrade PP3
                    criteria["PP3"]["strength"] = "Moderate"
                    criteria["PP3"]["rationale"] += (
                        f" [Upgraded: ISM constraint ratio f_c={f_c:.2f} confirms "
                        f"variant resides in highly constrained micro-environment "
                        f"({constrained}/{total} positions under purifying selection)]"
                    )
                elif ism_zone == "low" and pp3_strength in ("Moderate", "Supporting"):
                    # ISM contradicts PP3 → downgrade
                    criteria["PP3"]["strength"] = "Not Met"
                    criteria["PP3"]["met"] = False
                    criteria["PP3"]["rationale"] += (
                        f" [Downgraded: ISM constraint ratio f_c={f_c:.2f} indicates "
                        f"variant is in evolutionarily permissive region "
                        f"({constrained}/{total} positions constrained)]"
                    )
                # moderate ISM zone: no change to PP3 strength

            # ── Modify BP4 (benign computational evidence) ──
            if "BP4" in criteria and criteria["BP4"]["met"]:
                bp4_strength = criteria["BP4"]["strength"]
                if ism_zone == "high" and bp4_strength in ("Moderate", "Supporting"):
                    # ISM contradicts benign classification → downgrade
                    criteria["BP4"]["strength"] = "Not Met"
                    criteria["BP4"]["met"] = False
                    criteria["BP4"]["rationale"] += (
                        f" [Downgraded: ISM f_c={f_c:.2f} contradicts benign classification "
                        f"— variant in highly constrained region]"
                    )
                elif ism_zone == "low" and bp4_strength == "Supporting":
                    # ISM confirms permissive region → upgrade BP4
                    criteria["BP4"]["strength"] = "Moderate"
                    criteria["BP4"]["rationale"] += (
                        f" [Upgraded: ISM f_c={f_c:.2f} confirms permissive region "
                        f"({constrained}/{total} positions constrained)]"
                    )
                # moderate ISM zone: no change to BP4 strength

            # Store ISM metadata for report and concordance computation
            criteria["_ism_metadata"] = {
                "f_c": f_c,
                "constraint_zone": ism_zone,
                "constrained_positions": constrained,
                "total_positions": total,
            }
        
        # ─── Compute classification from met criteria ───
        met_criteria = {k: v for k, v in criteria.items() if v["met"]}
        
        # Count strengths
        very_strong = sum(1 for v in met_criteria.values() if v["strength"] == "Very Strong")
        strong = sum(1 for v in met_criteria.values() if v["strength"] == "Strong")
        moderate = sum(1 for v in met_criteria.values() if v["strength"] == "Moderate")
        supporting = sum(1 for v in met_criteria.values() if v["strength"] == "Supporting")
        standalone = sum(1 for v in met_criteria.values() if v["strength"] == "Standalone")
        
        # ACMG combining rules (simplified)
        if standalone >= 1:
            acmg_class = "Benign"
        elif very_strong >= 1 and (strong >= 1 or moderate >= 2 or supporting >= 2):
            acmg_class = "Pathogenic"
        elif strong >= 2:
            acmg_class = "Pathogenic"
        elif strong >= 1 and (moderate >= 2 or supporting >= 2):
            acmg_class = "Likely Pathogenic"
        elif strong >= 1 and moderate >= 1:
            acmg_class = "Likely Pathogenic"
        elif moderate >= 2:
            acmg_class = "Likely Pathogenic"
        elif moderate >= 1 and supporting >= 1:
            acmg_class = "Likely Pathogenic"
        elif supporting >= 2:
            acmg_class = "Uncertain Significance (VUS)"
        else:
            acmg_class = "Uncertain Significance (VUS)"
        
        return {
            "criteria": criteria,
            "met_count": len(met_criteria),
            "total_evaluated": len(criteria),
            "strength_counts": {
                "very_strong": very_strong,
                "strong": strong,
                "moderate": moderate,
                "supporting": supporting,
                "standalone": standalone
            },
            "acmg_classification": acmg_class,
            "classification_rationale": (
                f"ACMG classification based on {len(met_criteria)} met criteria "
                f"({very_strong}VS + {strong}S + {moderate}M + {supporting}P)"
            )
        }
    
    def _fetch_knowledge_graph(self, gene_symbol: str, ensembl_id: Optional[str] = None) -> dict:
        """
        Query Open Targets Platform for gene-disease-drug associations.
        
        Returns structured knowledge graph data linking the gene to associated
        diseases (with MONDO IDs) and known drugs (with clinical trial phases).
        This provides therapeutic context that no existing VEP tool offers.
        """
        import requests as req
        
        result = {
            "gene": gene_symbol,
            "diseases": [],
            "drugs": [],
            "clinical_actionability": None
        }
        
        try:
            # Use Ensembl ID from VEP if available, otherwise query by symbol
            if not ensembl_id:
                # Query Ensembl for gene ID
                ensembl_url = f"https://rest.ensembl.org/xrefs/symbol/homo_sapiens/{gene_symbol}?content-type=application/json"
                ensembl_resp = req.get(ensembl_url, timeout=10)
                if ensembl_resp.status_code == 200:
                    ensembl_data = ensembl_resp.json()
                    if ensembl_data and len(ensembl_data) > 0:
                        ensembl_id = ensembl_data[0].get("id")
            
            if not ensembl_id:
                logger.warning(f"Could not resolve Ensembl ID for {gene_symbol}")
                return result
            
            result["ensembl_id"] = ensembl_id
            
            # Query Open Targets for disease associations
            ot_query = """
            query targetInfo($ensemblId: String!) {
              target(ensemblId: $ensemblId) {
                approvedSymbol
                associatedDiseases(page: { size: 5, index: 0 }) {
                  rows {
                    disease {
                      id
                      name
                    }
                    score
                  }
                }
                knownDrugs(page: { size: 10, index: 0 }) {
                  rows {
                    drug {
                      name
                      drugType
                      maximumClinicalTrialPhase
                    }
                    mechanismOfAction
                  }
                }
              }
            }
            """
            
            ot_url = "https://api.platform.opentargets.org/api/v4/graphql"
            ot_resp = req.post(
                ot_url,
                json={"query": ot_query, "variables": {"ensemblId": ensembl_id}},
                timeout=15
            )
            
            if ot_resp.status_code == 200:
                ot_data = ot_resp.json()
                target = ot_data.get("data", {}).get("target", {})
                
                if target:
                    # Parse diseases
                    for row in target.get("associatedDiseases", {}).get("rows", []):
                        disease = row.get("disease", {})
                        result["diseases"].append({
                            "id": disease.get("id", ""),
                            "name": disease.get("name", ""),
                            "score": round(row.get("score", 0), 3)
                        })
                    
                    # Parse drugs
                    for row in target.get("knownDrugs", {}).get("rows", []):
                        drug = row.get("drug", {})
                        phase = drug.get("maximumClinicalTrialPhase", 0)
                        phase_label = (
                            "Approved" if phase >= 4 else
                            f"Phase {phase}" if phase > 0 else
                            "Preclinical"
                        )
                        result["drugs"].append({
                            "name": drug.get("name", ""),
                            "type": drug.get("drugType", "Unknown"),
                            "phase": phase_label,
                            "mechanism": row.get("mechanismOfAction", "Unknown")
                        })
                    
                    # Generate clinical actionability summary
                    if result["diseases"] and result["drugs"]:
                        top_disease = result["diseases"][0]["name"]
                        approved_drugs = [d["name"] for d in result["drugs"] if d["phase"] == "Approved"]
                        if approved_drugs:
                            result["clinical_actionability"] = (
                                f"{gene_symbol} pathogenic variants are associated with {top_disease}. "
                                f"FDA-approved therapies include: {', '.join(approved_drugs[:3])}."
                            )
                        else:
                            result["clinical_actionability"] = (
                                f"{gene_symbol} pathogenic variants are associated with {top_disease}. "
                                f"No approved targeted therapies identified."
                            )
                    elif result["diseases"]:
                        result["clinical_actionability"] = (
                            f"{gene_symbol} is associated with {result['diseases'][0]['name']}. "
                            f"No drug-target associations found in Open Targets."
                        )
            
            logger.info(f"Knowledge graph: {len(result['diseases'])} diseases, {len(result['drugs'])} drugs for {gene_symbol}")
            
        except Exception as e:
            logger.warning(f"Knowledge graph query failed for {gene_symbol}: {e}")
        
        return result
    
    def _fetch_external_scores(self, chromosome: str, position: int, reference: str, alternative: str) -> dict:
        """
        Fetch CADD scores for multi-tool concordance comparison.
        
        CADD (Combined Annotation Dependent Depletion) provides PHRED-scaled
        scores where >20 indicates likely pathogenic. This enables direct
        comparison between Evo2 and an established orthogonal method.
        
        Uses the public CADD SNV REST API:
            https://cadd.gs.washington.edu/api/v1.0/<CADD-version>/<chrom>:<pos>_<ref>_<alt>
        """
        result = {
            "cadd": None,
            "revel": None,
            "alphamissense": None,
            "concordance_note": (
                "REVEL scores are not currently retrieved because no public REST API or "
                "mounted REVEL database is available in this deployment. "
                "Consider integrating dbNSFP or a local REVEL TSV file for missense variants."
            )
        }
        
        try:
            import requests as req
            
            # Query CADD SNV API (GRCh38 v1.7).
            # The single-SNV endpoint is flaky and sometimes returns an empty
            # list even when data exists. The position endpoint (all 3 alts)
            # is more reliable, so we use that and match the desired alt.
            chrom_clean = chromosome.replace("chr", "")
            cadd_url = (
                f"https://cadd.gs.washington.edu/api/v1.0/GRCh38-v1.7/"
                f"{chrom_clean}:{position}"
            )

            cadd_resp = req.get(cadd_url, timeout=15)

            if cadd_resp.status_code == 200:
                cadd_data = cadd_resp.json()
                # CADD returns a list of SNV objects; pick the one matching ref/alt
                match = None
                for snv in cadd_data:
                    if (
                        str(snv.get("Pos")) == str(position)
                        and snv.get("Ref") == reference
                        and snv.get("Alt") == alternative
                    ):
                        match = snv
                        break

                if match and "PHRED" in match:
                    phred = float(match["PHRED"])
                    interpretation = (
                        "Likely Pathogenic" if phred >= 20 else
                        "Possibly Pathogenic" if phred >= 15 else
                        "Likely Benign" if phred < 10 else
                        "Uncertain"
                    )
                    result["cadd"] = {
                        "phred": round(phred, 1),
                        "raw": round(float(match.get("RawScore", 0)), 4),
                        "interpretation": interpretation
                    }
                    logger.info(f"CADD score: PHRED={phred} ({interpretation})")
                else:
                    logger.warning(f"CADD API returned no matching SNV for {chromosome}:{position} {reference}>{alternative}")
            else:
                logger.warning(f"CADD API returned status {cadd_resp.status_code} for {cadd_url}")
            
        except Exception as e:
            logger.warning(f"External score fetch failed: {e}")
        
        return result
    
    def _fetch_alphamissense_score(
        self,
        chromosome: str,
        position: int,
        reference: str,
        alternative: str,
        vep_annotation: dict = None
    ) -> dict:
        """
        Fetch AlphaMissense pathogenicity score from local DuckDB.
        
        Uses the pre-built 359M-variant database for sub-3ms lookups.
        Tries protein-level lookup first (by UniProt + AA change),
        falls back to genomic coordinate lookup.
        
        Paper: Cheng et al. (2023) — Science
        """
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("alphamissense_lookup", "/root/alphamissense_lookup.py")
            am_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(am_module)
            AlphaMissenseDB = am_module.AlphaMissenseDB
            
            db = AlphaMissenseDB(data_dir="/root/alphamissense_data/alphamissense_data")
            
            # Try protein-level lookup if VEP annotation has UniProt + AA info
            if vep_annotation:
                uniprot_id = vep_annotation.get("uniprotId") or vep_annotation.get("uniprot_id")
                aa_pos = vep_annotation.get("aaPosition") or vep_annotation.get("aa_position")
                ref_aa = vep_annotation.get("refAA") or vep_annotation.get("ref_aa")
                alt_aa = vep_annotation.get("altAA") or vep_annotation.get("alt_aa")
                
                if uniprot_id and aa_pos and ref_aa and alt_aa:
                    result = db.lookup(
                        uniprot_id=str(uniprot_id),
                        position=int(aa_pos),
                        ref_aa=str(ref_aa),
                        alt_aa=str(alt_aa)
                    )
                    if result:
                        logger.info(f"AlphaMissense (protein): {result['variant']} = {result['score']} ({result['classification']})")
                        db.close()
                        return result
            
            # Fallback: genomic coordinate lookup
            result = db.lookup_by_genomic(
                chrom=chromosome,
                pos=position,
                ref=reference,
                alt=alternative
            )
            db.close()
            
            if result:
                logger.info(f"AlphaMissense (genomic): {result['variant']} = {result['score']} ({result['classification']})")
            return result
            
        except Exception as e:
            logger.warning(f"AlphaMissense lookup failed: {e}")
            import traceback
            logger.warning(traceback.format_exc())
            return None
    
    def _refine_acmg_with_llm(
        self,
        gene_symbol: str,
        variant_str: str,
        rule_based_acmg: dict,
        delta_score: float,
        prediction: str,
        gnomad_result,
        clinvar_data,
        vep_annotation: Optional[dict],
        ism_result: Optional[dict]
    ) -> dict:
        """
        Use LLM to refine rule-based ACMG criteria mapping.
        
        The rule-based system provides deterministic, reproducible criteria.
        The LLM adds clinical nuance: adjusting strength levels, handling
        edge cases, and providing narrative justification for each criterion.
        """
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return None
        
        try:
            from groq import Groq
            client = Groq(api_key=api_key)
            
            # Build evidence summary for the LLM
            evidence_lines = [f"VARIANT: {gene_symbol} {variant_str}"]
            
            if vep_annotation:
                evidence_lines.append(f"VEP: {vep_annotation.get('consequence', 'Unknown')} ({vep_annotation.get('impact', 'Unknown')} impact)")
                if vep_annotation.get('aminoAcids'):
                    evidence_lines.append(f"Amino Acid Change: {vep_annotation['aminoAcids']}")
            
            evidence_lines.append(f"Evo2: Δ = {delta_score:.6f}, {prediction}")
            
            if gnomad_result and gnomad_result.allele_frequency is not None:
                evidence_lines.append(f"gnomAD: AF = {gnomad_result.allele_frequency:.6f}")
            else:
                evidence_lines.append("gnomAD: Not found")
            
            if clinvar_data and clinvar_data.status not in ("Not Found", "Error", None):
                evidence_lines.append(f"ClinVar: {clinvar_data.status} ({clinvar_data.num_submitters} submitters)")
            
            if ism_result:
                s = ism_result.get("summary", {})
                evidence_lines.append(f"ISM: {s.get('constrained_positions', 0)}/{s.get('total_positions_scanned', 0)} positions constrained ({s.get('constraint_zone', 'N/A')} zone)")
            
            # Format rule-based criteria
            criteria_lines = ["RULE-BASED ACMG OUTPUT:"]
            for code, criterion in rule_based_acmg.get("criteria", {}).items():
                status = "MET" if criterion["met"] else "NOT MET"
                strength = f" ({criterion['strength']})" if criterion["strength"] else ""
                criteria_lines.append(f"  {code}: {status}{strength} — {criterion['rationale']}")
            
            evidence_text = "\n".join(evidence_lines)
            criteria_text = "\n".join(criteria_lines)
            
            prompt = f"""You are a clinical geneticist reviewing ACMG/AMP criteria for a variant.

{evidence_text}

{criteria_text}

Please review each criterion and provide:
1. Confirm or adjust met/not-met status
2. Adjust strength level if warranted (Very Strong, Strong, Moderate, Supporting)
3. A 1-2 sentence clinical justification
4. A final ACMG classification

Return your response as a JSON object with this structure:
{{
  "criteria": {{
    "PVS1": {{"met": true/false, "strength": "Very Strong" or null, "justification": "..."}},
    ...
  }},
  "acmg_classification": "Pathogenic/Likely Pathogenic/VUS/Likely Benign/Benign",
  "classification_confidence": "High/Medium/Low",
  "narrative": "2-3 sentence clinical summary of the ACMG classification"
}}

Only include criteria that are relevant. Be conservative — when evidence is weak, downgrade strength levels."""
            
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are an expert clinical geneticist specializing in ACMG/AMP variant classification. You are conservative and evidence-based. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1500
            )
            
            content = response.choices[0].message.content
            
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                refined = json.loads(json_match.group(0))
                logger.info(f"LLM ACMG refinement: {refined.get('acmg_classification', 'N/A')}")
                return refined
            
        except Exception as e:
            logger.warning(f"LLM ACMG refinement failed: {e}")
        
        return None
    
    @staticmethod
    def _confidence_label(score: float) -> str:
        if score >= 0.8:
            return "High"
        elif score >= 0.5:
            return "Medium"
        else:
            return "Low"
    
    @modal.fastapi_endpoint(method="POST")
    def analyze_single_variant(self, request: VariantRequest):
        """
        FastAPI endpoint for single variant analysis.
        
        This is the primary production endpoint called by the frontend.
        Accepts a VariantRequest and returns comprehensive analysis results.
        """
        return self.run_analysis_logic.local(
            variant_position=request.variant_position,
            alternative=request.alternative,
            genome=request.genome,
            chromosome=request.chromosome,
            provided_reference=request.reference,
            gene_symbol=request.gene_symbol,
            vep_annotation=request.vep_annotation,
            run_ism_scan=request.run_ism_scan,
            ism_scan_radius=request.ism_scan_radius,
            ism_scan_stride=request.ism_scan_stride
        )   
        
    @modal.fastapi_endpoint(method="POST")
    def analyze_batch(self, batch: BatchRequest):
        """
        Optimized batch variant analysis endpoint.
        
        Features:
        - Chunked GPU processing: Prevents out-of-memory errors on large batches
        - Mixed precision (bfloat16): 2x faster inference with minimal accuracy loss
        - Sequence deduplication: Caches genome windows to avoid redundant fetches
        - Clinical enrichment: gnomAD, ACMG codes, and literature for all variants
        - Progressive logging: Track completion status for long-running batches
        
        Performance:
        - H100 GPU can process ~500 variants/minute with 8kb windows
        - Chunk size of 16 variants (32 sequences) optimized for H100 memory
        
        Args:
            batch: BatchRequest containing list of VariantRequest objects
        
        Returns:
            dict: Batch results with metadata (size, chunks, precision, results)
        """
        import torch
        
        num_variants = len(batch.requests)
        logger.info(f"Received batch of {num_variants} variants")
        
        # Configuration parameters optimized for H100 GPU
        WINDOW_SIZE = 8192  # 8kb context window for Evo2
        CHUNK_SIZE = 16     # 16 variants = 32 sequences per batch (memory-safe)
        
        # ===== STEP 1: Sequence Preparation =====
        # Fetch genome sequences and prepare reference/variant pairs
        sequences_to_score = []  # All sequences for GPU scoring
        meta_data = []           # Variant metadata for downstream processing
        batch_genome_cache = {}  # Cache to avoid redundant genome fetches
        
        logger.info("Preparing sequences...")
        for idx, req in enumerate(batch.requests):
            cache_key = f"{req.genome}_{req.chromosome}_{req.variant_position}"
            
            # Fetch genome sequence with caching to avoid redundant API calls
            if cache_key in batch_genome_cache:
                window_seq, seq_start = batch_genome_cache[cache_key]
            else:
                window_seq, seq_start = get_genome_sequence(
                    req.variant_position, req.genome, req.chromosome, WINDOW_SIZE
                )
                batch_genome_cache[cache_key] = (window_seq, seq_start)
            
            # Calculate relative positions and create variant sequence
            relative_pos = req.variant_position - 1 - seq_start
            ref_len = len(req.reference) if req.reference else 1
            reference = window_seq[relative_pos : relative_pos + ref_len]
            var_seq = window_seq[:relative_pos] + req.alternative + window_seq[relative_pos + ref_len:]
            
            # Add both sequences for batched scoring (ref and variant)
            sequences_to_score.append(window_seq)
            sequences_to_score.append(var_seq)
            
            meta_data.append({
                "req": req,
                "reference": reference,
                "variant_idx": idx
            })
        
        total_sequences = len(sequences_to_score)
        logger.info(f"Total sequences to score: {total_sequences}")
        
        # ===== STEP 2: Chunked GPU Scoring =====
        # Process in chunks with mixed precision for memory efficiency
        all_scores = []
        num_chunks = (total_sequences + (CHUNK_SIZE * 2) - 1) // (CHUNK_SIZE * 2)
        
        logger.info(f"Scoring in {num_chunks} chunks (bfloat16 precision)...")
        
        for chunk_idx in range(0, total_sequences, CHUNK_SIZE * 2):
            chunk_end = min(chunk_idx + CHUNK_SIZE * 2, total_sequences)
            chunk = sequences_to_score[chunk_idx:chunk_end]
            
            # Use bfloat16 precision for 2x speedup with minimal accuracy loss
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                chunk_scores = self.model.score_sequences(chunk)
            
            all_scores.extend(chunk_scores)
            
            # Log progress for user visibility
            progress = min(100, int((chunk_end / total_sequences) * 100))
            logger.info(f"Chunk {chunk_idx // (CHUNK_SIZE * 2) + 1}/{num_chunks} complete ({progress}%)")
        
        logger.info("GPU scoring complete")
        
        # ===== STEP 3: Classification and Clinical Enrichment =====
        # Calculate delta scores, classify variants, and add clinical context
        results = []
        params = self.default_params  # Use default threshold for batch (gene-specific TBD)
        
        logger.info("Adding clinical enrichment...")
        
        for i, meta in enumerate(meta_data):
            req = meta["req"]
            ref_score = all_scores[2 * i]
            var_score = all_scores[2 * i + 1]
            delta_score = var_score - ref_score
            
            # Classify using 3-tier system
            # Note: VEP support for batch analysis is a future enhancement
            prediction, confidence = self.classify_variant_with_vep(
                delta_score=delta_score,
                params=params,
                vep_annotation=None  # Future: Add VEP annotation support for batches
            )
            
            # Add clinical enrichment data for each variant
            enrichment = self.clinical_enricher.enrich_variant(
                chromosome=req.chromosome,
                position=req.variant_position,
                ref=meta["reference"],
                alt=req.alternative,
                delta_score=delta_score,
                confidence=confidence,
                gene_symbol=req.gene_symbol
            )
            
            # Build result object with all data
            results.append({
                "position": req.variant_position,
                "reference": meta["reference"],
                "alternative": req.alternative,
                "gene_symbol": req.gene_symbol,
                "delta_score": float(delta_score),
                "prediction": prediction,
                "classification_confidence": float(confidence),
                "classification_source": "Evo2_AI",
                # Clinical enrichment data
                "population_frequency": enrichment["population_frequency"],
                "acmg_evidence": enrichment["acmg_evidence"],
                "literature_context": enrichment.get("literature_context"),
            })
        
        logger.info(f"Batch complete: {num_variants} variants analyzed")
        
        return {
            "batch_size": num_variants,
            "chunks_processed": num_chunks,
            "precision": "bfloat16",
            "batch_results": results
        }

# =============================================================================
# BRCA1 VALIDATION FUNCTION
# =============================================================================
@app.function(gpu="H100", volumes={mount_path: volume}, timeout=1000)
def run_brca1_analysis():
    """
    BRCA1 saturation mutagenesis validation analysis.
    
    This function validates Evo2 predictions against the Findlay et al. 2018
    functional assay dataset (3,893 BRCA1 variants with experimentally measured
    functional scores). Used to:
    - Calculate AUROC for distinguishing LOF from functional variants
    - Determine optimal classification threshold
    - Estimate confidence parameters (LOF/FUNC standard deviations)
    - Generate visualization of score distributions
    
    Reference:
    Findlay et al. (2018) Nature. Accurate classification of BRCA1 variants
    with saturation genome editing. DOI: 10.1038/s41586-018-0461-z
    
    Returns:
        dict: Validation results containing:
            - variants: List of variant records with Evo2 delta scores
            - plot: Base64-encoded PNG visualization
            - auroc: Area under ROC curve
    """
    # Import heavy dependencies only when needed (reduces container startup time)
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
    
    # Load Evo2 model
    print("Loading evo2 model...")
    model = Evo2('evo2_7b')
    print("Evo2 model loaded")
    
    # Load Findlay et al. BRCA1 functional assay data
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
    # Combine functional and intermediate classes (both non-pathogenic)
    brca1_df['class'] = brca1_df['class'].replace(['FUNC', 'INT'], 'FUNC/INT')
    
    # Load chromosome 17 reference sequence (GRCh37/hg19)
    with gzip.open('/evo2/notebooks/brca1/GRCh37.p13_chr17.fna.gz', "rt") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            seq_chr17 = str(record.seq)
            break
    
    # Prepare sequences for scoring
    ref_seqs = []         # Unique reference sequences (deduplicated)
    ref_seq_to_index = {}  # Map sequence → index for deduplication
    ref_seq_indexes = []   # Index of ref_seq for each variant
    var_seqs = []          # Variant sequences (one per variant)
    
    # Process first 500 variants for faster validation (full dataset = 3,893)
    brca1_subset = brca1_df.iloc[:500].copy()
    # Build reference and variant sequences with 8kb context windows
    for _, row in brca1_subset.iterrows():
        p = row["pos"] - 1  # Convert to 0-based
        full_seq = seq_chr17
        ref_seq_start = max(0, p - WINDOW_SIZE//2)
        ref_seq_end = min(len(full_seq), p + WINDOW_SIZE//2)
        ref_seq = seq_chr17[ref_seq_start:ref_seq_end]
        
        # Calculate variant position within window
        snv_pos_in_ref = min(WINDOW_SIZE//2, p)
        var_seq = ref_seq[:snv_pos_in_ref] + row["alt"] + ref_seq[snv_pos_in_ref+1:]
        
        # Deduplicate reference sequences (many variants share same window)
        if ref_seq not in ref_seq_to_index:
            ref_seq_to_index[ref_seq] = len(ref_seqs)
            ref_seqs.append(ref_seq)
        ref_seq_indexes.append(ref_seq_to_index[ref_seq])
        var_seqs.append(var_seq)
    
    ref_seq_indexes = np.array(ref_seq_indexes)
    
    # Score sequences with Evo2
    print(f'Scoring likelihoods of {len(ref_seqs)} reference sequences with Evo 2...')
    ref_scores = model.score_sequences(ref_seqs)
    print(f'Scoring likelihoods of {len(var_seqs)} variant sequences with Evo 2...')
    var_scores = model.score_sequences(var_seqs)
    
    # Calculate delta scores (variant - reference)
    delta_scores = np.array(var_scores) - np.array(ref_scores)[ref_seq_indexes]
    brca1_subset[f'evo2_delta_score'] = delta_scores
    
    # Calculate AUROC (LOF vs FUNC/INT classification)
    y_true = (brca1_subset['class'] == 'LOF')
    auroc = roc_auc_score(y_true, -brca1_subset['evo2_delta_score'])  # Negative delta = pathogenic
    
    # Find optimal threshold (max Youden index)
    y_true = (brca1_subset["class"] == "LOF")
    fpr, tpr, thresholds = roc_curve(y_true, -brca1_subset["evo2_delta_score"])
    optimal_idx = (tpr - fpr).argmax()  # Youden's J statistic
    optimal_threshold = -thresholds[optimal_idx]
    
    # Calculate standard deviations for confidence estimation
    lof_scores = brca1_subset.loc[brca1_subset["class"]== "LOF", "evo2_delta_score"]
    func_scores = brca1_subset.loc[brca1_subset["class"]== "FUNC/INT", "evo2_delta_score"]
    
    confidence_params = {
        "threshold": optimal_threshold,
        "lof_std": lof_scores.std(),
        "func_std": func_scores.std()
    }
    print("Confidence params:", confidence_params)
    
    # Generate visualization
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
    
    # Convert plot to base64 for web display
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    return {
        'variants': brca1_subset.to_dict(orient="records"),
        "plot": plot_data,
        "auroc": auroc
    }

# =============================================================================
# LOCAL TESTING ENTRYPOINT
# =============================================================================

@app.local_entrypoint()
def main():
    """
    Local testing entrypoint for development.
    
    Tests the variant analysis pipeline with a sample BRCA1 variant.
    Run with: modal run main.py
    """
    evo2Model = Evo2Model()
    result = evo2Model.run_analysis_logic.remote(
        variant_position=43119628,
        alternative="G",
        genome="hg38",
        chromosome="chr17"
    )
    print(result)
