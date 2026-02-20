"""
Configuration for Somatic Driver Analysis Project
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = DATA_DIR / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# Ensure directories exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR, FIGURES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Modal Evo2 endpoint (from parent main.py)
MODAL_EVO2_ENDPOINT = os.getenv("MODAL_EVO2_ENDPOINT", "https://karsoham529--variant-analysis-evo2model-analyze-batch.modal.run")

# Data sources
DATA_SOURCES = {
    "india_project_2013": {
        "url": "https://www.nature.com/articles/ng.2764",  # Placeholder - update with actual data URL
        "local_path": PROCESSED_DATA_DIR / "india_variants.csv",
        "description": "Indian OSCC WGS (50 patients)",
        "format": "CSV"
    },
    "gse213862": {
        "path": "../../data/india/GSE213862_clinical.csv",
        "description": "Indian OSCC RNA-seq (45 patients)",
        "format": "CSV"
    },
    "tcga_hnsc": {
        "url": "https://portal.gdc.cancer.gov/projects/TCGA-HNSC",
        "description": "TCGA Head and Neck (500+ patients)",
        "format": "MAF"
    }
}

# Analysis parameters
ANALYSIS_CONFIG = {
    "evo2": {
        "batch_size": 100,
        "window_size": 8192,
        "genome_build": "hg38"
    },
    "filtering": {
        "min_quality": 30,
        "min_depth": 100,
        "min_vaf": 0.05,
        "max_vaf": 0.95
    },
    "drivers": {
        "known_genes": ["TP53", "FAT1", "CASP8", "NOTCH1", "PIK3CA", "MYC", "CDKN2A"],
        "score_threshold": -0.5,  # Evo2 delta LL threshold for high-impact
        "top_n": 20  # Top N drivers per patient
    },
    "survival": {
        "pds_top_variants": 10,  # Top variants for Prognostic Driver Score
        "stratification_method": "median"  # or "tertile"
    }
}

# Figure settings
FIGURE_CONFIG = {
    "dpi": 300,
    "format": ["png", "pdf"],
    "style": "seaborn-v0_8-paper",
    "font_size": 10,
    "figure_size": (12, 8)
}

# Benchmark tools
BENCHMARK_TOOLS = {
    "alphamissense": {
        "database_path": RAW_DATA_DIR / "alphamissense_predictions.tsv",
        "url": "https://storage.googleapis.com/alphamissense-public/AlphaMissense_hg38.tsv.gz"
    },
    "cadd": {
        "database_path": RAW_DATA_DIR / "cadd_scores.tsv.gz",
        "url": "https://krishna.gs.washington.edu/download/CADD/v1.6/"
    }
}

# Statistical thresholds
STATS_CONFIG = {
    "significance_level": 0.05,
    "fdr_method": "fdr_bh",  # Benjamini-Hochberg
    "min_sample_size": 10
}
