"""
Download ENCODE ATAC-seq peak data - FIXED VERSION
Uses ENCODE API for dynamic file discovery, validates downloads
"""

import os
import requests
import pandas as pd
from pathlib import Path
import gzip
import logging
import time
from typing import Optional, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

DATA_DIR = Path(__file__).parent / "data" / "atac"
DATA_DIR.mkdir(parents=True, exist_ok=True)

ENCODE_BASE = "https://www.encodeproject.org"
RETRIES = 3
TARGET_ASSEMBLY = "GRCh38"  # Match your reference genome

# Search by EXPERIMENT accession (more stable than file IDs)
# Verified ENCODE4 experiments with GRCh38 data
TARGET_EXPERIMENTS = {
    "K562": "ENCSR868FGK",      # K562 ATAC-seq (ENCODE4, Snyder lab)
    "GM12878": "ENCSR637XSC",   # GM12878 ATAC-seq (ENCODE4)
    "HepG2": "ENCSR506LLK",     # HepG2 ATAC-seq (ENCODE4)
    "A549": "ENCSR113YFR",      # A549 ATAC-seq (ENCODE4)
    "IMR90": "ENCSR653SYG",     # IMR90 fibroblast (ENCODE4)
}


def encode_request(url: str, retries: int = RETRIES) -> dict:
    """Make ENCODE API request with retry logic."""
    for attempt in range(retries):
        try:
            response = requests.get(
                url, 
                headers={"Accept": "application/json"},
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            if attempt < retries - 1:
                wait = 2 ** attempt  # Exponential backoff
                logging.warning(f"Retry {attempt + 1}/{retries} after {wait}s: {e}")
                time.sleep(wait)
            else:
                raise


def get_atac_files(experiment_id: str) -> List[dict]:
    """
    Get optimal ATAC-seq narrowPeak files from experiment.
    FIXED: Uses API to discover files dynamically
    """
    url = f"{ENCODE_BASE}/experiments/{experiment_id}/?format=json"
    data = encode_request(url)
    
    files = data.get("files", [])
    peak_files = []
    
    for file_ref in files:
        # Resolve file reference
        if isinstance(file_ref, str):
            file_url = f"{ENCODE_BASE}{file_ref}?format=json"
            file_data = encode_request(file_url)
        else:
            file_data = file_ref
        
        # Filter for optimal narrowPeak
        output_type = file_data.get("output_type", "")
        file_type = file_data.get("file_type", "")
        assembly = file_data.get("assembly", "")
        
        is_narrowpeak = "narrowPeak" in str(output_type) or "narrowPeak" in str(file_type)
        is_optimal = "optimal" in str(output_type).lower() or "idr" in str(output_type).lower()
        is_target_assembly = assembly == TARGET_ASSEMBLY
        
        if is_narrowpeak and is_optimal and is_target_assembly:
            peak_files.append({
                "accession": file_data.get("accession"),
                "href": file_data.get("href"),
                "output_type": output_type,
                "assembly": assembly
            })
    
    # Prefer "optimal IDR" over just "peaks"
    idr_files = [f for f in peak_files if "idr" in f["output_type"].lower()]
    return idr_files if idr_files else peak_files


def download_encode_file(file_info: dict, output_path: Path) -> bool:
    """Download file from ENCODE with validation."""
    accession = file_info["accession"]
    href = file_info["href"]
    
    # Construct download URL
    if href.startswith("/files/"):
        url = f"{ENCODE_BASE}{href}@@download/{accession}.bed.gz"
    else:
        url = f"{ENCODE_BASE}/files/{accession}/@@download/{accession}.bed.gz"
    
    if output_path.exists():
        # Validate existing file
        if validate_peak_file(output_path):
            logging.info(f"Valid existing file: {output_path.name}")
            return True
        else:
            logging.warning(f"Corrupt existing file, re-downloading: {output_path.name}")
            output_path.unlink()
    
    # Download with retry
    for attempt in range(RETRIES):
        try:
            logging.info(f"Downloading {accession} (attempt {attempt + 1})...")
            response = requests.get(url, stream=True, timeout=120)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Validate download
            if validate_peak_file(output_path):
                logging.info(f"✓ Downloaded and validated: {output_path.name}")
                return True
            else:
                output_path.unlink()
                raise ValueError("Downloaded file failed validation")
                
        except Exception as e:
            if attempt < RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                logging.error(f"Failed to download {accession}: {e}")
                return False
    
    return False


def validate_peak_file(path: Path) -> bool:
    """Validate that file is non-empty, parseable narrowPeak."""
    try:
        if not path.exists() or path.stat().st_size == 0:
            return False
        
        # Try parsing first 10 lines
        if str(path).endswith('.gz'):
            with gzip.open(path, 'rt') as f:
                lines = [next(f) for _ in range(10)]
        else:
            with open(path) as f:
                lines = [next(f) for _ in range(10)]
        
        # Check format: should have 10 tab-separated columns
        for line in lines:
            cols = line.strip().split('\t')
            if len(cols) != 10:
                return False
        
        return True
    except Exception:
        return False


def parse_narrowpeak(bed_path: Path, max_peaks: int = 10000) -> pd.DataFrame:
    """
    Parse narrowPeak to DataFrame with quality filtering.
    FIXED: Filters by qValue, takes top peaks
    """
    columns = [
        "chrom", "start", "end", "name", "score", "strand",
        "signal", "pvalue", "qvalue", "peak_offset"
    ]
    
    # Read file
    if str(bed_path).endswith('.gz'):
        with gzip.open(bed_path, 'rt') as f:
            df = pd.read_csv(f, sep='\t', header=None, names=columns)
    else:
        df = pd.read_csv(bed_path, sep='\t', header=None, names=columns)
    
    # Filter to standard chromosomes
    standard_chroms = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
    df = df[df["chrom"].isin(standard_chroms)]
    
    # Quality filter: qValue > 2 (p < 0.01)
    df = df[df["qvalue"] > 2]
    
    # Sort by signal, take top N
    df = df.nlargest(max_peaks, "signal")
    
    # Calculate peak center
    df["center"] = df["start"] + df["peak_offset"]
    
    # Validate
    assert (df["center"] >= df["start"]).all()
    assert (df["center"] <= df["end"]).all()
    
    logging.info(f"Parsed {len(df)} high-quality peaks from {bed_path.name}")
    logging.info(f"  Signal range: {df['signal'].min():.2f} - {df['signal'].max():.2f}")
    logging.info(f"  Q-value range: {df['qvalue'].min():.2f} - {df['qvalue'].max():.2f}")
    
    return df


def main():
    logging.info("=" * 60)
    logging.info("ENCODE ATAC-seq Download (Fixed)")
    logging.info("=" * 60)
    
    success_count = 0
    
    for cell_type, experiment_id in TARGET_EXPERIMENTS.items():
        logging.info(f"\n{'='*40}")
        logging.info(f"Processing {cell_type}: {experiment_id}")
        
        try:
            # Get files from experiment
            files = get_atac_files(experiment_id)
            
            if not files:
                logging.error(f"No ATAC-seq files found for {cell_type}")
                continue
            
            logging.info(f"Found {len(files)} candidate files")
            
            # Try downloading first available file
            downloaded = False
            for file_info in files:
                output_path = DATA_DIR / f"{cell_type.lower()}_peaks.bed.gz"
                
                if download_encode_file(file_info, output_path):
                    # Parse and save as CSV
                    df = parse_narrowpeak(output_path)
                    csv_path = DATA_DIR / f"{cell_type.lower()}_peaks.csv"
                    df.to_csv(csv_path, index=False)
                    
                    logging.info(f"✓ Saved {len(df)} peaks to {csv_path.name}")
                    success_count += 1
                    downloaded = True
                    break
            
            if not downloaded:
                logging.error(f"Failed to download any file for {cell_type}")
                
        except Exception as e:
            logging.error(f"Error processing {cell_type}: {e}")
    
    # Summary
    logging.info(f"\n{'='*60}")
    logging.info(f"SUMMARY: {success_count}/{len(TARGET_EXPERIMENTS)} cell types downloaded")
    
    for f in DATA_DIR.glob("*_peaks.csv"):
        df = pd.read_csv(f)
        logging.info(f"  {f.name}: {len(df)} peaks")


if __name__ == "__main__":
    main()
