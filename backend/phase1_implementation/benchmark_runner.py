"""
Phase 1 / Week 2: ClinVar 4K Benchmark Runner
===============================================
Runs Evo2-7B on 4,000 stratified ClinVar variants via the deployed
Modal endpoint. Computes AUROC, AUPRC, calibration curves, and
gene-stratified performance.

Dataset: clinvar_benchmark_4k_stratified.csv
         1,000 Pathogenic + 1,000 Benign + 1,000 VUS + 1,000 Conflicting

Papers:
- Nguyen et al. (2024) — bioRxiv: Evo2
- Cheng et al. (2023) — Science: AlphaMissense
- De la Vega et al. (2021) — Genome Medicine: GEM

Usage:
    python benchmark_runner.py

Output:
    helixmind_benchmark_results.csv — Raw predictions
    benchmark_metrics.json — AUROC, AUPRC, calibration
    benchmark_figures/ — ROC, PR, calibration plots
"""

import os
import sys
import json
import time
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================

# Modal endpoint (deployed)
MODAL_SINGLE_URL = "https://karsoham529--variant-analysis-evo2model-analyze-single-variant.modal.run"

# Input/output
INPUT_FILE = Path(__file__).parent.parent / "clinvar_benchmark_4k_stratified.csv"
OUTPUT_FILE = Path(__file__).parent / "helixmind_benchmark_results.csv"
CHECKPOINT_FILE = Path(__file__).parent / "benchmark_checkpoint.json"
METRICS_FILE = Path(__file__).parent / "analysis" / "benchmark_metrics.json"

# Benchmark settings
BATCH_SIZE = 50  # Variants per batch (Modal processes 16 per GPU chunk)
REQUEST_DELAY = 0.5  # Seconds between requests (rate limiting)
DEFAULT_GENOME = "hg38"

# =============================================================================
# VARIANT PARSING
# =============================================================================

def parse_variant_id(variant_id: str) -> dict:
    """
    Parse variant ID format: chr17-43045629-C-T
    Returns dict with chromosome, position, reference, alternative.
    """
    parts = variant_id.split("-")
    
    if len(parts) < 4:
        raise ValueError(f"Invalid variant ID format: {variant_id}")
    
    return {
        "chromosome": parts[0],
        "variant_position": int(parts[1]),
        "reference": parts[2],
        "alternative": parts[3],
        "genome": DEFAULT_GENOME,
    }


def call_endpoint(variant_data: dict) -> dict:
    """Call the Modal single variant endpoint."""
    try:
        response = requests.post(
            MODAL_SINGLE_URL,
            json=variant_data,
            timeout=90
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {"error": str(e), "variant_id": variant_data.get("variant_position")}


# =============================================================================
# CHECKPOINT MANAGEMENT
# =============================================================================

def save_checkpoint(processed_count: int, results: List[dict]):
    """Save progress for crash recovery."""
    checkpoint = {
        "processed_count": processed_count,
        "timestamp": datetime.now().isoformat(),
        "last_variant": results[-1].get("variant_id", "unknown") if results else None
    }
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f)
    
    # Save partial results
    if results:
        pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)


def load_checkpoint() -> Tuple[int, List[str]]:
    """Load checkpoint if exists. Returns (processed_count, processed_ids)."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, "r") as f:
            checkpoint = json.load(f)
            return checkpoint.get("processed_count", 0), []
    return 0, []


# =============================================================================
# MAIN BENCHMARK
# =============================================================================

def run_benchmark():
    """Run the full 4K variant benchmark."""
    print("=" * 60)
    print("🧬 HelixMind ClinVar 4K Benchmark")
    print("=" * 60)
    print(f"Endpoint: {MODAL_SINGLE_URL}")
    print(f"Input: {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print()
    
    # Load data
    print("📂 Loading benchmark dataset...")
    df = pd.read_csv(INPUT_FILE)
    total = len(df)
    print(f"   {total} variants loaded")
    print(f"   Distribution:")
    for label, count in df["ClinicalSignificance"].value_counts().items():
        print(f"     {label}: {count}")
    print()
    
    # Check checkpoint
    processed_count, _ = load_checkpoint()
    if processed_count > 0:
        print(f"📂 Resuming from checkpoint: {processed_count}/{total} processed")
        df = df.iloc[processed_count:]
    else:
        print("🆕 Starting fresh benchmark run.")
    
    # Run benchmark
    results = []
    start_time = time.perf_counter()
    
    for i, (_, row) in enumerate(df.iterrows()):
        variant_id = row["variant_id"]
        clinical_sig = row["ClinicalSignificance"]
        
        try:
            variant_data = parse_variant_id(variant_id)
        except ValueError as e:
            print(f"   [{i+1}/{total}] ⚠️  Skipping {variant_id}: {e}")
            continue
        
        # Call endpoint
        print(f"   [{i+1}/{total}] {variant_id} ({clinical_sig})...", end=" ", flush=True)
        
        api_result = call_endpoint(variant_data)
        
        if "error" in api_result:
            print(f"❌ {api_result['error']}")
            results.append({
                "variant_id": variant_id,
                "clinical_significance": clinical_sig,
                "error": api_result["error"],
                "prediction": None,
                "delta_score": None,
                "classification_confidence": None
            })
        else:
            prediction = api_result.get("prediction", "Unknown")
            delta = api_result.get("delta_score")
            confidence = api_result.get("classification_confidence")
            print(f"✓ {prediction} (Δ={delta:.6f})" if delta else f"✓ {prediction}")
            
            results.append({
                "variant_id": variant_id,
                "clinical_significance": clinical_sig,
                "prediction": prediction,
                "delta_score": delta,
                "classification_confidence": confidence,
                "classification_source": api_result.get("classification_source", "Unknown"),
                "gnomad_af": api_result.get("population_frequency", {}).get("gnomad_af"),
            })
        
        # Save checkpoint every 100 variants
        if (i + 1) % 100 == 0:
            save_checkpoint(processed_count + i + 1, results)
            elapsed = time.perf_counter() - start_time
            rate = (i + 1) / elapsed
            eta = (total - processed_count - i - 1) / rate
            print(f"   💾 Checkpoint saved. Rate: {rate:.1f} variants/s, ETA: {eta/60:.0f} min")
        
        # Rate limiting
        time.sleep(REQUEST_DELAY)
    
    # Final save
    elapsed = time.perf_counter() - start_time
    save_checkpoint(total, results)
    
    print()
    print("=" * 60)
    print("✅ Benchmark Complete")
    print("=" * 60)
    print(f"Total variants: {len(results)}")
    print(f"Successful: {sum(1 for r in results if r.get('prediction'))}")
    print(f"Failed: {sum(1 for r in results if r.get('error'))}")
    print(f"Time: {elapsed/60:.1f} minutes")
    print(f"Results saved to: {OUTPUT_FILE}")
    print()
    print("Next step: Run analysis/benchmark_metrics.py")


if __name__ == "__main__":
    run_benchmark()
