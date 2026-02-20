"""
HelixMind ClinVar Benchmark Script (Option A: HTTP Client)

This script calls your DEPLOYED Modal endpoint via HTTP.
It does NOT modify your backend - completely safe.

Progress visibility:
- Console shows real-time progress
- Saves partial results every batch (crash-safe)
- Final results saved to helixmind_benchmark_results.csv

Usage:
    python batch_benchmark.py

Requirements:
    pip install requests pandas tqdm
"""

import requests
import pandas as pd
import time
import json
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

# Your deployed Modal endpoint (batch version)
# Based on your single endpoint pattern, the batch endpoint should be:
MODAL_BATCH_URL = "https://karsoham529--variant-analysis-benchmark-evo2model-analyze-batch.modal.run"

# Alternative: Use single variant endpoint (slower but more reliable)
MODAL_SINGLE_URL = "https://karsoham529--variant-analysis-evo2model-analyze-single-variant.modal.run"

# Batch size - how many variants to send per API call
# Your backend processes 16 variants per GPU chunk, so 50-100 is efficient
BATCH_SIZE = 100

# Default genome (hg38 is standard, hg19 for older data)
DEFAULT_GENOME = "hg38"

# Input/Output files
INPUT_FILE = "clinvar_benchmark_4k_stratified.csv"
OUTPUT_FILE = "helixmind_benchmark_results.csv"
CHECKPOINT_FILE = "benchmark_checkpoint.json"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def parse_variant_id(variant_id: str) -> dict:
    """
    Parse variant ID format: chr17-43045629-C-T
    Returns dict with chromosome, position, reference, alternative
    """
    parts = variant_id.split("-")
    
    if len(parts) < 4:
        raise ValueError(f"Invalid variant ID format: {variant_id}")
    
    # Handle cases like chr17-43045629-CA-C (indels with multi-char alleles)
    return {
        "chromosome": parts[0],
        "variant_position": int(parts[1]),
        "reference": parts[2],
        "alternative": parts[3],
        "genome": DEFAULT_GENOME,
    }


def call_single_variant(variant_data: dict) -> dict:
    """Call the single variant endpoint"""
    try:
        response = requests.post(
            MODAL_SINGLE_URL,
            json=variant_data,
            timeout=60  # 60 second timeout per variant
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {"error": str(e), "variant_id": variant_data.get("variant_position")}


def call_batch_endpoint(variants: list) -> dict:
    """Call the batch endpoint with multiple variants"""
    batch_request = {"requests": variants}
    
    try:
        response = requests.post(
            MODAL_BATCH_URL,
            json=batch_request,
            timeout=300  # 5 minute timeout for batch
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {"error": str(e)}


def save_checkpoint(processed_ids: list, results: list):
    """Save progress checkpoint (crash recovery)"""
    checkpoint = {
        "processed_count": len(processed_ids),
        "processed_ids": processed_ids[-100:],  # Keep last 100 for reference
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f)
    
    # Also save partial results
    if results:
        pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)


def load_checkpoint() -> set:
    """Load checkpoint if exists"""
    if Path(CHECKPOINT_FILE).exists():
        with open(CHECKPOINT_FILE, "r") as f:
            checkpoint = json.load(f)
            print(f"📂 Found checkpoint: {checkpoint['processed_count']} variants already processed")
            return set(checkpoint.get("processed_ids", []))
    return set()


# =============================================================================
# MAIN BENCHMARK FUNCTION
# =============================================================================

def run_benchmark_single_mode():
    """
    Run benchmark using single variant endpoint (safer, more reliable)
    Shows progress in console.
    """
    print("=" * 60)
    print("🧬 HelixMind ClinVar Benchmark")
    print("=" * 60)
    
    # 1. Load input data
    print(f"\n📂 Loading {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"❌ Error: {INPUT_FILE} not found!")
        print("   Run prepare_clinvar_data_advanced.py first.")
        return
    
    total_variants = len(df)
    print(f"✅ Loaded {total_variants} variants")
    print(f"   Breakdown: {df['ClinicalSignificance'].value_counts().to_dict()}")
    
    # 2. Load checkpoint (if resuming)
    already_processed = load_checkpoint()
    
    # 3. Process variants
    results = []
    errors = []
    start_time = time.time()
    
    print(f"\n🚀 Starting benchmark (Single Variant Mode)...")
    print(f"   Endpoint: {MODAL_SINGLE_URL}")
    print(f"   Progress will be shown below:\n")
    
    for idx, row in df.iterrows():
        variant_id = row["variant_id"]
        ground_truth = row["ClinicalSignificance"]
        
        # Skip if already processed (resume capability)
        if variant_id in already_processed:
            continue
        
        # Parse variant
        try:
            variant_data = parse_variant_id(variant_id)
        except ValueError as e:
            errors.append({"variant_id": variant_id, "error": str(e)})
            continue
        
        # Call API
        api_start = time.time()
        response = call_single_variant(variant_data)
        api_time = time.time() - api_start
        
        # Process response
        if "error" in response:
            errors.append({
                "variant_id": variant_id,
                "error": response["error"]
            })
            result_row = {
                "variant_id": variant_id,
                "ground_truth": ground_truth,
                "evo2_score": None,
                "prediction": "ERROR",
                "confidence": None,
                "processing_time": api_time,
                "error": response["error"]
            }
        else:
            result_row = {
                "variant_id": variant_id,
                "ground_truth": ground_truth,
                "evo2_score": response.get("delta_score"),
                "prediction": response.get("prediction"),
                "confidence": response.get("classification_confidence"),
                "processing_time": api_time,
                "acmg_code": response.get("acmg_evidence", {}).get("code") if response.get("acmg_evidence") else None,
                "gnomad_af": response.get("population_frequency", {}).get("gnomad_af") if response.get("population_frequency") else None,
            }
        
        results.append(result_row)
        already_processed.add(variant_id)
        
        # Progress display
        completed = idx + 1
        elapsed = time.time() - start_time
        rate = completed / elapsed if elapsed > 0 else 0
        eta = (total_variants - completed) / rate if rate > 0 else 0
        
        # Progress bar
        progress_pct = (completed / total_variants) * 100
        bar_length = 30
        filled = int(bar_length * completed / total_variants)
        bar = "█" * filled + "░" * (bar_length - filled)
        
        print(f"\r[{bar}] {progress_pct:5.1f}% | {completed}/{total_variants} | "
              f"Speed: {rate:.1f}/s | ETA: {eta/60:.1f}min | "
              f"Errors: {len(errors)}", end="", flush=True)
        
        # Save checkpoint every 50 variants
        if completed % 50 == 0:
            save_checkpoint(list(already_processed), results)
    
    # 4. Final save
    print("\n")
    print("=" * 60)
    
    # Save final results
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_FILE, index=False)
    
    # Summary
    total_time = time.time() - start_time
    successful = len([r for r in results if r.get("evo2_score") is not None])
    
    print(f"🎉 BENCHMARK COMPLETE!")
    print(f"   Total variants: {total_variants}")
    print(f"   Successful: {successful}")
    print(f"   Errors: {len(errors)}")
    print(f"   Total time: {total_time/60:.1f} minutes")
    print(f"   Avg speed: {total_variants/total_time:.2f} variants/second")
    print(f"\n📄 Results saved to: {OUTPUT_FILE}")
    print(f"   Next step: Run analyze_results.py")
    
    # Cleanup checkpoint
    if Path(CHECKPOINT_FILE).exists():
        Path(CHECKPOINT_FILE).unlink()


# =============================================================================
# BATCH MODE (FASTER - Uses analyze_batch endpoint)
# =============================================================================

def run_benchmark_batch_mode():
    """
    Run benchmark using BATCH endpoint (faster, cheaper)
    Sends 50 variants at a time to the GPU.
    """
    print("=" * 60)
    print("🧬 HelixMind ClinVar Benchmark (BATCH MODE)")
    print("=" * 60)
    
    # 1. Load input data
    print(f"\n📂 Loading {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"❌ Error: {INPUT_FILE} not found!")
        print("   Run prepare_clinvar_data_advanced.py first.")
        return
    
    total_variants = len(df)
    print(f"✅ Loaded {total_variants} variants")
    print(f"   Breakdown: {df['ClinicalSignificance'].value_counts().to_dict()}")
    
    # 2. Prepare all variant requests
    all_requests = []
    ground_truth_map = {}
    
    for idx, row in df.iterrows():
        variant_id = row["variant_id"]
        ground_truth = row["ClinicalSignificance"]
        ground_truth_map[variant_id] = ground_truth
        
        try:
            parsed = parse_variant_id(variant_id)
            all_requests.append({
                "variant_id": variant_id,  # Keep for mapping
                **parsed
            })
        except ValueError as e:
            print(f"⚠️ Skipping invalid variant: {variant_id} - {e}")
    
    print(f"✅ Prepared {len(all_requests)} valid variants for processing")
    
    # 3. Process in batches
    results = []
    errors = []
    start_time = time.time()
    
    num_batches = (len(all_requests) + BATCH_SIZE - 1) // BATCH_SIZE
    
    print(f"\n🚀 Starting benchmark (BATCH MODE)...")
    print(f"   Endpoint: {MODAL_BATCH_URL}")
    print(f"   Batch size: {BATCH_SIZE} variants")
    print(f"   Total batches: {num_batches}")
    print(f"   Progress will be shown below:\n")
    
    for batch_idx in range(0, len(all_requests), BATCH_SIZE):
        batch_end = min(batch_idx + BATCH_SIZE, len(all_requests))
        batch = all_requests[batch_idx:batch_end]
        batch_num = (batch_idx // BATCH_SIZE) + 1
        
        # Prepare batch request (remove variant_id, keep only API fields)
        api_batch = []
        for v in batch:
            api_batch.append({
                "variant_position": v["variant_position"],
                "alternative": v["alternative"],
                "genome": v["genome"],
                "chromosome": v["chromosome"],
                "reference": v["reference"],
            })
        
        # Call batch API
        batch_start = time.time()
        try:
            response = requests.post(
                MODAL_BATCH_URL,
                json={"requests": api_batch},
                timeout=600  # 10 min timeout for batch
            )
            response.raise_for_status()
            batch_response = response.json()
            batch_time = time.time() - batch_start
            
            # Process batch results
            batch_results = batch_response.get("batch_results", [])
            
            for i, res in enumerate(batch_results):
                original = batch[i]
                variant_id = original["variant_id"]
                
                results.append({
                    "variant_id": variant_id,
                    "ground_truth": ground_truth_map.get(variant_id, "Unknown"),
                    "evo2_score": res.get("delta_score"),
                    "prediction": res.get("prediction"),
                    "confidence": res.get("classification_confidence"),
                    "acmg_code": res.get("acmg_evidence", {}).get("code") if res.get("acmg_evidence") else None,
                    "gnomad_af": res.get("population_frequency", {}).get("gnomad_af") if res.get("population_frequency") else None,
                })
                
        except requests.RequestException as e:
            batch_time = time.time() - batch_start
            print(f"\n❌ Batch {batch_num} failed: {e}")
            # Mark all variants in this batch as errors
            for v in batch:
                errors.append({"variant_id": v["variant_id"], "error": str(e)})
                results.append({
                    "variant_id": v["variant_id"],
                    "ground_truth": ground_truth_map.get(v["variant_id"], "Unknown"),
                    "evo2_score": None,
                    "prediction": "ERROR",
                    "confidence": None,
                    "error": str(e)
                })
        
        # Progress display
        completed = batch_end
        elapsed = time.time() - start_time
        rate = completed / elapsed if elapsed > 0 else 0
        eta = (total_variants - completed) / rate if rate > 0 else 0
        
        # Progress bar
        progress_pct = (completed / total_variants) * 100
        bar_length = 30
        filled = int(bar_length * completed / total_variants)
        bar = "█" * filled + "░" * (bar_length - filled)
        
        print(f"\r[{bar}] {progress_pct:5.1f}% | Batch {batch_num}/{num_batches} | "
              f"{completed}/{total_variants} variants | "
              f"Speed: {rate:.1f}/s | ETA: {eta/60:.1f}min", end="", flush=True)
        
        # Save checkpoint every batch
        save_checkpoint([r["variant_id"] for r in results], results)
    
    # 4. Final save
    print("\n\n" + "=" * 60)
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_FILE, index=False)
    
    # Summary
    total_time = time.time() - start_time
    successful = len([r for r in results if r.get("evo2_score") is not None])
    
    print(f"🎉 BENCHMARK COMPLETE!")
    print(f"   Total variants: {total_variants}")
    print(f"   Successful: {successful}")
    print(f"   Errors: {len(errors)}")
    print(f"   Total time: {total_time/60:.1f} minutes")
    print(f"   Avg speed: {total_variants/total_time:.2f} variants/second")
    print(f"\n📄 Results saved to: {OUTPUT_FILE}")
    print(f"   Next step: Run analyze_results.py")
    
    # Cleanup checkpoint
    if Path(CHECKPOINT_FILE).exists():
        Path(CHECKPOINT_FILE).unlink()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Use BATCH mode (faster) - switch to run_benchmark_single_mode() if issues
    run_benchmark_batch_mode()

