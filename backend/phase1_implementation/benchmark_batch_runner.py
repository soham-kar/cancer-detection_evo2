"""
4K ClinVar Benchmark — Batch Endpoint Runner
===============================================
Calls Modal /analyze_batch endpoint (16 variants per GPU call) for cost-efficient
Evo2 inference. Post-processes locally with DuckDB AlphaMissense + consensus.

Cost: ~$1.50 for 4K variants (~40 min)
Output: helixmind_benchmark_results.csv (raw Evo2 scores + labels)

Usage:
    python benchmark_batch_runner.py

Then run locally:
    python benchmark_postprocess.py  # Adds AlphaMissense + consensus
"""

import requests
import pandas as pd
import json
import time
from pathlib import Path
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================

BATCH_URL = "https://karsoham529--variant-analysis-evo2model-analyze-batch.modal.run"
INPUT_CSV = Path("clinvar_benchmark_4k_stratified.csv")
OUTPUT_DIR = Path("backend/phase1_implementation")
OUTPUT_CSV = OUTPUT_DIR / "helixmind_benchmark_results.csv"
CHECKPOINT = OUTPUT_DIR / "benchmark_checkpoint.json"
BATCH_SIZE = 16  # Variants per batch call (matches Modal chunk size)

def parse_variant_id(vid: str) -> dict:
    """Parse 'chr17-43045629-C-T' into dict."""
    parts = vid.split("-")
    return {
        "chromosome": parts[0],
        "variant_position": int(parts[1]),
        "reference": parts[2],
        "alternative": parts[3],
        "genome": "hg38",
    }

def call_batch_endpoint(batch_requests: list) -> list:
    """Call Modal batch endpoint with list of VariantRequest dicts."""
    payload = {"requests": batch_requests}
    try:
        r = requests.post(BATCH_URL, json=payload, timeout=300)
        r.raise_for_status()
        data = r.json()
        # Response format: {"batch_results": [...], "metadata": {...}}
        return data.get("batch_results", [])
    except Exception as e:
        print(f"  Batch error: {e}")
        return [{"error": str(e)}] * len(batch_requests)

def main():
    df = pd.read_csv(INPUT_CSV)
    total = len(df)
    print(f"Benchmarking {total} variants via batch endpoint...")
    print(f"Batch size: {BATCH_SIZE} | Estimated time: {total/BATCH_SIZE*2:.0f} batch calls")
    print(f"Output dir: {OUTPUT_DIR}")
    print("=" * 60)

    # Ensure output dir exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    start_idx = 0
    results = []
    if CHECKPOINT.exists():
        with open(CHECKPOINT) as f:
            ck = json.load(f)
            start_idx = ck.get("processed", 0)
            if OUTPUT_CSV.exists():
                results = pd.read_csv(OUTPUT_CSV).to_dict("records")
        print(f"Resuming from variant {start_idx}/{total}")

    t_start = time.perf_counter()
    batch_num = 0

    for i in range(start_idx, total, BATCH_SIZE):
        batch_end = min(i + BATCH_SIZE, total)
        batch_df = df.iloc[i:batch_end]

        # Build batch payload
        batch_requests = []
        for _, row in batch_df.iterrows():
            data = parse_variant_id(row["variant_id"])
            # Omit optional fields (gene_symbol, vep_annotation) to avoid Pydantic v2 422 errors
            batch_requests.append(data)

        # Call endpoint
        batch_results = call_batch_endpoint(batch_requests)

        # Extract results
        for idx, (_, row) in enumerate(batch_df.iterrows()):
            vid = row["variant_id"]
            label = row["ClinicalSignificance"]

            if idx < len(batch_results):
                res = batch_results[idx]
                pred = res.get("prediction", "ERROR")
                delta = res.get("delta_score")
                ref = res.get("reference")
            else:
                pred = "ERROR"
                delta = None
                ref = None

            results.append({
                "variant_id": vid,
                "label": label,
                "prediction": pred,
                "delta_score": delta,
                "reference": ref,
            })

        batch_num += 1

        # Checkpoint every batch
        pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
        with open(CHECKPOINT, "w") as f:
            json.dump(
                {"processed": batch_end, "timestamp": datetime.now().isoformat()},
                f,
                indent=2
            )

        elapsed = time.perf_counter() - t_start
        rate = batch_end / elapsed if elapsed > 0 else 0
        eta = (total - batch_end) / rate if rate > 0 else 0
        print(f"  [{batch_end}/{total}] Batch {batch_num} done | {rate:.1f} var/s | ETA: {eta/60:.0f}m")

    elapsed = time.perf_counter() - t_start
    print(f"\nDone! {total} variants in {elapsed/60:.1f} minutes")
    print(f"Results: {OUTPUT_CSV}")
    print(f"\nNext step: python benchmark_postprocess.py")

if __name__ == "__main__":
    main()
