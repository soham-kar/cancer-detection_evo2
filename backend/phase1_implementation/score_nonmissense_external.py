"""
score_nonmissense_external.py
==============================
Score non-missense ClinVar variants via Modal Evo2 batch endpoint.
Handles cold starts, retries, and checkpointing.

Usage:
    python backend/phase1_implementation/score_nonmissense_external.py
"""

import requests
import pandas as pd
import json
import time
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

BATCH_URL = "https://karsoham529--variant-analysis-evo2model-analyze-batch.modal.run"
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_CSV = SCRIPT_DIR / "clinvar" / "external_nonmissense_test.csv"
OUTPUT_CSV = SCRIPT_DIR / "clinvar" / "external_nonmissense_scored.csv"
CHECKPOINT = SCRIPT_DIR / "clinvar" / "nonmissense_checkpoint.json"
BATCH_SIZE = 16
MAX_RETRIES = 5
COLD_START_TIMEOUT = 300  # 5 min for H100 cold start
WARM_TIMEOUT = 120


def parse_variant_id(vid: str) -> dict:
    """Parse 'chr1-1035275-A-G' into VariantRequest dict."""
    parts = vid.split("-")
    return {
        "chromosome": parts[0],
        "variant_position": int(parts[1]),
        "reference": parts[2],
        "alternative": parts[3],
        "genome": "hg38",
    }


def call_batch_endpoint(batch_requests: list, timeout: int) -> list:
    """Call Modal batch endpoint with retry on cold start."""
    payload = {"requests": batch_requests}

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.post(BATCH_URL, json=payload, timeout=timeout)
            if r.status_code == 200:
                data = r.json()
                return data.get("batch_results", [])
            elif r.status_code == 422:
                # Pydantic validation error — print details once
                if attempt == 1:
                    print(f"    422 response: {r.text[:300]}")
                time.sleep(5 * attempt)
            else:
                print(f"    HTTP {r.status_code}, retrying...")
                time.sleep(10 * attempt)
        except requests.exceptions.ReadTimeout:
            print(f"    Timeout (cold start?), retrying with longer timeout...")
            timeout = min(timeout + 120, 600)
            time.sleep(15 * attempt)
        except Exception as e:
            print(f"    Error: {e}, retrying...")
            time.sleep(10 * attempt)

    # All retries exhausted
    print(f"    FAILED after {MAX_RETRIES} attempts")
    return [{"error": "max_retries_exceeded"}] * len(batch_requests)


def main():
    df = pd.read_csv(INPUT_CSV)
    total = len(df)
    print(f"Scoring {total} non-missense variants via Modal Evo2...")
    print(f"Batch size: {BATCH_SIZE} | ~{total // BATCH_SIZE + 1} batches")
    print(f"Input: {INPUT_CSV}")
    print(f"Output: {OUTPUT_CSV}")
    print("=" * 60)

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

    # Skip warmup — first batch will trigger cold start naturally
    first_batch = True

    t_start = time.perf_counter()
    batch_num = 0

    for i in range(start_idx, total, BATCH_SIZE):
        batch_end = min(i + BATCH_SIZE, total)
        batch_num += 1
        total_batches = (total - start_idx + BATCH_SIZE - 1) // BATCH_SIZE

        print(f"\nBatch {batch_num}/{total_batches} (variants {i+1}-{batch_end})...", end=" ", flush=True)

        # Build payload
        batch_requests = []
        for _, row in df.iloc[i:batch_end].iterrows():
            batch_requests.append(parse_variant_id(row["variant_id"]))

        # Call endpoint (longer timeout for first batch = cold start)
        timeout = COLD_START_TIMEOUT if first_batch else WARM_TIMEOUT
        batch_results = call_batch_endpoint(batch_requests, timeout)
        first_batch = False

        # Extract results
        scored = 0
        for j, (_, row) in enumerate(df.iloc[i:batch_end].iterrows()):
            vid = row["variant_id"]
            label = row["label"]

            if j < len(batch_results) and "error" not in batch_results[j]:
                res = batch_results[j]
                delta = res.get("delta_score")
                pred = res.get("prediction", "")
                scored += 1
            else:
                delta = None
                pred = "ERROR"

            results.append({
                "variant_id": vid,
                "label": label,
                "prediction": pred,
                "delta_score": delta,
                "reference": row["reference"],
                "alphamissense_score": None,  # Non-missense = no AM
            })

        print(f"{scored}/{len(batch_requests)} scored")

        # Save checkpoint
        with open(CHECKPOINT, "w") as f:
            json.dump({"processed": batch_end, "total": total}, f)

        # Save intermediate results
        pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)

    elapsed = time.perf_counter() - t_start
    print(f"\n{'=' * 60}")
    print(f"Complete! {len(results)} variants scored in {elapsed:.0f}s")

    # Final save
    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_CSV, index=False)

    scored_count = out_df["delta_score"].notna().sum()
    print(f"Successfully scored: {scored_count}/{len(out_df)}")
    print(f"Output: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
