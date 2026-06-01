"""
Small batch test with checkpointing + results saved to phase1 folder.
Tests the Modal /analyze_batch endpoint and saves incremental results.
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
URL = "https://karsoham529--variant-analysis-evo2model-analyze-batch.modal.run"
INPUT_CSV = Path("clinvar_benchmark_4k_stratified.csv")
OUTPUT_DIR = Path("backend/phase1_implementation")
OUTPUT_CSV = OUTPUT_DIR / "test_batch_results.csv"
CHECKPOINT = OUTPUT_DIR / "test_checkpoint.json"
BATCH_SIZE = 16
TEST_SIZE = 32  # Run on first 32 variants (2 batches) for quick validation

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
    """Call Modal batch endpoint."""
    payload = {"requests": batch_requests}
    try:
        r = requests.post(URL, json=payload, timeout=300)
        r.raise_for_status()
        data = r.json()
        return data.get("batch_results", [])
    except requests.exceptions.HTTPError as e:
        # Print full error detail for 422 debugging
        try:
            err_body = e.response.json()
            print(f"  422 DETAIL: {json.dumps(err_body, indent=2)[:800]}")
        except Exception:
            print(f"  422 BODY: {e.response.text[:500]}")
        return [{"error": str(e)}] * len(batch_requests)
    except Exception as e:
        print(f"  ERROR: {e}")
        return [{"error": str(e)}] * len(batch_requests)

def main():
    df = pd.read_csv(INPUT_CSV)
    test_df = df.head(TEST_SIZE)
    total = len(test_df)

    print(f"Testing batch endpoint with {total} variants ({total//BATCH_SIZE} batches)")
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
        batch_df = test_df.iloc[i:batch_end]

        # Build payload — omit optional fields instead of sending None
        # (Pydantic v2 rejects None for str fields)
        batch_requests = []
        for _, row in batch_df.iterrows():
            data = parse_variant_id(row["variant_id"])
            # gene_symbol and vep_annotation are optional with defaults;
            # omitting them avoids 422 validation errors
            batch_requests.append(data)

        print(f"\nBatch {batch_num+1}: variants {i+1}-{batch_end}")
        batch_results = call_batch_endpoint(batch_requests)

        # Extract and store
        for idx, (_, row) in enumerate(batch_df.iterrows()):
            vid = row["variant_id"]
            label = row["ClinicalSignificance"]

            if idx < len(batch_results):
                res = batch_results[idx]
                pred = res.get("prediction", "ERROR")
                delta = res.get("delta_score")
                ref = res.get("reference")
                conf = res.get("classification_confidence")
            else:
                pred, delta, ref, conf = "ERROR", None, None, None

            results.append({
                "variant_id": vid,
                "label": label,
                "prediction": pred,
                "delta_score": delta,
                "reference": ref,
                "confidence": conf,
            })

        batch_num += 1

        # SAVE after every batch (incremental download)
        pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
        with open(CHECKPOINT, "w") as f:
            json.dump({
                "processed": batch_end,
                "timestamp": datetime.now().isoformat(),
                "batches_done": batch_num
            }, f, indent=2)

        elapsed = time.perf_counter() - t_start
        rate = batch_end / elapsed if elapsed > 0 else 0
        print(f"  Saved {OUTPUT_CSV} | {batch_end}/{total} done | {rate:.1f} var/s")

    elapsed = time.perf_counter() - t_start
    print(f"\n{'='*60}")
    print(f"Done! {total} variants in {elapsed:.1f}s")
    print(f"Results: {OUTPUT_CSV}")
    print(f"Checkpoint: {CHECKPOINT}")

    # Quick summary
    out_df = pd.read_csv(OUTPUT_CSV)
    print(f"\nSummary:")
    print(f"  Total rows: {len(out_df)}")
    print(f"  Predictions: {out_df['prediction'].value_counts().to_dict()}")
    print(f"  Avg delta_score: {out_df['delta_score'].mean():.6f}")

if __name__ == "__main__":
    main()
