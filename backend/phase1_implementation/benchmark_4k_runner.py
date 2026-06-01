"""
Full 4K ClinVar Benchmark Runner — Background execution with checkpointing.
Uses the single-variant endpoint (includes AlphaMissense + consensus).
"""
import requests
import pandas as pd
import json
import time
from pathlib import Path
from datetime import datetime

URL = "https://karsoham529--variant-analysis-evo2model-analyze-single-variant.modal.run"
INPUT_CSV = Path("clinvar_benchmark_4k_stratified.csv")
OUTPUT_CSV = Path("helixmind_benchmark_results.csv")
CHECKPOINT = Path("benchmark_checkpoint.json")

def parse_variant_id(vid):
    parts = vid.split("-")
    return {
        "chromosome": parts[0],
        "variant_position": int(parts[1]),
        "reference": parts[2],
        "alternative": parts[3],
        "genome": "hg38",
    }

def call_endpoint(data):
    try:
        r = requests.post(URL, json=data, timeout=90)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def main():
    df = pd.read_csv(INPUT_CSV)
    total = len(df)
    
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
    
    print(f"Benchmarking {total} variants...")
    print(f"Output: {OUTPUT_CSV}")
    print("=" * 60)
    
    t_start = time.perf_counter()
    
    for i in range(start_idx, total):
        row = df.iloc[i]
        vid = row["variant_id"]
        label = row["ClinicalSignificance"]
        
        data = parse_variant_id(vid)
        resp = call_endpoint(data)
        
        # Extract fields
        pred = resp.get("prediction", "ERROR")
        delta = resp.get("delta_score")
        conf = resp.get("classification_confidence")
        am = resp.get("external_scores", {}).get("alphamissense")
        consensus = resp.get("multi_model_consensus", {})
        
        result = {
            "variant_id": vid,
            "label": label,
            "prediction": pred,
            "delta_score": delta,
            "confidence": conf,
            "alphamissense_score": am["score"] if am else None,
            "alphamissense_class": am["classification"] if am else None,
            "consensus_class": consensus.get("consensus_classification"),
            "consensus_confidence": consensus.get("consensus_confidence"),
            "models_agree": consensus.get("models_agree"),
            "models_total": consensus.get("models_total"),
            "error": resp.get("error"),
        }
        results.append(result)
        
        # Save checkpoint every 50
        if (i + 1) % 50 == 0 or i == total - 1:
            pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
            with open(CHECKPOINT, "w") as f:
                json.dump({"processed": i + 1, "timestamp": datetime.now().isoformat()}, f)
            elapsed = time.perf_counter() - t_start
            rate = (i + 1 - start_idx) / elapsed
            eta = (total - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{total}] {vid} | {pred:20s} | AM: {am['score']:.3f if am else 'N/A':6s} | Consensus: {consensus.get('consensus_classification', 'N/A'):15s} | ETA: {eta/60:.0f}m")
    
    elapsed = time.perf_counter() - t_start
    print(f"\nDone! {total} variants in {elapsed/60:.1f} minutes")
    print(f"Results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
