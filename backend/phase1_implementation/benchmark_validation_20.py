"""Quick 20-variant validation before full 4K benchmark."""
import requests
import pandas as pd
import time

URL = "https://karsoham529--variant-analysis-evo2model-analyze-single-variant.modal.run"

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

# Load 4K dataset
df = pd.read_csv("clinvar_benchmark_4k_stratified.csv")

# Sample 5 per class
sampled = df.groupby("ClinicalSignificance").head(5)
print(f"Testing {len(sampled)} variants (5 per class)...\n")

results = []
for i, row in sampled.iterrows():
    vid = row["variant_id"]
    label = row["ClinicalSignificance"]
    data = parse_variant_id(vid)
    
    t0 = time.perf_counter()
    resp = call_endpoint(data)
    t1 = time.perf_counter()
    
    pred = resp.get("prediction", "ERROR")
    delta = resp.get("delta_score")
    am = resp.get("external_scores", {}).get("alphamissense")
    consensus = resp.get("multi_model_consensus", {}).get("consensus_classification")
    
    am_score = am["score"] if am else None
    am_str = f"{am_score:.4f}" if am_score is not None else "N/A"
    
    print(f"{i+1:2d}. {vid:30s} | Label: {label:20s} | Pred: {pred:20s} | Delta: {delta:10.6f} | AM: {am_str:8s} | Consensus: {consensus or 'N/A':15s} | {t1-t0:.1f}s")
    
    results.append({
        "variant_id": vid,
        "label": label,
        "prediction": pred,
        "delta_score": delta,
        "alphamissense_score": am["score"] if am else None,
        "consensus": consensus,
        "time": t1-t0
    })

# Summary
print("\n" + "="*60)
print("Validation Summary")
print("="*60)
for label in df["ClinicalSignificance"].unique():
    subset = [r for r in results if r["label"] == label]
    preds = [r["prediction"] for r in subset]
    print(f"{label:20s}: {preds}")

avg_time = sum(r["time"] for r in results) / len(results)
print(f"\nAverage time: {avg_time:.1f}s")
print(f"Total time: {sum(r['time'] for r in results):.1f}s")
