"""Quick test of batch endpoint with 16 variants."""
import requests
import pandas as pd

URL = "https://karsoham529--variant-analysis-evo2model-analyze-batch.modal.run"

def parse_variant_id(vid):
    parts = vid.split("-")
    return {
        "chromosome": parts[0],
        "variant_position": int(parts[1]),
        "reference": parts[2],
        "alternative": parts[3],
        "genome": "hg38",
    }

# Load first 16 variants
df = pd.read_csv("clinvar_benchmark_4k_stratified.csv")
batch = df.head(16)

requests_list = [parse_variant_id(row["variant_id"]) for _, row in batch.iterrows()]

print(f"Sending batch of {len(requests_list)} variants...")
resp = requests.post(URL, json={"requests": requests_list}, timeout=120)
print(f"Status: {resp.status_code}")

if resp.status_code == 200:
    data = resp.json()
    print(f"Batch size: {data.get('batch_size')}")
    print(f"Chunks: {data.get('chunks_processed')}")
    print(f"Precision: {data.get('precision')}")
    results = data.get("batch_results", [])
    print(f"Results returned: {len(results)}")
    for i, r in enumerate(results[:3]):
        print(f"  {i+1}. {r.get('prediction')} | delta={r.get('delta_score'):.6f}")
else:
    print(f"Error: {resp.text[:500]}")
