"""Test Phase 1 deployment: AlphaMissense + Consensus on BRCA1 variant."""
import requests, json, time

url = "https://karsoham529--variant-analysis-evo2model-analyze-single-variant.modal.run"

payload = {
    "variant_position": 43045629,
    "alternative": "C",
    "genome": "hg38",
    "chromosome": "chr17",
    "reference": "T",
    "gene_symbol": "BRCA1",
    "vep_annotation": {
        "consequence": "missense_variant",
        "impact": "MODERATE",
        "aminoAcids": "L/A",
        "uniprotId": "P38398",
        "aaPosition": 718,
        "refAA": "L",
        "altAA": "A",
        "transcriptId": "ENST00000471181",
        "codons": "Ctg/Gcg"
    }
}

print("Sending test request (BRCA1 L718A)...")
t0 = time.perf_counter()
resp = requests.post(url, json=payload, timeout=180)
t1 = time.perf_counter()

print(f"Status: {resp.status_code}")
print(f"Time: {t1-t0:.1f}s")

if resp.status_code == 200:
    data = resp.json()
    print(f"Prediction: {data.get('prediction')}")
    print(f"Delta: {data.get('delta_score')}")
    
    es = data.get("external_scores", {})
    if es.get("alphamissense"):
        am = es["alphamissense"]
        print(f"AlphaMissense: score={am['score']}, class={am['classification']}")
    else:
        print("AlphaMissense: NOT FOUND")
    
    if es.get("cadd"):
        print(f"CADD: PHRED={es['cadd']['phred']}")
    
    mc = data.get("multi_model_consensus")
    if mc:
        print(f"Consensus: {mc['consensus_classification']} ({mc['models_agree']}/{mc['models_total']} models)")
    else:
        print("Consensus: NOT COMPUTED")
else:
    print(f"Error: {resp.text[:500]}")
