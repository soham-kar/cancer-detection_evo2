import requests

endpoint = "https://karsoham529--variant-analysis-evo2model-analyze-single-variant.modal.run"

test_variant = {
    "variant_position": 186627650,
    "alternative": "GA",
    "genome": "hg38",
    "chromosome": "chr4",
    "reference": "A",  # Fixed: actual reference is A, not G
    "gene_symbol": "FAT1"
}

print(f"Testing: {endpoint}")
print("Sending test variant...")

try:
    response = requests.post(endpoint, json=test_variant, timeout=60)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Success! Delta score: {result.get('delta_score')}")
    else:
        print(f"Error: {response.text}")
except Exception as e:
    print(f"Failed: {e}")
