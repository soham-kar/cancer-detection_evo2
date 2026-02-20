"""
Simple test workflow for modal_extract.py
Generates mock data, uploads, runs extraction, downloads, validates
"""
import json
import random
import subprocess
from pathlib import Path

def generate_dna(length=8000):
    return ''.join(random.choice('ATCG') for _ in range(length))

def main():
    print("=" * 60)
    print("MODAL EXTRACT - SIMPLE TEST")
    print("=" * 60)
    
    # Step 1: Generate test data
    print("\n[1/5] Generating test data (5 samples)...")
    Path("data/features").mkdir(parents=True, exist_ok=True)
    
    data = [{
        'seq_id': f'test_{i}',
        'sequence_8kb': generate_dna(8000),
        'chrom': 'chr1',
        'center': 1000000 + i * 10000
    } for i in range(5)]
    
    with open("data/features/test_input.json", 'w') as f:
        json.dump(data, f)
    print("✅ Created data/features/test_input.json")
    
    # Step 2: Upload to Modal
    print("\n[2/5] Uploading to Modal volume...")
    result = subprocess.run([
        "modal", "volume", "put", "--force", "crispr-data",
        "data/features/test_input.json", "/data/test_input.json"
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Upload failed: {result.stderr}")
        return
    print("✅ Uploaded to /data/test_input.json")
    
    # Step 3: Run extraction
    print("\n[3/5] Running Evo2 extraction (this takes ~1 min)...")
    result = subprocess.run([
        "modal", "run", "modal_extract.py::process_dataset",
        "--input-path", "/data/test_input.json",
        "--output-path", "/data/test_output.json",
        "--limit", "5"
    ], capture_output=True, text=True)
    
    print(result.stdout)
    if result.returncode != 0:
        print(f"❌ Extraction failed: {result.stderr}")
        return
    
    # Step 4: Download results
    print("\n[4/5] Downloading results...")
    result = subprocess.run([
        "modal", "volume", "get", "crispr-data",
        "/data/test_output.json", "data/features/"
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Download failed: {result.stderr}")
        return
    print("✅ Downloaded to data/features/test_output.json")
    
    # Step 5: Validate
    print("\n[5/5] Validating results...")
    with open("data/features/test_output.json") as f:
        results = json.load(f)
    
    success = sum(1 for x in results if x.get('success'))
    print(f"Total: {len(results)}, Success: {success}")
    
    if success == 0:
        print("❌ All failed!")
        print(f"Error: {results[0].get('error')}")
        return
    
    if success < len(results):
        print(f"⚠️  {len(results) - success} failures")
    
    print(f"Sample score: {results[0].get('evo2_score')}")
    print("\n" + "=" * 60)
    print("✅ TEST PASSED - Ready for full run")
    print("=" * 60)
    print("\nTo run full extraction:")
    print("  1. Uncomment 'data = data[:5]' in modal_extract.py")
    print("  2. Run: modal run modal_extract.py")

if __name__ == "__main__":
    main()
