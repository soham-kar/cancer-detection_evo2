"""
Quick test script for modal_extract.py
Tests with 5 samples before running full 1000
"""
import subprocess
import json
from pathlib import Path

def test_extraction():
    print("=" * 60)
    print("MODAL EXTRACT TEST (5 samples)")
    print("=" * 60)
    
    # Check if input exists
    input_file = Path("data/features/input1k.json")
    if not input_file.exists():
        print(f"ERROR: {input_file} not found")
        print("Create test input first")
        return False
    
    print("\n>>> Running Modal extraction (5 samples)...")
    result = subprocess.run([
        "modal", "run", "modal_extract.py::process_dataset",
        "--input-path", "/data/input1k.json",
        "--output-path", "/data/test_5.json",
        "--limit", "5"
    ], capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    if result.returncode != 0:
        print("❌ Extraction failed")
        return False
    
    print("\n>>> Downloading results...")
    result = subprocess.run([
        "modal", "volume", "get", "--force",
        "crispr-data", "test_5.json", "data/features/"
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print("❌ Download failed:", result.stderr)
        return False
    
    print("\n>>> Validating results...")
    try:
        with open("data/features/test_5.json") as f:
            data = json.load(f)
        
        success = sum(1 for x in data if x.get('success'))
        print(f"Total: {len(data)}, Success: {success}")
        
        if success == 0:
            print("❌ All extractions failed")
            print("First error:", data[0].get('error'))
            return False
        
        if success < len(data):
            print(f"⚠️  {len(data) - success} failures")
            for i, item in enumerate(data):
                if not item.get('success'):
                    print(f"  Failed {i}: {item.get('error')}")
        
        print(f"\n✅ Test passed: {success}/{len(data)} successful")
        print(f"Sample score: {data[0].get('evo2_score')}")
        return True
        
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return False

if __name__ == "__main__":
    success = test_extraction()
    
    if success:
        print("\n" + "=" * 60)
        print("✅ TEST PASSED - Ready for full run")
        print("=" * 60)
        print("\nRun full extraction:")
        print("  modal run modal_extract.py")
    else:
        print("\n" + "=" * 60)
        print("❌ TEST FAILED - Fix issues before full run")
        print("=" * 60)
