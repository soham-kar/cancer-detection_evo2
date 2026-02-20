"""
Test Single Variant with Evo2 on Modal

Quick test to verify Evo2 scoring works
"""
import modal

# Import the working image
from modal_evo2_production import evo2_image, volume, mount_path

test_app = modal.App("test-evo2")

@test_app.function(image=evo2_image, gpu="H100", volumes={mount_path: volume}, timeout=600)
def test_single_score():
    from evo2 import Evo2
    import numpy as np
    
    print("Loading Evo2...")
    model = Evo2('evo2_7b')
    print("✅ Model loaded!")
    
    # Simple test: score a short sequence
    ref_seq = "ATCGATCGATCG"
    alt_seq = "ATCGTTCGATCG"  # T->T mutation at position 5
    
    print(f"\nTest sequences:")
    print(f"  REF: {ref_seq}")
    print(f"  ALT: {alt_seq}")
    
    try:
        score = model.delta_log_likelihood(ref_seq, alt_seq, position=5)
        print(f"\n✅ Score: {score}")
        return {"success": True, "score": float(score)}
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

@test_app.local_entrypoint()
def main():
    print("="*70)
    print("EVO2 SINGLE VARIANT TEST")
    print("="*70)
    result = test_single_score.remote()
    print(f"\nResult: {result}")
    print("="*70)
