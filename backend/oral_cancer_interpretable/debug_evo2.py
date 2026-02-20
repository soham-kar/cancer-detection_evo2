"""
Debug Evo2 connection and scoring.
Verify that the deployed H100 function is reachable and returning real scores.
"""
import modal

app = modal.App("debug-evo2")
volume = modal.Volume.from_name("oral-cancer-model")

@app.function(image=modal.Image.debian_slim().pip_install("modal"), gpu="T4", volumes={"/model": volume}, timeout=600)
def debug_evo2_connection():
    """Verify real Evo2 is accessible and working"""
    import modal
    
    print("Testing Evo2 connection...")
    
    # Try to lookup the class from the deployed app
    try:
        # Note: Use Cls.from_name since we are calling a class in another app
        # survival-guided-crispr-evo2 is the app name
        # Evo2OffTargetScorer is the class name
        Evo2Scorer = modal.Cls.from_name("survival-guided-crispr-evo2", "Evo2OffTargetScorer")
        print("✓ Evo2 class found")
    except Exception as e:
        print(f"✗ Cannot find Evo2 class: {e}")
        return {"status": "error", "message": "Evo2 not found"}
    
    # Test a single call
    test_grna = "GAGGGTCATTTCCCCTAGCG"
    test_target = "GAGGGTCATTTCCCCTAGCG"  # Perfect match (should be high risk)
    
    print(f"\nTesting with perfect match...")
    print(f"gRNA:  {test_grna}")
    print(f"Target: {test_target}")
    
    try:
        # Instantiate and call remote method
        scorer_instance = Evo2Scorer()
        result = scorer_instance.score_offtarget.remote(test_grna, test_target)
        
        print(f"\n✓ Evo2 response received:")
        print(f"  Score: {result.get('evo2_score', 'N/A')}")
        print(f"  Delta LL: {result.get('delta_ll', 'N/A')}")
        print(f"  Risk level: {result.get('risk_level', 'N/A')}")
        print(f"  Error/Warning: {result.get('error', 'None')}")
        
        # Verify it's not mock data
        if result.get('delta_ll') is None:
            print("⚠️  WARNING: delta_ll is None - using fallback!")
            score_type = "fallback"
        else:
            print("✓ Real Evo2 is working (delta_ll present)")
            score_type = "real"
            
        return {
            "status": "success", 
            "result": result,
            "type": score_type
        }
            
    except Exception as e:
        print(f"✗ Evo2 call failed: {e}")
        return {"status": "error", "message": str(e)}

@app.local_entrypoint()
def main():
    result = debug_evo2_connection.remote()
    print("\n" + "="*60)
    print(result)
