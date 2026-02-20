
import modal
import sys
import os

# Add directory to path to find the module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from survival_crispr_with_evo2 import app, therapeutic_index_pipeline_real_evo2

@app.local_entrypoint()
def run_pipeline():
    print("🚀 Launching Survival-Guided CRISPR Pipeline with Real Evo2...")
    print("   - Survival Model: T4 GPU")
    print("   - Evo2 Scorer: H100 GPU")
    
    # Run pipeline
    result = therapeutic_index_pipeline_real_evo2.remote()
    
    print("\n✅ Pipeline Completed!")
    print(f"Top Recommendation: {result['top_recommendation']['guide_id']}")
    print(f"Clinical Summary: {result['clinical_summary']}")
