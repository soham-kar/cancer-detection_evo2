
"""
Batch analysis of therapeutic potential on high-MYC patient cohort.
Runs on T4 GPU, calls H100 Evo2 scorer via the pipeline.
"""
import modal
import pandas as pd
import time

app = modal.App("batch-crispr-analysis")
image = modal.Image.debian_slim(python_version="3.11").pip_install("pandas")

@app.function(image=image, gpu="T4", timeout=3600)
def analyze_patient_batch(patient_indices: list):
    """
    Run therapeutic index pipeline on multiple patients
    """
    # Lookup the pipeline function from the DEPLOYED app
    import modal as modal_lib
    
    # Note: Use Function.from_name for a @app.function
    # Our pipeline is a @app.function called 'therapeutic_index_pipeline_real_evo2'
    try:
        Pipeline = modal_lib.Function.from_name("survival-guided-crispr-evo2", "therapeutic_index_pipeline_real_evo2")
        print("Connected to deployed pipeline.")
    except Exception as e:
        print(f"Error connecting to pipeline: {e}")
        return []
    
    results = []
    for idx in patient_indices:
        print(f"Processing patient index {idx}...")
        try:
            # Call remote pipeline
            result = Pipeline.remote(patient_idx=idx, top_k_guides=5)
            results.append(result)
        except Exception as e:
            print(f"Error on patient {idx}: {e}")
            results.append({'patient_idx': idx, 'candidate': False, 'reason': str(e)})
        
        # Avoid rate limits if any
        time.sleep(1)
    
    return results

@app.local_entrypoint()
def main(n_patients: int = 10):
    print(f"Starting batch analysis on {n_patients} high-MYC patients...")
    # Analyze first N high-MYC patients
    patient_ids = list(range(n_patients))
    
    start_time = time.time()
    batch_results = analyze_patient_batch.remote(patient_ids)
    duration = time.time() - start_time
    
    # Create summary table
    summary = []
    for r in batch_results:
        # Check if successful candidate
        if r.get('candidate'):
            top = r.get('top_recommendation', {})
            summary.append({
                'patient_idx': r.get('patient_idx'),
                'myc_expr': r.get('myc_norm_expr', 0), # Optional if returned
                'survival_benefit': r.get('survival_benefit', 0),
                'best_guide': top.get('guide_id', 'N/A'),
                'therapeutic_index': top.get('therapeutic_index', 0),
                'safety_score': top.get('safety_score', 0),
                'category': top.get('category', 'UNKNOWN')
            })
        else:
             summary.append({
                'patient_idx': r.get('patient_idx'),
                'candidate': False,
                'survival_benefit': r.get('survival_benefit', 0), 
                'therapeutic_index': 0,
                'category': 'REJECT',
                'reason': r.get('reason')
            })
    
    df = pd.DataFrame(summary)
    output_filename = "crispr_therapeutic_results.csv"
    df.to_csv(output_filename, index=False)
    
    print(f"\nAnalyzed {len(summary)} patients in {duration:.1f}s")
    print("\nResults Summary:")
    print(df[['patient_idx', 'survival_benefit', 'therapeutic_index', 'category']])
    
    print(f"\nPriority candidates: {sum(df['category'] == 'PRIORITY')}")
    print(f"Acceptable candidates: {sum(df['category'] == 'ACCEPTABLE')}")
    print(f"Rejected: {sum(df['category'] == 'REJECT')}")
    print(f"\nResults saved to {output_filename}")
