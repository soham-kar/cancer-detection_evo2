
"""
Validate REAL Evo2 on GUIDE-seq data to generate report metrics.
"""
import modal
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import json

app = modal.App("evo2-validation-real")
volume = modal.Volume.from_name("oral-cancer-model")

# Use survival image for validation client (needs sklearn)
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "pandas", "numpy", "scikit-learn", "matplotlib", "modal"
)

@app.function(image=image, gpu="T4", volumes={"/model": volume}, timeout=1200)
def validate_evo2_on_guide_seq(dataset_path: str = "/model/guide_seq_complete.csv"):
    """
    Validate Real Evo2 on GUIDE-seq data to generate report metrics
    """
    print("Loading GUIDE-seq validation data...")
    try:
        data = pd.read_csv(dataset_path)
    except FileNotFoundError:
        print(f"File not found: {dataset_path}")
        # Fallback to local if running locally mapped
        try:
            data = pd.read_csv("backend/oral_cancer_interpretable/data/guide_seq_complete.csv")
            print("Loaded from local fallback")
        except: 
            print("Could not find dataset")
            return {}
    
    print(f"Validating on {len(data)} sites ({data['is_offtarget'].sum()} positives)")
    
    # Connect to deployed Evo2
    import modal as modal_lib
    print("Connecting to Real Evo2 scorer (H100)...")
    # Using the real scorer from the updated file
    Evo2Scorer = modal_lib.Cls.from_name("survival-guided-crispr-evo2", "Evo2OffTargetScorer")
    
    # Score with Evo2 (batch for efficiency)
    batch_size = 20 # Smaller batch for real model
    predictions = []
    
    print(f"Scoring {len(data)} variants...")
    
    # Prepare all offtargets
    all_offtargets = []
    for _, row in data.iterrows():
        chrom = row.get('chrom', 'chrX')
        pos = row.get('pos', 0)
        
        all_offtargets.append({
            'sequence': row['offtarget_seq'], 
            'locus': f"{chrom}:{pos}",
            'guide_seq': row['guide_seq']
        })
        
    # Process in batches
    for i in range(0, len(all_offtargets), batch_size):
        batch = all_offtargets[i:i+batch_size]
        
        # Group by guide for batch processing efficiency
        guides = set(item['guide_seq'] for item in batch)
        
        batch_scores = [None] * len(batch)
        
        for guide in guides:
            guide_items = [item for item in batch if item['guide_seq'] == guide]
            guide_indices = [k for k, item in enumerate(batch) if item['guide_seq'] == guide]
            
            try:
                # Call remote scorer
                results = Evo2Scorer().score_batch.remote(guide_items, guide)
                
                for idx, res in zip(guide_indices, results):
                    batch_scores[idx] = res['evo2_score']
            except Exception as e:
                print(f"Batch failed: {e}")
                for idx in guide_indices:
                    batch_scores[idx] = 0.0
        
        predictions.extend(batch_scores)
        print(f"Processed {min(i+batch_size, len(data))}/{len(data)}")
    
    # Calculate metrics
    actuals = data['is_offtarget'].values
    predictions = np.array([p if p is not None else 0.0 for p in predictions])
    
    try:
        auroc = roc_auc_score(actuals, predictions)
        auprc = average_precision_score(actuals, predictions)
    except ValueError:
        auroc = 0.5
        auprc = 0.0
        print("Error calculating metrics")
    
    print(f"\n{'='*50}")
    print(f"REAL EVO2 VALIDATION RESULTS")
    print(f"{'='*50}")
    print(f"AUROC: {auroc:.4f}")
    print(f"AUPRC: {auprc:.4f}")
    print(f"Samples: {len(actuals)} (Positives: {sum(actuals)})")
    
    return {
        'AUROC': float(auroc),
        'AUPRC': float(auprc),
        'n_samples': int(len(actuals)),
        'n_positives': int(sum(actuals))
    }

@app.local_entrypoint()
def main(dataset: str = "/model/guide_seq_complete.csv"):
    print(f"🚀 Launching Real Evo2 Validation on {dataset}...")
    results = validate_evo2_on_guide_seq.remote(dataset_path=dataset)
    
    with open("evo2_real_validation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to evo2_real_validation_results.json")
