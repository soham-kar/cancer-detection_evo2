"""
Validate Evo2 on GUIDE-seq data to generate report metrics.
"""
import modal
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import json

app = modal.App("evo2-validation")
volume = modal.Volume.from_name("oral-cancer-model")

# Use survival image for validation client (needs sklearn)
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "pandas", "numpy", "scikit-learn", "matplotlib", "modal"
)

@app.function(image=image, gpu="T4", volumes={"/model": volume}, timeout=600)
def validate_evo2_on_guide_seq(dataset_path: str = "/model/guide_seq_complete.csv"):
    """
    Validate Evo2 on GUIDE-seq data to generate report metrics
    """
    print("Loading GUIDE-seq validation data...")
    try:
        data = pd.read_csv(dataset_path)
    except FileNotFoundError:
        print(f"File not found: {dataset_path}")
        return {}
    
    # Filter for high-confidence off-targets or all
    # For mock data, trust 'is_offtarget' column
    print(f"Validating on {len(data)} sites ({data['is_offtarget'].sum()} positives)")
    
    # Connect to deployed Evo2
    # Use Cls.from_name for cross-app lookup
    import modal as modal_lib
    print("Connecting to Evo2 scorer...")
    Evo2Scorer = modal_lib.Cls.from_name("survival-guided-crispr-evo2", "Evo2OffTargetScorer")
    
    # Score with Evo2 (batch for efficiency)
    batch_size = 50
    predictions = []
    
    print(f"Scoring {len(data)} variants...")
    
    # Prepare all offtargets
    all_offtargets = []
    for _, row in data.iterrows():
        # Handle optional locus info
        chrom = row['chrom'] if 'chrom' in row else 'chrX'
        pos = row['pos'] if 'pos' in row else 0
        
        all_offtargets.append({
            'sequence': row['offtarget_seq'], 
            'locus': f"{chrom}:{pos}",
            'guide_seq': row['guide_seq']
        })
        
    # Process in batches
    for i in range(0, len(all_offtargets), batch_size):
        batch = all_offtargets[i:i+batch_size]
        # Group by guide_seq (assuming batch has same guide or we handle it)
        # score_batch takes ONE guide_seq. We must group by guide.
        # Simplification: Invoke per item or group by guide
        
        # Group batch by guide
        guides = set(item['guide_seq'] for item in batch)
        
        batch_scores = [None] * len(batch)
        
        for guide in guides:
            guide_items = [item for item in batch if item['guide_seq'] == guide]
            guide_indices = [k for k, item in enumerate(batch) if item['guide_seq'] == guide]
            
            # Call remote scorer
            results = Evo2Scorer().score_batch.remote(guide_items, guide)
            
            for idx, res in zip(guide_indices, results):
                batch_scores[idx] = res['evo2_score']
        
        predictions.extend(batch_scores)
        print(f"Processed {min(i+batch_size, len(data))}/{len(data)}")
    
    # Calculate metrics
    actuals = data['is_offtarget'].values
    predictions = np.array(predictions)
    
    try:
        auroc = roc_auc_score(actuals, predictions)
        auprc = average_precision_score(actuals, predictions)
    except ValueError:
        auroc = 0.5
        auprc = 0.0
        print("Error calculating metrics (maybe only one class present)")
    
    # Generate ROC curve data
    fpr, tpr, thresholds = roc_curve(actuals, predictions)
    
    results = {
        'AUROC': float(auroc),
        'AUPRC': float(auprc),
        'n_samples': len(actuals),
        'n_positives': int(sum(actuals)),
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'thresholds': thresholds.tolist()
    }
    
    print(f"\n{'='*50}")
    print(f"EVO2 VALIDATION RESULTS")
    print(f"{'='*50}")
    print(f"AUROC: {auroc:.3f}")
    print(f"AUPRC: {auprc:.3f}")
    print(f"Samples: {len(actuals)} (Positives: {sum(actuals)})")
    
    return results

@app.local_entrypoint()
def main():
    results = validate_evo2_on_guide_seq.remote()
    
    with open("evo2_validation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to evo2_validation_results.json")
