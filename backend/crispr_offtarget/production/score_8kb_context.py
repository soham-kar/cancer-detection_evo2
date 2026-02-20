"""
Score CRISPR Off-Targets with 8kb Genomic Context using Evo2

Since Kleinstiver data doesn't have real genomic coordinates, we'll:
1. Create synthetic random context (±4kb flanking)
2. Embed the 20bp target in the center
3. Score with Evo2 via Modal

This tests whether Evo2 gets better signal with longer context.
"""

import modal
import pandas as pd
import numpy as np
from pathlib import Path
import json
import ast

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results" / "modal_evo2_8kb"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Import local helpers
from crispr_scorer import (
    apply_position_weights, 
    calculate_seed_penalty, 
    estimate_confidence,
    POSITION_WEIGHTS
)


def generate_random_context(length: int, seed: int = 42) -> str:
    """Generate random DNA sequence for context."""
    np.random.seed(seed)
    return ''.join(np.random.choice(['A', 'C', 'G', 'T'], length))


def create_8kb_sequences(grna: str, target: str, context_seed: int) -> tuple:
    """
    Create 8kb sequences with gRNA/target embedded in center.
    
    Returns (ref_seq, var_seq) where:
    - ref_seq: 8kb context with perfect gRNA match
    - var_seq: 8kb context with target (with mismatches)
    """
    CONTEXT_LEN = 8192  # 8kb total
    SEQ_LEN = 20  # Standard gRNA length
    
    # Generate consistent random context (same for ref and var)
    np.random.seed(context_seed)
    left_flank = ''.join(np.random.choice(['A', 'C', 'G', 'T'], (CONTEXT_LEN - SEQ_LEN - 3) // 2))
    right_flank = ''.join(np.random.choice(['A', 'C', 'G', 'T'], (CONTEXT_LEN - SEQ_LEN - 3) // 2))
    
    pam = "TGG"  # Standard PAM
    
    # Reference: perfect gRNA match in context
    ref_seq = left_flank + grna.upper()[:20] + pam + right_flank
    
    # Variant: target with mismatches in same context
    var_seq = left_flank + target.upper()[:20] + pam + right_flank
    
    return ref_seq, var_seq


# Modal App for 8kb scoring
evo2_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12"
    )
    .apt_install(
        "build-essential", "cmake", "ninja-build", "libcudnn8",
        "libcudnn8-dev", "git", "gcc", "g++"
    )
    .env({"CC": "/usr/bin/gcc", "CXX": "/usr/bin/g++"})
    .pip_install("packaging", "wheel", "setuptools", "ninja")
    .run_commands(
        "git clone --recurse-submodules https://github.com/ArcInstitute/evo2.git && "
        "cd evo2 && pip install ."
    )
    .run_commands(
        "MAX_JOBS=6 pip install flash-attn==2.8.0.post2 --no-build-isolation",
        "MAX_JOBS=6 pip install transformer_engine[pytorch]==2.8.0 --no-build-isolation",
        gpu="L40S"
    )
    .pip_install(
        "biopython", "huggingface_hub", "torch", "vtx>=0.0.8", 
        "pandas", "scikit-learn", "numpy"
    )
    .env({"PYTHONPATH": "/root"})
)

app = modal.App("evonator-crispr-8kb", image=evo2_image)
volume = modal.Volume.from_name("hf_cache", create_if_missing=True)
mount_path = "/root/.cache/huggingface"


@app.cls(
    image=evo2_image,
    gpu="H100",
    timeout=3600,
    volumes={mount_path: volume},
    scaledown_window=1200,
)
class CRISPRScorer8KB:
    @modal.enter()
    def load_model(self):
        import torch
        print("Loading Evo2 model for 8kb context scoring...")
        from evo2 import Evo2
        self.model = Evo2("evo2_7b")
        print("Evo2 model loaded successfully")
    
    @modal.method()
    def score_batch_8kb(self, rows: list) -> list:
        """Score batch of rows with 8kb context."""
        import torch
        import numpy as np
        
        print(f"\n📊 Modal: Scoring {len(rows)} sites with 8kb context...")
        
        results = []
        batch_size = 4  # Smaller batch due to longer sequences
        
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i+batch_size]
            
            # Prepare sequences
            ref_seqs = []
            var_seqs = []
            
            for j, row in enumerate(batch):
                ref_seq, var_seq = create_8kb_sequences(
                    row['grna_sequence'][:20],
                    row['target_sequence'][:20],
                    context_seed=i + j + 42
                )
                ref_seqs.append(ref_seq)
                var_seqs.append(var_seq)
            
            # Score with Evo2
            with torch.no_grad():
                all_seqs = ref_seqs + var_seqs
                scores = self.model.score_sequences(all_seqs)
                
                ref_scores = scores[:len(ref_seqs)]
                var_scores = scores[len(ref_seqs):]
            
            # Process results
            for j, row in enumerate(batch):
                # Parse mismatch positions
                positions = row.get('mismatch_positions', [])
                if isinstance(positions, str):
                    try:
                        positions = ast.literal_eval(positions)
                    except:
                        positions = []
                
                # Calculate delta
                delta = var_scores[j] - ref_scores[j]
                
                # Apply position weighting
                weights = apply_position_weights(positions) if positions else np.array([1.0])
                weight_factor = np.mean(weights)
                weighted_delta = float(delta * weight_factor)
                
                # Confidence and seed penalty
                delta_per_pos = delta / max(len(positions), 1)
                delta_scores = [float(delta_per_pos * w) for w in weights] if positions else []
                confidence = estimate_confidence(delta_scores, positions)
                seed_penalty = calculate_seed_penalty(positions) if positions else 0
                
                results.append({
                    'grna_name': row.get('grna_name', ''),
                    'target_sequence': row.get('target_sequence', ''),
                    'mismatches': len(positions),
                    'weighted_delta_ll': weighted_delta,
                    'raw_delta': float(delta),
                    'confidence': confidence,
                    'seed_penalty': seed_penalty,
                    'is_validated': row.get('is_validated', False),
                    'context_length': 8192
                })
            
            print(f"   Processed {min(i + batch_size, len(rows))}/{len(rows)}")
        
        print(f"✅ Modal scored {len(results)} sites with 8kb context")
        return results


@app.local_entrypoint()
def main(
    input_file: str = "kleinstiver_balanced.csv",
    output_file: str = "kleinstiver_8kb_scored.csv",
    batch_size: int = 20
):
    """Score with 8kb context."""
    print("=" * 60)
    print("EVO2 CRISPR SCORING WITH 8KB CONTEXT")
    print("=" * 60)
    
    # Load data
    input_path = DATA_DIR / input_file
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        return
    
    df = pd.read_csv(input_path)
    print(f"\n📂 Loaded {len(df)} samples from {input_file}")
    
    # Convert to list of dicts
    rows = df.to_dict('records')
    
    # Split into batches
    batches = []
    for i in range(0, len(rows), batch_size):
        batches.append(rows[i:i+batch_size])
    
    print(f"   Created {len(batches)} batches of size {batch_size}")
    
    # Score via Modal
    model = CRISPRScorer8KB()
    
    all_results = []
    for batch_idx, batch_results in enumerate(model.score_batch_8kb.map(batches, order_outputs=True)):
        all_results.extend(batch_results)
        print(f"   Batch {batch_idx + 1}/{len(batches)} complete")
    
    # Save results
    results_df = pd.DataFrame(all_results)
    output_path = RESULTS_DIR / output_file
    results_df.to_csv(output_path, index=False)
    print(f"\n✅ Saved: {output_path}")
    
    # Calculate metrics
    from sklearn.metrics import roc_auc_score, average_precision_score
    
    scores = results_df['weighted_delta_ll'].values
    raw_deltas = results_df['raw_delta'].values
    labels = results_df['is_validated'].astype(int).values
    
    auroc_weighted = roc_auc_score(labels, scores)
    auroc_raw = roc_auc_score(labels, raw_deltas)
    
    print(f"\n{'='*60}")
    print("RESULTS (8KB CONTEXT)")
    print("="*60)
    print(f"AUROC (weighted): {auroc_weighted:.4f}")
    print(f"AUROC (raw delta): {auroc_raw:.4f}")
    
    # Save metrics
    metrics = {
        'context_length': 8192,
        'auroc_weighted': float(auroc_weighted),
        'auroc_raw': float(auroc_raw),
        'n_samples': len(results_df)
    }
    with open(RESULTS_DIR / 'metrics_8kb.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Saved: {RESULTS_DIR / 'metrics_8kb.json'}")


if __name__ == "__main__":
    # Local test
    main()
