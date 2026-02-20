"""
Step 3: Score all ref/alt sequence pairs with Evo2 on Modal GPU.

Reads:   data/sequences.json  (produced by 02_prepare_sequences.py)
Outputs: data/delta_scores.json

Run with:
    modal run 03_modal_score.py

The script:
  1. Uploads sequences.json to the Modal job
  2. Scores all unique ref sequences once (deduplication saves GPU time)
  3. Scores all alt sequences
  4. Computes delta_score = alt_score - ref_score for each variant
  5. Saves results locally
"""

import json
import modal
import subprocess
import sys
from pathlib import Path

# ─── Paths ─────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data"
IN_JSON  = DATA_DIR / "sequences.json"
OUT_JSON = DATA_DIR / "delta_scores.json"

# ─── Modal Image (reuses same setup as main.py) ──────────────────────────
def _build_cuda_kernels():
    import os, subprocess, sys
    os.environ["MAX_JOBS"] = "6"
    for pkg in ("flash-attn==2.8.0.post2", "transformer_engine[pytorch]==2.8.0"):
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "--no-build-isolation"])

evo2_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install("build-essential", "cmake", "ninja-build", "libcudnn8", "libcudnn8-dev", "git", "gcc", "g++")
    .env({"CC": "/usr/bin/gcc", "CXX": "/usr/bin/g++"})
    .pip_install("packaging", "wheel", "setuptools", "ninja")
    .run_commands(
        "git clone --recurse-submodules https://github.com/ArcInstitute/evo2.git && "
        "cd evo2 && pip install ."
    )
    .run_function(_build_cuda_kernels, gpu="L40S", memory=32768, cpu=8, timeout=3600)
    .pip_install("torch", "numpy")
)

volume     = modal.Volume.from_name("hf_cache", create_if_missing=True)
MOUNT_PATH = "/root/.cache/huggingface"

app = modal.App("clinvar-vus-scorer", image=evo2_image)


# ─── Remote scoring function ─────────────────────────────────────────────
@app.function(
    gpu="H100",
    volumes={MOUNT_PATH: volume},
    timeout=7200,          # 2 h — plenty for 5k variants
    memory=32768,
)
def score_sequences_remote(sequences: list[str]) -> list[float]:
    """Score a list of DNA sequences with Evo2. Returns mean log-likelihood per sequence."""
    from evo2 import Evo2
    print(f"Loading Evo2 model…")
    model = Evo2("evo2_7b")
    print(f"Scoring {len(sequences)} sequences…")
    scores = model.score_sequences(sequences)
    return [float(s) for s in scores]


# ─── Local entrypoint ────────────────────────────────────────────────────
@app.local_entrypoint()
def main(test: bool = False):
    """
    Args:
        --test  Run on 10 variants only, output → data/delta_scores_test.json
    """
    out_path = DATA_DIR / ("delta_scores_test.json" if test else "delta_scores.json")
    limit    = 10 if test else None

    print(f"Loading {IN_JSON}…")
    with open(IN_JSON) as fh:
        data = json.load(fh)

    unique_refs: list[str] = data["unique_refs"]
    variants:    list[dict] = data["variants"]

    if limit:
        variants    = variants[:limit]
        # Only keep ref windows actually needed for this subset
        needed_idxs = set(v["ref_idx"] for v in variants)
        # Remap ref_idx to a smaller list
        idx_map     = {old: new for new, old in enumerate(sorted(needed_idxs))}
        unique_refs = [unique_refs[i] for i in sorted(needed_idxs)]
        for v in variants:
            v["ref_idx"] = idx_map[v["ref_idx"]]
        print(f"  [TEST MODE] Capped to {limit} variants, {len(unique_refs)} unique ref windows")
    else:
        print(f"  Unique ref windows : {len(unique_refs):,}")
        print(f"  Total variants     : {len(variants):,}")

    # Score unique ref sequences (deduplication saves GPU time)
    print("\nScoring reference sequences…")
    ref_scores = score_sequences_remote.remote(unique_refs)
    print(f"  Done. {len(ref_scores)} ref scores")

    # Score alt sequences
    alt_seqs = [v["alt_seq"] for v in variants]
    print(f"\nScoring {len(alt_seqs):,} alternate sequences…")
    alt_scores = score_sequences_remote.remote(alt_seqs)
    print(f"  Done.")

    # Compute deltas
    results = []
    for v, alt_score in zip(variants, alt_scores):
        ref_score = ref_scores[v["ref_idx"]]
        delta     = alt_score - ref_score
        results.append({
            "id":          v["id"],
            "gene":        v["gene"],
            "chrom":       v["chrom"],
            "pos":         v["pos"],
            "ref":         v["ref"],
            "alt":         v["alt"],
            "ref_score":   ref_score,
            "alt_score":   alt_score,
            "delta_score": delta,
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)

    print(f"\nSaved {len(results):,} delta scores → {out_path}")
    if test:
        print("TEST PASSED ✓  Run without --test for the full dataset.")
        # Print a preview
        for r in results:
            flag = "PATH" if r["delta_score"] < -0.0009178519 else "BEN "
            print(f"  [{flag}]  {r['gene']:10s}  {r['chrom']}:{r['pos']}  Δ={r['delta_score']:.6f}")
    else:
        print("Done. Next: python 04_classify_and_plot.py")
