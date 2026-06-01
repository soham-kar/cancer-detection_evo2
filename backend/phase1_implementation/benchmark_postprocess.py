"""
Benchmark Post-Processing: Add AlphaMissense + Consensus (Local, Free)
=======================================================================
Reads raw Evo2 results from helixmind_benchmark_results.csv,
queries local DuckDB for AlphaMissense scores, computes consensus,
and saves enriched output.

Usage:
    python benchmark_postprocess.py

Requires:
    - alphamissense.duckdb (built locally, ~9 GB)
    - helixmind_benchmark_results.csv (from benchmark_batch_runner.py)
"""

import pandas as pd
import sys
from pathlib import Path

# Add phase1_implementation to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from alphamissense_lookup import AlphaMissenseDB
from consensus_engine import ConsensusEngine

# Paths — resolve to absolute paths so they work regardless of cwd
REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # d:\project\biotech-evo2
PHASE1_DIR = REPO_ROOT / "backend" / "phase1_implementation"
DB_PATH = PHASE1_DIR / "alphamissense_data" / "alphamissense.duckdb"
INPUT_CSV = PHASE1_DIR / "helixmind_benchmark_results.csv"
OUTPUT_CSV = PHASE1_DIR / "helixmind_benchmark_results_enriched.csv"

def main():
    if not DB_PATH.exists():
        print(f"❌ DuckDB not found at {DB_PATH}")
        print(f"   Please ensure alphamissense.duckdb exists in {PHASE1_DIR / 'alphamissense_data'}")
        return

    print("Loading AlphaMissense DB...")
    db = AlphaMissenseDB(data_dir=str(PHASE1_DIR / "alphamissense_data"))
    engine = ConsensusEngine()

    print(f"Reading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    total = len(df)
    print(f"Processing {total} variants...")

    enriched = []
    for i, row in df.iterrows():
        vid = row["variant_id"]
        pred = row["prediction"]
        delta = row["delta_score"]

        # Parse variant ID for genomic lookup
        parts = vid.split("-")
        chrom, pos, ref, alt = parts[0], int(parts[1]), parts[2], parts[3]

        # Query AlphaMissense (genomic fallback — no VEP data in benchmark)
        am = db.lookup_by_genomic(chrom, pos, ref, alt, genome="hg38")

        # Compute consensus
        consensus = engine.compute_consensus(
            evo2_prediction=pred,
            evo2_confidence=0.5 if pd.notna(delta) else 0.0,
            alphamissense_score=am["score"] if am else None,
            alphamissense_confidence=am["am_class"] if am else None,
            cadd_phred=None,
            gene_symbol=None,
            variant_str=f"{ref}>{alt}"
        )
        consensus_dict = engine.to_dict(consensus)

        enriched.append({
            **row.to_dict(),
            "alphamissense_score": am["score"] if am else None,
            "alphamissense_class": am["classification"] if am else None,
            "consensus_class": consensus_dict["consensus_classification"],
            "consensus_confidence": consensus_dict["consensus_confidence"],
            "models_agree": consensus_dict["models_agree"],
            "models_total": consensus_dict["models_total"],
        })

        if (i + 1) % 500 == 0:
            print(f"  Processed {i+1}/{total}")

    db.close()

    out_df = pd.DataFrame(enriched)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved enriched results to {OUTPUT_CSV}")
    print(f"  Rows: {len(out_df)}")
    print(f"  Columns: {list(out_df.columns)}")

if __name__ == "__main__":
    main()
