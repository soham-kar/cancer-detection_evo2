"""
Modal CPU job to pre-build AlphaMissense DuckDB from TSV files.
Run once: modal run backend/phase1_implementation/build_alphamissense_db.py
"""
import modal
import time

app = modal.App("alphamissense-db-builder")

# Mount the data volume (read TSV files, write DuckDB)
alphamissense_vol = modal.Volume.from_name("alphamissense_data", create_if_missing=True)

image = modal.Image.debian_slim().pip_install("duckdb", "pandas")

@app.function(
    image=image,
    volumes={"/root/alphamissense_data": alphamissense_vol},
    timeout=1800,  # 30 minutes
    cpu=4,
    memory=16384,  # 16 GB
)
def build_db():
    import os, sys, gzip
    from pathlib import Path
    import duckdb
    import pandas as pd

    data_dir = Path("/root/alphamissense_data/alphamissense_data")
    db_path = data_dir / "alphamissense.duckdb"

    print("=" * 60)
    print("🔨 Building AlphaMissense DuckDB Index")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"Output: {db_path}")
    print()

    tsv_files = list(data_dir.glob("*.tsv.gz"))
    if not tsv_files:
        print("❌ No TSV files found.")
        return 0

    print(f"Found {len(tsv_files)} TSV files.")

    # Remove old DB if exists
    if db_path.exists():
        print("🗑️  Removing old database...")
        db_path.unlink()

    conn = duckdb.connect(str(db_path))
    conn.execute("SET memory_limit = '12GB'")
    conn.execute("SET threads = 4")
    conn.execute("SET preserve_insertion_order = false")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS alphamissense_genomic (
            chrom VARCHAR, pos BIGINT, ref VARCHAR, alt VARCHAR,
            genome VARCHAR, uniprot_id VARCHAR, transcript_id VARCHAR,
            protein_variant VARCHAR, am_pathogenicity FLOAT, am_class VARCHAR
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS alphamissense_protein (
            uniprot_id VARCHAR, protein_variant VARCHAR,
            am_pathogenicity FLOAT, am_class VARCHAR
        )
    """)

    total_rows = 0
    start_time = time.perf_counter()

    for i, tsv_file in enumerate(tsv_files):
        fname = tsv_file.name
        print(f"   [{i+1}/{len(tsv_files)}] {fname}...", end=" ", flush=True)

        # Skip non-variant files
        if "isoforms" in fname.lower():
            print("⏭️  Skipped (isoforms)")
            continue
        if "gene_" in fname.lower():
            print("⏭️  Skipped (gene mapping)")
            continue
        if "hg19" in fname and "hg38" not in fname:
            print("⏭️  Skipped (hg19)")
            continue

        # Peek header
        with gzip.open(tsv_file, "rt") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                header = line.strip().split("\t")
                break

        is_genomic = len(header) >= 10

        try:
            if is_genomic and "hg38" in fname:
                chunks = pd.read_csv(
                    tsv_file, sep="\t", comment="#", compression="gzip",
                    chunksize=300000, low_memory=False,
                    names=["chrom","pos","ref","alt","genome","uniprot_id",
                           "transcript_id","protein_variant","am_pathogenicity","am_class"],
                    header=0
                )
                file_rows = 0
                for chunk in chunks:
                    conn.execute("INSERT INTO alphamissense_genomic SELECT * FROM chunk")
                    file_rows += len(chunk)
                total_rows += file_rows
                print(f"✓ ({file_rows:,} genomic rows)")

            elif not is_genomic:
                chunks = pd.read_csv(
                    tsv_file, sep="\t", comment="#", compression="gzip",
                    chunksize=300000, low_memory=False,
                    names=["uniprot_id","protein_variant","am_pathogenicity","am_class"],
                    header=0
                )
                file_rows = 0
                for chunk in chunks:
                    conn.execute("INSERT INTO alphamissense_protein SELECT * FROM chunk")
                    file_rows += len(chunk)
                total_rows += file_rows
                print(f"✓ ({file_rows:,} protein rows)")
            else:
                print("⏭️  Skipped")
        except Exception as e:
            print(f"⚠️  Error: {e}")
            continue

    # Commit + create indexes
    conn.commit()
    print("\n🔍 Creating indexes...")
    try:
        conn.execute("CREATE INDEX idx_protein_lookup ON alphamissense_protein(uniprot_id, protein_variant)")
        print("   ✅ Protein index")
    except Exception as e:
        print(f"   ⚠️  Protein index: {e}")
    try:
        conn.execute("CREATE INDEX idx_genomic_lookup ON alphamissense_genomic(chrom, pos, ref, alt)")
        print("   ✅ Genomic index")
    except Exception as e:
        print(f"   ⚠️  Genomic index: {e}")

    conn.close()
    elapsed = time.perf_counter() - start_time
    db_size = db_path.stat().st_size / (1024**3)
    print(f"\n✅ Done: {total_rows:,} variants in {elapsed:.1f}s")
    print(f"   DB size: {db_size:.2f} GB")
    return total_rows

@app.local_entrypoint()
def main():
    print("Starting AlphaMissense DuckDB build...")
    result = build_db.remote()
    print(f"Build complete: {result:,} variants")
