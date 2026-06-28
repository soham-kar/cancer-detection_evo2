"""
One-shot Modal function to download AlphaMissense DuckDB from Google Drive
to the persistent Modal volume at /root/alphamissense_data/alphamissense.duckdb

Usage:
    modal run download_am_db.py
"""
import modal
import os

# Use the same volume as main.py
am_volume = modal.Volume.from_name("alphamissense_data", create_if_missing=True)
MOUNT_PATH = "/root/alphamissense_data"

# Google Drive file ID for alphamissense.duckdb (8.94 GB)
# Shared publicly: https://drive.google.com/drive/folders/17EUT7UDqs21WCBuy_MpsjsDx3p9v7kc_
FILE_ID = "1kgfJnQ3raGBbEDIbbqWp-7NumP16cUC8"
OUTPUT_PATH = f"{MOUNT_PATH}/alphamissense.duckdb"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("gdown>=5.2.0", "duckdb>=1.0.0")
)

app = modal.App("download-am-db", image=image, volumes={MOUNT_PATH: am_volume})


@app.function(volumes={MOUNT_PATH: am_volume}, timeout=7200)  # 2 hour timeout for 9GB download
def download_alphamissense_db():
    import gdown
    import duckdb
    import time
    
    # Check if already downloaded
    if os.path.exists(OUTPUT_PATH):
        size_gb = os.path.getsize(OUTPUT_PATH) / (1024**3)
        print(f"✅ File already exists at {OUTPUT_PATH} ({size_gb:.2f} GB)")
        
        # Verify integrity
        try:
            conn = duckdb.connect(OUTPUT_PATH, read_only=True)
            count = conn.execute("SELECT COUNT(*) FROM alphamissense_genomic").fetchone()[0]
            conn.close()
            print(f"✅ DB integrity check: {count:,} genomic variants")
            return f"Already present: {size_gb:.2f} GB, {count:,} variants"
        except Exception as e:
            print(f"⚠️ DB integrity check failed: {e}")
            print("Re-downloading...")
            os.remove(OUTPUT_PATH)
    
    print(f"📥 Downloading AlphaMissense DuckDB (8.94 GB) from Google Drive...")
    print(f"   File ID: {FILE_ID}")
    print(f"   Output: {OUTPUT_PATH}")
    
    start = time.perf_counter()
    
    # Download with progress
    gdown.download(
        f"https://drive.google.com/uc?id={FILE_ID}&confirm=t",
        output=OUTPUT_PATH,
        quiet=False
    )
    
    elapsed = time.perf_counter() - start
    size_gb = os.path.getsize(OUTPUT_PATH) / (1024**3)
    speed_mbps = (size_gb * 1024) / elapsed if elapsed > 0 else 0
    
    print(f"✅ Download complete: {size_gb:.2f} GB in {elapsed:.0f}s ({speed_mbps:.1f} MB/s)")
    
    # Verify integrity
    print("🔍 Verifying database integrity...")
    conn = duckdb.connect(OUTPUT_PATH, read_only=True)
    g_count = conn.execute("SELECT COUNT(*) FROM alphamissense_genomic").fetchone()[0]
    p_count = conn.execute("SELECT COUNT(*) FROM alphamissense_protein").fetchone()[0]
    
    # Test lookup
    test = conn.execute("""
        SELECT chrom, pos, ref, alt, am_pathogenicity, am_class 
        FROM alphamissense_genomic 
        WHERE chrom = 'chr17' AND pos = 43071092 AND ref = 'C' AND alt = 'A'
        LIMIT 1
    """).fetchone()
    
    conn.close()
    
    print(f"✅ Genomic variants: {g_count:,}")
    print(f"✅ Protein variants: {p_count:,}")
    print(f"✅ Total: {g_count + p_count:,}")
    print(f"✅ Test lookup (chr17:43071092 C>A): score={test[4]}, class={test[5]}")
    
    am_volume.commit()
    print("✅ Volume committed to Modal persistent storage")
    
    return {
        "status": "success",
        "size_gb": round(size_gb, 2),
        "download_time_s": round(elapsed, 1),
        "genomic_variants": g_count,
        "protein_variants": p_count,
        "test_lookup": {"score": test[4], "class": test[5]}
    }


@app.local_entrypoint()
def main():
    result = download_alphamissense_db.remote()
    print("\n" + "=" * 60)
    print("RESULT:", result)
    print("=" * 60)
