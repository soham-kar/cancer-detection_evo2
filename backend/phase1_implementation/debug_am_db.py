"""Debug AlphaMissense DB lookup on Modal."""
import modal

app = modal.App("am-debug")
vol = modal.Volume.from_name("alphamissense_data")
image = modal.Image.debian_slim().pip_install("duckdb")

@app.function(image=image, volumes={"/root/alphamissense_data": vol}, timeout=60)
def debug_lookup():
    import duckdb
    from pathlib import Path
    
    db_path = Path("/root/alphamissense_data/alphamissense_data/alphamissense.duckdb")
    print(f"DB exists: {db_path.exists()}")
    print(f"DB size: {db_path.stat().st_size / (1024**3):.2f} GB")
    
    conn = duckdb.connect(str(db_path))
    
    # Check tables
    tables = conn.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'").fetchall()
    print(f"Tables: {tables}")
    
    # Count rows
    for t in tables:
        tn = t[0]
        cnt = conn.execute(f"SELECT COUNT(*) FROM {tn}").fetchone()[0]
        print(f"  {tn}: {cnt:,} rows")
    
    # Try lookup BRCA1 G718C
    result = conn.execute(
        "SELECT * FROM alphamissense_protein WHERE uniprot_id = 'P38398' AND protein_variant = 'G718C' LIMIT 5"
    ).fetchall()
    print(f"BRCA1 G718C results: {result}")
    
    # Try a simpler query
    result2 = conn.execute(
        "SELECT * FROM alphamissense_protein WHERE uniprot_id = 'P38398' LIMIT 3"
    ).fetchall()
    print(f"BRCA1 any variant: {result2}")
    
    # Check column names
    cols = conn.execute("PRAGMA table_info('alphamissense_protein')").fetchall()
    print(f"Protein table columns: {cols}")
    
    conn.close()

@app.local_entrypoint()
def main():
    debug_lookup.remote()
