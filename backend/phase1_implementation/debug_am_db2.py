"""Debug AlphaMissense DB for BRCA1 position 718."""
import modal

app = modal.App("am-debug2")
vol = modal.Volume.from_name("alphamissense_data")
image = modal.Image.debian_slim().pip_install("duckdb")

@app.function(image=image, volumes={"/root/alphamissense_data": vol}, timeout=60)
def debug_lookup():
    import duckdb
    from pathlib import Path
    
    db_path = Path("/root/alphamissense_data/alphamissense_data/alphamissense.duckdb")
    conn = duckdb.connect(str(db_path))
    
    # Check if position 718 exists for P38398
    result = conn.execute(
        "SELECT protein_variant, am_pathogenicity, am_class FROM alphamissense_protein "
        "WHERE uniprot_id = 'P38398' AND protein_variant LIKE '%718%' LIMIT 10"
    ).fetchall()
    print(f"BRCA1 pos 718 variants: {result}")
    
    # Count total BRCA1 variants
    cnt = conn.execute(
        "SELECT COUNT(*) FROM alphamissense_protein WHERE uniprot_id = 'P38398'"
    ).fetchone()[0]
    print(f"Total BRCA1 variants: {cnt}")
    
    # Check genomic table for chr17:43045629
    result2 = conn.execute(
        "SELECT * FROM alphamissense_genomic WHERE chrom = 'chr17' AND pos = 43045629 LIMIT 5"
    ).fetchall()
    print(f"Genomic chr17:43045629: {result2}")
    
    # Check without 'chr' prefix
    result3 = conn.execute(
        "SELECT * FROM alphamissense_genomic WHERE chrom = '17' AND pos = 43045629 LIMIT 5"
    ).fetchall()
    print(f"Genomic 17:43045629: {result3}")
    
    conn.close()

@app.local_entrypoint()
def main():
    debug_lookup.remote()
