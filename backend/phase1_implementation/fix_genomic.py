"""Re-import hg38 genomic file into alphamissense_genomic table."""
import duckdb, pandas as pd, time

db = duckdb.connect('alphamissense_data/alphamissense.duckdb')
db.execute("SET memory_limit = '8GB'")
db.execute("SET threads = 2")

# Drop and recreate genomic table
db.execute("DROP TABLE IF EXISTS alphamissense_genomic")
db.execute("""
    CREATE TABLE alphamissense_genomic (
        chrom VARCHAR, pos BIGINT, ref VARCHAR, alt VARCHAR,
        genome VARCHAR, uniprot_id VARCHAR, transcript_id VARCHAR,
        protein_variant VARCHAR, am_pathogenicity FLOAT, am_class VARCHAR
    )
""")

print("Importing AlphaMissense_hg38.tsv.gz (613 MB compressed)...")
t0 = time.perf_counter()

chunks = pd.read_csv(
    'alphamissense_data/AlphaMissense_hg38.tsv.gz',
    sep='\t', comment='#', compression='gzip',
    chunksize=200000, low_memory=False,
    names=["chrom","pos","ref","alt","genome","uniprot_id",
           "transcript_id","protein_variant","am_pathogenicity","am_class"],
    header=0
)

total = 0
for chunk in chunks:
    db.execute("INSERT INTO alphamissense_genomic SELECT * FROM chunk")
    total += len(chunk)
    if total % 5000000 == 0:
        print(f"  {total:,} rows...")

print(f"✓ {total:,} genomic rows imported in {time.perf_counter()-t0:.0f}s")

# Create index
print("Creating genomic index...")
db.execute("SET memory_limit = '4GB'")
db.execute("SET threads = 1")
db.execute(
    "CREATE INDEX IF NOT EXISTS idx_genomic_lookup "
    "ON alphamissense_genomic(chrom, pos, ref, alt)"
)
print("✓ Genomic index created")

# Verify
c = db.execute("SELECT COUNT(*) FROM alphamissense_genomic").fetchone()[0]
print(f"\nGenomic table: {c:,} rows")

# Test lookup
r = db.execute(
    "SELECT am_pathogenicity, am_class, uniprot_id, protein_variant "
    "FROM alphamissense_genomic "
    "WHERE chrom = 'chr1' AND pos = 69094 AND ref = 'G' AND alt = 'T' LIMIT 1"
).fetchone()
print(f"Test lookup: {r}")

db.close()
print("Done!")
