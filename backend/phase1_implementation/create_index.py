"""Create protein index on AlphaMissense database with low memory."""
import duckdb

db = duckdb.connect('alphamissense_data/alphamissense.duckdb')
db.execute("SET memory_limit = '4GB'")
db.execute("SET threads = 1")

print("Creating protein index (287M rows, 4GB limit, 1 thread)...")
db.execute(
    "CREATE INDEX IF NOT EXISTS idx_protein_lookup "
    "ON alphamissense_protein(uniprot_id, protein_variant)"
)
print("Done!")
db.close()
