"""Benchmark AlphaMissense lookup speed without index."""
import duckdb, time

db = duckdb.connect('alphamissense_data/alphamissense.duckdb')
db.execute("SET threads = 2")

# Test 1: Protein lookup without index (287M rows)
t0 = time.perf_counter()
r = db.execute(
    "SELECT am_pathogenicity, am_class FROM alphamissense_protein "
    "WHERE uniprot_id = 'P38398' AND protein_variant = 'G718C' LIMIT 1"
).fetchone()
t1 = time.perf_counter()
print(f"Protein lookup (no index, 287M rows): {(t1-t0)*1000:.1f} ms")
print(f"  Result: {r}")

# Test 2: Another lookup
t0 = time.perf_counter()
r = db.execute(
    "SELECT am_pathogenicity, am_class FROM alphamissense_protein "
    "WHERE uniprot_id = 'Q8NH21' AND protein_variant = 'V2L' LIMIT 1"
).fetchone()
t1 = time.perf_counter()
print(f"Protein lookup 2 (no index): {(t1-t0)*1000:.1f} ms")
print(f"  Result: {r}")

# Test 3: Genomic lookup WITH index
t0 = time.perf_counter()
r = db.execute(
    "SELECT am_pathogenicity, am_class, uniprot_id, protein_variant "
    "FROM alphamissense_genomic "
    "WHERE chrom = 'chr17' AND pos = 43045629 AND ref = 'C' AND alt = 'T' LIMIT 1"
).fetchone()
t1 = time.perf_counter()
print(f"Genomic lookup (indexed): {(t1-t0)*1000:.1f} ms")
print(f"  Result: {r}")

# Stats
c = db.execute("SELECT COUNT(*) FROM alphamissense_protein").fetchone()[0]
print(f"\nProtein table: {c:,} rows")
c = db.execute("SELECT COUNT(*) FROM alphamissense_genomic").fetchone()[0]
print(f"Genomic table: {c:,} rows")

db.close()
