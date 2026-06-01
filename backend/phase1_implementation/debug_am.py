import duckdb
from pathlib import Path

db_path = Path('backend/phase1_implementation/alphamissense_data/alphamissense.duckdb')
con = duckdb.connect(str(db_path), read_only=True)

# Check chrom format
result = con.execute('SELECT DISTINCT chrom FROM alphamissense LIMIT 10').fetchall()
print('Chrom formats:', [r[0] for r in result])

# Check a specific variant that should exist
result = con.execute("SELECT chrom, pos, ref, alt, am_pathogenicity FROM alphamissense WHERE chrom = '1' AND pos = 69134 LIMIT 5").fetchall()
print('Query with chrom=1, pos=69134:', result)

result = con.execute("SELECT chrom, pos, ref, alt, am_pathogenicity FROM alphamissense WHERE chrom = 'chr1' AND pos = 69134 LIMIT 5").fetchall()
print('Query with chrom=chr1, pos=69134:', result)

con.close()
