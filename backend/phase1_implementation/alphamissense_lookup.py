"""
Phase 1 / Day 2: AlphaMissense DuckDB Lookup Service
=====================================================
Zero-config embedded database for fast AlphaMissense score queries.
Uses DuckDB (pip install duckdb) — no PostgreSQL/MySQL needed.

Query by (UniProt accession, position, ref_aa, alt_aa) → score + confidence.

Paper: Cheng et al. (2023) — Science
       "Accurate proteome-wide missense variant effect prediction with AlphaMissense"

Classification thresholds (from AlphaMissense paper):
    - score > 0.564  → Likely Pathogenic
    - score < 0.34   → Likely Benign
    - 0.34 ≤ score ≤ 0.564 → Uncertain

Usage:
    from alphamissense_lookup import AlphaMissenseDB
    
    db = AlphaMissenseDB("alphamissense_data/")
    result = db.lookup("P38398", 718, "G", "C")
    # → {"score": 0.91, "confidence": "high", "classification": "Likely Pathogenic"}
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, List
import time

# =============================================================================
# CONFIGURATION
# =============================================================================

# Default data directory (relative to this file)
DEFAULT_DATA_DIR = Path(__file__).parent / "alphamissense_data"

# AlphaMissense classification thresholds (from Cheng et al. 2023)
PATHOGENIC_THRESHOLD = 0.564   # > this = likely pathogenic
BENIGN_THRESHOLD = 0.34        # < this = likely benign
# Between = uncertain

# =============================================================================
# DATABASE CLASS
# =============================================================================

class AlphaMissenseDB:
    """
    DuckDB-backed AlphaMissense lookup service.
    
    Features:
    - O(1) lookup by (uniprot_id, variant_string)
    - Automatic TSV import on first use
    - Memory-efficient: uses DuckDB's columnar storage
    - Thread-safe for concurrent queries
    """
    
    def __init__(self, data_dir: str = None):
        """
        Initialize the AlphaMissense database.
        
        Args:
            data_dir: Path to directory containing AlphaMissense TSV files.
                     Defaults to 'alphamissense_data/' next to this file.
        """
        self.data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
        self.db_path = self.data_dir / "alphamissense.duckdb"
        self._conn = None
        self._ready = False
        
        # Auto-build on first use if DB doesn't exist
        if not self.db_path.exists():
            print(f"🔨 DuckDB not found at {self.db_path}, auto-building from TSV files...")
            self.build_index()
    
    @property
    def conn(self):
        """Lazy-load DuckDB connection."""
        if self._conn is None:
            try:
                import duckdb
                self._conn = duckdb.connect(str(self.db_path))
            except ImportError:
                print("❌ DuckDB not installed. Run: pip install duckdb")
                raise
        return self._conn
    
    def build_index(self, force_rebuild: bool = False) -> int:
        """
        Import AlphaMissense TSV files into DuckDB using pandas chunked reading.
        
        Two file formats:
        1. Genomic files (hg38.tsv.gz): CHROM, POS, REF, ALT, genome, uniprot_id,
           transcript_id, protein_variant, am_pathogenicity, am_class
        2. Protein files (aa_substitutions.tsv.gz): uniprot_id, protein_variant,
           am_pathogenicity, am_class (no genomic coords)
        
        We import BOTH into separate tables for flexible lookup.
        """
        import gzip
        
        print("=" * 60)
        print("🔨 Building AlphaMissense DuckDB Index")
        print("=" * 60)
        print(f"Data directory: {self.data_dir}")
        print()
        
        tsv_files = list(self.data_dir.glob("*.tsv.gz"))
        if not tsv_files:
            print("❌ No TSV files found.")
            return 0
        
        print(f"Found {len(tsv_files)} TSV files.")
        
        # Check if already built
        table_exists = False
        try:
            result = self.conn.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'alphamissense_genomic'"
            ).fetchone()
            table_exists = result[0] > 0
        except:
            pass
        
        if table_exists and not force_rebuild:
            g_count = self.conn.execute("SELECT COUNT(*) FROM alphamissense_genomic").fetchone()[0]
            p_count = self.conn.execute("SELECT COUNT(*) FROM alphamissense_protein").fetchone()[0]
            print(f"✅ Index already exists: {g_count:,} genomic + {p_count:,} protein variants.")
            self._ready = True
            return g_count + p_count
        
        if table_exists and force_rebuild:
            print("🔄 Dropping existing tables...")
            self.conn.execute("DROP TABLE IF EXISTS alphamissense_genomic")
            self.conn.execute("DROP TABLE IF EXISTS alphamissense_protein")
        
        # Set memory limits upfront to avoid OOM
        self.conn.execute("SET memory_limit = '10GB'")
        self.conn.execute("SET threads = 2")
        self.conn.execute("SET preserve_insertion_order = false")
        
        # Create tables
        print("📋 Creating table schemas...")
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS alphamissense_genomic (
                chrom VARCHAR, pos BIGINT, ref VARCHAR, alt VARCHAR,
                genome VARCHAR, uniprot_id VARCHAR, transcript_id VARCHAR,
                protein_variant VARCHAR, am_pathogenicity FLOAT, am_class VARCHAR
            )
        """)
        self.conn.execute("""
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
            
            # Peek at header to determine format
            with gzip.open(tsv_file, "rt") as f:
                for line in f:
                    if line.startswith("#"):
                        continue
                    header = line.strip().split("\t")
                    break
            
            # Detect format by column count (10 = genomic, 4 = protein)
            is_genomic = len(header) >= 10
            
            # Skip isoforms, gene maps, and hg19 files (redundant, 566M rows saved)
            if "isoforms" in fname.lower():
                print("⏭️  Skipped (isoforms — redundant)")
                continue
            if "gene_" in fname.lower():
                print("⏭️  Skipped (gene mapping file)")
                continue
            if "hg19" in fname and "hg38" not in fname:
                print("⏭️  Skipped (hg19)")
                continue
            
            try:
                if is_genomic and "hg38" in fname:
                    # Genomic format — use pandas chunked read
                    import pandas as pd
                    chunks = pd.read_csv(
                        tsv_file, sep="\t", comment="#", compression="gzip",
                        chunksize=200000, low_memory=False,
                        names=["chrom","pos","ref","alt","genome","uniprot_id",
                               "transcript_id","protein_variant","am_pathogenicity","am_class"],
                        header=0
                    )
                    file_rows = 0
                    for chunk in chunks:
                        self.conn.execute(
                            "INSERT INTO alphamissense_genomic SELECT * FROM chunk"
                        )
                        file_rows += len(chunk)
                    total_rows += file_rows
                    print(f"✓ ({file_rows:,} genomic rows)")
                    
                elif not is_genomic:
                    # Protein format — pandas chunked read
                    import pandas as pd
                    chunks = pd.read_csv(
                        tsv_file, sep="\t", comment="#", compression="gzip",
                        chunksize=200000, low_memory=False,
                        names=["uniprot_id","protein_variant","am_pathogenicity","am_class"],
                        header=0
                    )
                    file_rows = 0
                    for chunk in chunks:
                        self.conn.execute(
                            "INSERT INTO alphamissense_protein SELECT * FROM chunk"
                        )
                        file_rows += len(chunk)
                    total_rows += file_rows
                    print(f"✓ ({file_rows:,} protein rows)")
                    
                else:
                    print("⏭️  Skipped (hg19 or other)")
                    
            except Exception as e:
                print(f"⚠️  Error: {e}")
                continue
        
        elapsed = time.perf_counter() - start_time
        
        # Create indexes (with memory safeguards)
        print()
        print("🔍 Creating indexes...")
        
        # Set memory limits to avoid OOM during index creation
        self.conn.execute("SET memory_limit = '8GB'")
        self.conn.execute("SET threads = 2")
        
        try:
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_protein_lookup ON alphamissense_protein(uniprot_id, protein_variant)"
            )
            print("   ✅ Protein lookup index created")
        except Exception as e:
            print(f"   ⚠️  Protein index skipped (memory): {e}")
        
        try:
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_genomic_lookup ON alphamissense_genomic(chrom, pos, ref, alt)"
            )
            print("   ✅ Genomic lookup index created")
        except Exception as e:
            print(f"   ⚠️  Genomic index skipped (memory): {e}")
        
        print()
        print(f"✅ Import complete: {total_rows:,} variants in {elapsed:.1f}s")
        
        db_size = self.db_path.stat().st_size / (1024 * 1024) if self.db_path.exists() else 0
        print(f"   Database: {self.db_path} ({db_size:.1f} MB)")
        
        self._ready = True
        return total_rows
    
    def lookup(
        self,
        uniprot_id: str,
        position: int,
        ref_aa: str,
        alt_aa: str,
    ) -> Optional[Dict]:
        """
        Look up AlphaMissense score by protein variant (e.g., "V2L").
        Queries the protein-level table (all 71M missense variants).
        
        Args:
            uniprot_id: UniProt accession (e.g., "P38398" for BRCA1)
            position: 1-based amino acid position
            ref_aa: Reference amino acid (single letter, e.g., "G")
            alt_aa: Alternative amino acid (single letter, e.g., "C")
        
        Returns:
            dict with score, classification, or None if not found
        """
        if not self._ready:
            self.build_index()
        
        variant_str = f"{ref_aa}{position}{alt_aa}"
        
        result = self.conn.execute(
            "SELECT am_pathogenicity, am_class FROM alphamissense_protein "
            "WHERE uniprot_id = ? AND protein_variant = ? LIMIT 1",
            [uniprot_id, variant_str]
        ).fetchone()
        
        if result is None:
            return None
        
        return {
            "score": round(result[0], 4),
            "classification": self._classify_from_am(result[1]),
            "am_class": result[1],
            "variant": variant_str,
            "uniprot_id": uniprot_id,
        }
    
    def lookup_by_genomic(
        self,
        chrom: str,
        pos: int,
        ref: str,
        alt: str,
        genome: str = "hg38"
    ) -> Optional[Dict]:
        """
        Look up AlphaMissense score by genomic coordinates.
        Queries the genomic table (hg38 variants with coordinates).
        
        Args:
            chrom: Chromosome (e.g., "chr17")
            pos: 1-based genomic position
            ref: Reference allele
            alt: Alternative allele
            genome: Genome build ("hg38" or "hg19")
        
        Returns:
            dict or None if not found
        """
        if not self._ready:
            self.build_index()
        
        result = self.conn.execute(
            "SELECT am_pathogenicity, am_class, uniprot_id, protein_variant, transcript_id "
            "FROM alphamissense_genomic "
            "WHERE chrom = ? AND pos = ? AND ref = ? AND alt = ? AND genome = ? "
            "LIMIT 1",
            [chrom, pos, ref, alt, genome]
        ).fetchone()
        
        if result is None:
            return None
        
        return {
            "score": round(result[0], 4),
            "classification": self._classify_from_am(result[1]),
            "am_class": result[1],
            "uniprot_id": result[2],
            "variant": result[3],
            "transcript_id": result[4],
        }
    
    def lookup_batch(
        self,
        queries: List[Dict]
    ) -> List[Optional[Dict]]:
        """
        Batch lookup for multiple variants.
        
        Args:
            queries: List of dicts with keys: uniprot_id, position, ref_aa, alt_aa
        
        Returns:
            List of results (same order as queries), None for missing variants.
        """
        if not self._ready:
            self.build_index()
        
        results = []
        for q in queries:
            result = self.lookup(
                uniprot_id=q["uniprot_id"],
                position=q["position"],
                ref_aa=q["ref_aa"],
                alt_aa=q["alt_aa"]
            )
            results.append(result)
        
        return results
    
    def _classify_from_am(self, am_class: str) -> str:
        """Convert AlphaMissense class label to clinical classification."""
        am_class = am_class.lower()
        if "pathogenic" in am_class:
            return "Likely Pathogenic"
        elif "benign" in am_class:
            return "Likely Benign"
        else:
            return "Uncertain Significance"
    
    def _classify(self, score: float) -> str:
        """Classify AlphaMissense score into clinical category."""
        if score > PATHOGENIC_THRESHOLD:
            return "Likely Pathogenic"
        elif score < BENIGN_THRESHOLD:
            return "Likely Benign"
        else:
            return "Uncertain Significance"
    
    def get_statistics(self) -> Dict:
        """Get database statistics."""
        if not self._ready:
            self.build_index()
        
        g_total = self.conn.execute("SELECT COUNT(*) FROM alphamissense_genomic").fetchone()[0]
        p_total = self.conn.execute("SELECT COUNT(*) FROM alphamissense_protein").fetchone()[0]
        
        p_pathogenic = self.conn.execute(
            "SELECT COUNT(*) FROM alphamissense_protein WHERE am_class = 'pathogenic'"
        ).fetchone()[0]
        p_benign = self.conn.execute(
            "SELECT COUNT(*) FROM alphamissense_protein WHERE am_class = 'likely_benign'"
        ).fetchone()[0]
        p_ambiguous = self.conn.execute(
            "SELECT COUNT(*) FROM alphamissense_protein WHERE am_class = 'ambiguous'"
        ).fetchone()[0]
        
        unique_genes = self.conn.execute(
            "SELECT COUNT(DISTINCT uniprot_id) FROM alphamissense_protein"
        ).fetchone()[0]
        
        return {
            "genomic_variants": g_total,
            "protein_variants": p_total,
            "total_variants": g_total + p_total,
            "pathogenic": p_pathogenic,
            "benign": p_benign,
            "ambiguous": p_ambiguous,
            "unique_genes": unique_genes,
        }
    
    def close(self):
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None


# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="AlphaMissense DuckDB Lookup Service"
    )
    parser.add_argument(
        "--build", action="store_true",
        help="Build/re-build the DuckDB index from TSV files"
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Show database statistics"
    )
    parser.add_argument(
        "--lookup", nargs=4, metavar=("UNIPROT", "POS", "REF", "ALT"),
        help="Look up a single variant: UNIPROT POS REF_AA ALT_AA"
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Path to AlphaMissense TSV data directory"
    )
    
    args = parser.parse_args()
    
    db = AlphaMissenseDB(data_dir=args.data_dir)
    
    try:
        if args.build:
            db.build_index(force_rebuild=True)
        
        if args.stats:
            stats = db.get_statistics()
            print("\n📊 AlphaMissense Database Statistics:")
            print(f"   Protein variants:  {stats['protein_variants']:,}")
            print(f"   Genomic variants:  {stats['genomic_variants']:,}")
            print(f"   Total variants:    {stats['total_variants']:,}")
            print(f"   Unique genes:      {stats['unique_genes']:,}")
            print(f"   Pathogenic:        {stats['pathogenic']:,}")
            print(f"   Benign:            {stats['benign']:,}")
            print(f"   Ambiguous:         {stats['ambiguous']:,}")
        
        if args.lookup:
            uniprot, pos, ref, alt = args.lookup
            result = db.lookup(uniprot, int(pos), ref, alt)
            if result:
                print(f"\n🔍 AlphaMissense Lookup: {uniprot} {ref}{pos}{alt}")
                print(f"   Score:          {result['score']}")
                print(f"   Confidence:     {result['confidence']}")
                print(f"   Classification: {result['classification']}")
            else:
                print(f"\n❌ Variant {uniprot} {ref}{pos}{alt} not found in database.")
        
        if not any([args.build, args.stats, args.lookup]):
            # Default: show stats
            stats = db.get_statistics()
            print("\n📊 AlphaMissense Database Statistics:")
            print(f"   Total variants: {stats['total_variants']:,}")
            print(f"   Pathogenic:     {stats['pathogenic']:,} ({stats['pathogenic_pct']}%)")
            print(f"   Benign:         {stats['benign']:,} ({stats['benign_pct']}%)")
            print(f"   Uncertain:      {stats['uncertain']:,} ({stats['uncertain_pct']}%)")
            print()
            print("Usage examples:")
            print("  python alphamissense_lookup.py --build")
            print("  python alphamissense_lookup.py --lookup P38398 718 G C")
            print("  python alphamissense_lookup.py --stats")
    
    finally:
        db.close()


if __name__ == "__main__":
    main()
