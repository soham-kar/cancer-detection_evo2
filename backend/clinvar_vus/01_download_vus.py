"""
Step 1: Download ClinVar variant summary and filter for VUS (Variants of Uncertain Significance).

Downloads:
    https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz

Outputs:
    data/clinvar_vus_raw.tsv  — filtered VUS variants with chr, pos, ref, alt, gene
"""

import urllib.request
import gzip
import csv
import os
import shutil
from pathlib import Path

# ─── Config ────────────────────────────────────────────────────────────────
FTP_URL = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz"
DATA_DIR = Path(__file__).parent / "data"
RAW_GZ   = DATA_DIR / "variant_summary.txt.gz"
OUT_TSV  = DATA_DIR / "clinvar_vus_raw.tsv"

# Columns we care about (GRCh38 coords):
KEEP_COLS = [
    "GeneSymbol", "ClinicalSignificance", "ClinSigSimple",
    "Chromosome", "Start", "ReferenceAlleleVCF", "AlternateAlleleVCF",
    "Assembly", "Type", "Name", "RS# (dbSNP)", "VariationID"
]

# Only single‐nucleotide variants on GRCh38 that are VUS
ASSEMBLY_FILTER = "GRCh38"
TYPE_FILTER     = "single nucleotide variant"
VUS_KEYWORDS    = ["uncertain significance", "uncertain_significance", "vus"]

MAX_VARIANTS = 5000   # cap to keep data manageable; set None for all


def download_clinvar():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_GZ.exists():
        print(f"[skip] {RAW_GZ} already exists — delete to re-download")
        return

    print(f"Downloading ClinVar variant summary (~300 MB)…")
    print(f"  URL: {FTP_URL}")
    urllib.request.urlretrieve(FTP_URL, RAW_GZ)
    print(f"  Saved → {RAW_GZ}")


def filter_vus():
    print("Filtering for VUS SNVs on GRCh38…")
    kept = []

    with gzip.open(RAW_GZ, "rt", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        available = reader.fieldnames or []

        # Gracefully handle column name differences across ClinVar releases
        use_cols = [c for c in KEEP_COLS if c in available]
        missing  = [c for c in KEEP_COLS if c not in available]
        if missing:
            print(f"  [warn] Missing columns (not critical): {missing}")

        for row in reader:
            # Assembly filter
            if row.get("Assembly", "").strip() != ASSEMBLY_FILTER:
                continue
            # Variant type filter
            if TYPE_FILTER not in row.get("Type", "").lower():
                continue
            # VUS filter
            sig = row.get("ClinicalSignificance", "").lower()
            if not any(kw in sig for kw in VUS_KEYWORDS):
                continue
            # Must have chr, pos, ref, alt
            chrom = row.get("Chromosome", "").strip()
            pos   = row.get("Start", "").strip()
            ref   = row.get("ReferenceAlleleVCF", "").strip()
            alt   = row.get("AlternateAlleleVCF", "").strip()
            if not all([chrom, pos, ref, alt]):
                continue
            if ref in ("na", "N/A", ".") or alt in ("na", "N/A", "."):
                continue

            kept.append({c: row.get(c, "").strip() for c in use_cols})

            if MAX_VARIANTS and len(kept) >= MAX_VARIANTS:
                print(f"  [cap] Reached {MAX_VARIANTS} variants — stopping early")
                break

    print(f"  Found {len(kept):,} VUS SNVs")

    # Write output
    if not kept:
        print("  ERROR: No variants found. Check the column names above.")
        return

    fieldnames = list(kept[0].keys())
    with open(OUT_TSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(kept)

    print(f"  Saved → {OUT_TSV}  ({OUT_TSV.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    download_clinvar()
    filter_vus()
    print("\nDone. Next: python 02_prepare_sequences.py")
