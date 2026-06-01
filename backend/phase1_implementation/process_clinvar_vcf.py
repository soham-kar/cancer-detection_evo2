"""
process_clinvar_vcf.py
=====================
Parse ClinVar VCF, filter high-quality SNPs with clear P/B labels,
query AlphaMissense DuckDB, and produce a CSV ready for external validation.

Usage:
    python backend/phase1_implementation/process_clinvar_vcf.py \
        --vcf backend/phase1_implementation/clinvar/clinvar_20260523.vcf.gz \
        --out backend/phase1_implementation/clinvar/external_clinvar_candidates.csv \
        --max-variants 10000

Output columns (matching training format):
    variant_id, label, delta_score, alphamissense_score, reference, prediction

Note: delta_score (Evo2) will be NaN for all variants since we cannot
re-run Evo2 on new data without the model. CEFN v2 is designed to handle
missing predictors gracefully.
"""

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import duckdb

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_VCF = SCRIPT_DIR / "clinvar" / "clinvar_20260523.vcf.gz"
DEFAULT_OUT = SCRIPT_DIR / "clinvar" / "external_clinvar_candidates.csv"
DEFAULT_DB = SCRIPT_DIR / "alphamissense_data" / "alphamissense.duckdb"

# Original dataset for overlap check
ORIGINAL_CSV = SCRIPT_DIR / "helixmind_benchmark_results_enriched.csv"

# Label mapping
VALID_LABELS = {
    "Pathogenic", "Likely_pathogenic",
    "Benign", "Likely_benign",
}

LABEL_MAP = {
    "Pathogenic": "Pathogenic",
    "Likely_pathogenic": "Pathogenic",
    "Benign": "Benign",
    "Likely_benign": "Benign",
}

# Review status quality (higher = better)
REVSTAT_QUALITY = {
    "no_assertion_criteria_provided": 0,
    "criteria_provided": 1,
    "criteria_provided,_single_submitter": 2,
    "criteria_provided,_multiple_submitters": 3,
    "criteria_provided,_conflicting_classifications": 3,
    "reviewed_by_expert_panel": 4,
    "practice_guideline": 5,
}


# =============================================================================
# 1. VCF PARSING
# =============================================================================

def parse_info(info_str: str) -> Dict[str, str]:
    """Parse VCF INFO field into key-value dict."""
    result = {}
    for item in info_str.split(";"):
        if "=" in item:
            k, v = item.split("=", 1)
            result[k] = v
        else:
            result[item] = ""
    return result


def parse_clinvar_vcf(
    vcf_path: str,
    max_variants: Optional[int] = None,
    min_revstat: int = 2,
) -> pd.DataFrame:
    """
    Parse ClinVar VCF and filter for high-quality SNPs with clear P/B labels.

    Returns DataFrame with columns:
        chrom, pos, id, ref, alt, clnsig, clnrevstat, revstat_score, allele_id
    """
    records = []
    total = 0
    kept = 0
    skipped_reasons = {
        "not_snp": 0,
        "low_revstat": 0,
        "invalid_label": 0,
        "conflicting": 0,
        "multi_allelic": 0,
    }

    print(f"Parsing {vcf_path}...")
    with gzip.open(vcf_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue

            total += 1
            if total % 500000 == 0:
                print(f"  Processed {total:,} records, kept {kept:,}...")

            parts = line.strip().split("\t")
            if len(parts) < 8:
                continue

            chrom, pos, vid, ref, alt, qual, filt, info = parts[:8]

            # Must be SNP (single nucleotide, no indels)
            if len(ref) != 1 or len(alt) != 1:
                skipped_reasons["not_snp"] += 1
                continue

            # Skip multi-allelic (comma in ALT)
            if "," in alt:
                skipped_reasons["multi_allelic"] += 1
                continue

            info_dict = parse_info(info)

            # Get CLNSIG
            clnsig = info_dict.get("CLNSIG", "")
            if not clnsig:
                skipped_reasons["invalid_label"] += 1
                continue

            # Handle multiple classifications (pipe-separated)
            sigs = set(clnsig.split("|"))

            # Skip if any conflicting or uncertain
            if any(s in sigs for s in ["Conflicting_classifications_of_pathogenicity",
                                         "Conflicting_interpretations_of_pathogenicity",
                                         "Uncertain_significance",
                                         "not_provided",
                                         "drug_response",
                                         "risk_factor",
                                         "protective",
                                         "Affects",
                                         "association",
                                         "other"]):
                skipped_reasons["conflicting"] += 1
                continue

            # Must have exactly one valid P/B label
            valid_sigs = sigs & VALID_LABELS
            if len(valid_sigs) != 1:
                skipped_reasons["invalid_label"] += 1
                continue

            raw_label = list(valid_sigs)[0]
            mapped_label = LABEL_MAP[raw_label]

            # Get review status
            revstat = info_dict.get("CLNREVSTAT", "")
            revstat_score = 0
            for key, score in REVSTAT_QUALITY.items():
                if key in revstat:
                    revstat_score = max(revstat_score, score)

            if revstat_score < min_revstat:
                skipped_reasons["low_revstat"] += 1
                continue

            # Get allele ID
            allele_id = info_dict.get("ALLELEID", "")

            records.append({
                "chrom": chrom,
                "pos": int(pos),
                "id": vid,
                "ref": ref,
                "alt": alt,
                "label": mapped_label,
                "raw_label": raw_label,
                "revstat": revstat,
                "revstat_score": revstat_score,
                "allele_id": allele_id,
            })
            kept += 1

            if max_variants and kept >= max_variants:
                print(f"  Reached max variants limit ({max_variants})")
                break

    print(f"\nParsing complete:")
    print(f"  Total records: {total:,}")
    print(f"  Kept: {kept:,}")
    for reason, count in skipped_reasons.items():
        print(f"  Skipped ({reason}): {count:,}")

    df = pd.DataFrame(records)
    return df


# =============================================================================
# 2. ALPHAMISSENSE LOOKUP
# =============================================================================

def query_alphamissense(
    df: pd.DataFrame,
    db_path: str,
) -> pd.DataFrame:
    """
    Query AlphaMissense DuckDB for each variant.
    Adds 'alphamissense_score' and 'alphamissense_class' columns.
    """
    if not Path(db_path).exists():
        print(f"WARNING: AlphaMissense DB not found at {db_path}")
        df["alphamissense_score"] = np.nan
        df["alphamissense_class"] = None
        return df

    print(f"\nQuerying AlphaMissense DB ({db_path})...")
    con = duckdb.connect(db_path, read_only=True)

    # Build lookup table in DuckDB
    # Create a temporary table with our variants
    variants = df[["chrom", "pos", "ref", "alt"]].copy()
    variants["chrom"] = variants["chrom"].astype(str)
    # AlphaMissense DB uses "chr1" format; VCF uses "1" format
    variants["chrom"] = "chr" + variants["chrom"]
    variants["pos"] = variants["pos"].astype(int)

    # Register as a DuckDB relation
    con.register("query_variants", variants)

    # Query with JOIN
    result = con.execute("""
        SELECT
            q.chrom,
            q.pos,
            q.ref,
            q.alt,
            a.am_pathogenicity,
            a.am_class
        FROM query_variants q
        LEFT JOIN alphamissense a
            ON q.chrom = a.chrom
            AND q.pos = a.pos
            AND q.ref = a.ref
            AND q.alt = a.alt
    """).fetchdf()

    con.close()

    # Merge back
    # Note: result has chrom with 'chr' prefix, but df has without
    score_map = {}
    class_map = {}
    for _, row in result.iterrows():
        # Strip 'chr' prefix to match original df format
        chrom = str(row["chrom"]).replace("chr", "")
        key = (chrom, int(row["pos"]), row["ref"], row["alt"])
        score_map[key] = row["am_pathogenicity"]
        class_map[key] = row["am_class"]

    df["alphamissense_score"] = df.apply(
        lambda r: score_map.get((str(r["chrom"]), int(r["pos"]), r["ref"], r["alt"])),
        axis=1,
    )
    df["alphamissense_class"] = df.apply(
        lambda r: class_map.get((str(r["chrom"]), int(r["pos"]), r["ref"], r["alt"])),
        axis=1,
    )

    matched = df["alphamissense_score"].notna().sum()
    print(f"  AlphaMissense matches: {matched:,} / {len(df):,} ({matched/len(df)*100:.1f}%)")

    return df


# =============================================================================
# 3. OVERLAP CHECK
# =============================================================================

def check_overlap(df: pd.DataFrame, original_csv: str) -> pd.DataFrame:
    """Remove any variants already present in the original dataset."""
    if not Path(original_csv).exists():
        print(f"WARNING: Original CSV not found at {original_csv}")
        return df

    original = pd.read_csv(original_csv)
    original_ids = set(original["variant_id"].dropna().unique())

    df["variant_id"] = df.apply(
        lambda r: f"chr{r['chrom']}-{r['pos']}-{r['ref']}-{r['alt']}",
        axis=1,
    )

    overlap = df["variant_id"].isin(original_ids)
    n_overlap = overlap.sum()

    if n_overlap > 0:
        print(f"\nRemoving {n_overlap:,} variants that overlap with original dataset...")
        df = df[~overlap].copy()
    else:
        print(f"\nNo overlap with original dataset ({len(original_ids):,} original variants).")

    return df


# =============================================================================
# 4. OUTPUT FORMATTING
# =============================================================================

def format_output(df: pd.DataFrame) -> pd.DataFrame:
    """
    Format DataFrame to match training CSV columns.
    delta_score (Evo2) is set to NaN since we don't have Evo2 for new data.
    prediction is set to the label string for compatibility.
    """
    out = pd.DataFrame({
        "variant_id": df["variant_id"],
        "label": df["label"],
        "prediction": df["label"],  # placeholder; not used by CEFN v2
        "delta_score": np.nan,       # Evo2 not available for new variants
        "reference": df["ref"],
        "alphamissense_score": df["alphamissense_score"],
        "alphamissense_class": df["alphamissense_class"],
    })
    return out


# =============================================================================
# 5. MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Process ClinVar VCF for external validation")
    parser.add_argument("--vcf", type=str, default=str(DEFAULT_VCF),
                        help="Path to ClinVar VCF.gz")
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT),
                        help="Output CSV path")
    parser.add_argument("--db", type=str, default=str(DEFAULT_DB),
                        help="Path to AlphaMissense DuckDB")
    parser.add_argument("--max-variants", type=int, default=10000,
                        help="Max variants to keep (for testing)")
    parser.add_argument("--min-revstat", type=int, default=2,
                        help="Minimum review status score (0-5)")
    parser.add_argument("--original-csv", type=str, default=str(ORIGINAL_CSV),
                        help="Original dataset CSV for overlap check")
    args = parser.parse_args()

    print("=" * 70)
    print("ClinVar VCF Processing for External Validation")
    print("=" * 70)

    # 1. Parse VCF
    df = parse_clinvar_vcf(args.vcf, args.max_variants, args.min_revstat)
    if len(df) == 0:
        print("ERROR: No variants passed filtering.")
        sys.exit(1)

    # 2. Check overlap
    df = check_overlap(df, args.original_csv)
    if len(df) == 0:
        print("ERROR: All variants overlap with original dataset.")
        sys.exit(1)

    # 3. Query AlphaMissense
    df = query_alphamissense(df, args.db)

    # 4. Format output
    out_df = format_output(df)

    # 5. Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(f"\n{'='*70}")
    print(f"Output saved to {out_path}")
    print(f"  Total variants: {len(out_df):,}")
    print(f"  Labels: P={(out_df['label']=='Pathogenic').sum()}, "
          f"B={(out_df['label']=='Benign').sum()}")
    print(f"  With AlphaMissense: {out_df['alphamissense_score'].notna().sum():,} "
          f"({out_df['alphamissense_score'].notna().mean()*100:.1f}%)")
    print(f"  Without Evo2 (by design): {out_df['delta_score'].isna().sum():,}")
    print(f"\nNOTE: Evo2 scores are not available for new ClinVar variants.")
    print(f"      CEFN v2 will evaluate using AlphaMissense only.")
    print(f"      This tests the model's robustness to missing predictors.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
