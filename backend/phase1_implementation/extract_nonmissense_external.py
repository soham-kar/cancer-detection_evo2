"""
extract_nonmissense_external.py
===============================
Extract non-missense ClinVar variants (stop_gained, splice_donor/acceptor,
frameshift) from the 20260523 VCF, score them via Modal Evo2 batch endpoint,
and produce a CSV ready for external_validation.py.

Usage:
    python backend/phase1_implementation/extract_nonmissense_external.py \
        --max-variants 100 \
        --out backend/phase1_implementation/clinvar/external_nonmissense.csv
"""

import argparse
import gzip
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
import requests

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_VCF = SCRIPT_DIR / "clinvar" / "clinvar_20260523.vcf.gz"
DEFAULT_OUT = SCRIPT_DIR / "clinvar" / "external_nonmissense.csv"
ORIGINAL_CSV = SCRIPT_DIR / "helixmind_benchmark_results_enriched.csv"

# Modal batch endpoint
BATCH_URL = "https://karsoham529--variant-analysis-evo2model-analyze-batch.modal.run"
BATCH_SIZE = 16  # variants per Modal call

# Non-missense consequences we want (from MC field in VCF)
NONMISSENSE_SO = {
    "stop_gained",           # SO:0001587
    "splice_donor_variant",  # SO:0001575
    "splice_acceptor_variant",  # SO:0001574
    "frameshift_variant",    # SO:0001589
    "start_lost",            # SO:0002012
    "stop_lost",             # SO:0001578
}

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

# Review status quality
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
# 1. VCF PARSING — NON-MISSENSE ONLY
# =============================================================================

def parse_info(info_str: str) -> Dict[str, str]:
    result = {}
    for item in info_str.split(";"):
        if "=" in item:
            k, v = item.split("=", 1)
            result[k] = v
        else:
            result[item] = ""
    return result


def is_nonmissense(mc_field: str) -> bool:
    """Check if MC field contains any non-missense consequence."""
    if not mc_field:
        return False
    consequences = set(mc_field.split(","))
    # MC format: SO:0001587|stop_gained,SO:0001575|splice_donor_variant
    for c in consequences:
        parts = c.split("|")
        if len(parts) >= 2:
            so_name = parts[1].strip()
            if so_name in NONMISSENSE_SO:
                return True
    return False


def parse_clinvar_nonmissense(
    vcf_path: str,
    max_variants: int = 100,
    min_revstat: int = 2,
) -> pd.DataFrame:
    """Parse ClinVar VCF for non-missense SNPs with clear P/B labels."""
    records = []
    total = 0
    kept = 0
    skipped = {"not_snp": 0, "not_nonmissense": 0, "low_revstat": 0,
               "invalid_label": 0, "conflicting": 0, "multi_allelic": 0}

    print(f"Parsing {vcf_path} for non-missense variants...")
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

            # Must be SNP
            if len(ref) != 1 or len(alt) != 1:
                skipped["not_snp"] += 1
                continue

            # Skip multi-allelic
            if "," in alt:
                skipped["multi_allelic"] += 1
                continue

            info_dict = parse_info(info)

            # Must be non-missense
            mc = info_dict.get("MC", "")
            if not is_nonmissense(mc):
                skipped["not_nonmissense"] += 1
                continue

            # Get CLNSIG
            clnsig = info_dict.get("CLNSIG", "")
            if not clnsig:
                skipped["invalid_label"] += 1
                continue

            sigs = set(clnsig.split("|"))

            # Skip conflicting/uncertain
            if any(s in sigs for s in [
                "Conflicting_classifications_of_pathogenicity",
                "Conflicting_interpretations_of_pathogenicity",
                "Uncertain_significance", "not_provided",
                "drug_response", "risk_factor", "protective",
                "Affects", "association", "other",
            ]):
                skipped["conflicting"] += 1
                continue

            # Must have exactly one valid P/B label
            valid_sigs = sigs & VALID_LABELS
            if len(valid_sigs) != 1:
                skipped["invalid_label"] += 1
                continue

            raw_label = list(valid_sigs)[0]
            mapped_label = LABEL_MAP[raw_label]

            # Review status
            revstat = info_dict.get("CLNREVSTAT", "")
            revstat_score = 0
            for key, score in REVSTAT_QUALITY.items():
                if key in revstat:
                    revstat_score = max(revstat_score, score)

            if revstat_score < min_revstat:
                skipped["low_revstat"] += 1
                continue

            # Get gene info
            gene_info = info_dict.get("GENEINFO", "")

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
                "consequence": mc,
                "gene": gene_info.split(":")[0] if ":" in gene_info else gene_info,
            })
            kept += 1

            if max_variants and kept >= max_variants:
                print(f"  Reached max variants limit ({max_variants})")
                break

    print(f"\nParsing complete:")
    print(f"  Total records scanned: {total:,}")
    print(f"  Non-missense kept: {kept:,}")
    for reason, count in skipped.items():
        print(f"  Skipped ({reason}): {count:,}")

    return pd.DataFrame(records)


# =============================================================================
# 2. OVERLAP CHECK
# =============================================================================

def check_overlap(df: pd.DataFrame, original_csv: str) -> pd.DataFrame:
    """Remove variants already in the original dataset."""
    if not Path(original_csv).exists():
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
        print(f"Removing {n_overlap:,} overlapping variants...")
        df = df[~overlap].copy()
    else:
        print(f"No overlap with original dataset.")

    return df


# =============================================================================
# 3. EVO2 SCORING VIA MODAL
# =============================================================================

def call_evo2_batch(variants: List[Dict]) -> List[Dict]:
    """Call Modal batch endpoint for Evo2 delta scores."""
    requests_list = []
    for v in variants:
        requests_list.append({
            "chromosome": v["chrom"],
            "variant_position": v["pos"],
            "reference": v["ref"],
            "alternative": v["alt"],
            "genome": "hg38",
        })

    payload = {"requests": requests_list}

    for attempt in range(3):
        try:
            r = requests.post(BATCH_URL, json=payload, timeout=600)
            r.raise_for_status()
            data = r.json()
            return data.get("batch_results", [])
        except Exception as e:
            print(f"  Batch attempt {attempt+1} failed: {e}")
            if attempt < 2:
                time.sleep(10)

    return [{"error": "all_retries_failed"} for _ in variants]


def score_variants_with_evo2(df: pd.DataFrame) -> pd.DataFrame:
    """Score all variants via Modal batch endpoint."""
    variants = df.to_dict("records")
    total = len(variants)
    results = []

    print(f"\nScoring {total} variants via Modal Evo2 endpoint...")
    print(f"  Batch size: {BATCH_SIZE}, Batches: {(total + BATCH_SIZE - 1) // BATCH_SIZE}")

    for i in range(0, total, BATCH_SIZE):
        batch = variants[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"  Batch {batch_num}/{total_batches} ({len(batch)} variants)...", end=" ")

        batch_results = call_evo2_batch(batch)
        results.extend(batch_results)

        n_success = sum(1 for r in batch_results if "delta_score" in r)
        print(f"{n_success}/{len(batch)} scored")

        if i + BATCH_SIZE < total:
            time.sleep(2)  # Rate limiting

    # Extract delta scores
    delta_scores = []
    predictions = []
    for r in results:
        if "delta_score" in r:
            delta_scores.append(r["delta_score"])
            predictions.append(r.get("prediction", ""))
        else:
            delta_scores.append(np.nan)
            predictions.append("")

    df["delta_score"] = delta_scores
    df["prediction"] = predictions

    n_scored = df["delta_score"].notna().sum()
    print(f"\n  Successfully scored: {n_scored}/{total} ({n_scored/total*100:.1f}%)")

    return df


# =============================================================================
# 4. OUTPUT FORMATTING
# =============================================================================

def format_output(df: pd.DataFrame) -> pd.DataFrame:
    """Format to match training CSV columns for external_validation.py."""
    out = pd.DataFrame({
        "variant_id": df["variant_id"],
        "label": df["label"],
        "prediction": df["prediction"],
        "delta_score": df["delta_score"],
        "reference": df["ref"],
        "alphamissense_score": np.nan,  # Non-missense = no AlphaMissense
    })
    return out


# =============================================================================
# 5. MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extract non-missense ClinVar variants for external validation"
    )
    parser.add_argument("--vcf", type=str, default=str(DEFAULT_VCF))
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    parser.add_argument("--max-variants", type=int, default=100)
    parser.add_argument("--min-revstat", type=int, default=2)
    parser.add_argument("--original-csv", type=str, default=str(ORIGINAL_CSV))
    parser.add_argument("--skip-evo2", action="store_true",
                        help="Skip Evo2 scoring (for testing)")
    args = parser.parse_args()

    print("=" * 70)
    print("Non-Missense External Validation Pipeline")
    print("=" * 70)

    # 1. Parse VCF for non-missense
    df = parse_clinvar_nonmissense(args.vcf, args.max_variants, args.min_revstat)
    if len(df) == 0:
        print("ERROR: No non-missense variants found.")
        sys.exit(1)

    print(f"\nLabel distribution:")
    print(df["label"].value_counts())

    # 2. Check overlap
    df = check_overlap(df, args.original_csv)
    if len(df) == 0:
        print("ERROR: All variants overlap with original dataset.")
        sys.exit(1)

    # 3. Score with Evo2
    if not args.skip_evo2:
        df = score_variants_with_evo2(df)
    else:
        print("\nSkipping Evo2 scoring (--skip-evo2 flag)")
        df["delta_score"] = np.nan
        df["prediction"] = ""

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
    print(f"  With Evo2 scores: {out_df['delta_score'].notna().sum():,}")
    print(f"  With AlphaMissense: 0 (by design — non-missense)")
    print(f"\nNext: Run external_validation.py on this CSV")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
