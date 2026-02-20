"""
Step 2: Prepare ref/alt sequence pairs for all VUS variants.

For each variant in data/clinvar_vus_raw.tsv:
  - Fetch an 8 kb genomic window centred on the position (UCSC API)
  - Build the reference sequence and the alternate (mutant) sequence
  - Verify the reference allele matches the genome

Outputs:
    data/sequences.json — list of {id, gene, chrom, pos, ref, alt, ref_seq, alt_seq, ref_idx}

ref_idx is used for deduplication: many variants share the same 8kb window
(same chromosome region), so we store unique ref_seqs and share indices.
"""

import json
import time
import requests
import csv
from pathlib import Path

# ─── Config ────────────────────────────────────────────────────────────────
DATA_DIR    = Path(__file__).parent / "data"
IN_TSV      = DATA_DIR / "clinvar_vus_raw.tsv"
OUT_JSON    = DATA_DIR / "sequences.json"

WINDOW_SIZE = 8192          # same as main.py
UCSC_API    = "https://api.genome.ucsc.edu/getData/sequence"
GENOME      = "hg38"        # GRCh38
MAX_RETRIES = 3
SLEEP_RETRY = 2.0           # seconds between retries
SLEEP_REQ   = 0.3           # polite pause between UCSC requests


def fetch_window(chrom: str, position: int) -> tuple[str, int]:
    """Return (sequence, seq_start) for an 8 kb window centred on position."""
    half  = WINDOW_SIZE // 2
    start = max(0, position - 1 - half)          # 0-based
    end   = position - 1 + half + 1              # exclusive

    url = f"{UCSC_API}?genome={GENOME};chrom={chrom};start={start};end={end}"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            data = r.json()
            seq  = data.get("dna", "").upper()
            if not seq:
                raise ValueError("Empty sequence returned")
            return seq, start
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise
            print(f"    [retry {attempt}] {e}")
            time.sleep(SLEEP_RETRY)


def build_alt_seq(ref_seq: str, seq_start: int, position: int, ref: str, alt: str) -> str:
    """Substitute ref → alt at the correct position within the window."""
    rel = position - 1 - seq_start              # 0-based relative index
    if rel < 0 or rel >= len(ref_seq):
        raise ValueError(f"Position {position} outside window [{seq_start}, {seq_start + len(ref_seq)})")
    actual = ref_seq[rel : rel + len(ref)]
    if actual.upper() != ref.upper():
        raise ValueError(f"Ref mismatch at pos {position}: genome={actual}, ClinVar={ref}")
    return ref_seq[:rel] + alt + ref_seq[rel + len(ref):]


def load_vus_variants():
    variants = []
    with open(IN_TSV, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            chrom = row.get("Chromosome", "").strip()
            if not chrom.startswith("chr"):
                chrom = f"chr{chrom}"
            try:
                pos = int(row.get("Start", ""))
            except ValueError:
                continue
            variants.append({
                "id":    row.get("VariationID", ""),
                "gene":  row.get("GeneSymbol", ""),
                "chrom": chrom,
                "pos":   pos,
                "ref":   row.get("ReferenceAlleleVCF", "").upper(),
                "alt":   row.get("AlternateAlleleVCF", "").upper(),
            })
    return variants


def main():
    print(f"Loading variants from {IN_TSV}…")
    variants = load_vus_variants()
    print(f"  {len(variants):,} VUS variants to prepare")

    # Deduplicate ref windows: many nearby variants share the same 8kb window
    ref_seq_cache: dict[str, int] = {}  # window_key → index in unique_refs
    unique_refs:   list[str]      = []

    records  = []
    skipped  = 0

    for i, v in enumerate(variants, 1):
        if i % 100 == 0:
            print(f"  [{i}/{len(variants)}] processed …  (skipped={skipped})")

        try:
            ref_seq, seq_start = fetch_window(v["chrom"], v["pos"])
            alt_seq = build_alt_seq(ref_seq, seq_start, v["pos"], v["ref"], v["alt"])
        except Exception as e:
            print(f"    [skip] {v['id']} {v['chrom']}:{v['pos']} — {e}")
            skipped += 1
            continue

        # Deduplicate ref sequences
        cache_key = f"{v['chrom']}:{seq_start}"
        if cache_key not in ref_seq_cache:
            ref_seq_cache[cache_key] = len(unique_refs)
            unique_refs.append(ref_seq)
        ref_idx = ref_seq_cache[cache_key]

        records.append({
            **v,
            "ref_idx": ref_idx,
            "alt_seq": alt_seq,
        })

        time.sleep(SLEEP_REQ)

    print(f"\nPrepared {len(records):,} variants  (skipped {skipped})")
    print(f"Unique reference windows: {len(unique_refs):,}")

    out = {
        "variants":     records,
        "unique_refs":  unique_refs,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as fh:
        json.dump(out, fh)

    print(f"Saved → {OUT_JSON}  ({OUT_JSON.stat().st_size // 1024:,} KB)")
    print("\nDone. Next: modal run 03_modal_score.py")


if __name__ == "__main__":
    main()
