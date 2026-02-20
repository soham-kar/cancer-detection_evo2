"""
Step 4: Classify VUS variants and generate publication-quality figures.

Reads:   data/delta_scores.json  (produced by 03_modal_score.py)
Outputs:
  data/vus_classified.csv         — full results table with predictions
  results/vus_summary.png         — bar chart: Pathogenic vs Benign counts
  results/delta_distribution.png  — histogram of delta scores
  results/gene_breakdown.png      — top-20 genes, stacked bar chart

Classification threshold calibrated from BRCA1 analysis in main.py:
  delta < -0.0009178519  →  Likely Pathogenic
  delta >= -0.0009178519 →  Likely Benign
"""

import json
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─── Config ────────────────────────────────────────────────────────────────
DATA_DIR     = Path(__file__).parent / "data"
RESULTS_DIR  = Path(__file__).parent / "results"

IN_JSON      = DATA_DIR / "delta_scores.json"
OUT_CSV      = DATA_DIR / "vus_classified.csv"

# Threshold calibrated from BRCA1 (main.py run_brca1_analysis)
THRESHOLD    = -0.0009178519

PALETTE = {
    "Likely Pathogenic": "#D62728",   # red
    "Likely Benign":     "#2CA02C",   # green
}


# ─── Load & classify ────────────────────────────────────────────────────────
def load_and_classify():
    with open(IN_JSON) as fh:
        records = json.load(fh)

    for r in records:
        r["prediction"] = (
            "Likely Pathogenic" if r["delta_score"] < THRESHOLD else "Likely Benign"
        )
    return records


def save_csv(records):
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fields = ["id", "gene", "chrom", "pos", "ref", "alt",
              "delta_score", "ref_score", "alt_score", "prediction"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(records)
    print(f"Saved classified table → {OUT_CSV}")


# ─── Plot 1: Summary bar chart ───────────────────────────────────────────────
def plot_summary(records):
    paths = defaultdict(int)
    benign = defaultdict(int)
    for r in records:
        if r["prediction"] == "Likely Pathogenic":
            paths["Likely Pathogenic"] += 1
        else:
            benign["Likely Benign"] += 1

    total  = len(records)
    n_path = paths["Likely Pathogenic"]
    n_ben  = benign["Likely Benign"]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(
        ["Likely Pathogenic", "Likely Benign"],
        [n_path, n_ben],
        color=[PALETTE["Likely Pathogenic"], PALETTE["Likely Benign"]],
        edgecolor="white",
        linewidth=1.5,
        width=0.5,
    )
    for bar, count in zip(bars, [n_path, n_ben]):
        pct = count / total * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.01,
            f"{count:,}\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=12, fontweight="bold"
        )

    ax.set_ylabel("Number of VUS Variants", fontsize=12)
    ax.set_title(
        f"Evo2 Classification of ClinVar VUS Variants\n(n={total:,}, threshold={THRESHOLD:.7f})",
        fontsize=13, fontweight="bold"
    )
    ax.set_ylim(0, max(n_path, n_ben) * 1.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = RESULTS_DIR / "vus_summary.png"
    fig.savefig(out, dpi=150)
    print(f"Saved → {out}")
    plt.close(fig)


# ─── Plot 2: Delta-score distribution ───────────────────────────────────────
def plot_delta_distribution(records):
    path_deltas  = [r["delta_score"] for r in records if r["prediction"] == "Likely Pathogenic"]
    benign_deltas = [r["delta_score"] for r in records if r["prediction"] == "Likely Benign"]

    fig, ax = plt.subplots(figsize=(8, 5))

    bins = np.linspace(
        min(r["delta_score"] for r in records),
        max(r["delta_score"] for r in records),
        60
    )
    ax.hist(path_deltas,  bins=bins, color=PALETTE["Likely Pathogenic"],
            alpha=0.7, label=f"Likely Pathogenic (n={len(path_deltas):,})")
    ax.hist(benign_deltas, bins=bins, color=PALETTE["Likely Benign"],
            alpha=0.7, label=f"Likely Benign (n={len(benign_deltas):,})")
    ax.axvline(THRESHOLD, color="black", linestyle="--", linewidth=1.5,
               label=f"Threshold ({THRESHOLD:.7f})")

    ax.set_xlabel("Evo2 Δ Score  (alt − ref log-likelihood)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Distribution of Evo2 Delta Scores for ClinVar VUS", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = RESULTS_DIR / "delta_distribution.png"
    fig.savefig(out, dpi=150)
    print(f"Saved → {out}")
    plt.close(fig)


# ─── Plot 3: Per-gene breakdown (top 20 genes) ─────────────────────────────
def plot_gene_breakdown(records):
    gene_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"Likely Pathogenic": 0, "Likely Benign": 0})
    for r in records:
        gene = r.get("gene") or "Unknown"
        gene_counts[gene][r["prediction"]] += 1

    # Sort by total variants, take top 20
    top_genes = sorted(gene_counts.items(), key=lambda x: sum(x[1].values()), reverse=True)[:20]
    genes      = [g for g, _ in top_genes]
    n_path     = [c["Likely Pathogenic"] for _, c in top_genes]
    n_benign   = [c["Likely Benign"]     for _, c in top_genes]

    x    = np.arange(len(genes))
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x, n_path,            color=PALETTE["Likely Pathogenic"], label="Likely Pathogenic")
    ax.bar(x, n_benign, bottom=n_path, color=PALETTE["Likely Benign"],     label="Likely Benign")

    ax.set_xticks(x)
    ax.set_xticklabels(genes, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Number of VUS Variants", fontsize=12)
    ax.set_title("Top 20 Genes — Evo2 VUS Classification Breakdown", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = RESULTS_DIR / "gene_breakdown.png"
    fig.savefig(out, dpi=150)
    print(f"Saved → {out}")
    plt.close(fig)


# ─── Main ───────────────────────────────────────────────────────────────────
def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading delta scores from {IN_JSON}…")
    records = load_and_classify()
    total    = len(records)

    n_path  = sum(1 for r in records if r["prediction"] == "Likely Pathogenic")
    n_benign = total - n_path

    print(f"\n{'='*50}")
    print(f"  Total VUS variants   : {total:,}")
    print(f"  Likely Pathogenic    : {n_path:,}  ({n_path/total*100:.1f}%)")
    print(f"  Likely Benign        : {n_benign:,}  ({n_benign/total*100:.1f}%)")
    print(f"  Threshold used       : {THRESHOLD}")
    print(f"{'='*50}\n")

    save_csv(records)
    plot_summary(records)
    plot_delta_distribution(records)
    plot_gene_breakdown(records)

    print("\nAll done!")
    print(f"  Results: {RESULTS_DIR.resolve()}")
    print(f"  Full table: {OUT_CSV.resolve()}")


if __name__ == "__main__":
    main()
