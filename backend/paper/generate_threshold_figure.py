"""
Generate a publication-quality figure for Section 5.5: Gene-Specific Thresholds.

Creates a visualization showing the decision boundaries (pathogenic / VUS / benign)
for each gene based on Evo2 delta score thresholds.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# ── Publication style ──
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ── Gene thresholds ──
genes = ["TP53", "BRCA1", "BRCA2", "MSH2", "PTEN"]
tau = [-0.003, -0.007, -0.006, -0.007, -0.004]
sigma_lof = [0.0015, 0.0015, 0.0015, 0.0015, 0.0015]
sigma_func = [0.0009, 0.0009, 0.0009, 0.0009, 0.0009]
mechanisms = [
    "Dominant-negative\n(Li-Fraumeni)",
    "Haploinsufficient\n(Findlay validated)",
    "Haploinsufficient",
    "Lynch syndrome\n(high penetrance)",
    "Haploinsufficient\n(Cowden)",
]

# ── Colors ──
C_PATHO = "#C73E1D"   # red
C_VUS = "#F18F01"      # orange
C_BENIGN = "#2E86AB"   # blue
C_DARK = "#2B2D42"

OUT_DIR = Path(__file__).resolve().parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE: Two-panel — (A) Threshold bar chart, (B) Decision boundary visualization
# ═══════════════════════════════════════════════════════════════════════════

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)

# ── Panel A: Threshold bar chart with confidence bands ──
x = np.arange(len(genes))
width = 0.35

bars_tau = ax1.barh(x, tau, height=0.5, color=C_PATHO, edgecolor="white",
                     linewidth=0.5, label=r"$\tau_g$ (pathogenic threshold)", zorder=3)

# Add ±σ_LOF error bars
ax1.errorbar(tau, x, xerr=sigma_lof, fmt="none", color=C_DARK,
             capsize=4, capthick=1.2, lw=1.2, zorder=4, label=r"$\pm \sigma_{\mathrm{LOF}}$")

# Add benign threshold markers (positive τ)
for i, t in enumerate(tau):
    ax1.plot(-t, i, "s", color=C_BENIGN, markersize=8, zorder=4,
             markeredgecolor="white", markeredgewidth=0.5)

# Add a benign threshold legend entry manually
benign_patch = mpatches.Patch(color=C_BENIGN, label=r"$|\tau_g|$ (benign threshold)")
ax1.legend(handles=[
    mpatches.Patch(color=C_PATHO, label=r"$\tau_g$ (pathogenic)"),
    mpatches.Patch(color=C_BENIGN, label=r"$|\tau_g|$ (benign)"),
    plt.Line2D([0], [0], color=C_DARK, lw=1.2, marker="|", markersize=8,
               label=r"$\pm \sigma_{\mathrm{LOF}}$"),
], loc="lower right", frameon=False)

ax1.set_yticks(x)
ax1.set_yticklabels([f"*{g}" for g in genes], style="italic")
ax1.set_xlabel(r"Evo2 $\Delta$ score threshold")
ax1.set_title("A  Gene-specific classification thresholds", fontweight="bold")
ax1.axvline(x=0, color="gray", lw=0.8, ls="--", alpha=0.5)
ax1.set_xlim(-0.012, 0.012)
ax1.invert_yaxis()
ax1.grid(True, alpha=0.2, ls="--", axis="x")

# ── Panel B: Decision boundary visualization ──
# For each gene, show the delta score axis with colored regions:
#   red = pathogenic (Δ < τ), orange = VUS (τ ≤ Δ ≤ |τ|), blue = benign (Δ > |τ|)

for i, (gene, t, mech) in enumerate(zip(genes, tau, mechanisms)):
    y = len(genes) - i - 1  # top to bottom

    # Pathogenic region
    ax2.barh(y, abs(t), left=t, height=0.6, color=C_PATHO, alpha=0.7,
             edgecolor="white", linewidth=0.5)
    # VUS region
    ax2.barh(y, abs(t) - abs(t), left=t, height=0.6, color=C_VUS, alpha=0.5,
             edgecolor="white", linewidth=0.5)
    # Actually VUS is between τ and |τ|
    vus_left = t
    vus_width = abs(t) - abs(t)  # This is 0... need to fix
    # τ is negative, |τ| is positive. VUS region is from τ to |τ|
    vus_left = min(t, abs(t))
    vus_width = abs(t) - t  # from τ to |τ|
    ax2.barh(y, vus_width, left=vus_left, height=0.6, color=C_VUS, alpha=0.5,
             edgecolor="white", linewidth=0.5)
    # Benign region
    ax2.barh(y, 0.012 - abs(t), left=abs(t), height=0.6, color=C_BENIGN, alpha=0.7,
             edgecolor="white", linewidth=0.5)

    # Threshold labels
    ax2.text(t, y, f"  τ={t:.3f}", va="center", ha="right", fontsize=7,
             color=C_PATHO, fontweight="bold")
    ax2.text(abs(t), y, f"  |τ|={abs(t):.3f}", va="center", ha="left", fontsize=7,
             color=C_BENIGN, fontweight="bold")

    # Mechanism label on the right
    ax2.text(0.013, y, mech, va="center", ha="left", fontsize=7.5,
             color=C_DARK, style="italic")

ax2.set_yticks(range(len(genes)))
ax2.set_yticklabels([f"*{g}" for g in reversed(genes)], style="italic")
ax2.set_xlabel(r"Evo2 $\Delta$ score")
ax2.set_xlim(-0.012, 0.025)
ax2.set_title("B  Decision boundaries by gene", fontweight="bold")
ax2.axvline(x=0, color="gray", lw=0.8, ls="--", alpha=0.5)

# Legend for Panel B
ax2.legend(handles=[
    mpatches.Patch(color=C_PATHO, alpha=0.7, label="Pathogenic"),
    mpatches.Patch(color=C_VUS, alpha=0.5, label="VUS (uncertain)"),
    mpatches.Patch(color=C_BENIGN, alpha=0.7, label="Benign"),
], loc="lower right", frameon=False)

ax2.grid(True, alpha=0.2, ls="--", axis="x")

# Save
fig.savefig(OUT_DIR / "figure_gene_thresholds.png", format="png")
fig.savefig(OUT_DIR / "figure_gene_thresholds.pdf", format="pdf")
plt.close(fig)
print(f"Saved: {OUT_DIR / 'figure_gene_thresholds.png'}")
print(f"Saved: {OUT_DIR / 'figure_gene_thresholds.pdf'}")