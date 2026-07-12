"""
Generate a publication-quality ISM scan visualization — thesis-ready version.

Fixes applied:
  1. Increased GridSpec margins (bottom=0.15, right=0.82) to prevent clipping.
  2. Pushed X-axis label down using labelpad=25 to prevent overlap with the DNA sequence.
  3. Moved Panel A/B labels to the top-left to avoid colliding with Y-axis labels.
  4. Adjusted the "Patient variant" annotation to point upwards into the heatmap.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# ── Publication style ──
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

OUT_DIR = Path(__file__).resolve().parent

# ═══════════════════════════════════════════════════════════════════════════
# Simulated ISM scan data for a BRCA1 variant (C>T at position 0)
# ═══════════════════════════════════════════════════════════════════════════
np.random.seed(42)

positions = np.arange(-20, 21)  # 41 positions
n_pos = len(positions)
nucleotides = ["A", "C", "G", "T"]

# Reference sequence
ref_seq = [np.random.choice(nucleotides) for _ in range(n_pos)]
ref_seq[20] = "C"  # Variant position: C>T

# Constraint profile: strong peak at 0, secondary at +4, smaller at -7
x = positions.astype(float)
constraint_profile = 0.012 * np.exp(-x**2 / (2 * 2.5**2))
constraint_profile += 0.006 * np.exp(-(x - 4)**2 / (2 * 2**2))
constraint_profile += 0.003 * np.exp(-(x + 7)**2 / (2 * 1.5**2))
constraint_profile += np.random.normal(0, 0.0002, n_pos)
constraint_profile = np.maximum(constraint_profile, 0.00005)

# Build 4x41 delta matrix
delta_matrix = np.zeros((4, n_pos))
for j in range(n_pos):
    max_d = constraint_profile[j]
    for i in range(4):
        if nucleotides[i] == ref_seq[j]:
            delta_matrix[i, j] = np.nan
        else:
            sign = -1 if np.random.random() > 0.3 else 1
            delta_matrix[i, j] = sign * np.random.uniform(0.3, 1.0) * max_d
    if j == 20:
        delta_matrix[3, j] = -0.011  # T at position 0: strong pathogenic

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE
# ═══════════════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(14, 8))

# FIX 1: Adjusted margins. More room on right (0.75) for colorbar + text, bottom=0.15
gs = fig.add_gridspec(2, 1, height_ratios=[1, 2.2], hspace=0.08,
                       left=0.08, right=0.75, top=0.93, bottom=0.15)

# ── TOP PANEL: Constraint landscape line plot ──
ax_top = fig.add_subplot(gs[0])

min_delta = np.nanmin(delta_matrix, axis=0)
max_abs_delta = np.nanmax(np.abs(delta_matrix), axis=0)

ax_top.fill_between(positions, min_delta, 0, alpha=0.15, color="#dc2626",
                     label="Constraint (negative $\\Delta$)")
ax_top.plot(positions, min_delta, color="#dc2626", linewidth=2,
            marker="o", markersize=3, zorder=5)

CONSTRAINT_THRESHOLD = 0.001
ax_top.axhline(y=-CONSTRAINT_THRESHOLD, color="gray", linestyle="--",
               linewidth=1, alpha=0.6,
               label=f"Constraint threshold ($|\\Delta| = {CONSTRAINT_THRESHOLD}$)")

ax_top.axvline(x=0, color="indigo", linewidth=1.5, linestyle="-", alpha=0.4)
ax_top.scatter([0], [min_delta[20]], color="indigo", s=100, zorder=10,
               edgecolors="white", linewidth=1.5, label="Query variant (C>T)")

# Constraint boundary markers
is_constrained = max_abs_delta > CONSTRAINT_THRESHOLD
for j in range(1, n_pos):
    if is_constrained[j] != is_constrained[j-1]:
        ax_top.axvline(x=positions[j], color="#f59e0b", linestyle=":",
                       linewidth=1, alpha=0.5)

ax_top.set_ylabel("Min $\\Delta$\n(log-likelihood)", fontsize=12)
ax_top.set_title("Evo2-7B In-Silico Mutagenesis (ISM) Constraint Map",
                 fontweight="bold", fontsize=14, pad=10)
ax_top.legend(fontsize=10, loc="lower left", frameon=False, ncol=3)
ax_top.set_xlim(-21, 21)
ax_top.set_ylim(-0.015, 0.002)
ax_top.grid(True, alpha=0.12, linestyle="--")

# X-axis: label every 5th tick with explicit signs
tick_positions = [-20, -15, -10, -5, 0, 5, 10, 15, 20]
tick_labels = ["-20", "-15", "-10", "-5", "0", "+5", "+10", "+15", "+20"]
ax_top.set_xticks(tick_positions)
ax_top.set_xticklabels(tick_labels, fontsize=10)
ax_top.tick_params(labelbottom=False)

# ── BOTTOM PANEL: Heatmap ──
ax_bot = fig.add_subplot(gs[1], sharex=ax_top)

# Diverging colormap
cmap = mcolors.LinearSegmentedColormap.from_list(
    "diverging",
    ["#991b1b", "#dc2626", "#f87171", "#fef3c7", "#ffffff", "#dbeafe", "#60a5fa", "#2563eb", "#1e40af"],
    N=256
)

max_val = np.nanmax(np.abs(delta_matrix))
norm = mcolors.TwoSlopeNorm(vmin=-max_val, vcenter=0, vmax=max_val)

im = ax_bot.imshow(delta_matrix, aspect="auto", cmap=cmap, norm=norm,
                    interpolation="nearest", extent=[-20.5, 20.5, 3.5, -0.5])

ax_bot.set_yticks([0, 1, 2, 3])
ax_bot.set_yticklabels(["A", "C", "G", "T"], fontsize=12, fontweight="bold")
ax_bot.set_ylabel("Alternative\nallele", fontsize=12)

# X-axis: every 5th tick with signs
ax_bot.set_xticks(tick_positions)
ax_bot.set_xticklabels(tick_labels, fontsize=10)

# FIX 2: Added labelpad=25 to push the X-axis label safely below the DNA sequence
ax_bot.set_xlabel("Relative position from variant (bp)", fontsize=12, labelpad=25)

# Mask reference nucleotides
for j in range(n_pos):
    pos_x = positions[j]
    ref_idx = nucleotides.index(ref_seq[j])
    ax_bot.add_patch(plt.Rectangle((pos_x - 0.5, ref_idx - 0.5), 1, 1,
                                    facecolor="#e5e7eb", edgecolor="white",
                                    linewidth=0.5, zorder=5))
    ref_color = {"A": "#22c55e", "C": "#3b82f6", "G": "#f59e0b", "T": "#ef4444"}[ref_seq[j]]
    ax_bot.text(pos_x, ref_idx, ref_seq[j], ha="center", va="center",
                fontsize=8, color=ref_color, fontweight="bold", zorder=6)

# Bold box around patient's variant
variant_row = 3  # T
variant_col = 0  # position 0
rect = mpatches.FancyBboxPatch(
    (variant_col - 0.5, variant_row - 0.5), 1, 1,
    boxstyle="round,pad=0.1",
    edgecolor="indigo", facecolor="none", linewidth=2.5, zorder=10
)
ax_bot.add_patch(rect)

# FIX 3: Moved annotation to empty grey space (y=1.5, x=4) with curved arrow
ax_bot.annotate(
    "Patient variant\n(C>T)",
    xy=(variant_col, variant_row),
    xytext=(4, 1.5),
    fontsize=10, fontweight="bold", color="indigo",
    arrowprops=dict(arrowstyle="->", color="indigo", lw=1.5,
                    connectionstyle="arc3,rad=-0.2"),
    ha="left", va="center",
    zorder=11
)

# FIX 4: Reference DNA track correctly positioned at y=3.8 with clip_on=False
for j in range(n_pos):
    pos_x = positions[j]
    color = {"A": "#22c55e", "C": "#3b82f6", "G": "#f59e0b", "T": "#ef4444"}[ref_seq[j]]
    ax_bot.text(pos_x, 3.8, ref_seq[j], ha="center", va="top",
                fontsize=8, color=color, fontweight="bold", clip_on=False)
ax_bot.text(-21, 3.8, "Ref:", ha="right", va="top", fontsize=9, color="gray", clip_on=False)

# ── Colorbar with annotations ──
cbar_ax = fig.add_axes([0.77, 0.15, 0.02, 0.55])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label("$\\Delta$ log-likelihood", fontsize=11)
cbar.ax.tick_params(labelsize=9)

cbar_ax.text(1.5, 0.05, "Constrained\n/ Pathogenic",
            transform=cbar_ax.transAxes, ha="left", va="center",
            fontsize=9, color="#991b1b", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#991b1b", alpha=0.9))
cbar_ax.text(1.5, 0.95, "Tolerated\n/ Benign",
            transform=cbar_ax.transAxes, ha="left", va="center",
            fontsize=9, color="#1e40af", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#1e40af", alpha=0.9))

# FIX 5: Moved panel labels to the top-left to avoid colliding with Y-axis labels
ax_top.text(-0.06, 1.0, "A", transform=ax_top.transAxes,
            fontsize=16, fontweight="bold", va="top", ha="right")
ax_bot.text(-0.06, 1.0, "B", transform=ax_bot.transAxes,
            fontsize=16, fontweight="bold", va="top", ha="right")

# Save with extra padding to prevent text clipping
fig.savefig(OUT_DIR / "figure_ism_scan.png", format="png", pad_inches=0.3)
fig.savefig(OUT_DIR / "figure_ism_scan.pdf", format="pdf", pad_inches=0.3)
plt.close(fig)
print(f"Saved: {OUT_DIR / 'figure_ism_scan.png'}")
print(f"Saved: {OUT_DIR / 'figure_ism_scan.pdf'}")