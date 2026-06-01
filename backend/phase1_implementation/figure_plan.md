# Figure Generation Plan — CEFN v2 Paper

## Overview

This document specifies the exact requirements for each publication-ready figure. All figures should be generated as **SVG** (primary) and **PNG** (fallback) at 300 DPI. Use a consistent color palette across all figures.

**Color Palette** (consistent across all figures):
- CEFN v2: `#2E86AB` (blue)
- Evo2-only: `#A23B72` (magenta)
- AlphaMissense: `#F18F01` (orange)
- DST+BMA: `#C73E1D` (red)
- Simple Average: `#6A4C93` (purple)
- Baseline / Reference: `#8D99AE` (gray)
- Ideal / Diagonal: `#2B2D42` (dark, dashed)

---

## Figure 2: ROC Curves — Internal Test + Temporal External

### Purpose
Demonstrate discriminative performance of CEFN v2 against all baselines on both evaluation tiers.

### Data Sources
- Internal test (chr19-22, n=397): `cefn_v2_outputs/external_validation_structural.json`
- Temporal external (balanced, n=118): `cefn_v2_outputs/external_validation_balanced.json`
- Need to re-run baselines on temporal external to get their ROC curves (or use point estimates)

### Layout
**Two-panel figure** (side by side):
- **Panel A**: Internal test set (chr19-22)
- **Panel B**: Temporal external validation (balanced, n=118)

### Curves to Plot (each panel)
| Curve | Color | Style | Data Source |
|-------|-------|-------|-------------|
| CEFN v2 | Blue | Solid, 2px | Model predictions from `external_validation.py` |
| Evo2-only | Magenta | Dashed, 1.5px | Platt-scaled delta scores |
| AlphaMissense (missense only) | Orange | Dotted, 1.5px | Raw AM scores |
| DST+BMA | Red | Dash-dot, 1.5px | Consensus engine output |
| Simple Average | Purple | Dashed, 1px | (Evo2 + AM) / 2 |
| Random | Gray | Dotted, 1px | Diagonal y=x |

### Annotations
- AUROC value in legend for each curve: `CEFN v2 (AUROC = 0.986)`
- 95% confidence intervals (if computable) shown as shaded regions or error bars
- Panel labels: **A** and **B** in top-left corner, bold, 14pt

### Axes
- X-axis: False Positive Rate (0–1)
- Y-axis: True Positive Rate (0–1)
- Square aspect ratio (equal axes)
- Grid: light gray, dashed, 0.5px

### Output
- `figures/figure2_roc_curves.svg`
- `figures/figure2_roc_curves.png` (300 DPI, 180mm width)

---

## Figure 3: Calibration Curves (Reliability Diagram)

### Purpose
Visually prove that CEFN v2 probabilities are well-calibrated and trustworthy for clinical decision-making.

### Data Sources
- CEFN v2 predictions: `external_validation_structural.json` (or re-run model to get probabilities)
- Evo2-only: Platt-scaled scores from same test set
- Need predicted probabilities and true labels for both

### Layout
**Two-panel figure**:
- **Panel A**: Reliability diagram (CEFN v2 vs. Evo2-only)
- **Panel B**: Histogram of predicted probabilities (distribution check)

### Panel A: Reliability Diagram
- 10 equal-width bins (0.0–0.1, 0.1–0.2, ..., 0.9–1.0)
- X-axis: Mean predicted probability per bin
- Y-axis: Observed fraction of positives per bin
- **Perfect calibration**: diagonal dashed line (y=x)
- **CEFN v2**: blue circles with error bars (95% CI per bin)
- **Evo2-only**: magenta squares with error bars
- Include **ECE value** as text annotation in top-right

### Panel B: Probability Histogram
- X-axis: Predicted probability of pathogenicity (0–1)
- Y-axis: Count / density
- **CEFN v2**: blue histogram, semi-transparent
- **Evo2-only**: magenta histogram, semi-transparent
- Vertical dashed line at threshold 0.5
- Shows whether probabilities are spread out (good) or clustered (bad)

### Annotations
- ECE values: `CEFN v2 ECE = 0.044`, `Evo2 ECE = —` (if computable)
- Panel labels: **A**, **B**

### Output
- `figures/figure3_calibration.svg`
- `figures/figure3_calibration.png` (300 DPI, 180mm width)

---

## Figure 4: VUS Rate by Variant Type

### Status
✅ **Already generated**: `cefn_v2_outputs/vus_by_vartype.png`

### Required Modifications
- Recreate with publication-quality styling (consistent color palette)
- Add error bars (if multiple runs available) or leave as point estimates
- Include sample size (n) above each bar
- Add horizontal reference lines for success criteria (5% missense, 2% non-missense)

### Layout
- Grouped bar chart: missense, splice_site, synonymous, frameshift, nonsense, other
- Y-axis: VUS rate (0–10%)
- X-axis: Variant type
- Bars: CEFN v2 (blue), with n labeled above each bar
- Horizontal dashed lines: 5% (missense threshold), 2% (non-missense threshold)

### Output
- `figures/figure4_vus_by_vartype.svg` (replace existing)

---

## Figure 5: Ablation Study Results

### Purpose
Visual proof that every architectural component matters. More impactful than Table 3 alone.

### Data Source
- `ablation_outputs/ablation_results.json`

### Layout
**Grouped bar chart** with 3 metric groups:
- **Primary Y-axis (left)**: AUROC (0.5–1.0)
- **Secondary Y-axis (right)**: ECE (0–0.15) and VUS rate (0–60%)

### X-axis Categories (6 bars)
1. Full CEFN v2
2. − Prior network
3. − Predictor-specific missing (shared token)
4. − Deep Sets (flat MLP)
5. − Fixed α_VUS (learnable)
6. − Platt scaling

### Visual Design
- **Full CEFN v2**: solid blue bar, slightly wider, with star (*) symbol
- **Ablated variants**: lighter blue / gray bars
- **Critical ablations** (learnable VUS, no Platt): red outline or hatching
- **AUROC**: blue bars, left axis
- **ECE**: orange line with circles, right axis
- **VUS rate**: red line with triangles, right axis

### Annotations
- Horizontal dashed line at AUROC = 0.96 (success threshold)
- Horizontal dashed line at ECE = 0.05 (success threshold)
- Text callouts for critical drops: "−0.016 AUROC", "+0.024 ECE", etc.

### Output
- `figures/figure5_ablation.svg`
- `figures/figure5_ablation.png` (300 DPI, 180mm width)

---

## Figure 6: Decision Curve Analysis (DCA)

### Purpose
Show clinical net benefit — increasingly expected by clinical genomics journals (*Genetics in Medicine*, *NPJ Genomic Medicine*).

### Concept
Decision curve analysis plots **net benefit** vs. **threshold probability** (clinical decision threshold).

$$\text{Net Benefit} = \frac{\text{TP}}{N} - \frac{\text{FP}}{N} \times \frac{p_t}{1 - p_t}$$

Where $p_t$ is the threshold probability (e.g., 0.1 = treat if >10% pathogenic).

### Data Needed
- CEFN v2 predicted probabilities on internal test set
- Evo2-only probabilities on same set
- True labels

### Curves to Plot
| Strategy | Color | Style | Description |
|----------|-------|-------|-------------|
| CEFN v2 | Blue | Solid, 2px | Model net benefit |
| Evo2-only | Magenta | Dashed, 1.5px | Baseline net benefit |
| Treat all | Gray | Dotted, 1px | Assume all pathogenic |
| Treat none | Gray | Dotted, 1px | Assume all benign |

### X-axis
- Threshold probability $p_t$: 0 to 1 (or 0 to 0.5 for clinical relevance)
- Log scale optional for better resolution at low thresholds

### Y-axis
- Net benefit: −0.1 to 0.5 (or auto-scaled)

### Key Clinical Thresholds
- Vertical dashed lines at:
  - $p_t = 0.05$ (screening threshold — high sensitivity)
  - $p_t = 0.20$ (diagnostic threshold — balanced)
  - $p_t = 0.50$ (treatment threshold — high specificity)

### Annotations
- Shaded region where CEFN v2 > Evo2-only (clinical advantage)
- Text box: "CEFN v2 provides net benefit across 0.05–0.80 threshold range"

### Output
- `figures/figure6_decision_curve.svg`
- `figures/figure6_decision_curve.png` (300 DPI, 180mm width)

---

## Supplementary Figures

| Figure | Source | Status |
|--------|--------|--------|
| **Supp. Fig. 1**: Training curves (loss, AUROC, ECE per epoch) | `cefn_v2_outputs/training_curves.png` | ✅ Exists, needs restyling |
| **Supp. Fig. 2**: Method comparison bar chart | `cefn_v2_outputs/comparison.png` | ✅ Exists, can be used as-is or restyled |
| **Supp. Fig. 3**: Delta score distribution | `figures/delta_distribution.png` | ✅ Exists |
| **Supp. Fig. 4**: PR curves | `figures/pr_curves.png` | ✅ Exists |

---

## Implementation Order

| Priority | Figure | Effort | Blockers |
|----------|--------|--------|----------|
| 1 | Figure 3 (Calibration) | Low | Need predicted probabilities from test set |
| 2 | Figure 2 (ROC) | Low | Need baseline ROC curves on temporal external |
| 3 | Figure 5 (Ablation) | Low | Data ready in `ablation_results.json` |
| 4 | Figure 6 (DCA) | Medium | Need predicted probabilities + threshold sweep |
| 5 | Figure 4 (VUS) | Low | Restyle existing `vus_by_vartype.png` |
| 6 | Supp. figures | Low | Restyle existing plots |

---

## Technical Notes

### Libraries
- **matplotlib** (primary): full control over SVG output
- **seaborn** (optional): statistical styling
- **scikit-learn**: `roc_curve`, `precision_recall_curve`, `calibration_curve`

### Output Specifications
- **Format**: SVG (vector) + PNG (raster, 300 DPI)
- **Width**: 180 mm (single column) or 360 mm (double column)
- **Font**: Arial or Helvetica, 8–10 pt for labels, 6–7 pt for tick labels
- **Line width**: 1.5–2 px for primary curves, 1 px for secondary
- **No borders**: use `spines['top'].set_visible(False)` etc.

### Data Access Pattern
All figures should read from:
- `cefn_v2_outputs/*.json` (metrics)
- `cefn_v2_outputs/cefn_v2_model.pt` (model weights for inference)
- `ablation_outputs/ablation_results.json` (ablation)
- `clinvar/structural_external_test.csv` (test data for probability extraction)
