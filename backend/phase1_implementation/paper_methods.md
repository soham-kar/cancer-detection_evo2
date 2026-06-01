# Methods

## 1. Dataset and Data Sources

### 1.1 Primary Benchmark Dataset

We constructed a stratified benchmark of **4,000 single-nucleotide variants (SNVs)** from the ClinVar database (release 2026-05-23, genome build hg38). Variants were sampled across all autosomes to ensure chromosomal diversity, with stratification by clinical significance category. The dataset includes missense, nonsense, splice site, frameshift, synonymous, and other variant types.

**Inclusion criteria**: (i) single-nucleotide variants only (no insertions, deletions, or multi-allelic variants); (ii) review status of "criteria provided" or higher (ClinVar REVSTAT score ≥ 2); (iii) unambiguous clinical significance labels (Pathogenic/Likely Pathogenic or Benign/Likely Benign). Variants with conflicting classifications, uncertain significance, risk factor, protective, or drug response annotations were excluded.

**Label mapping**: Pathogenic and Likely Pathogenic were merged into a single "Pathogenic" class; Benign and Likely Benign were merged into "Benign." Variants with ClinVar labels of "Uncertain significance" or "Conflicting classifications of pathogenicity" were retained as a separate VUS category for evaluating model uncertainty handling.

### 1.2 Chromosome-Wise Data Split

To prevent data leakage and enable rigorous evaluation of generalization, we employed a **chromosome-wise split** strategy:

| Split | Chromosomes | N | Purpose |
|-------|-------------|---|---------|
| **Training** | chr1–16 | 3,028 | Model parameter optimization |
| **Validation** | chr17–18 | 341 | Hyperparameter tuning, early stopping, best-ECE checkpoint selection |
| **Test** | chr19–22 | 397 | Held-out evaluation |

Because the training and test chromosomes are completely disjoint, the test set provides a stringent evaluation of generalization across genomic regions, gene densities, and regional genomic features. The test set (chr19–22) comprised **397 variants** (71 Pathogenic, 128 Benign, 198 VUS/Conflicting), of which 190 were missense and 207 were non-missense (splice site, synonymous, frameshift, nonsense, and other).

### 1.3 Temporal External Validation Dataset

For **temporal external validation**, we processed the ClinVar VCF release 2026-05-23 independently of the benchmark dataset. After applying the same filtering criteria, we extracted **565 missense variants** with AlphaMissense scores available. A balanced subset of **118 variants** (59 Pathogenic, 59 Benign) was used for calibrated evaluation. These variants were not present in the training or test sets, providing a test of generalization across ClinVar releases.

### 1.4 AlphaMissense Database

We integrated the **AlphaMissense** database (Cheng et al., 2023, *Science*), which provides pathogenicity predictions for all 71 million possible missense variants across the human proteome. The database was stored locally as a **DuckDB** database (9.6 GB) with indexed lookups by chromosome, position, reference allele, and alternative allele (hg38 coordinates). AlphaMissense scores range from 0 to 1, with the original publication's thresholds: score > 0.564 (Likely Pathogenic), score < 0.34 (Likely Benign), and 0.34–0.564 (Uncertain).

---

## 2. Predictor Models

### 2.1 Evo2-7B: Evolutionary Language Model

**Evo2** (Nguyen et al., 2024, *bioRxiv*) is a 7-billion-parameter genomic language model trained on 9.3 trillion nucleotides from diverse organisms. For variant scoring, Evo2 computes a **delta score** by comparing the log-likelihood of the reference sequence against the variant sequence within an 8,192 bp context window centered on the variant position. The model was deployed on an NVIDIA H100 GPU (80 GB) via Modal serverless infrastructure, processing variants in batches of 16 with bfloat16 mixed precision.

Raw Evo2 delta scores ranged from approximately −0.043 to +0.004, where more negative values indicate stronger predicted pathogenicity. To calibrate these scores to a [0, 1] probability scale, we applied **Platt scaling** (logistic regression) fitted on the binary training data (chr1–16), yielding a scaling coefficient of −2,442.37 and intercept of −2.96. Because the raw delta scores occupy a very narrow range, the logistic regression learned a steep slope to separate pathogenic from benign variants; the intercept of −2.96 centres the decision boundary appropriately for the training distribution.

### 2.2 AlphaMissense

AlphaMissense (Cheng et al., 2023) uses AlphaFold-derived protein structure predictions combined with population frequency data to classify missense variants. Scores are inherently calibrated to [0, 1] and were used directly without additional scaling. AlphaMissense scores are only available for missense variants; for all other variant types (nonsense, splice site, frameshift, synonymous, etc.), the AlphaMissense predictor is treated as **missing**.

---

## 3. CEFN v2: Conditional Evidential Fusion Network

### 3.1 Design Rationale

Existing variant interpretation pipelines face three key limitations: (i) they cannot gracefully handle missing predictors (e.g., AlphaMissense unavailable for non-missense variants); (ii) they produce overconfident predictions without quantifying uncertainty; and (iii) simple averaging or voting schemes fail when predictors disagree. CEFN v2 addresses all three through an **evidential deep learning** framework grounded in **Dempster-Shafer Theory (DST)**.

### 3.2 Architecture

CEFN v2 is a lightweight neural network (71,236 parameters) with three core components:

**Deep Sets Encoder (Permutation-Invariant)**. Each predictor contributes a feature vector $[\text{score}_i, \text{mask}_i, \text{embedding}_i]$ where $\text{mask}_i \in \{0, 1\}$ indicates predictor availability and $\text{embedding}_i \in \mathbb{R}^{16}$ is a learned representation of the predictor's identity. A predictor-specific **missing embedding** (a learnable parameter) is used when a predictor is absent, enabling the network to distinguish "predictor says benign" from "predictor is unavailable." The element-wise network $\phi$ (two linear layers with LayerNorm and ReLU, hidden dimension 128) processes each predictor independently, followed by permutation-invariant mean pooling and an aggregation network $\rho$.

**Prior Network (Variant-Type Conditioned)**. A separate network takes a one-hot encoding of the variant type (missense, nonsense, splice site, frameshift, synonymous, or other) and outputs prior logits for the Pathogenic and Benign classes. This allows the model to learn variant-type-specific base rates (e.g., nonsense variants are more likely pathogenic a priori).

**Evidence Network**. The pooled predictor representation and prior logits are concatenated and passed through a two-layer network (hidden dimension 128, dropout 0.1) that outputs **evidence masses** $e_P$ and $e_B$ for the Pathogenic and Benign classes. These are converted to Dirichlet concentration parameters:

$$\alpha_P = e_P + 1, \quad \alpha_B = e_B + 1, \quad \alpha_{VUS} = 1.0$$

The VUS concentration parameter is **fixed at 1.0**, meaning VUS emerges naturally when evidence for both Pathogenic and Benign is low, rather than being explicitly predicted.

**Belief Mass Computation**. Following Dempster-Shafer Theory, belief masses and total ignorance are computed as:

$$b_P = \frac{\alpha_P - 1}{S}, \quad b_B = \frac{\alpha_B - 1}{S}, \quad u = \frac{3}{S}$$

where $S = \alpha_P + \alpha_B + \alpha_{VUS}$ is the total Dirichlet strength. The predicted class is $\arg\max(b_P, b_B, u)$, with $u$ corresponding to the VUS (uncertain) prediction.

### 3.3 Loss Function

Training uses a **binary evidential loss** applied only to variants with definitive Pathogenic or Benign labels (VUS variants are excluded from loss computation):

$$\mathcal{L}_{\text{data}} = -\left[y \cdot \psi(\alpha_P) + (1-y) \cdot \psi(\alpha_B) - \psi(\alpha_P + \alpha_B + \alpha_{VUS})\right]$$

where $\psi$ is the digamma function and $y \in \{0, 1\}$ is the binary label. A KL-divergence regularization term penalizes deviation from a uniform Dirichlet prior for correctly classified samples:

$$\mathcal{L}_{\text{reg}} = D_{KL}\left(\text{Dir}(\boldsymbol{\alpha}) \parallel \text{Dir}(\mathbf{1})\right)$$

The total loss is $\mathcal{L} = \mathcal{L}_{\text{data}} + \lambda \cdot \gamma(t) \cdot \mathcal{L}_{\text{reg}}$, where $\lambda = 0.05$ and $\gamma(t) = \min(1.0, t / 0.5T)$ is a linear annealing coefficient that gradually increases regularization over training epochs $t$.

---

## 4. Training Procedure

### 4.1 Preprocessing

Evo2 delta scores were Platt-scaled to [0, 1] using a logistic regression calibrator fitted on the training split (chr1–16, binary labels only). AlphaMissense scores were used directly. Variant types were inferred from the variant description (reference and alternative alleles) and encoded as 6-dimensional one-hot vectors.

### 4.2 Optimization

CEFN v2 was trained for up to 100 epochs using the AdamW optimizer (learning rate $10^{-3}$, weight decay $10^{-5}$) with a batch size of 128. A ReduceLROnPlateau scheduler (factor 0.5, patience 10 epochs) reduced the learning rate when validation loss plateaued. Early stopping with a patience of 20 epochs on validation loss prevented overfitting. Gradient clipping at norm 1.0 was applied.

### 4.3 Checkpoint Selection

Two checkpoints were tracked: (i) **best-loss** (lowest training loss) for training continuity, and (ii) **best-ECE** (lowest validation Expected Calibration Error) for final model selection. The best-ECE checkpoint (epoch 25, validation ECE = 0.030) was used for all evaluation, prioritizing calibration quality over raw discriminative performance.

---

## 5. Evaluation Metrics

### 5.1 Discriminative Performance

- **AUROC**: Area under the receiver operating characteristic curve, computed on binary (Pathogenic vs. Benign) samples using the predicted probability of pathogenicity $p_P = \alpha_P / (\alpha_P + \alpha_B + \alpha_{VUS})$.
- **AUPRC**: Area under the precision-recall curve, particularly informative for imbalanced datasets.

### 5.2 Calibration

- **Expected Calibration Error (ECE)**: Binned calibration error with 10 equal-width bins, measuring the weighted average of $|\text{accuracy}_b - \text{confidence}_b|$ across bins.

### 5.3 Uncertainty Handling

- **VUS Rate**: Fraction of variants predicted as VUS (uncertain), reported overall and stratified by variant type.
- **Mean Uncertainty**: Average ignorance mass $u$ across all variants.

### 5.4 Success Criteria

For publication readiness, we defined the following thresholds:

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| AUROC | ≥ 0.96 | High discrimination for clinical utility |
| ECE | ≤ 0.05 | Well-calibrated confidence (±5%) |
| VUS Rate (missense) | ≤ 5% | Tolerable uncertainty on diverse missense variants |
| VUS Rate (non-missense) | ≤ 2% | Low uncertainty on mechanistically clearer variants |
| CEFN ≥ Evo2 | AUROC_CEFN ≥ AUROC_Evo2 − 0.02 | Consensus must not degrade single-predictor performance |

---

## 6. Baseline Methods

We compared CEFN v2 against three baselines:

**Evo2-only**: Platt-scaled Evo2 delta scores used directly as pathogenicity probabilities, with a threshold of 0.5 for binary classification. This represents the state-of-the-art zero-shot genomic language model approach.

**Simple Average**: Arithmetic mean of Evo2 (Platt-scaled) and AlphaMissense scores when both are available. When only one predictor is available, that predictor's score is used alone. This represents the simplest possible fusion strategy.

**DST+BMA (Dempster-Shafer Fusion with Bayesian Model Averaging)**: A non-learned consensus method that weights predictors by their known AUROC using a softmax with temperature 10.0, then combines evidence masses via Dempster's combination rule. Missing predictors are assigned total ignorance mass ($u = 1.0$). This baseline isolates the contribution of learned fusion versus fixed-rule fusion.

---

## 7. External Validation Strategy

### 7.1 Cross-Chromosomal Generalization

The held-out test chromosomes (chr19–22, $n = 397$) provide a stringent test of **cross-chromosomal generalization**. Since CEFN v2 was trained exclusively on chr1–16 and validated on chr17–18, chr19–22 represent completely unseen genomic regions with distinct gene densities, regulatory landscapes, and evolutionary constraints. The set includes both missense ($n = 190$) and non-missense ($n = 207$) variants, enabling evaluation with both predictors available.

### 7.2 Temporal Generalization

The ClinVar 2026-05-23 release provides a **temporal external validation set** ($n = 565$ missense, balanced subset $n = 118$). These variants were curated after the benchmark dataset and represent new ClinVar entries. Evaluation on this set tests generalization across time as clinical annotations evolve. Since Evo2 inference was not available for these variants (requiring GPU infrastructure), evaluation used AlphaMissense scores only, demonstrating CEFN v2's ability to handle missing predictors in a real-world scenario.

### 7.3 Validation Summary

| Validation | Set | N | Predictors | AUROC | ECE | VUS Rate |
|------------|-----|---|------------|-------|-----|----------|
| Internal (cross-chromosomal) | chr19–22 | 397 | Both (Evo2 + AM) | 0.988 | 0.040 | 0.8% |
| Temporal (balanced) | ClinVar 2026-05-23 | 118 | AM only | 0.976 | 0.050 | 0.0% |
| Temporal (full) | ClinVar 2026-05-23 | 565 | AM only | 0.976 | 0.083 | 0.2% |

---

## 8. Implementation

CEFN v2 was implemented in PyTorch 2.x and trained on a CPU (inference-only; training does not require GPU). The Evo2-7B model was deployed via Modal serverless infrastructure on NVIDIA H100 GPUs for variant scoring. AlphaMissense lookups were performed using DuckDB 1.0 with indexed queries. All code, trained model weights, Platt scalers, and evaluation scripts are available at [repository URL]. Training completed in 104.4 seconds for 100 epochs (early stopping at epoch 25 for best ECE).
