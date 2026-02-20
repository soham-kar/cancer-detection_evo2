# Progress Checklist: Nature Paper Execution

## Week 1: Setup & Data Preparation

### Environment Setup
- [ ] Install Python dependencies (`pip install -r requirements.txt`)
- [ ] Test Evo2 backend connection
- [ ] Update `config.py` with correct Modal endpoint
- [ ] Run test pipeline with mock data
- [ ] Verify 3 figures generated successfully

### Data Acquisition
- [ ] Search ICGC for India Project 2013 data
- [ ] Download variant files (VCF or TSV)
- [ ] Place in `data/raw/india_project_2013/`
- [ ] Load GSE213862 clinical metadata
- [ ] Verify patient counts: ~50 Indian patients

### Data Formatting
- [ ] Convert VCF to standardized CSV format
- [ ] Columns: patient_id, chrom, pos, ref, alt, gene, consequence, vaf
- [ ] Filter: quality > 30, depth > 100, 0.05 < VAF < 0.95
- [ ] Save to `data/processed/india_variants.csv`
- [ ] Count: Expect 10,000-50,000 variants

**Deliverable**: `data/processed/india_variants.csv` ready for scoring

---

## Week 2: Evo2 Scoring

### Pre-Scoring Checks
- [ ] Verify Modal service is deployed
- [ ] Test single variant scoring
- [ ] Estimate cost (variants / 100 * $0.50)
- [ ] Confirm budget approval

### Batch Scoring
- [ ] Run: `python scripts/02_evo2_batch_scoring.py`
- [ ] Monitor progress (check logs every hour)
- [ ] Save intermediate results (every 10 batches)
- [ ] Handle any failures (retry failed batches)
- [ ] Verify all variants scored

### Quality Control
- [ ] Check score distribution (mean, std, range)
- [ ] Identify outliers (scores < -10 or > 10)
- [ ] Verify no missing scores
- [ ] Save: `data/results/evo2_scores_all_variants.csv`

**Deliverable**: All variants scored with Evo2 delta LL

**Estimated Cost**: $20-50
**Estimated Time**: 2-4 hours compute time

---

## Week 3: Core Analyses

### Indel Analysis (Innovation #2)
- [ ] Run: `python scripts/04_indel_analysis.py`
- [ ] Count SNVs vs indels
- [ ] Identify high-impact indels (top 1%)
- [ ] FAT1 indel analysis (expect 10-20 indels)
- [ ] Statistical test: indels vs synonymous (p < 0.05)
- [ ] Generate Figure 1 (4 panels)

**Key Finding**: Evo2 scores X indels, AlphaMissense scores 0

### Gender Analysis (Innovation #4)
- [ ] Run: `python scripts/06_gender_analysis.py`
- [ ] Merge with clinical data (sex, age, tobacco)
- [ ] Compare driver frequencies (Fisher's exact)
- [ ] CASP8 enrichment in females (expect OR > 2)
- [ ] Statistical significance (p < 0.05)
- [ ] Generate Figure 3 (4 panels)

**Key Finding**: CASP8 enriched X-fold in females (p = Y)

### Survival Analysis (Innovation #5)
- [ ] Run: `python scripts/07_survival_integration.py`
- [ ] Calculate PDS for all patients
- [ ] Stratify by PDS (high vs low)
- [ ] Kaplan-Meier curves
- [ ] Log-rank test (expect p < 0.05)
- [ ] Cox model (expect HR > 1.5)
- [ ] Generate Figure 4 (4 panels)

**Key Finding**: High PDS = X-fold increased death risk

**Deliverable**: 3 main figures + results tables

---

## Week 4: Population Comparison (Optional)

### TCGA Download
- [ ] Download TCGA HNSCC mutation data
- [ ] Format to match Indian data
- [ ] Score with Evo2 (if not pre-scored)

### Comparison Analysis
- [ ] Compare driver frequencies (Indian vs TCGA)
- [ ] FAT1: 28% Indian vs 12% TCGA
- [ ] Statistical tests (Fisher's exact)
- [ ] Generate Figure 2 (population comparison)

**Key Finding**: FAT1 enriched X-fold in Indians

---

## Week 5: Figure Refinement

### Figure 1: Indel Advantage
- [ ] Panel A: Score distributions (SNV vs indel)
- [ ] Panel B: Top genes with indels
- [ ] Panel C: FAT1 indel positions
- [ ] Panel D: AlphaMissense comparison
- [ ] Export as PNG (300 dpi) and PDF (vector)

### Figure 2: Population Comparison (if done)
- [ ] Panel A: TCGA driver landscape
- [ ] Panel B: Indian driver landscape
- [ ] Panel C: Frequency comparison
- [ ] Panel D: Evo2 score comparison

### Figure 3: Gender Analysis
- [ ] Panel A: Odds ratio forest plot
- [ ] Panel B: Frequency by sex
- [ ] Panel C: CASP8 scores by sex
- [ ] Panel D: Top genes by sex

### Figure 4: Survival Analysis
- [ ] Panel A: Kaplan-Meier curves
- [ ] Panel B: PDS distribution
- [ ] Panel C: PDS vs survival time
- [ ] Panel D: Cox hazard ratios

**Deliverable**: 4 publication-quality figures

---

## Week 6-7: Manuscript Writing

### Results Section
- [ ] Cohort description (N patients, N variants)
- [ ] Evo2 scoring summary (mean, range)
- [ ] Indel findings (N indels, top genes)
- [ ] FAT1 validation (frequency, scores)
- [ ] Gender findings (CASP8, OR, p-value)
- [ ] Survival findings (PDS, HR, p-value)

### Methods Section
- [ ] Variant calling/filtering
- [ ] Evo2 scoring procedure
- [ ] Statistical tests used
- [ ] Survival analysis methods
- [ ] Code availability statement

### Discussion
- [ ] Indel advantage over AlphaMissense
- [ ] Population-specific drivers
- [ ] Clinical implications
- [ ] Limitations
- [ ] Future directions

### Supplementary Materials
- [ ] Supplementary Table 1: All high-impact variants
- [ ] Supplementary Table 2: Gender comparison (all genes)
- [ ] Supplementary Table 3: Patient characteristics
- [ ] Supplementary Figure 1: QQ plots
- [ ] Supplementary Figure 2: Additional validations

**Deliverable**: Complete manuscript draft

---

## Week 7: Submission Preparation

### Code Repository
- [ ] Create GitHub repository
- [ ] Upload all scripts
- [ ] Add README with usage instructions
- [ ] Include example data
- [ ] Add license (MIT or similar)

### Data Availability
- [ ] Upload processed data to Zenodo/Figshare
- [ ] Get DOI for dataset
- [ ] Add data availability statement

### Submission Materials
- [ ] Main manuscript (Word/LaTeX)
- [ ] All figures (high-res)
- [ ] Supplementary materials
- [ ] Cover letter
- [ ] Suggested reviewers list

### Target Journals (in order)
1. **Nature Communications** (IF: 16.6)
   - Open access
   - Broad readership
   - Accepts computational studies

2. **Clinical Cancer Research** (IF: 11.5)
   - Cancer-focused
   - Clinical relevance
   - Accepts biomarker studies

3. **Genome Medicine** (IF: 10.4)
   - Genomics-focused
   - Open access
   - Accepts AI/ML studies

**Deliverable**: Complete submission package

---

## Success Metrics

### Minimum for Publication
- [ ] >5,000 variants scored
- [ ] >50 high-impact indels
- [ ] At least 1 novel finding (FAT1, CASP8, or MYC)
- [ ] Survival correlation (p < 0.05)
- [ ] 3 main figures

### For High-Impact Journal (Nature Communications)
- [ ] >10,000 variants scored
- [ ] Multiple novel findings (FAT1 + CASP8 + survival)
- [ ] Strong survival correlation (p < 0.01, HR > 1.5)
- [ ] Population comparison (Indian vs TCGA)
- [ ] 4 main figures + 5 supplementary

---

## Budget Tracking

| Item | Estimated | Actual | Notes |
|------|-----------|--------|-------|
| Evo2 compute | $20-50 | | H100 GPU on Modal |
| Data download | $0 | | Public datasets |
| Publication fee | $0-5000 | | If open access |
| **Total** | **$20-5050** | | |

---

## Timeline Summary

| Week | Phase | Status | Deliverable |
|------|-------|--------|-------------|
| 1 | Data Prep | ⬜ | Formatted variants |
| 2 | Evo2 Scoring | ⬜ | All scores |
| 3 | Core Analyses | ⬜ | 3 figures |
| 4 | Population | ⬜ | Figure 2 (optional) |
| 5 | Figures | ⬜ | Final figures |
| 6-7 | Writing | ⬜ | Manuscript |
| 7 | Submission | ⬜ | Submit! |

---

## Notes & Blockers

### Current Blockers
- [ ] Need variant data (India Project 2013)
- [ ] Need Modal endpoint URL
- [ ] Need survival time data (if not in GSE213862)

### Resolved Issues
- [x] Evo2 backend ready (main.py)
- [x] Analysis scripts written
- [x] Figure generation automated

### Questions for PI/Collaborators
- [ ] Budget approval for Evo2 compute?
- [ ] Access to India Project 2013 data?
- [ ] Target journal preference?
- [ ] Authorship order?

---

**Last Updated**: [Date]
**Next Review**: [Date]
**Status**: Setup Complete, Ready for Data
