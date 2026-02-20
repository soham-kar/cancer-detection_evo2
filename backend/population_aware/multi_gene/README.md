# Multi-Gene Evo2 Validation

**Organized structure for multi-gene analysis**

---

## 📁 Folder Structure

```
multi_gene/
├── brca1_findlay/          # BRCA1 (Findlay experimental data) - COMPLETE
│   └── [Reference only - results in ../results/evo2_results/]
│
├── brca2_clinvar/          # BRCA2 (ClinVar clinical data)
│   ├── brca2_curate_clinvar.py     # Step 1: Curate variants
│   ├── brca2_fetch_sequences.py    # Step 2: Fetch genomic sequences
│   ├── brca2_score_evo2.py         # Step 3: Score with Evo2
│   └── brca2_validate.py           # Step 4: Cross-validation
│
├── palb2_clinvar/          # PALB2 (ClinVar clinical data)
│   ├── palb2_curate_clinvar.py     # Step 1: Curate variants
│   ├── palb2_fetch_sequences.py    # Step 2: Fetch genomic sequences
│   ├── palb2_score_evo2.py         # Step 3: Score with Evo2
│   └── palb2_validate.py           # Step 4: Cross-validation
│
└── meta_analysis/          # Cross-gene comparison
    ├── compare_genes.py            # Meta-analysis
    ├── create_figures.py           # Publication figuresю    └── manuscript_tables.py        # Tables for paper
```

---

## 🚀 Execution Order

### **Week 1: BRCA2 & PALB2 Curation**

#### Day 1: Download ClinVar (ONE-TIME)
```powershell
# Download ClinVar VCF (~2GB, takes 10-15 min)
Invoke-WebRequest -Uri "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz" -OutFile "data/clinvar.vcf.gz"

# Download index
Invoke-WebRequest -Uri "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz.tbi" -OutFile "data/clinvar.vcf.gz.tbi"
```

#### Day 2: Curate BRCA2 & PALB2
```bash
# BRCA2
cd multi_gene/brca2_clinvar
python brca2_curate_clinvar.py

# PALB2
cd ../palb2_clinvar
python palb2_curate_clinvar.py
```

**CRITICAL:** Check sample sizes before proceeding!
- BRCA2 needs ≥800 for robust analysis
- PALB2 needs ≥400 for exploratory analysis
- If PALB2 < 300, EXCLUDE from study

---

## 📊 Data Sources

| Gene | Data Source | Validation Type | Expected N |
|------|-------------|----------------|------------|
| BRCA1 | Findlay 2018 | Experimental (gold standard) | 3,644 |
| BRCA2 | ClinVar | Clinical (realistic) | 1,200-1,800 |
| PALB2 | ClinVar | Clinical (exploratory) | 400-600 |

**Note:** Different data sources allow assessment of Evo2 under both ideal (experimental) and realistic (clinical) conditions.

---

## ✅ Progress Tracking

- [x] Folder structure created
- [x] BRCA2 curation script ready
- [x] PALB2 curation script ready
- [ ] ClinVar VCF downloaded
- [ ] BRCA2 variants curated
- [ ] PALB2 variants curated
- [ ] Sample sizes verified
- [ ] Sequences fetched
- [ ] Evo2 scoring complete
- [ ] Cross-validation done
- [ ] Meta-analysis figures generated

---

## 🎯 Decision Criteria

After curation (Day 2), decide:

1. **If BRCA2 ≥ 800 AND PALB2 ≥ 400:** → Proceed with 3-gene analysis
2. **If BRCA2 ≥ 800 AND PALB2 < 400:** → 2-gene analysis (BRCA1 + BRCA2)
3. **If BRCA2 < 800:** → Pivot to BRCA1-only paper (already complete!)

---

## 📝 Next Steps

1. **Wait for ClinVar download to complete** (~10-15 min)
2. **Run curation scripts** (Day 2)
3. **Verify sample sizes** before committing to 4-week timeline
