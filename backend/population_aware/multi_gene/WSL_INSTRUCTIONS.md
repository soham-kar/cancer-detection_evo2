# Multi-Gene Curation - WSL Instructions

**Problem:** `pysam` doesn't work on Windows  
**Solution:** Run curation in WSL (Ubuntu)

---

## 🐧 Setup WSL Environment

### 1. Open WSL (Ubuntu)
```bash
wsl
```

### 2. Navigate to project
```bash
cd /mnt/d/project/biotech-evo2/backend/population_aware
```

### 3. Install pysam (in WSL)
```bash
pip install pysam pandas
```

---

## 📊 Run BRCA2 Curation

```bash
cd multi_gene/brca2_clinvar
python brca2_curate_clinvar.py
```

**Expected output:**
```
BRCA2 CLINVAR CURATION
======================================================================

📂 Reading ClinVar VCF: ../../data/clinvar.vcf.gz
🔍 Extracting BRCA2 variants (chr13:32315086-32400268)...

📊 BRCA2 Curation Results:
  Total BRCA2 variants (2+ stars): 1,500-2,000
  Pathogenic: 500-700
  Benign: 1,000-1,300
  VUS (will exclude): 200-300

  ✅ Usable for validation: 1,200-1,800
     Pathogenic: 500-700
     Benign: 700-1,100

  ✅ Sample size sufficient for robust validation

💾 Saved to multi_gene/brca2_clinvar/brca2_clinvar_curated.csv
```

---

## 📊 Run PALB2 Curation

```bash
cd ../palb2_clinvar
python palb2_curate_clinvar.py
```

**Expected output:**
```
PALB2 CLINVAR CURATION
======================================================================

📊 PALB2 Curation Results:
  Total PALB2 variants (2+ stars): 400-600
  Pathogenic: 150-250
  Benign: 250-350

  ✅ Usable for validation: 400-550
  
  ⚠️  WARNING: Sample size <500
     Report as EXPLORATORY ONLY
```

---

## 🎯 Decision Point

After curation, check sample sizes:

```bash
# Check BRCA2
wc -l ../brca2_clinvar/brca2_clinvar_curated.csv

# Check PALB2
wc -l palb2_clinvar_curated.csv
```

**Decision criteria:**
- **BRCA2 ≥ 800 usable?** → Proceed with multi-gene
- **PALB2 ≥ 400 usable?** → Include as exploratory
- **PALB2 < 300?** → Exclude, focus on BRCA1+BRCA2 only

---

## 📁 Files Created (in WSL)

After successful curation:
```
multi_gene/
├── brca2_clinvar/
│   └── brca2_clinvar_curated.csv  (~1,500-2,000 variants)
└── palb2_clinvar/
    └── palb2_clinvar_curated.csv  (~400-600 variants)
```

---

## 🔜 Next Steps (After Curation)

1. **Verify sample sizes** (see decision criteria above)
2. **Fetch genomic sequences** (will create separate script)
3. **Score with Evo2** (reuse Modal pipeline)
4. **Cross-validation** (5-fold CV)

---

## ⚠️ Important Notes

- **Run in WSL, not PowerShell** (pysam requirement)
- **ClinVar VCF path:** `../../data/clinvar.vcf.gz` (relative to gene folders)
- **Check sample sizes** before proceeding to avoid wasted work
- **PALB2 may be underpowered** - that's OK, report as exploratory

---

## 🚀 Quick Start (Copy-Paste)

```bash
# In WSL
wsl
cd /mnt/d/project/biotech-evo2/backend/population_aware

# Install dependencies (if needed)
pip install pysam pandas

# Run BRCA2 curation
cd multi_gene/brca2_clinvar
python brca2_curate_clinvar.py

# Run PALB2 curation
cd ../palb2_clinvar
python palb2_curate_clinvar.py

# Check results
cd ..
echo "BRCA2 variants: $(wc -l < brca2_clinvar/brca2_clinvar_curated.csv)"
echo "PALB2 variants: $(wc -l < palb2_clinvar/palb2_clinvar_curated.csv)"
```

**Ready to curate! Switch to WSL and run the commands above.** 🐧
