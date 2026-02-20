# Quick Start Guide: Somatic Driver Analysis

## 🚀 Getting Started (5 minutes)

### Step 1: Setup Environment
```bash
cd backend/somatic_driver_analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Test Your Evo2 Backend
```bash
# Make sure your main.py Modal service is deployed
cd ..
modal deploy main.py

# Get your endpoint URL
modal app show variant-analysis

# Update config.py with your endpoint
# Edit config.py and set MODAL_EVO2_ENDPOINT = "your-url-here"
```

### Step 3: Create Mock Data (for testing)
```bash
cd somatic_driver_analysis
python scripts/01_data_preparation.py --source mock
```

This creates `data/processed/mock_variants.csv` with 20 patients and ~50 variants.

### Step 4: Score Variants with Evo2
```bash
# Test connection first
python scripts/02_evo2_batch_scoring.py \
    --input data/processed/mock_variants.csv \
    --test

# If connection works, run scoring
python scripts/02_evo2_batch_scoring.py \
    --input data/processed/mock_variants.csv \
    --output data/results/evo2_scores.csv
```

**Expected time**: 2-5 minutes for mock data
**Expected cost**: $0.50-1.00

### Step 5: Run Analyses
```bash
# Indel analysis (key innovation)
python scripts/04_indel_analysis.py \
    --input data/results/evo2_scores.csv

# Gender analysis
python scripts/06_gender_analysis.py \
    --input data/results/evo2_scores.csv \
    --clinical data/processed/gse213862_clinical.csv

# Survival analysis
python scripts/07_survival_integration.py \
    --input data/results/evo2_scores.csv \
    --clinical data/processed/gse213862_clinical.csv
```

### Step 6: View Results
```bash
# Figures are in figures/
ls figures/

# Results tables are in data/results/
ls data/results/
```

---

## 📊 Working with Real Data

### Option A: Use Published Indian OSCC Data

1. **Download India Project 2013 data**:
   - Visit: https://www.nature.com/articles/ng.2764
   - Download supplementary tables with variant calls
   - Or search ICGC: https://dcc.icgc.org/

2. **Format the data**:
   ```bash
   # Place VCF or CSV files in data/raw/india_project_2013/
   
   # If VCF format, convert to CSV:
   python scripts/convert_vcf_to_csv.py \
       --input data/raw/india_project_2013/*.vcf \
       --output data/processed/india_variants.csv
   ```

3. **Score with Evo2**:
   ```bash
   python scripts/02_evo2_batch_scoring.py \
       --input data/processed/india_variants.csv \
       --output data/results/evo2_scores_india.csv
   ```

### Option B: Use TCGA HNSCC Data

1. **Download from GDC**:
   ```bash
   # Install GDC client
   # https://gdc.cancer.gov/access-data/gdc-data-transfer-tool
   
   # Download TCGA-HNSC mutations
   gdc-client download -m tcga_hnsc_manifest.txt
   ```

2. **Process MAF files**:
   ```bash
   python scripts/process_tcga_maf.py \
       --input data/raw/tcga_hnsc/*.maf \
       --output data/processed/tcga_variants.csv
   ```

---

## 🔧 Troubleshooting

### Issue: "Connection failed" when scoring
**Solution**: 
1. Check Modal service is running: `modal app list`
2. Get correct endpoint: `modal app show variant-analysis`
3. Update `config.py` with correct URL

### Issue: "No clinical data found"
**Solution**:
1. Make sure GSE213862 clinical data is at: `../../data/india/GSE213862_clinical.csv`
2. Or provide path explicitly: `--clinical /path/to/clinical.csv`

### Issue: "Not enough variants for analysis"
**Solution**:
1. Use mock data for testing: `--source mock`
2. Or download real data (see "Working with Real Data" above)

### Issue: "Out of memory" during scoring
**Solution**:
1. Reduce batch size in `config.py`: `batch_size: 50` (default is 100)
2. Or process in smaller chunks

---

## 📈 Expected Outputs

### After Step 4 (Scoring):
- `data/results/evo2_scores.csv` - All variants with Evo2 delta LL scores

### After Step 5 (Analysis):
- `data/results/high_impact_indels.csv` - Top deleterious indels
- `data/results/fat1_indels.csv` - FAT1-specific indels (Indian driver)
- `data/results/gender_driver_comparison.csv` - Sex-specific drivers
- `data/results/patient_survival_with_pds.csv` - Prognostic scores

### Figures:
- `figures/figure_indel_analysis.png` - Indel vs SNV comparison
- `figures/figure_gender_analysis.png` - Sex-stratified drivers
- `figures/figure_survival_analysis.png` - Kaplan-Meier curves

---

## 💡 Tips for Nature Paper

### Key Analyses to Highlight:

1. **Indel Advantage** (Innovation #2):
   - Show AlphaMissense cannot score indels
   - Evo2 scores 100+ indels with high confidence
   - FAT1 frameshift indels = Indian-specific driver

2. **Gender Differences** (Innovation #4):
   - CASP8 enriched in females (confirm Kolar et al.)
   - Different driver landscapes by sex
   - Clinical implication: sex-specific therapies

3. **Prognostic Value** (Innovation #5):
   - PDS predicts survival (p < 0.05)
   - Hazard ratio > 1.5
   - Better than expression-only models

### Figures for Paper:
- **Figure 1**: Indel analysis (4 panels)
- **Figure 2**: Population comparison (Indian vs TCGA)
- **Figure 3**: Gender analysis (4 panels)
- **Figure 4**: Survival curves (4 panels)

---

## 📞 Need Help?

1. Check `README.md` for project overview
2. Review `config.py` for all settings
3. Look at script docstrings for detailed usage
4. Create GitHub issue if stuck

---

## ⏱️ Timeline Estimate

- **Week 1**: Data preparation (real data download)
- **Week 2**: Evo2 scoring (10K-50K variants)
- **Week 3**: Run all analyses
- **Week 4**: Generate figures
- **Week 5**: Write manuscript

**Total**: 5 weeks (vs. 12 weeks in original roadmap)

**Cost**: $20-50 (vs. $100-200 in original estimate)

---

## 🎯 Success Criteria

✓ Scored >5,000 unique variants
✓ Identified >50 high-impact indels
✓ Found sex-specific drivers (p < 0.05)
✓ PDS correlates with survival (p < 0.05)
✓ Generated 4 publication-quality figures

**Ready for Nature Communications submission!**
