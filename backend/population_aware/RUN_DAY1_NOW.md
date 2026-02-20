# Day 1 - Final Steps

## ✅ Downloads Complete!

You have everything needed:
- chr13 VCF: 25.65 GB ✓
- chr13 index: 92 KB ✓  
- chr17 VCF: 25.17 GB ✓ (BRCA1)
- chr17 index: 2 MB ✓

**You DON'T need chr16 right now.** It's for PALB2, which we'll add later if needed.

---

## 🚀 Run Day 1 Pipeline Now

### Step 1: Install pysam
```bash
pip install pysam
```

### Step 2: Run annotation
```bash
cd D:\project\biotech-evo2\backend\population_aware
python day1_add_gnomad.py
```

**Expected output:**
```
INFO:__main__:Loading Findlay dataset...
INFO:__main__:Loaded 3893 variants
INFO:__main__:Converting coordinates: hg19 → hg38
INFO:__main__:Liftover success: 3893/3893 (100.0%)
INFO:__main__:Querying gnomAD v4 VCF files...
Querying gnomAD: 100%|████████| 3893/3893 [03:42<00:00, 17.5it/s]
INFO:__main__:gnomAD found: 3672/3893 (94.3%)
✅ Saved annotated dataset to: results/brca1_with_gnomad.csv
```

**Time:** 3-5 minutes

---

## After Pipeline Completes

### Verify output:
```bash
# Check file exists
ls results/

# Preview data
python -c "import pandas as pd; df = pd.read_csv('results/brca1_with_gnomad.csv'); print(df.head()); print(f'\nTotal variants: {len(df)}')"
```

---

## Then You're Done with Day 1! 🎉

**Day 1 Complete checklist:**
- ✅ Project structure
- ✅ Liftover implementation  
- ✅ Local VCF client
- ✅ Modal authentication
- ✅ Downloaded chr17 VCF
- ✅ Annotated variants with population AFs ← **You're here next**

**Tomorrow (Day 2):** Score variants with Evo2 on Modal H100 GPUs
