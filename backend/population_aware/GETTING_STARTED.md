# Day 1: Getting Started

## Prerequisites

1. **Install dependencies:**
```bash
cd backend/population_aware
pip install -r requirements.txt
```

2. **Verify data file exists:**
```bash
# The Findlay dataset should be at:
backend/evo2/notebooks/brca1/41586_2018_461_MOESM3_ESM.xlsx
```

## Running Day 1

### Option 1: Full Pipeline (30-60 minutes)
```bash
cd backend/population_aware
python day1_add_gnomad.py
```

### Option 2: Test Components First (Recommended)
```bash
# Test liftover
python -c "from utils.liftover import test_liftover; test_liftover()"

# Test gnomAD client
python -c "from utils.gnomad_client import test_gnomad_client; test_gnomad_client()"

# If both pass, run full pipeline
python day1_add_gnomad.py
```

## Expected Output

```
Loading Findlay dataset from ...
Loaded 3893 variants
Class distribution: {'FUNC': 2204, 'INT': 522, 'LOF': 1167}

Converting coordinates: hg19 → hg38
Liftover success: 3782/3893 (97.2%)

Querying gnomAD v4 for population frequencies...
100%|████████████████████| 3893/3893 [30:15<00:00,  2.14it/s]
gnomAD found: 1245/3893 (32.0%)

============================================================
ANNOTATION SUMMARY
============================================================
Total variants: 3893
Liftover success: 3782 (97.2%)
gnomAD found: 1245 (32.0%)

Population AF coverage:
  NFE: 1089 variants (28.0%)
  AFR: 287 variants (7.4%)
  EAS: 156 variants (4.0%)
  SAS: 198 variants (5.1%)
  AMR: 234 variants (6.0%)

Variants with both EUR and AFR frequencies: 189
============================================================

✅ Saved annotated dataset to: backend/population_aware/results/brca1_with_gnomad.csv
```

## Troubleshooting

### Issue: Low gnomAD hit rate (<20%)
- **Cause:** Many Findlay variants are ultra-rare (not in gnomAD)
- **Solution:** This is expected. We'll filter to common variants (AF > 0.001) in Day 4

### Issue: Liftover failures (>10%)
- **Cause:** Structural variants or regions with assembly differences
- **Solution:** Flag these and exclude from downstream analysis

### Issue: API timeout/rate limit errors
- **Cause:** gnomAD server congestion
- **Solution:** Increase `rate_limit_delay` to 0.5 in `gnomad_client.py`

## Next Steps

After Day 1 completes, you should have `brca1_with_gnomad.csv` with columns:
- `chrom`, `pos_hg19`, `pos_hg38`, `ref`, `alt`
- `func_score`, `func_class` (from Findlay)
- `af_nfe`, `af_afr`, `af_eas`, `af_sas`, `af_amr` (from gnomAD)
- `liftover_success`, `gnomad_found` (quality flags)

This file is ready for **Day 2: Evo2 Scoring**.
