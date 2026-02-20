# Quick Fix Guide - Evo2 Innovation

## Status: ✅ FIXED - Ready for Testing

### What Was Fixed

1. ✅ **Completed truncated main function** in `modal_extract.py`
   - Added complete local_entrypoint with proper error handling
   - Added download instructions in output

2. ✅ **BFloat16 conversion** already fixed (lines 115-116)
   - Proper conversion: BFloat16 → Float32 → Float16
   - Prevents overflow errors

3. ✅ **Tokenizer compatibility** already handled
   - Multiple fallback paths for different tokenizer APIs
   - Byte encoding as last resort

### Remaining Issues

#### 1. Missing Input Data
**Problem**: `/data/input1k.json` may not exist in Modal volume

**Solution A - Use Existing Test Data**:
```bash
# Upload test data
modal volume put crispr-data data/features/test_input.json /data/input1k.json

# Run extraction
modal run modal_extract.py
```

**Solution B - Generate New Test Data**:
```bash
# Generate test sequences
python simple_test.py

# This will:
# 1. Create test_input.json
# 2. Upload to Modal
# 3. Run extraction
# 4. Download results
# 5. Validate
```

#### 2. CHANGE-seq Dataset Missing
**Problem**: Full dataset not available

**Solution**: Download from Nature Biotechnology paper
```bash
# Manual download required from:
# https://www.nature.com/articles/s41587-020-0555-7
# Supplementary Data 2: CHANGE-seq off-target sites

# Or use Kleinstiver dataset as alternative:
cd backend/crispr_offtarget/data
# Use existing Kleinstiver data for quantitative analysis
```

---

## Testing Workflow

### Step 1: Quick Test (5 samples, ~1 minute, $0.03)
```bash
cd backend/crispr_offtarget/evo2_innovation

# Option A: Use test script
python test_modal.py

# Option B: Use simple test (generates mock data)
python simple_test.py
```

**Expected Output**:
```
=============================================================
MODAL EXTRACT TEST (5 samples)
=============================================================

>>> Running Modal extraction (5 samples)...
Processing 5 items from /data/input1k.json
  5/5 done

>>> Downloading results...
>>> Validating results...
Total: 5, Success: 5
Sample score: -1.084908
✅ TEST PASSED - Ready for full run
```

### Step 2: Medium Test (100 samples, ~15 minutes, $0.50)
```bash
# Use run_modal_extraction.py with limit
python run_modal_extraction.py --limit 100
```

### Step 3: Full Run (1000+ samples, ~3 hours, $6)
```bash
# Only run after successful medium test
python run_modal_extraction.py --full
```

---

## Validation Checklist

Before full run, verify:
- [ ] Test with 5 samples passes
- [ ] All 5 extractions successful
- [ ] Evo2 scores are reasonable (-5 to 5 range)
- [ ] No errors in Modal logs
- [ ] Cache system working (check for .npz files)

---

## Troubleshooting

### Issue: "File not found: /data/input1k.json"
**Solution**: Upload input data first
```bash
modal volume put crispr-data data/features/test_input.json /data/input1k.json
```

### Issue: "Tokenizer error"
**Check**: Modal logs for tokenizer type
```bash
modal logs evo2-chromatin-innovation-fixed
```

**Expected**: Should see "Tokenizer type: <class '...'>"

### Issue: "GPU OOM"
**Solution**: Already handled with try-catch and torch.cuda.empty_cache()
- Batch size is 1 per call (safe for H100)
- If still OOM, reduce sequence length or use smaller model

### Issue: "Model access error"
**Check**: Verify `self.model.model` exists
```python
# In load_model():
print(f"Model attributes: {dir(self.model)}")
```

---

## Cost Optimization

### Current Settings
- GPU: H100 (expensive but fast)
- Scaledown: 120 seconds (2 minutes idle before shutdown)
- Batch size: 1 sequence per call

### Optimization Options

**Option 1: Use L40S instead of H100**
```python
@app.cls(
    gpu="L40S",  # Change from H100
    # ... rest same
)
```
- Cost: ~50% cheaper
- Speed: ~2x slower
- Best for: Testing and development

**Option 2: Increase batch size**
```python
# In process_dataset function
BATCH_SIZE = 4  # Process 4 sequences at once
```
- Requires: More VRAM testing
- Risk: Potential OOM errors
- Benefit: ~2-3x faster

**Option 3: Use caching aggressively**
```python
# Already implemented in extract_with_cache()
# Rerun same dataset = instant results from cache
```

---

## Next Steps After Testing

### If Test Passes ✅
1. Run medium test (100 samples)
2. Validate chromatin correlation
3. Train chromatin-aware model
4. Benchmark vs heuristics

### If Test Fails ❌
1. Check Modal logs: `modal logs evo2-chromatin-innovation-fixed`
2. Verify input data format
3. Test tokenizer compatibility
4. Report issue with full error trace

---

## Expected Results

### Feature Extraction Output
```json
{
  "seq_id": "seq_0",
  "evo2_score": -1.084908,
  "seq_length": 8000,
  "success": true,
  "source": "fresh"  // or "cache" if cached
}
```

### Cached Features (NPZ file)
```python
data = np.load("features/seq_0.npz")
# Contains:
# - evo2_score: float
# - global_embedding: [512] float16
# - center_embedding: [512] float16
```

---

## Performance Benchmarks

### Extraction Speed (H100)
- Single sequence: ~10 seconds
- 100 sequences: ~15 minutes (with caching)
- 1000 sequences: ~2.5 hours (with caching)

### Storage Requirements
- Per sequence: ~1 KB (compressed NPZ)
- 1000 sequences: ~1 MB
- 100K sequences: ~100 MB

### Cost Estimates
- Test (5 samples): $0.03
- Medium (100 samples): $0.50
- Full (1000 samples): $6.00
- Large (10K samples): $60.00

---

## Contact & Support

### Modal.com Issues
- Check status: https://status.modal.com
- Docs: https://modal.com/docs

### Evo2 Model Issues
- GitHub: https://github.com/ArcInstitute/evo2
- Paper: https://arcinstitute.org/evo2

### Project Issues
- Check ANALYSIS.md for detailed architecture
- Check TEST_GUIDE.md for testing procedures
- Check CODE_ANALYSIS_REPORT.md for comprehensive review
