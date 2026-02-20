# Quick Test Guide for modal_extract.py

## Status: ✅ BFloat16 Fixed, Ready for Testing

### What's Fixed
- ✅ BFloat16 → float32 → float16 conversion (lines 115-116)
- ✅ Completed truncated main function
- ✅ Added test mode with `--limit` parameter
- ✅ Added validation step

### Test Before Full Run

**Cost**: $0.03 (5 samples) vs $6 (1000 samples)

```bash
# 1. Test with 5 samples first
python test_modal.py

# 2. If test passes, run full extraction
modal run modal_extract.py
```

### Expected Output (Test)
```
MODAL EXTRACT TEST (5 samples)
>>> Running Modal extraction (5 samples)...
Processing 5 items from /data/input1k.json
  5/5 done
>>> Downloading results...
>>> Validating results...
Total: 5, Success: 5
Sample score: -2.345
✅ TEST PASSED - Ready for full run
```

### If Test Fails

Check Modal logs:
```bash
modal logs evo2-chromatin-innovation-fixed
```

Common issues:
1. **Tokenizer error**: Check `load_model()` output for tokenizer type
2. **Model access error**: Verify `self.model.model` exists
3. **Memory error**: Reduce batch size (already at 1 per call)

### Verification Checklist

Before full run:
- [ ] Test with 5 samples passes
- [ ] All 5 extractions successful
- [ ] Evo2 scores are reasonable (-5 to 5 range)
- [ ] No errors in Modal logs

### Full Run Command

```bash
modal run modal_extract.py
```

Expected:
- Time: ~3 hours (1000 samples × 10 sec/sample)
- Cost: ~$6 (H100 GPU time)
- Output: `data/features/evo2_feats_1k.json`
