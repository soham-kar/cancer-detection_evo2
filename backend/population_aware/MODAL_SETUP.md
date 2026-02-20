# Modal Setup Guide for Population-Aware Calibration

## Step 1: Install Modal CLI

```bash
pip install modal
```

## Step 2: Authenticate with Modal

```bash
modal token new
```

This will:
1. Open your browser
2. Ask you to log in to Modal (use your new account)
3. Generate an API token
4. Save it locally to `~/.modal.toml`

## Step 3: Verify Authentication

```bash
modal profile list
```

You should see your profile listed.

## Step 4: Create Modal App

Modal apps are defined in Python files. We'll create two:
1. `modal_score_evo2.py` - For Day 2-3: Score variants with Evo2-7B
2. `modal_train.py` - For Day 5: Train the calibrator (if needed on GPU)

## Step 5: Test Modal Connection

```bash
# Run a simple test
modal run backend/population_aware/modal_score_evo2.py::test
```

Expected output:
```
✓ App initialized
✓ Test function completed
```

## Next Steps

After Modal is authenticated:
1. **Today (Day 1):** Finish downloading chr17 and run `day1_add_gnomad.py` locally
2. **Tomorrow (Day 2):** Deploy Evo2 scoring to Modal H100 GPUs
3. **Day 3:** Process all 3,893 variants (~30 min on Modal)

---

## Cost Estimate

Modal charges for GPU usage:
- **H100 GPU:** ~$4/hour
- **Estimated time:** 30 minutes for 3,893 variants
- **Total cost:** ~$2 for Day 2-3

This is much cheaper than renting your own GPU!

---

## Environment Variables Needed

For the population_aware project, you'll need:
- No external secrets for Day 2-3 (Evo2 is open-source)
- Redis URL (optional, for caching)

These will be set in the Modal script.
