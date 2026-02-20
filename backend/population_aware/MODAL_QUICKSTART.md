# Modal Quick Start

## 1. Install and Authenticate (Do this now)

```bash
# Install Modal
pip install modal

# Authenticate (opens browser)
modal token new
```

**Follow the browser prompts to log in with your new Modal account.**

## 2. Verify Setup

```bash
# Check your profile
modal profile list

# Test the deployment script
modal run backend/population_aware/modal_score_evo2.py::test
```

Expected output:
```
✓ Modal app initialized successfully!
✓ GPU image configured
✓ Data volume mounted

Ready for Day 2 variant scoring!
```

## 3. What This Sets Up

**For Day 2-3 (Evo2 Scoring):**
- H100 GPU instance
- Evo2-7B model (~7 billion parameters)
- Automated variant scoring pipeline
- Cost: ~$2 total

**NOT needed yet:**
- Redis (we're using local files for now)
- External APIs (gnomAD is local VCFs)
- Secrets (Evo2 is open-source)

## 4. Timeline

- **Today:** Authenticate Modal ✓
- **Day 1 (finish):** Complete local annotation with gnomAD
- **Day 2:** Deploy Evo2 scoring to Modal
- **Day 3:** Score all 3,893 variants (~30 min on Modal)

## Troubleshooting

**"modal: command not found"**
```bash
pip install modal
```

**"No token found"**
```bash
modal token new
# Follow browser prompts
```

**"ImportError: No module named modal"**
```bash
python -m pip install modal
```

---

Once authenticated, you're ready for Day 2! 🚀
