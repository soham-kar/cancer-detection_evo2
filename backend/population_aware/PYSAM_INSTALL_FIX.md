# pysam Installation Fix for Windows

## Problem
pysam requires C++ compilation on Windows, which fails without build tools.

## Solution Options

### Option 1: Use Conda (RECOMMENDED)
```bash
# Install conda if you don't have it
# Download from: https://docs.conda.io/en/latest/miniconda.html

# Then install pysam
conda install -c bioconda pysam
```

### Option 2: Install Microsoft C++ Build Tools
1. Download: https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. Install "Desktop development with C++"
3. Restart terminal
4. Run: `pip install pysam`

### Option 3: Use cyvcf2 instead (Alternative library)
```bash
pip install cyvcf2
```

Then modify `gnomad_client_local.py` to use cyvcf2 instead of pysam.

### Option 4: Use WSL2 (Linux on Windows)
```bash
# In WSL2 terminal
pip install pysam  # Works perfectly in Linux
```

---

## Quick Test: Which Option Works?

Run this to check if conda is available:
```bash
conda --version
```

If conda is available, use Option 1.
If not, use Option 3 (cyvcf2) as the quickest workaround.

---

## Recommended: Option 3 (cyvcf2)

This is the fastest solution for Windows:

```bash
pip install cyvcf2
```

I'll update the code to support cyvcf2 automatically.
