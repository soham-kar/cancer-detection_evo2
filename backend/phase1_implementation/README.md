# Phase 1 Implementation: Multi-Model Consensus + Benchmarking

**Target:** Publication-ready in 2 weeks  
**Target Journal:** *Bioinformatics* or *BMC Medical Genomics*  
**Research Gaps:** #7 (AlphaMissense), #1 (Benchmarking), #3 (Counterfactuals)

---

## 📁 File Structure

```
phase1_implementation/
├── README.md                          ← This file
├── download_alphamissense.py          ← Day 1: Download AlphaMissense DB (2GB)
├── alphamissense_lookup.py            ← Day 2: DuckDB lookup service
├── consensus_engine.py                ← Day 3-4: Multi-model consensus + LLM explainer
├── multi-model-consensus.tsx          ← Day 5: Frontend consensus panel
├── benchmark_runner.py                ← Week 2: ClinVar 4K benchmark runner
├── ism_full_scan.py                   ← Gap #3: 20-AA counterfactual scanner
├── alphamissense_data/                ← Downloaded AlphaMissense TSV files
├── analysis/
│   ├── benchmark_metrics.py           ← AUROC, AUPRC, sensitivity/specificity
│   ├── calibration.py                 ← Reliability diagrams, ECE, Brier score
│   ├── generate_figures.py            ← Publication figures (300 DPI)
│   └── figures/                       ← Generated PNG files
├── helixmind_benchmark_results.csv    ← Raw benchmark output
└── benchmark_checkpoint.json          ← Crash recovery checkpoint
```

---

## 📚 Referenced Papers

| Gap | Paper | Journal | Key Finding |
|:---:|-------|---------|-------------|
| #7 | Cheng et al. (2023) | *Science* | AlphaMissense: 71M missense scores, CC BY 4.0 |
| #7 | Ruzicka et al. (2025) | *medRxiv* | Clinical evaluation: reduces workload, needs multi-model |
| #7 | Medge et al. (2026) | *SSRN* | Pathoscope: SHAP on GBMs, no foundation models |
| #1 | Nguyen et al. (2024) | *bioRxiv* | Evo2-7B: 1Mbp context, zero-shot variant prediction |
| #1 | De la Vega et al. (2021) | *Genome Medicine* | GEM: traditional ML, no foundation models |
| #3 | Frazer et al. (2021) | *Nature Genetics* | ISM: research technique, not clinical feature |
| #3 | Weile et al. (2023) | *Cell* | Base editing: experimental AA scans, limited genes |

---

## 🚀 Quick Start

### Prerequisites
```bash
pip install duckdb pandas numpy scikit-learn matplotlib requests groq
```

### Step 1: Download AlphaMissense Database (Day 1)
```bash
cd phase1_implementation
python download_alphamissense.py
```
Downloads ~2GB of pre-computed AlphaMissense scores from Google Cloud Storage.
If `gsutil` is not installed, choose the HTTP fallback option.

### Step 2: Build DuckDB Index (Day 2)
```bash
python alphamissense_lookup.py --build
python alphamissense_lookup.py --stats
```
Imports all TSV files into a DuckDB database with indexed lookups.

### Step 3: Test Consensus Engine (Day 3-4)
```bash
python consensus_engine.py
```
Runs 3 test cases showing full agreement, disagreement, and partial data.

### Step 4: Run Benchmark (Week 2)
```bash
# Ensure Modal endpoint is deployed first:
# cd ../backend && modal deploy main.py

python benchmark_runner.py
```
Runs Evo2-7B on all 4,000 ClinVar variants. Saves checkpoints every 100 variants.

### Step 5: Compute Metrics
```bash
python analysis/benchmark_metrics.py
python analysis/calibration.py
python analysis/generate_figures.py
```

---

## 📊 Expected Outputs

### After Benchmark
| Metric | Expected Range | What It Means |
|--------|:-------------:|---------------|
| AUROC | 0.75–0.85 | Discriminative power (P vs B) |
| AUPRC | 0.70–0.85 | Better for imbalanced data |
| Sensitivity @ 90% Spec | 0.50–0.70 | Clinical sensitivity |
| ECE | 0.05–0.15 | Calibration error (lower = better) |
| Brier Score | 0.10–0.20 | Overall prediction accuracy |

### Paper-Ready Claims
1. **"First multi-model consensus platform combining Evo2-7B, AlphaMissense, CADD, and REVEL with explainable disagreement resolution"**
2. **"Evo2-7B achieves AUROC of X.XX on ClinVar 4K stratified benchmark"**
3. **"Calibration analysis reveals well-calibrated confidence scores (ECE = 0.0X)"**
4. **"Full 20-amino-acid counterfactual analysis reveals position-specific constraint patterns"**

---

## 🔧 Integration with Main Codebase

### Backend (`backend/main.py`)
Add to `run_analysis_logic()` after `_fetch_external_scores()`:
```python
# NEW: AlphaMissense lookup
if vep_annotation and vep_annotation.get("uniprotId"):
    from phase1_implementation.alphamissense_lookup import AlphaMissenseDB
    am_db = AlphaMissenseDB("phase1_implementation/alphamissense_data/")
    am_score = am_db.lookup(
        uniprot_id=vep_annotation["uniprotId"],
        position=vep_annotation.get("aaPosition"),
        ref_aa=vep_annotation.get("refAA"),
        alt_aa=vep_annotation.get("altAA")
    )
    if am_score:
        external_scores["alphamissense"] = am_score

# NEW: Multi-model consensus
from phase1_implementation.consensus_engine import ConsensusEngine
engine = ConsensusEngine()
consensus = engine.compute_consensus(
    evo2_prediction=prediction,
    evo2_confidence=confidence,
    alphamissense_score=external_scores.get("alphamissense", {}).get("score"),
    cadd_phred=external_scores.get("cadd", {}).get("phred"),
)
result["multi_model_consensus"] = engine.to_dict(consensus)
```

### Frontend (`frontend/src/components/variant-analysis.tsx`)
Add after the ToolConcordance component:
```tsx
import { MultiModelConsensus, buildConsensusData } from "@/phase1_implementation/multi-model-consensus";

// In the render:
{analysisResult && (
  <MultiModelConsensus
    data={buildConsensusData(analysisResult)}
  />
)}
```

---

## ⚠️ Notes

- **AlphaMissense data**: ~2GB download. One-time cost. CC BY 4.0 license.
- **Benchmark cost**: ~$200 Modal GPU credits for 4,000 variants.
- **Benchmark time**: ~3-4 hours at 0.5s delay between variants.
- **Crash recovery**: Checkpoints saved every 100 variants. Resume automatically.
- **DuckDB**: Zero-config embedded database. No PostgreSQL/MySQL needed.
