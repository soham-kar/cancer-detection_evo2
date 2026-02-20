# Comparison: main.py vs modal_score_evo2.py

## Overview

**`backend/main.py`**: Production web service for clinical variant interpretation  
**`population_aware/modal_score_evo2.py`**: Research pipeline for population calibration

---

## Key Differences

| Feature | `main.py` (Production) | `modal_score_evo2.py` (Research) |
|---------|------------------------|----------------------------------|
| **Purpose** | Real-time clinical variant analysis API | Batch scoring for research dataset |
| **Users** | Doctors/clinicians via web UI | Data scientists/researchers |
| **Input** | Single variant requests (API) | 3,893 variants (CSV file) |
| **Output** | JSON response with clinical interpretation | Scored dataset (CSV) |
| **Dependencies** | Clinical enrichment, PubMed RAG, Redis | Evo2 model only |
| **Secrets** | Redis, Entrez, Groq API keys | None (open-source only) |
| **GPU** | H100 (50-80GB memory for caching) | H100 (for inference speed) |
| **Cache** | Redis (variant scores + PubMed results) | File-based (Modal Volume) |
| **Runtime** | Always-on FastAPI server | Run once, then terminate |
| **Cost** | ~$10-20/day (always running) | ~$2 one-time (30 min batch job) |

---

## Architecture Comparison

### `main.py` (Clinical API)

```
┌─────────────┐
│  Frontend   │  (Next.js web app)
└──────┬──────┘
       │ HTTP POST /api/variant
       v
┌─────────────────────────────────┐
│     main.py (FastAPI API)       │
│  ┌──────────────────────────┐   │
│  │ 1. Validate variant       │   │
│  │ 2. Fetch genome sequence  │   │  ◄── Redis cache
│  │ 3. Score with Evo2        │   │
│  │ 4. Enrich with ClinVar    │   │  ◄── PubMed API
│  │ 5. Generate LLM summary   │   │  ◄── Groq API
│  └──────────────────────────┘   │
└─────────────────────────────────┘
       │
       v
  JSON response to user
```

**Key components:**
- `StructuredLogger`: Production logging (DataDog-compatible)
- `VariantRequest`: Pydantic validation
- `Evo2Model`: Full clinical workflow
- `clinical_enrichment.py`: ClinVar + PubMed RAG
- FastAPI endpoints: `/predict`, `/batch`, `/status`

---

### `modal_score_evo2.py` (Research Pipeline)

```
┌────────────────────────────┐
│   day1_add_gnomad.py       │  (Run locally)
│   Output: brca1_gnomad.csv │
└──────────┬─────────────────┘
           │ Upload to Modal Volume
           v
┌────────────────────────────┐
│  modal_score_evo2.py       │  (Run on Modal H100)
│  ┌──────────────────────┐  │
│  │ 1. Load CSV          │  │
│  │ 2. Batch variants    │  │
│  │ 3. Score with Evo2   │  │  (No ClinVar, no PubMed)
│  │ 4. Save to CSV       │  │
│  └──────────────────────┘  │
└──────────┬─────────────────┘
           │ Download scored CSV
           v
┌────────────────────────────┐
│   day5_train_calibrator.py │  (Run locally)
└────────────────────────────┘
```

**Key components:**
- `score_variant_batch`: Simple Evo2 scoring only
- `score_all_variants`: Batch processing orchestrator
- Modal Volume: File storage (no Redis)
- No API, no web server, no clinical enrichment

---

## Code Comparison

### `main.py` - Clinical Variant Analysis
```python
# Full pipeline with clinical context
@app.function(gpu="H100", timeout=300, keep_warm=2)
class Evo2Model:
    def score_variant(self, variant: VariantRequest):
        # 1. Get genome sequence (with Redis cache)
        sequence = get_genome_sequence(...)
        
        # 2. Score with Evo2
        delta_score = self.model.score(ref) - self.model.score(alt)
        
        # 3. Clinical enrichment
        clinvar_data = fetch_clinvar_variants(gene_symbol)
        pubmed_refs = search_pubmed(variant)
        llm_summary = generate_clinical_summary(...)
        
        # 4. Return comprehensive report
        return {
            "evo2_score": delta_score,
            "clinvar_matches": clinvar_data,
            "references": pubmed_refs,
            "ai_summary": llm_summary,
            "clinical_significance": classify(delta_score)
        }
```

### `modal_score_evo2.py` - Research Scoring
```python
# Minimal scoring for research dataset
@app.function(gpu="H100", timeout=3600)
def score_variant_batch(variants_df):
    # Load model
    model = Evo2Model.from_pretrained("evo2_7b")
    
    # Score each variant (no clinical context)
    for variant in variants_df:
        ref_seq = get_sequence(variant)  # No cache
        alt_seq = mutate(ref_seq, variant)
        
        delta = model.score(alt_seq) - model.score(ref_seq)
        results.append({"evo2_delta": delta})
    
    return pd.DataFrame(results)
```

---

## When to Use Which?

### Use `main.py` when:
- ✅ Building a clinical decision support tool
- ✅ Need real-time variant interpretation
- ✅ Serving end-users (doctors, genetic counselors)
- ✅ Want comprehensive clinical context
- ✅ Need PubMed literature search
- ✅ Require production monitoring/logging

### Use `modal_score_evo2.py` when:
- ✅ Running research experiments
- ✅ Processing datasets offline
- ✅ Testing new calibration methods
- ✅ Need raw Evo2 scores without clinical enrichment
- ✅ Want minimal cost (batch processing)
- ✅ Publishing academic papers

---

## Summary

**`main.py`** is a **production clinical API** designed for real-time variant interpretation with full clinical context.

**`modal_score_evo2.py`** is a **research tool** for batch-scoring genomic datasets to test population-aware calibration methods.

They share the same underlying Evo2 model but serve completely different purposes:
- One is for **clinical care** (user-facing)
- One is for **research** (data science)

---

## Action Items

1. **Today:** Authenticate Modal (`modal token new`)
2. **Day 1 (finish):** Download chr17, run `day1_add_gnomad.py`
3. **Day 2:** Use `modal_score_evo2.py` to score variants
4. **Future:** Keep `main.py` for clinical API (separate from research)
