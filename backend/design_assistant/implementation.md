# HelixDesign — Multimodal Therapeutic Reasoning Assistant

> **A Nemotron-3 Ultra 550B + Evo2-7B powered assistant for variant-to-therapy reasoning**
>
> GitHub: [https://github.com/soham-kar/cancer-detection_evo2](https://github.com/soham-kar/cancer-detection_evo2)
>
> Built by Soham Kar · Last updated: 2026-06-28

---

## Table of Contents

1. [Research Gaps We Are Solving](#1-research-gaps-we-are-solving)
2. [How We Solve Each Gap](#2-how-we-solve-each-gap)
3. [System Architecture](#3-system-architecture)
4. [Core Modules — Detailed Implementation](#4-core-modules--detailed-implementation)
5. [Strategy Taxonomy & Routing Logic](#5-strategy-taxonomy--routing-logic)
6. [Evidence Graph Design](#6-evidence-graph-design)
7. [LLM Integration — Nemotron-3 Ultra 550B](#7-llm-integration--nemotron-3-ultra-550b)
8. [Frontend Integration](#8-frontend-integration)
9. [Proto-Tools Integration (Phase 2)](#9-proto-tools-integration-phase-2)
10. [Implementation Phases & Roadmap](#10-implementation-phases--roadmap)
11. [References & Papers](#11-references--papers)
12. [Vlogs, Talks & Tutorials](#12-vlogs-talks--tutorials)
13. [Code Resources & GitHub Links](#13-code-resources--github-links)
14. [API Shape & Contracts](#14-api-shape--contracts)
15. [Acceptance Criteria](#15-acceptance-criteria)
16. [Risks, Constraints & Future Work](#16-risks-constraints--future-work)

---

## 1. Research Gaps We Are Solving

### Gap 1: The Variant Interpretation-to-Action Gulf

**Problem:** Existing variant interpretation tools (CADD, REVEL, AlphaMissense, Evo2) produce pathogenicity scores but stop there. A clinician or researcher who receives a "VUS" (Variant of Uncertain Significance) classification has no systematic guidance on what to do next. There is no bridge from *"this variant is uncertain"* to *"here is the most promising therapeutic direction to investigate."*

**Scale of the problem:** Approximately 40-50% of all ClinVar submissions are VUS. For underrepresented populations, this fraction is even higher. Every VUS represents a potential missed therapeutic opportunity.

**Why existing tools fall short:**
- CADD (Kircher et al., 2014) → single numeric score, no therapeutic reasoning
- REVEL (Ioannidis et al., 2016) → ensemble score only for missense variants
- AlphaMissense (Cheng et al., 2023) → pathogenicity classification, no mechanism insight
- Evo2 (Nguyen et al., 2024) → evolutionary likelihood, no clinical synthesis
- ClinVar (Landrum et al., 2018) → crowd-sourced classifications, no reasoning layer

### Gap 2: Evidence Fragmentation Across Modalities

**Problem:** A single variant has evidence scattered across 8+ independent databases and tools:

| Source | What It Provides | Format | Access Method |
|--------|-----------------|--------|---------------|
| Evo2-7B | Evolutionary constraint score | Delta score (-∞ to +∞) | Modal H100 GPU |
| AlphaMissense | Structure-aware pathogenicity | Score 0-1 + classification | DuckDB local cache |
| CADD | Combined annotation score | Phred-scaled score | REST API (position endpoint) |
| ClinVar | Clinical consensus | Star-rated classification | NCBI E-utilities |
| gnomAD v4.1 | Population frequency | Allele frequency per population | GraphQL API |
| UniProtKB | Protein function & domains | Reviewed/Swiss-Prot entries | REST API |
| PubMed | Literature evidence | Article abstracts & PMIDs | Entrez E-utilities |
| VEP (Ensembl) | Molecular consequence | Transcript annotations | REST API |

**No existing system unifies all 8 into a single reasoning graph.** Each tool speaks a different "language" — scores, classifications, free text — and there is no common schema for cross-referencing them.

### Gap 3: The Therapeutic Reasoning Vacuum

**Problem:** Even when a variant is classified as pathogenic, there is no automated system that maps variant features to therapeutic strategies. The reasoning chain:

```
Variant → Molecular Mechanism → Druggable Target → Therapeutic Strategy
```

...is currently done entirely by human experts, one variant at a time. This doesn't scale to the millions of variants in ClinVar.

**Specific missing capabilities:**
- No automated mapping of missense-in-domain → structural rescue strategies
- No automated mapping of splice-site → ASO/splice-modulating therapy
- No automated mapping of surface-exposed missense → binder design
- No systematic way to rule out strategies based on variant features

### Gap 4: Black-Box AI Without Explainable Reasoning

**Problem:** Deep learning models (Evo2, AlphaMissense, Enformer) produce scores without explaining *why*. A delta score of -0.001 tells you nothing about which nucleotides matter, which protein domains are affected, or what the evolutionary context implies.

**What's missing:**
- Per-nucleotide importance (which positions drive the score?)
- Counterfactual analysis (what if it were a different base?)
- Domain-level impact (which protein function is disrupted?)
- Evidence provenance (which sources agree/disagree and why?)

### Gap 5: Proto's Vision Without an Implementation

**Problem:** The Proto paper (2025) laid out a compelling vision for composable, modular biological design programs — sequences, constraints, generators, and optimizers that can be combined like LEGO blocks. But Proto itself is a programming language and tool ecosystem, not a variant interpretation system. No one has applied Proto's design principles to the variant-to-therapy problem.

**What we take from Proto:**
- Modular composition of evidence sources
- Explicit reasoning graphs rather than black-box predictions
- Tool orchestration as a first-class concept
- Strategy selection as a constrained optimization problem

---

## 2. How We Solve Each Gap

### Solution 1: The Design Assistant Bridge

We built **HelixDesign**, a therapeutic reasoning layer that sits on top of the Evo2 variant analysis pipeline. It consumes a complete variant report (8+ evidence sources) and produces:

- **A recommended therapeutic strategy class** (from a taxonomy of 6)
- **Evidence-backed reasoning** with source attribution
- **Ranked next steps** for experimental validation
- **Confidence calibration** based on evidence quality
- **Limitations and caveats** for transparent uncertainty

**Key insight:** We don't replace any existing tool. We add a *reasoning layer* that consumes their outputs and synthesizes them.

### Solution 2: The Unified Evidence Graph

We built an `EvidenceGraph` data structure that normalizes all 8 evidence sources into a common schema:

```python
@dataclass
class EvidenceNode:
    source: str          # "Evo2", "AlphaMissense", "CADD", etc.
    signal: str          # "pathogenic", "benign", "uncertain", "missing"
    confidence: str      # "high", "medium", "low"
    detail: str          # Human-readable summary
    score: Optional[float]  # Normalized score when available
```

Each node is classified using source-specific thresholds:
- **Evo2**: delta_score < gene_threshold → pathogenic
- **AlphaMissense**: am_class = "pathogenic" / "likely_pathogenic" / "ambiguous" / "likely_benign" / "benign"
- **CADD**: phred ≥ 20 → pathogenic, 15-20 → uncertain, < 15 → benign
- **ClinVar**: "Pathogenic"/"Likely pathogenic" → pathogenic, "VUS" → uncertain, "Benign"/"Likely benign" → benign
- **gnomAD**: AF > 0.01 → benign (BA1), AF > 0.001 → uncertain, absent → uncertain
- **PubMed**: articles mentioning pathogenicity → pathogenic signal
- **UniProt**: domain annotation → contextual (not directly pathogenic/benign)
- **VEP**: frameshift/nonsense → pathogenic, missense → uncertain, synonymous → benign

### Solution 3: Strategy Taxonomy with Rule-Based + LLM Routing

We defined 6 therapeutic strategy classes and built a two-stage router:

**Stage 1 — Rule-Based (fast, deterministic, always available):**
Priority-ordered rules map variant features to strategies:
1. Splice-site/intronic → `splice_rescue`
2. Missense in known domain + ≥2 pathogenic signals → `structural_rescue`
3. Coding + ≥2 pathogenic signals → `allele_specific_targeting`
4. Missense + ≥1 pathogenic signal → `protein_binder_design`
5. Conflicting signals (P>0 and B>0) → `literature_and_evidence_review`
6. Default → `observe_and_reassess`

**Stage 2 — LLM Refinement (Nemotron-3 Ultra 550B):**
For medium/low confidence cases, Nemotron reviews the evidence graph and either confirms or refines the strategy. The LLM is constrained to use exact evidence counts (no hallucination) and returns structured JSON.

### Solution 4: Multi-Layer Explainability (XAI)

We built explainability at every layer:

| Layer | What It Explains | Implementation |
|-------|-----------------|----------------|
| ISM Scan | Which nucleotides drive the score | In-silico mutagenesis across ±N bp |
| Counterfactuals | What if it were A/C/G/T? | All 3 alternative alleles scored |
| ACMG Criteria | Which clinical criteria are met? | 10 criteria with strength levels |
| Domain Map | Which protein domain is affected? | SVG domain map with position mapping |
| Multi-Model Consensus | Do models agree? | Weighted voting with disagreement explanation |
| Evidence Confidence | How reliable is each source? | Per-source confidence with overall level |

### Solution 5: Proto-Inspired Modular Architecture

We adopted Proto's design philosophy without depending on Proto's runtime:

- **Modular composition**: Each evidence source is an independent node in the graph
- **Explicit reasoning**: The strategy selector is a transparent decision tree, not a black box
- **Tool orchestration**: Optional proto-tools calls enrich evidence when needed
- **Constrained optimization**: Strategy selection is framed as finding the best match given variant features

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER (Browser)                            │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              Saved Report Modal                             │  │
│  │  ┌─────────┐ ┌──────────┐ ┌────────────────────────────┐  │  │
│  │  │ Analysis│ │ Evidence │ │ Design Therapeutics [NEW]   │  │  │
│  │  │ Results │ │ Breakdown│ │ • Strategy Class            │  │  │
│  │  │         │ │          │ │ • Reasoning Summary         │  │  │
│  │  │         │ │          │ │ • Evidence Bullets          │  │  │
│  │  │         │ │          │ │ • Next Steps                │  │  │
│  │  │         │ │          │ │ • Limitations               │  │  │
│  │  └─────────┘ └──────────┘ └────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              │ HTTP POST                         │
│                              ▼                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │         /api/design-assistant (Next.js Route)              │  │
│  │  • Clerk auth • camelCase→snake_case • spawn Python       │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                               │ stdin JSON
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                   PYTHON BACKEND (design_assistant/)             │
│                                                                  │
│  ┌──────────┐    ┌──────────────────┐    ┌──────────────────┐  │
│  │ agent.py │───▶│ evidence_graph.py│───▶│strategy_selector │  │
│  │(orchestr)│    │ (8-source graph) │    │   (rule + LLM)   │  │
│  └──────────┘    └──────────────────┘    └────────┬─────────┘  │
│                                                   │             │
│                              ┌────────────────────┘             │
│                              ▼                                   │
│                    ┌──────────────────┐                          │
│                    │ nvidia_client.py │  Nemotron-3 Ultra 550B  │
│                    │ (OpenAI-compat)  │─────────────────────────│
│                    └──────────────────┘   integrate.api.nvidia  │
│                              │                                   │
│                              ▼                                   │
│                    ┌──────────────────┐                          │
│                    │response_builder  │                          │
│                    │ (structured JSON)│                          │
│                    └──────────────────┘                          │
└─────────────────────────────────────────────────────────────────┘
                               │ JSON response
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                   EXISTING EVO2 PIPELINE (unchanged)             │
│                                                                  │
│  Modal H100 GPU ←── main.py (Evo2-7B scoring)                   │
│  Modal CPU     ←── clinical_enrichment.py (gnomAD, ACMG, etc.)  │
│  Modal CPU     ←── pubmed_rag.py (HCR literature retrieval)     │
│  Modal CPU     ←── multimodal_rag.py (Llama-3-70B synthesis)    │
│  Groq Cloud    ←── Llama 3.3 70B (clinical summaries, ACMG)     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Core Modules — Detailed Implementation

### 4.1 `agent.py` — Main Orchestrator

**File:** `backend/design_assistant/agent.py` (~160 lines)

**Role:** Top-level entry point. Accepts a variant report dict, orchestrates the full pipeline, returns structured results.

```python
class DesignAssistant:
    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm

    def analyze(self, report: Dict[str, Any]) -> Dict[str, Any]:
        # Step 1: Build evidence graph from report
        graph = build_evidence_graph(report)

        # Step 2: Select strategy (rule-based)
        strategy = select_strategy(graph)

        # Step 3: Optionally refine with Nemotron
        if self.use_llm and strategy.confidence != "high":
            strategy = refine_with_llm(graph, strategy)

        # Step 4: Build structured response
        response = build_response(graph, strategy)

        return response
```

**Key design decisions:**
- LLM refinement is skipped for high-confidence cases (saves ~24s latency)
- The agent is stateless — each call is independent
- All evidence sources are optional; the graph gracefully handles missing data

### 4.2 `evidence_graph.py` — Evidence Graph Builder

**File:** `backend/design_assistant/evidence_graph.py` (~290 lines)

**Role:** Converts a raw variant report into a structured `EvidenceGraph` with classified nodes.

```python
@dataclass
class EvidenceNode:
    source: str
    signal: str       # pathogenic, benign, uncertain, missing
    confidence: str   # high, medium, low
    detail: str
    score: Optional[float] = None

class EvidenceGraph:
    nodes: List[EvidenceNode]
    consequence: str
    gene_symbol: str
    variant_id: str

    def pathogenic_count(self) -> int: ...
    def benign_count(self) -> int: ...
    def uncertain_count(self) -> int: ...
    def missing_count(self) -> int: ...
    def available_sources(self) -> int: ...
    def total_sources(self) -> int: ...
    def summary(self) -> str: ...
```

**Signal classification logic (per source):**

| Source | Pathogenic Signal | Benign Signal | Uncertain Signal |
|--------|------------------|---------------|------------------|
| Evo2 | delta < gene_threshold AND confidence ≥ 0.5 | delta > benign_threshold AND confidence ≥ 0.7 | Everything else |
| AlphaMissense | am_class in (pathogenic, likely_pathogenic) | am_class in (benign, likely_benign) | am_class = ambiguous |
| CADD | phred ≥ 20 | phred < 15 | 15 ≤ phred < 20 |
| ClinVar | Classification contains "Pathogenic" | Classification contains "Benign" | "Uncertain significance" or "VUS" |
| gnomAD | — (population data is contextual) | AF > 0.01 (BA1) | AF ≤ 0.01 or absent |
| ISM Scan | Mean |ΔS| > 0.1 across window | Mean |ΔS| < 0.01 | 0.01 ≤ mean |ΔS| ≤ 0.1 |
| PubMed | ≥2 articles mention pathogenicity | ≥2 articles suggest benign | <2 relevant articles |
| UniProt | — (structural context only) | — | Always contextual |

### 4.3 `strategy_selector.py` — Two-Stage Strategy Router

**File:** `backend/design_assistant/strategy_selector.py` (~250 lines)

**Stage 1 — Rule-Based Routing:**

```python
def select_strategy(graph: EvidenceGraph) -> StrategyResult:
    consequence = graph.consequence.lower()

    # Rule 1: Splice-related → splice_rescue
    if _is_splice_related(consequence):
        return StrategyResult(
            strategy_class="splice_rescue",
            confidence="high" if graph.pathogenic_count() >= 2 else "medium",
            reasoning="Splice-modulating therapies (ASOs, engineered U1 snRNA)..."
        )

    # Rule 2: Missense in domain + pathogenic → structural_rescue
    if _is_missense(consequence) and _has_domain(graph) and graph.pathogenic_count() >= 2:
        return StrategyResult(
            strategy_class="structural_rescue",
            confidence="high" if graph.pathogenic_count() >= 3 else "medium",
            reasoning="Structural rescue strategies (small molecule stabilizers...)"
        )

    # Rules 3-6: allele_specific_targeting, protein_binder_design,
    #           literature_review, observe_and_reassess
    ...
```

**Stage 2 — Nemotron LLM Refinement:**

```python
def refine_with_llm(graph, rule_result, client=None) -> StrategyResult:
    if rule_result.confidence == "high":
        return rule_result  # Skip LLM for clear cases

    system_prompt = """You are a clinical genomics expert...
    CRITICAL: Use ONLY the exact evidence counts provided.
    Return ONLY JSON: {"strategy_class": "...", "confidence": "...", "reasoning": "..."}"""

    user_prompt = f"""Evidence Graph Summary:
    {graph.summary}
    EXACT COUNTS: P={graph.pathogenic_count()}, B={graph.benign_count()},
    U={graph.uncertain_count()}, Missing={graph.missing_count()}
    Rule-based strategy: {rule_result.strategy_class} (confidence: {rule_result.confidence})
    Review and return JSON:"""

    result = client.reason_structured(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.1,  # Low temperature for consistency
        max_tokens=1024,
    )
    ...
```

### 4.4 `nvidia_client.py` — Nemotron-3 Ultra 550B Client

**File:** `backend/design_assistant/nvidia_client.py` (~230 lines)

**Role:** Thin wrapper around NVIDIA's integrate API. Uses OpenAI-compatible REST format.

```python
class NemotronClient:
    BASE_URL = "https://integrate.api.nvidia.com/v1"
    MODEL = "nvidia/nemotron-3-ultra-550b-a55b"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def reason(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        """General reasoning call. Returns model's text response."""
        ...

    def reason_structured(self, system_prompt: str, user_prompt: str, **kwargs) -> dict:
        """Reasoning call that parses JSON from the response."""
        ...

    def answer(self, question: str, context: str, **kwargs) -> str:
        """Q&A mode for follow-up questions."""
        ...
```

**Key features:**
- 1M token context window (enough for full reports + conversation history)
- ~24s latency for 600-token responses
- Returns both `reasoning_content` (chain-of-thought) and `content` (final answer)
- Temperature set to 0.1 for clinical consistency
- Fallback: Groq Llama 3.3 70B for cost-sensitive or latency-sensitive paths

### 4.5 `response_builder.py` — Structured Output Formatter

**File:** `backend/design_assistant/response_builder.py` (~210 lines)

**Role:** Converts the evidence graph + strategy result into the final JSON response.

```python
def build_response(graph: EvidenceGraph, strategy: StrategyResult) -> dict:
    return {
        "strategy_class": strategy.strategy_class,
        "confidence": strategy.confidence,
        "reasoning_summary": strategy.reasoning,
        "evidence_bullets": _build_evidence_bullets(graph),
        "recommended_next_steps": _build_next_steps(strategy),
        "tool_calls": [],  # Populated in Phase 2
        "limitations": _build_limitations(graph),
        "generated_at": datetime.utcnow().isoformat(),
    }
```

### 4.6 `run_api.py` — CLI Entry Point

**File:** `backend/design_assistant/run_api.py` (~50 lines)

**Role:** stdin/stdout bridge between Next.js and Python. Reads JSON from stdin, runs the assistant, prints JSON to stdout.

```python
import sys, json
from design_assistant.agent import DesignAssistant

if __name__ == "__main__":
    report = json.loads(sys.stdin.read())
    assistant = DesignAssistant(use_llm=True)
    result = assistant.analyze(report)
    print(json.dumps(result))
```

---

## 5. Strategy Taxonomy & Routing Logic

### 5.1 The Six Strategy Classes

| # | Strategy Class | When Applied | Therapeutic Approach | Example |
|---|---------------|-------------|---------------------|---------|
| 1 | `observe_and_reassess` | Insufficient evidence (<2 sources, all uncertain) | Wait for new evidence; re-analyze in 6-12 months | BRCA1 VUS with only Evo2 uncertain |
| 2 | `structural_rescue` | Missense in known protein domain + ≥2 pathogenic signals | Small molecule stabilizers, stapled peptides, protein engineering | TP53 R273H in DNA-binding domain |
| 3 | `splice_rescue` | Splice-site, intronic, or synonymous near exon boundary | ASOs, engineered U1 snRNA, small molecule splice modulators | BRCA1 c.5152+1G>A |
| 4 | `allele_specific_targeting` | High-confidence pathogenic coding variant | CRISPR, ASO, RNAi — selectively silence mutant allele | KRAS G12D |
| 5 | `protein_binder_design` | Surface-exposed missense with ≥1 pathogenic signal | Monobodies, DARPins, antibodies targeting mutant epitope | PIK3CA H1047R |
| 6 | `literature_and_evidence_review` | Conflicting signals (P>0 AND B>0) | Deep literature review, functional validation, expert panel | BRCA1 missense with split model predictions |

### 5.2 Routing Decision Tree

```
Variant Consequence
│
├─ Splice-site / intronic / synonymous@boundary?
│   └─ YES → splice_rescue
│
├─ Missense?
│   ├─ In known domain + ≥2 pathogenic?
│   │   └─ YES → structural_rescue
│   ├─ ≥1 pathogenic?
│   │   └─ YES → protein_binder_design
│   └─ Conflicting (P>0 & B>0)?
│       └─ YES → literature_and_evidence_review
│
├─ Coding (non-missense) + ≥2 pathogenic?
│   └─ YES → allele_specific_targeting
│
├─ Conflicting signals?
│   └─ YES → literature_and_evidence_review
│
└─ Default → observe_and_reassess
```

---

## 6. Evidence Graph Design

### 6.1 Node Schema

```python
@dataclass
class EvidenceNode:
    source: str          # "Evo2", "AlphaMissense", "CADD", "ClinVar",
                         # "gnomAD", "ISM_Scan", "PubMed", "UniProt"
    signal: str          # "pathogenic" | "benign" | "uncertain" | "missing"
    confidence: str      # "high" | "medium" | "low"
    detail: str          # Human-readable evidence summary
    score: Optional[float] = None  # Normalized score when applicable
```

### 6.2 Graph Summary Format

The graph produces a machine-readable summary used in LLM prompts:

```
Evidence Graph: BRCA1 A>C (chr17:43094169)
  Evo2: uncertain (delta=-0.0014, conf=36%)
  AlphaMissense: benign (score=0.27, likely_benign)
  CADD: uncertain (phred=11.8)
  ClinVar: uncertain (VUS, 0 submitters)
  gnomAD: uncertain (absent from database)
  ISM_Scan: missing
  PubMed: uncertain (4 articles, no specific variant data)
  UniProt: contextual (BRCA1, 2 domains: RING, BRCT)
Summary: P=0, B=1, U=6, Missing=1 | Available: 7/8
```

### 6.3 Edge Cases Handled

- **Missing sources**: Marked as `signal="missing"`, don't count toward pathogenic/benign totals
- **ClinVar "Uncertain significance"**: Correctly classified as uncertain (not missing)
- **gnomAD "absent"**: Classified as uncertain (absence of evidence ≠ evidence of absence)
- **AlphaMissense null for non-missense**: Marked as missing with note
- **CADD API failure**: Marked as missing with error detail

---

## 7. LLM Integration — Nemotron-3 Ultra 550B

### 7.1 Why Nemotron-3 Ultra 550B?

| Criterion | Nemotron-3 Ultra 550B | GPT-4o | Claude 3.5 Sonnet | Groq Llama 3.3 70B |
|-----------|----------------------|--------|-------------------|---------------------|
| Context window | 1M tokens | 128K | 200K | 128K |
| Reasoning quality | Excellent (hybrid Mamba-Transformer MoE) | Excellent | Excellent | Good |
| Latency (600 tokens) | ~24s | ~8s | ~10s | ~2s |
| Cost per 1M tokens | Free (NVIDIA research) | $5-15 | $3-15 | $0.59-0.79 |
| Tool calling | Yes | Yes | Yes | Limited |
| Medical/clinical safety | Good | Excellent | Excellent | Moderate |

**Decision:** Nemotron-3 Ultra 550B is the primary model because:
1. 1M context window fits full reports + conversation history
2. Free via NVIDIA research access
3. Hybrid Mamba-Transformer architecture provides strong reasoning
4. ~24s latency is acceptable for an assistant (not real-time)

**Fallback:** Groq Llama 3.3 70B for latency-sensitive paths (~2s)

### 7.2 Prompt Engineering Strategy

**Anti-hallucination measures:**
1. Exact evidence counts are injected into the prompt (P=0, B=1, U=6)
2. System prompt explicitly forbids inventing numbers
3. Temperature set to 0.1 for deterministic outputs
4. Structured JSON output format constrains the response
5. Rule-based result is always provided as a baseline

**Example system prompt:**
```
You are a clinical genomics expert reviewing therapeutic strategies.
CRITICAL RULES:
- Use ONLY the exact evidence counts provided. Do NOT invent or change numbers.
- If the evidence says P=0, B=1, U=6, write exactly "0 pathogenic, 1 benign, 6 uncertain".
- Never write "3 benign and 3 uncertain" unless the evidence literally says B=3, U=3.
- Be precise about which sources are available vs missing.
Return ONLY a JSON object.
```

### 7.3 Verified Test Results (2026-06-28)

**Test variant:** BRCA1 A>C (chr17:43094169), missense S454R

**Input evidence:** P=0, B=1 (AlphaMissense), U=6, Missing=1 (ISM)

**Nemotron output:**
```json
{
  "strategy_class": "observe_and_reassess",
  "confidence": "medium",
  "reasoning": "Evidence shows 0 pathogenic, 1 benign, 6 uncertain across 7/8 available sources. The single benign signal from AlphaMissense is contradicted by uncertain signals from Evo2, CADD, and ClinVar. Without stronger pathogenic evidence, therapeutic intervention is premature. Recommend re-analysis in 6-12 months."
}
```

**Assessment:** Correctly identified the evidence pattern, didn't hallucinate counts, chose the appropriate conservative strategy.

---

## 8. Frontend Integration

### 8.1 API Route: `/api/design-assistant`

**File:** `frontend/src/app/api/design-assistant/route.ts` (~230 lines)

**Flow:**
1. Clerk authentication check
2. Parse request body → extract `report` object
3. Normalize camelCase → snake_case field names
4. Reconstruct missing structured fields (clinvar_evidence, protein_context, multi_model_consensus)
5. Spawn Python child process: `python run_api.py`
6. Pipe normalized report via stdin
7. Collect stdout → parse JSON → return to frontend

**Key normalization mappings:**
```typescript
const fieldMap = {
  geneSymbol: "gene_symbol",
  deltaScore: "delta_score",
  classificationConfidence: "classification_confidence",
  externalScores: "external_scores",
  clinvarClassification: "clinvar_classification",
  // ... 20+ mappings
};
```

### 8.2 UI Component: `DesignTherapeutics`

**File:** `frontend/src/components/design-therapeutics.tsx` (~400 lines)

**Features:**
- DNA/helix iconography in the tab header
- Strategy class displayed with color-coded badge
- Confidence indicator (high/medium/low)
- Formatted reasoning with highlighted key terms
- Evidence bullets with source attribution
- Recommended next steps as numbered list
- Limitations section
- Loading state with animated spinner
- Error state with retry button

---

## 9. Proto-Tools Integration (Phase 2)

### 9.1 Tools to Integrate

| Tool | Purpose | When to Call | Expected Latency |
|------|---------|-------------|-----------------|
| **UniProt Fetch** | Protein function, domains, PTMs | Always (fills "Domains: none" gap) | ~1s |
| **AlphaFold DB Fetch** | Structure metadata, pLDDT confidence | structural_rescue, protein_binder_design | ~2s |
| **ESMFold2** | Structure prediction (no experimental structure) | structural_rescue when no PDB/AF exists | ~60s |
| **SpliceAI** | Splice site prediction (delta scores) | splice_rescue, intronic variants | ~5s |
| **Pangolin** | Splice site prediction (alternative to SpliceAI) | splice_rescue fallback | ~5s |
| **pDockQ2** | Protein-protein interface quality | protein_binder_design | ~10s |
| **IPSAE** | Interface prediction accuracy | protein_binder_design | ~10s |
| **USalign/TMalign** | Structural comparison (WT vs mutant) | structural_rescue | ~30s |

### 9.2 Tool Orchestrator Design

```python
class ToolOrchestrator:
    def __init__(self, tool_budget: int = 3):
        self.tool_budget = tool_budget
        self.results = []

    def enrich(self, graph: EvidenceGraph, strategy: StrategyResult) -> List[ToolResult]:
        needed = self._determine_needed_tools(graph, strategy)
        # Sort by relevance, cap at budget
        selected = needed[:self.tool_budget]
        # Execute in parallel where possible
        for tool in selected:
            result = self._execute_tool(tool, graph)
            self.results.append(result)
        return self.results
```

### 9.3 Proto-Tools Client

```python
class ProtoToolsClient:
    """Thin wrapper around proto-tools for evidence enrichment."""

    def fetch_uniprot(self, accession: str) -> dict: ...
    def fetch_alphafold(self, uniprot_id: str) -> dict: ...
    def run_esmfold(self, sequence: str) -> dict: ...
    def run_spliceai(self, chrom: str, pos: int, ref: str, alt: str) -> dict: ...
    def run_pdockq2(self, structure_path: str) -> dict: ...
```

---

## 10. Implementation Phases & Roadmap

### Phase 1: Reasoning Skeleton ✅ COMPLETED (2026-06-28)

| Deliverable | Status | File |
|------------|--------|------|
| Input normalization (camelCase→snake_case) | ✅ | `route.ts` |
| Evidence graph builder (8 sources) | ✅ | `evidence_graph.py` |
| Strategy selector (rule-based + LLM) | ✅ | `strategy_selector.py` |
| Nemotron-3 Ultra 550B client | ✅ | `nvidia_client.py` |
| Response builder | ✅ | `response_builder.py` |
| CLI entry point (stdin/stdout) | ✅ | `run_api.py` |
| Next.js API route | ✅ | `route.ts` |
| Design Therapeutics UI panel | ✅ | `design-therapeutics.tsx` |
| Saved report modal integration | ✅ | `saved-report-modal.tsx` |
| Field normalization & reconstruction | ✅ | `route.ts` |
| Evidence graph signal classification fix | ✅ | `evidence_graph.py` |
| LLM anti-hallucination prompts | ✅ | `strategy_selector.py` |
| Debug log removal & cleanup | ✅ | `route.ts` |

### Phase 2: Proto-Tools Integration 🔜 NEXT

| Deliverable | Estimated Effort | Dependencies |
|------------|-----------------|--------------|
| `tool_orchestrator.py` | 2-3 hours | None |
| `proto_tools_client.py` | 3-4 hours | proto-tools installed |
| UniProt Fetch integration | 1 hour | proto_tools_client |
| AlphaFold DB Fetch integration | 1 hour | proto_tools_client |
| SpliceAI/Pangolin integration | 2 hours | proto_tools_client |
| ESMFold2 integration | 2 hours | proto_tools_client |
| Evidence graph v2 (tool nodes) | 2 hours | tool_orchestrator |
| Response builder v2 (tool outputs) | 1 hour | tool_orchestrator |

### Phase 3: Richer Strategy Output

| Deliverable | Estimated Effort |
|------------|-----------------|
| Strategy differential (why rejected) | 2 hours |
| Confidence breakdown component | 2 hours |
| Per-source confidence visualization | 2 hours |
| Strategy comparison table | 2 hours |

### Phase 4: Follow-up Q&A

| Deliverable | Estimated Effort |
|------------|-----------------|
| Chat input in Design Therapeutics panel | 3 hours |
| Conversation context management | 3 hours |
| Nemotron Q&A mode integration | 2 hours |
| Session persistence | 2 hours |

### Phase 5: Demo Preparation

| Deliverable | Estimated Effort |
|------------|-----------------|
| 4 curated example variants | 2 hours |
| Loading/error state polish | 2 hours |
| Responsive layout fixes | 2 hours |
| Copy-to-clipboard | 1 hour |
| PDF export integration | 2 hours |

---

## 11. References & Papers

### Core Models & Tools

| Paper | Authors | Year | Key Contribution | Link |
|-------|---------|------|-----------------|------|
| **Evo2: Sequence modeling and design across the tree of life** | Nguyen et al. | 2024 | 7B-parameter genomic language model; zero-shot pathogenicity prediction | [bioRxiv](https://www.biorxiv.org/content/10.1101/2024.02.20.581202) |
| **Accurate proteome-wide missense variant effect prediction with AlphaMissense** | Cheng et al. | 2023 | AlphaFold-derived pathogenicity scores for all human missense variants | [Science](https://www.science.org/doi/10.1126/science.adg7492) |
| **A general framework for estimating the relative pathogenicity of human genetic variants** | Kircher et al. | 2014 | CADD: Combined Annotation Dependent Depletion | [Nature Genetics](https://www.nature.com/articles/ng.2892) |
| **REVEL: An Ensemble Method for Predicting the Pathogenicity of Rare Missense Variants** | Ioannidis et al. | 2016 | Ensemble score combining 13 individual tools | [AJHG](https://www.cell.com/ajhg/fulltext/S0002-9297(16)30370-6) |
| **Highly accurate protein structure prediction with AlphaFold** | Jumper et al. | 2021 | Revolutionary protein structure prediction | [Nature](https://www.nature.com/articles/s41586-021-03819-2) |
| **Evolutionary-scale prediction of atomic-level protein structure with a language model** | Lin et al. | 2023 | ESMFold: structure prediction from sequence alone | [Science](https://www.science.org/doi/10.1126/science.ade2574) |
| **Effective gene expression prediction from sequence by integrating long-range interactions** | Avsec et al. | 2021 | Enformer: sequence-to-expression model | [Nature Methods](https://www.nature.com/articles/s41592-021-01252-x) |

### Clinical & Population Genomics

| Paper | Authors | Year | Key Contribution | Link |
|-------|---------|------|-----------------|------|
| **Standards and guidelines for the interpretation of sequence variants** | Richards et al. | 2015 | ACMG/AMP variant classification guidelines | [Genetics in Medicine](https://www.nature.com/articles/gim201530) |
| **The mutational constraint spectrum quantified from variation in 141,456 humans** | Karczewski et al. | 2020 | gnomAD v2/v3: population frequency database | [Nature](https://www.nature.com/articles/s41586-020-2308-7) |
| **ClinVar: improving access to variant interpretations and supporting evidence** | Landrum et al. | 2018 | ClinVar: public archive of variant-clinical relationships | [Nucleic Acids Research](https://academic.oup.com/nar/article/46/D1/D1062/4584623) |
| **The Ensembl Variant Effect Predictor** | McLaren et al. | 2016 | VEP: comprehensive variant annotation | [Genome Biology](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-016-0974-4) |

### Splicing & Therapeutic Design

| Paper | Authors | Year | Key Contribution | Link |
|-------|---------|------|-----------------|------|
| **Predicting Splicing from Primary Sequence with Deep Learning** | Jaganathan et al. | 2019 | SpliceAI: deep learning splice site prediction | [Cell](https://www.cell.com/cell/fulltext/S0092-8674(18)31635-0) |
| **Pangolin: predicting tissue-specific splicing from sequence** | Zeng & Li | 2022 | Pangolin: tissue-aware splice prediction | [Genome Biology](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-022-02679-3) |
| **Antisense oligonucleotides: rising stars in the clinic** | Bennett et al. | 2021 | ASO therapeutic mechanisms and clinical applications | [Nature Reviews Drug Discovery](https://www.nature.com/articles/s41573-021-00157-2) |

### LLM & AI Reasoning

| Paper | Authors | Year | Key Contribution | Link |
|-------|---------|------|-----------------|------|
| **Nemotron-3 Ultra: A 550B Parameter Hybrid Mamba-Transformer Model** | NVIDIA | 2025 | 1M context, agentic reasoning, tool calling | [NVIDIA Blog](https://developer.nvidia.com/blog/nemotron-3-ultra-550b/) |
| **Llama 3: Open and Efficient Foundation Language Models** | Meta AI | 2024 | Llama 3.3 70B used for clinical summaries | [Meta AI](https://ai.meta.com/blog/meta-llama-3/) |
| **Proto: A Programming Language for Generative Biology** | Proto Team | 2025 | Composable biological design programs | [Proto](https://www.proto.bio/) |

---

## 12. Vlogs, Talks & Tutorials

### Evo2 & Genomic AI

| Title | Speaker/Channel | Year | Topic | Link |
|-------|----------------|------|-------|------|
| Evo2: Foundation Model for Biology | Brian Hie (Arc Institute) | 2024 | Evo2 architecture, training, and applications | [YouTube](https://www.youtube.com/results?search_query=evo2+foundation+model+biology+brian+hie) |
| Arc Institute Evo2 Launch | Arc Institute | 2024 | Official Evo2 release presentation | [Arc Institute](https://arcinstitute.org/news/blog/evo2) |
| NVIDIA Nemotron-3 Ultra Launch | NVIDIA | 2025 | 550B model capabilities and API access | [NVIDIA Developer](https://developer.nvidia.com/nemotron) |

### Variant Interpretation

| Title | Speaker/Channel | Year | Topic | Link |
|-------|----------------|------|-------|------|
| ACMG Variant Interpretation Guidelines Explained | ClinGen | 2020 | Step-by-step ACMG criteria application | [YouTube](https://www.youtube.com/results?search_query=acmg+variant+interpretation+guidelines+explained) |
| gnomAD v4: What's New | gnomAD Team | 2024 | gnomAD v4.1 features and API | [gnomAD Blog](https://gnomad.broadinstitute.org/news/) |

### Protein Design & Structure

| Title | Speaker/Channel | Year | Topic | Link |
|-------|----------------|------|-------|------|
| AlphaFold: The Revolution in Protein Structure | DeepMind | 2021 | AlphaFold architecture and impact | [YouTube](https://www.youtube.com/results?search_query=alphafold+revolution+protein+structure) |
| ESMFold: Protein Folding with Language Models | Meta AI | 2023 | ESMFold architecture and API | [Meta AI Blog](https://ai.meta.com/blog/esmfold-protein-folding/) |

### Proto & Generative Biology

| Title | Speaker/Channel | Year | Topic | Link |
|-------|----------------|------|-------|------|
| Proto: Getting Started | Proto Team | 2025 | Proto tutorial and language overview | [Proto Docs](https://docs.proto.bio/) |
| Generative Biology with Proto | Proto Team | 2025 | Design patterns for biological programs | [Proto Blog](https://www.proto.bio/blog) |

---

## 13. Code Resources & GitHub Links

### Our Repository

| Resource | Link |
|----------|------|
| **Main Repository** | [https://github.com/soham-kar/cancer-detection_evo2](https://github.com/soham-kar/cancer-detection_evo2) |
| **Design Assistant Module** | `backend/design_assistant/` |
| **Frontend API Route** | `frontend/src/app/api/design-assistant/route.ts` |
| **Design Therapeutics Panel** | `frontend/src/components/design-therapeutics.tsx` |
| **Implementation Plan** | `backend/design_assistant/implementation.md` |

### External Repositories

| Repository | Description | Link |
|-----------|-------------|------|
| **Evo2 (Arc Institute)** | Official Evo2 model and inference code | [https://github.com/ArcInstitute/evo2](https://github.com/ArcInstitute/evo2) |
| **Vortex (Zymrael)** | Evo2 model implementation used as submodule | [https://github.com/Zymrael/vortex](https://github.com/Zymrael/vortex) |
| **AlphaMissense** | Official AlphaMissense inference code | [https://github.com/google-deepmind/alphamissense](https://github.com/google-deepmind/alphamissense) |
| **CADD** | CADD scoring scripts and precomputed scores | [https://github.com/kircherlab/CADD-scripts](https://github.com/kircherlab/CADD-scripts) |
| **SpliceAI** | Deep learning splice site prediction | [https://github.com/Illumina/SpliceAI](https://github.com/Illumina/SpliceAI) |
| **ESMFold** | Meta's protein folding via language models | [https://github.com/facebookresearch/esm](https://github.com/facebookresearch/esm) |
| **gnomAD API** | gnomAD GraphQL API documentation | [https://gnomad.broadinstitute.org/api](https://gnomad.broadinstitute.org/api) |
| **ClinVar API** | NCBI ClinVar E-utilities | [https://www.ncbi.nlm.nih.gov/clinvar/](https://www.ncbi.nlm.nih.gov/clinvar/) |
| **Ensembl VEP** | Variant Effect Predictor REST API | [https://rest.ensembl.org/](https://rest.ensembl.org/) |
| **NVIDIA NIM API** | Nemotron-3 Ultra 550B API | [https://build.nvidia.com/nvidia/nemotron-3-ultra-550b-a55b](https://build.nvidia.com/nvidia/nemotron-3-ultra-550b-a55b) |
| **Groq API** | Llama 3.3 70B via Groq | [https://console.groq.com/](https://console.groq.com/) |
| **Proto** | Generative biology programming language | [https://github.com/proto-bio/proto](https://github.com/proto-bio/proto) |

---

## 14. API Shape & Contracts

### Request

```
POST /api/design-assistant
Content-Type: application/json
Authorization: Bearer <clerk_session_token>

{
  "report": {
    "geneSymbol": "BRCA1",
    "position": 43094169,
    "reference": "A",
    "alternative": "C",
    "prediction": "Uncertain significance",
    "deltaScore": -0.00144,
    "classificationConfidence": 0.36,
    "externalScores": {
      "cadd": { "phred": 11.8, "interpretation": "Uncertain" },
      "alphamissense": {
        "score": 0.2676,
        "classification": "Likely Benign",
        "am_class": "likely_benign"
      }
    },
    "clinvarClassification": "Uncertain significance",
    "evidenceConfidence": { ... },
    "clinicalSummary": "...",
    "acmgCriteria": { ... },
    "multiModelConsensus": { ... }
  }
}
```

### Response

```json
{
  "strategy_class": "observe_and_reassess",
  "confidence": "medium",
  "reasoning_summary": "Evidence shows 0 pathogenic, 1 benign, 6 uncertain across 7/8 available sources...",
  "evidence_bullets": [
    {
      "source": "AlphaMissense",
      "finding": "Likely Benign (score: 0.27)",
      "implication": "Suggests variant may not disrupt protein structure"
    },
    {
      "source": "Evo2-7B",
      "finding": "Uncertain significance (delta: -0.0014)",
      "implication": "Evolutionary constraint is weak at this position"
    }
  ],
  "recommended_next_steps": [
    {
      "priority": 1,
      "action": "Re-analyze in 6-12 months as new ClinVar submissions accumulate",
      "rationale": "Currently 0 submitters; additional clinical evidence may clarify classification"
    },
    {
      "priority": 2,
      "action": "Functional validation via saturation mutagenesis",
      "rationale": "ISM scan would reveal positional constraint across the domain"
    }
  ],
  "tool_calls": [],
  "limitations": [
    "No ISM scan data available for this variant",
    "REVEL scores unavailable (no public REST API)",
    "Only 4 PubMed articles found; none specific to this variant"
  ],
  "generated_at": "2026-06-28T13:05:00Z"
}
```

---

## 15. Acceptance Criteria

### v1 (Current — Phase 1 Complete)

- [x] Read an existing variant report from the database
- [x] Build an evidence graph from 8 data sources
- [x] Return one clear therapeutic strategy class
- [x] Provide a short, evidence-based explanation
- [x] Show at least one useful follow-up step
- [x] Preserve the current Evo2 analysis pipeline unchanged
- [x] Handle missing evidence sources gracefully
- [x] Anti-hallucination measures in LLM prompts
- [x] Frontend panel integrated into saved report modal
- [x] Clean git history with professional commit messages

### v2 (Phase 2 Target)

- [ ] At least 3 proto-tools integrated (UniProt, AlphaFold DB, SpliceAI)
- [ ] Tool outputs enrich evidence graph nodes
- [ ] Tool failures don't break the assistant
- [ ] "Domains: none" issue resolved via UniProt Fetch

### v3 (Phase 3-4 Target)

- [ ] Strategy differential shows why alternatives were rejected
- [ ] Follow-up Q&A works with conversation memory
- [ ] Confidence breakdown by evidence quality, model agreement, tool availability

### Demo-Ready (Phase 5 Target)

- [ ] 4 curated example variants showing different strategies
- [ ] All loading/error states polished
- [ ] PDF export includes Design Therapeutics section
- [ ] <5s time-to-first-meaningful-output for demo path

---

## 16. Risks, Constraints & Future Work

### Current Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Nemotron API rate limits or deprecation | Medium | Fallback to Groq Llama 3.3 70B; cache common responses |
| Proto-tools installation complexity | Medium | Wrap in Docker; provide setup script |
| LLM hallucination of evidence counts | Low | Exact counts injected in prompt; temperature=0.1; structured JSON output |
| Slow ESMFold2 (~60s) for live demo | High | Pre-compute for demo variants; show cached results |
| CRISPR off-target reliability | High | Deferred to future work; focus on non-CRISPR strategies first |

### Future Work

1. **CRISPR Design Module**: Once off-target prediction is reliable, add guide RNA design for allele-specific targeting
2. **Clinical Trial Matching**: Connect strategy recommendations to active clinical trials via ClinicalTrials.gov API
3. **Drug Repurposing**: Map affected domains/pathways to existing FDA-approved drugs
4. **Multi-Variant Analysis**: Analyze all variants in a gene simultaneously for combinatorial effects
5. **Population-Specific Calibration**: Adjust thresholds based on ancestry-specific gnomAD frequencies
6. **Real-time Collaboration**: Allow multiple researchers to discuss and annotate assistant outputs
7. **Automated Report Generation**: Produce publication-ready variant interpretation reports

---

## Appendix A: Environment Setup

### Prerequisites

```bash
# Python 3.10+
python --version

# Node.js 18+
node --version

# Modal CLI
pip install modal
modal token new

# NVIDIA API key (free research access)
# Get from: https://build.nvidia.com/nvidia/nemotron-3-ultra-550b-a55b
# Store in: frontend/.env.local as NVIDIA_API_KEY=...

# Groq API key (for clinical summaries)
# Get from: https://console.groq.com/
# Store in: backend/.env as GROQ_API_KEY=...
```

### Running the Assistant

```bash
# Start the Next.js dev server
cd frontend
npm run dev

# The assistant is called via the API route:
# POST http://localhost:3000/api/design-assistant
# Body: { "report": { ... } }

# Or test the Python backend directly:
cd backend
echo '{"gene_symbol":"BRCA1","position":43094169,...}' | python design_assistant/run_api.py
```

---

## Appendix B: Glossary

| Term | Definition |
|------|-----------|
| **VUS** | Variant of Uncertain Significance — a variant with insufficient evidence for pathogenic or benign classification |
| **ACMG** | American College of Medical Genetics and Genomics — publisher of variant interpretation guidelines |
| **Delta Score** | Evo2's metric: log-likelihood ratio between variant and reference sequences |
| **Phred Score** | CADD's scaled score: -10 × log₁₀(rank) |
| **ASO** | Antisense Oligonucleotide — short synthetic DNA/RNA that modulates splicing or gene expression |
| **ISM** | In-Silico Mutagenesis — systematically mutating each position and measuring score change |
| **pLDDT** | Predicted Local Distance Difference Test — AlphaFold's per-residue confidence metric |
| **HCR** | Hierarchical Cascading Retrieval — our two-level PubMed search architecture |
| **MoE** | Mixture of Experts — neural network architecture where different "experts" handle different inputs |

---

> **Document maintained by Soham Kar**
> GitHub: [https://github.com/soham-kar/cancer-detection_evo2](https://github.com/soham-kar/cancer-detection_evo2)
> Last updated: 2026-06-28
