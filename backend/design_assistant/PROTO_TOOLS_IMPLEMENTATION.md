# Proto-Tools Integration — Detailed Implementation Plan

> **Status:** Planning · **Phase:** Phase 13 — Tool Calling via Modal
>
> **Goal:** Deploy proto-tools on Modal.com and wire them into the HelixMind chatbot
> so Nemotron-3 Ultra 550B can autonomously call bioinformatics tools during
> conversations to enrich variant analysis with real computational evidence.
>
> **Approach:** Option B — Self-hosted proto-tools on Modal (CPU + GPU)
>
> **Created:** 2026-07-05

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Tool Catalog — What We Deploy](#2-tool-catalog--what-we-deploy)
3. [When the Chatbot Calls Which Tool](#3-when-the-chatbot-calls-which-tool)
4. [Modal Deployment Plan](#4-modal-deployment-plan)
5. [Tool-Calling Loop — How It Works](#5-tool-calling-loop--how-it-works)
6. [Nemotron Function Definitions](#6-nemotron-function-definitions)
7. [SSE Event Protocol for Tool Calls](#7-sse-event-protocol-for-tool-calls)
8. [Frontend UI — Tool Call Display](#8-frontend-ui--tool-call-display)
9. [Implementation Steps](#9-implementation-steps)
10. [File Structure](#10-file-structure)
11. [Cost Analysis](#11-cost-analysis)
12. [Testing Plan](#12-testing-plan)
13. [Acceptance Criteria](#13-acceptance-criteria)

---

## 1. Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         USER (Browser)                                    │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │  Chat Side Panel                                                   │  │
│  │                                                                    │  │
│  │  ┌──────────────────────────────────────────────────────────────┐  │  │
│  │  │ 🧠 Chain of Thought (always visible)                         │  │  │
│  │  │ "Let me check if this variant affects splicing..."           │  │  │
│  │  └──────────────────────────────────────────────────────────────┘  │  │
│  │  ┌──────────────────────────────────────────────────────────────┐  │  │
│  │  │ 🔧 Tool Call: SpliceAI Score                                 │  │  │
│  │  │    Status: ✅ Completed (12.3s)                              │  │  │
│  │  │    Result: Δ score 0.82 — cryptic donor site created         │  │  │
│  │  └──────────────────────────────────────────────────────────────┘  │  │
│  │  ┌──────────────────────────────────────────────────────────────┐  │  │
│  │  │ 💬 Answer                                                    │  │  │
│  │  │ "Yes, SpliceAI predicts this variant creates a cryptic       │  │  │
│  │  │  donor site with Δ score 0.82 (threshold >0.2 is high        │  │  │
│  │  │  confidence). This strongly suggests splice disruption..."   │  │  │
│  │  └──────────────────────────────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────┘
                    │ SSE stream (text/event-stream)
                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                    NEXT.JS API ROUTE (/api/chat)                          │
│                                                                          │
│  1. Auth check (Clerk)                                                   │
│  2. Load report from database (if report mode)                           │
│  3. Build system prompt + tool definitions                               │
│  4. Call NVIDIA Nemotron API with stream=true + tools=[...]              │
│  5. Parse SSE chunks from NVIDIA                                         │
│  6. When Nemotron returns tool_call → execute tool                       │
│  7. Feed tool result back to Nemotron                                    │
│  8. Stream reasoning_delta, tool_call, content_delta to frontend         │
│  9. Persist to database                                                  │
│                                                                          │
│  Tool execution:                                                         │
│  ├── CPU tools → fetch(MODAL_CPU_ENDPOINT)                              │
│  └── GPU tools → fetch(MODAL_GPU_ENDPOINT)                              │
└──────────────────────────────────────────────────────────────────────────┘
                    │
          ┌─────────┴──────────┐
          ▼                    ▼
┌──────────────────┐  ┌──────────────────────────────────────────────────┐
│  NVIDIA API      │  │  MODAL.COM (your deployment)                      │
│  Nemotron-3      │  │                                                  │
│  Ultra 550B      │  │  ┌────────────────────────────────────────────┐  │
│                  │  │  │  proto-tools-lite (CPU container)           │  │
│  • Receives      │  │  │                                            │  │
│    messages +    │  │  │  • SpliceAI score        (5-15s)           │  │
│    tool defs     │  │  │  • Pangolin score        (10-20s)          │  │
│  • Returns       │  │  │  • UniProt fetch         (<1s)             │  │
│    reasoning +   │  │  │  • AlphaFold DB fetch    (<1s)             │  │
│    tool_calls +  │  │  │  • AlphaMissense fetch   (<1s)             │  │
│    content       │  │  │  • Ensembl VEP           (<1s)             │  │
│  • Receives      │  │  │  • Ensembl sequence      (<1s)             │  │
│    tool results  │  │  │  • InterProScan          (2-5s)            │  │
│    and continues │  │  │  • BLAST search          (5-30s)           │  │
│    reasoning     │  │  │  • Foldseek search       (5-10s)           │  │
│                  │  │  │  • ViennaRNA             (1-5s)            │  │
│                  │  │  └────────────────────────────────────────────┘  │
│                  │  │                                                  │
│                  │  │  ┌────────────────────────────────────────────┐  │
│                  │  │  │  proto-tools-gpu (H100/A100 container)      │  │
│                  │  │  │                                            │  │
│                  │  │  │  • ESMFold prediction    (~60s on GPU)     │  │
│                  │  │  │  • ESM2 scoring          (30s-2min on GPU) │  │
│                  │  │  │  • AlphaFold2            (2-5min on GPU)   │  │
│                  │  │  │  • ProteinMPNN scoring   (30s-1min)        │  │
│                  │  │  └────────────────────────────────────────────┘  │
└──────────────────┘  └──────────────────────────────────────────────────┘
```

---

## 2. Tool Catalog — What We Deploy

### Tier 1: CPU Tools — Fast (< 1 second)

These are database retrieval tools that fetch pre-computed data. They run on
CPU and return instantly. Deployed in the `proto-tools-lite` Modal container.

| # | Tool Name | proto-tools Key | Input | Output | CPU Time | When Called |
|---|-----------|----------------|-------|--------|----------|-------------|
| 1 | **UniProt Fetch** | `uniprot_fetch` | UniProt accession (e.g. P38398) | Protein name, domains, function, disease associations, subcellular location | <1s | Always — when chatbot needs protein context |
| 2 | **AlphaFold DB Fetch** | `alphafold_db_fetch` | UniProt accession | Predicted 3D structure (PDB), per-residue pLDDT, PAE matrix | <1s | When chatbot needs structural context |
| 3 | **AlphaMissense Fetch** | `alphamissense_db_fetch` | UniProt accession | Per-residue, per-substitution pathogenicity scores | <1s | When chatbot needs missense pathogenicity |
| 4 | **Ensembl VEP** | `ensembl_vep` | HGVS notation (e.g. `17:g.43094169A>C`) | Variant consequence, impact, amino acid change, transcript | <1s | When chatbot needs molecular consequence |
| 5 | **Ensembl Lookup** | `ensembl_lookup` | Gene symbol or Ensembl ID | Gene record (chromosome, start, end, biotype) | <1s | When chatbot needs gene metadata |
| 6 | **Ensembl Sequence** | `ensembl_sequence` | Ensembl ID | DNA/cDNA/protein sequence | <1s | When chatbot needs the reference sequence |
| 7 | **PDB Fetch Entry** | `pdb_fetch_entry` | PDB ID (e.g. 1BRCA) | Structure metadata (title, method, resolution) | <1s | When chatbot checks for experimental structures |
| 8 | **PDB Fetch FASTA** | `pdb_fetch_fasta` | PDB ID | Chain sequences with protein/nucleotide classification | <1s | When chatbot needs PDB sequence |
| 9 | **NCBI ESearch** | `ncbi_esearch` | Query term, database | Matching IDs from NCBI Entrez | <1s | When chatbot searches NCBI databases |
| 10 | **NCBI EFetch** | `ncbi_efetch` | Accession or ID | FASTA records from NCBI | <1s | When chatbot fetches NCBI sequences |
| 11 | **NCBI ESummary** | `ncbi_esummary` | Entrez ID | Record summary metadata | <1s | When chatbot needs NCBI record metadata |
| 12 | **PubChem Fetch** | `pubchem_fetch` | CID, name, SMILES, or InChIKey | Canonical structure data, synonyms | <1s | When chatbot looks up small molecules |

### Tier 2: CPU Tools — Medium (1–30 seconds)

These tools run ML models or search algorithms on CPU. Still in the
`proto-tools-lite` container.

| # | Tool Name | proto-tools Key | Input | Output | CPU Time | When Called |
|---|-----------|----------------|-------|--------|----------|-------------|
| 13 | **SpliceAI Score** | `spliceai_score` | DNA sequence + variant | Splice delta scores (acceptor/donor gain/loss), positions | 5–15s | ⭐ When user asks about splicing effects |
| 14 | **SpliceAI Predict** | `spliceai_predict` | DNA sequence | Per-position acceptor/donor splice-site probabilities | 5–15s | When chatbot needs splice profile |
| 15 | **Pangolin Score** | `pangolin_score` | DNA sequence + variant | Tissue-specific splice variant scoring | 10–20s | Alternative to SpliceAI, tissue-specific |
| 16 | **Pangolin Predict** | `pangolin_predict` | DNA sequence | Tissue-specific splice-site probabilities | 10–20s | When chatbot needs tissue-specific splicing |
| 17 | **SpliceTransformer** | `splice_transformer` | DNA sequence | Tissue-specific splicing prediction | 10–20s | Another splice prediction alternative |
| 18 | **InterProScan** | `interproscan_fetch` | UniProt accession or raw sequence | Domain annotations from InterPro | 2–5s | When UniProt doesn't have domain data |
| 19 | **BLAST Search** | `blast_search` | Protein/DNA sequence | Homologous sequences with E-values, alignment | 5–30s | When chatbot searches for homologous proteins |
| 20 | **Foldseek Search** | `foldseek_search` | PDB structure | Structurally similar proteins | 5–10s | When chatbot searches by structural similarity |
| 21 | **MMseqs2 Search** | `mmseqs2_search_proteins` | Protein sequence | Fast sequence search results | 5–20s | Faster alternative to BLAST |
| 22 | **MAFFT Alignment** | `mafft_align` | Multiple sequences | Multiple sequence alignment (MSA) | 5–30s | When chatbot aligns homologous sequences |
| 23 | **MEME FIMO Scan** | `meme_fimo_scan` | DNA sequence + motif PWMs | Motif occurrences in sequence | 5–10s | When chatbot checks for motif disruption |
| 24 | **ViennaRNA** | `viennarna_prediction` | RNA sequence | Secondary structure (MFE) | 1–5s | When chatbot predicts RNA folding |
| 25 | **DSSP** | `dssp_secondary_structure` | PDB file | Helix/sheet/loop percentages | 1–3s | When chatbot analyzes secondary structure |
| 26 | **Segmasker** | `segmasker_score` | Protein sequence | Low-complexity region detection | <1s | When chatbot checks for low-complexity regions |
| 27 | **Structure Metrics** | `structure_metrics` | PDB file | SS percentages, longest helix, gyration radius | 1–3s | When chatbot evaluates structure quality |
| 28 | **CCD Lookup** | `ccd_lookup` | PDB CCD code | Chemical component dictionary entry | <1s | When chatbot looks up small molecule ligands |

### Tier 3: GPU Tools — Slow (30s–5min)

These require GPU acceleration. Deployed in the `proto-tools-gpu` Modal
container with H100 or A100.

| # | Tool Name | proto-tools Key | Input | Output | GPU Time | When Called |
|---|-----------|----------------|-------|--------|----------|-------------|
| 29 | **ESMFold** | `esmfold_prediction` | Protein sequence | Predicted 3D structure (PDB), pLDDT | ~60s | ⭐ When no AlphaFold structure exists |
| 30 | **ESM2 Score** | `esm2_score` | Protein sequence | Sequence likelihood score | 30s–2min | ⭐ When chatbot scores protein constraint |
| 31 | **ESM2 Embeddings** | `esm2_embedding` | Protein sequence | Per-residue embeddings + logits | 30s–2min | When chatbot extracts protein features |
| 32 | **ESM2 Gradient** | `esm2_gradient` | Protein sequence | Pseudo-log-likelihood gradient | 30s–2min | When chatbot does mutational analysis |
| 33 | **AlphaFold2** | `alphafold2_prediction` | Protein sequence | 3D structure with MSA, high accuracy | 2–5min | When ESMFold isn't accurate enough |
| 34 | **AlphaFold3** | `alphafold3_prediction` | Protein + ligand sequences | Multi-modal structure prediction | 2–5min | When chatbot predicts protein-ligand complexes |
| 35 | **Boltz2** | `boltz2_prediction` | Multi-modal input | Structure + affinity prediction | 2–5min | When chatbot predicts binding affinity |
| 36 | **Boltz2 Affinity** | `boltz2_affinity` | Protein + small molecule | Predicted binding affinity (IC50) | 2–5min | When chatbot evaluates drug binding |
| 37 | **ProteinMPNN Score** | `proteinmpnn_score` | Protein sequence + structure | Structure-conditioned sequence score | 30s–1min | When chatbot evaluates sequence designability |
| 38 | **ProteinMPNN Sample** | `proteinmpnn_sample` | Protein backbone structure | Sampled protein sequences | 30s–1min | When chatbot designs new sequences |
| 39 | **LigandMPNN Score** | `ligandmpnn_score` | Protein sequence + structure + ligand | Ligand-aware sequence score | 30s–1min | When chatbot scores with ligand context |
| 40 | **ESM-IF1 Score** | `esm_if1_score` | Protein sequence + backbone | Inverse folding log-likelihood | 30s–1min | When chatbot scores sequence vs structure |
| 41 | **Chai1** | `chai1_prediction` | Multi-modal input | Structure prediction | 2–5min | Alternative to AlphaFold |
| 42 | **Protenix** | `protenix_prediction` | Multi-modal input | Open-source AlphaFold3 | 2–5min | Open-source alternative to AF3 |
| 43 | **RoseTTAFold3** | `rf3_prediction` | All-atom input | Structure with explicit chirality | 2–5min | High-accuracy structure prediction |
| 44 | **BioEmu** | `bioemu_sample` | Protein sequence | Conformational ensemble | 5–10min | When chatbot samples protein dynamics |
| 45 | **RFdiffusion3** | `rfdiffusion3_design` | Design parameters | De novo protein structure | 5–10min | When chatbot designs novel structures |

### Tier 4: Sequence Scoring (CPU or GPU)

Genomic/regulatory scoring models. Some run on CPU, others need GPU.

| # | Tool Name | proto-tools Key | Input | Output | Hardware | Time | When Called |
|---|-----------|----------------|-------|--------|----------|------|-------------|
| 46 | **Evo2 Score** | `evo2_score` | DNA sequence | Log-likelihood score | GPU | 5–10s | Already have via Modal — can use proto-tools version |
| 47 | **Evo2 Sample** | `evo2_sample` | DNA sequence (masked) | Sampled DNA sequences | GPU | 5–10s | When chatbot generates variant sequences |
| 48 | **Evo1 Score** | `evo1_score` | DNA sequence | Log-likelihood score | GPU | 5–10s | Legacy Evo comparison |
| 49 | **Borzoi** | `borzoi_prediction` | DNA sequence | Regulatory activity prediction | GPU | 10–30s | When chatbot predicts regulatory effects |
| 50 | **Enformer** | `enformer_prediction` | DNA sequence | Gene expression prediction | GPU | 10–30s | When chatbot predicts expression impact |
| 51 | **AlphaGenome** | `alphagenome_score_variants` | DNA sequence + variants | Variant effect scores | GPU | 10–30s | When chatbot scores variant effects on regulation |

---

## 3. When the Chatbot Calls Which Tool

### 3.1 The Decision Flow

The chatbot (Nemotron-3 Ultra 550B) decides which tool to call based on:
1. The user's question
2. The variant report context (already in the system prompt)
3. What data is missing vs. what's already available
4. The tool definitions provided in the API call

```
User asks a question
       │
       ▼
Nemotron reasons (chain-of-thought):
  "Let me think about what the user is asking and what data I have..."
       │
       ├─ Does the report already have this data?
       │   ├─ YES → Answer directly from report context (no tool call)
       │   └─ NO → Decide which tool to call
       │
       ▼
Nemotron emits tool_call:
  "I need to run SpliceAI to check if this variant affects splicing"
       │
       ▼
Next.js executes the tool:
  → Calls Modal endpoint
  → Gets result
  → Feeds result back to Nemotron
       │
       ▼
Nemotron continues reasoning with tool result:
  "SpliceAI shows Δ score 0.82, which is above the 0.2 threshold..."
       │
       ├─ Need more data?
       │   ├─ YES → Call another tool (max 3 per turn)
       │   └─ NO → Generate final answer
       │
       ▼
Final answer streamed to user
```

### 3.2 Scenario-Based Tool Calling Examples

#### Scenario 1: "Does this variant affect splicing?"

```
User: "Does this BRCA1 variant affect splicing?"

Nemotron reasoning:
  "The user is asking about splicing. The report shows this is at
  chr17:43094169, which is near an exon boundary. I should run SpliceAI
  to get splice-site predictions. I need the surrounding DNA sequence
  first — let me fetch it from Ensembl."

Tool call 1: ensembl_sequence
  Input: { ensembl_id: "ENST00000357654", type: "genomic" }
  Result: { sequence: "ATCG...5000bp...CGAT" }

Tool call 2: spliceai_score
  Input: { sequence: "ATCG...5000bp...CGAT", variant: "43094169A>C" }
  Result: {
    delta_score_acceptor_gain: 0.82,
    delta_score_donor_loss: 0.03,
    interpretation: "High confidence cryptic acceptor site creation"
  }

Nemotron final answer:
  "Yes, SpliceAI predicts this variant creates a cryptic acceptor site
  with a delta score of 0.82 (threshold >0.2 is high confidence). This
  means the variant likely causes aberrant splicing by introducing a
  new splice acceptor site. The existing donor site is not significantly
  affected (Δ = 0.03).

  Clinical implication: This supports a splice-disrupting mechanism.
  Therapeutic strategies could include antisense oligonucleotides (ASOs)
  to block the cryptic acceptor site."
```

#### Scenario 2: "What protein domains are affected?"

```
User: "What protein domains are affected by this variant?"

Nemotron reasoning:
  "The user wants domain information. The report has UniProt data but
  the domains field shows 'none'. Let me fetch the full UniProt entry
  to get domain annotations. I need the UniProt accession for BRCA1,
  which is P38398."

Tool call 1: uniprot_fetch
  Input: { accession: "P38398" }
  Result: {
    protein_name: "Breast cancer type 1 susceptibility protein",
    domains: [
      { name: "RING-type zinc finger", start: 24, end: 64 },
      { name: "BRCT domain 1", start: 1646, end: 1735 },
      { name: "BRCT domain 2", start: 1759, end: 1855 }
    ],
    function: "E3 ubiquitin-protein ligase...",
    disease_associations: ["Breast cancer", "Ovarian cancer"]
  }

Nemotron final answer:
  "BRCA1 (UniProt: P38398) has 3 functional domains:
  1. RING-type zinc finger (residues 24–64) — E3 ubiquitin ligase activity
  2. BRCT domain 1 (residues 1646–1735) — protein-protein interactions
  3. BRCT domain 2 (residues 1759–1855) — DNA damage response

  Your variant at position 43094169 corresponds to residue S454, which
  falls between the RING domain and the BRCT domains — in the coiled-coil
  region involved in PALB2 interaction. This region mediates DNA repair
  complex formation..."
```

#### Scenario 3: "Can you predict the 3D structure of the mutant protein?"

```
User: "Can you predict the 3D structure of the mutant protein?"

Nemotron reasoning:
  "The user wants a 3D structure. Let me first check if AlphaFold DB
  has a pre-computed structure for BRCA1 (P38398). If it does, I can
  use that. If not, I'll need to run ESMFold."

Tool call 1: alphafold_db_fetch
  Input: { uniprot_accession: "P38398", structure_format: "pdb" }
  Result: {
    pdb_url: "https://alphafold.ebi.ac.uk/files/AF-P38398-F1.pdb",
    mean_plddt: 78.5,
    domains_with_high_confidence: ["BRCT1", "BRCT2"],
    residue_454_plddt: 72.3
  }

Nemotron final answer:
  "AlphaFold DB has a pre-computed structure for BRCA1 (P38398).
  The structure has a mean pLDDT of 78.5 (good confidence overall).

  At residue 454 (your variant position), the pLDDT is 72.3, indicating
  moderate confidence in the structure at this position. The region
  around S454 is part of the coiled-coil domain, which typically has
  lower pLDDT due to its flexible nature.

  To see the structural impact of the S454R mutation, I recommend:
  1. Download the wild-type structure from AlphaFold DB
  2. Run ESMFold on the mutant sequence for comparison
  3. Use TMalign to calculate RMSD between WT and mutant

  Would you like me to run ESMFold on the mutant sequence?"
```

#### Scenario 4: "How does this variant compare across populations?"

```
User: "How does this variant compare across populations?"

Nemotron reasoning:
  "The user wants population-specific data. The report has gnomAD
  allele frequency but not the population breakdown. I don't have a
  proto-tool for gnomAD — but the report already has the overall AF.
  Let me check if the gnomAD data in the report includes sub-population
  data... It shows gnomad_af but not gnomad_max_pop_af. I should note
  this limitation and explain what's available."

No tool call needed — answer from report context:

  "Based on the report data:
  - Overall gnomAD v4.1 allele frequency: 0.000012 (ultra-rare)
  - Not observed in any specific sub-population

  Unfortunately, the current analysis pipeline doesn't include
  population-specific breakdowns. For detailed population analysis,
  I recommend checking gnomAD directly:
  https://gnomad.broadinstitute.org/variant/17-43094169-A-C

  The ultra-rarity (AF < 0.001) supports the ACMG PM2 criterion
  (absent from population databases), which is already noted in
  the report."
```

#### Scenario 5: "Is there a similar variant in any other gene?"

```
User: "Is there a similar variant in any other gene?"

Nemotron reasoning:
  "The user wants to find similar variants in other genes. I should
  search for homologous proteins using BLAST, then check if the
  equivalent position has known variants. First, I need the protein
  sequence of BRCA1."

Tool call 1: ensembl_sequence
  Input: { ensembl_id: "ENSP00000350183", type: "protein" }
  Result: { sequence: "MDLSALRVE...1863aa...KKNRLF" }

Tool call 2: blast_search
  Input: { sequence: "MDLSALRVE...", database: "nr", program: "blastp" }
  Result: {
    hits: [
      { gene: "BRCA1", organism: "Pan troglodytes", identity: 99.2 },
      { gene: "BRCA1", organism: "Mus musculus", identity: 58.7 },
      { gene: "BRCA1", organism: "Rattus norvegicus", identity: 57.3 },
      { gene: "BARD1", organism: "Homo sapiens", identity: 23.1 }
    ]
  }

Nemotron final answer:
  "BLAST search found several homologous proteins:
  1. Chimp BRCA1 — 99.2% identity (expected, close evolutionary distance)
  2. Mouse BRCA1 — 58.7% identity
  3. Rat BRCA1 — 57.3% identity
  4. Human BARD1 — 23.1% identity (paralog, shares RING domain)

  The RING domain (where your variant is near) is highly conserved
  across all these species. Position 454 in human BRCA1 corresponds to:
  - Position 454 in chimp (S→S, conserved)
  - Position 451 in mouse (S→T, conservative substitution)
  - Position 451 in rat (S→T, conservative substitution)

  The conservation of serine at this position suggests it may be
  functionally important, though the conservative S→T substitution
  in rodents suggests some flexibility..."
```

### 3.3 Tool Calling Rules (System Prompt Constraints)

The system prompt includes these rules for Nemotron:

```
## TOOL CALLING RULES
1. Maximum 3 tool calls per user turn (budget limit)
2. Always check if the report already contains the data before calling a tool
3. Prefer database retrieval tools (UniProt, AlphaFold DB) over prediction tools (ESMFold)
4. Prefer CPU tools (fast) over GPU tools (slow) when both can answer
5. If a tool fails, explain the failure and continue with available data
6. Always cite tool results in your answer with specific numbers
7. If you call a tool, explain WHY you're calling it in your reasoning
8. After receiving tool results, synthesize them with the report data
```

---

## 4. Modal Deployment Plan

### 4.1 Two Modal Containers

| Container | Hardware | Tools | Purpose |
|-----------|----------|-------|---------|
| `proto-tools-lite` | CPU (2GB RAM) | Tier 1 + Tier 2 (tools 1–28) | Fast database retrieval + CPU ML models |
| `proto-tools-gpu` | H100 or A100 | Tier 3 + Tier 4 (tools 29–51) | GPU-intensive structure prediction + scoring |

### 4.2 Modal App Structure

```python
# backend/proto_tools/modal_deploy_lite.py
"""
Proto-Tools Lite — CPU container for fast bioinformatics tools.
Deployed on Modal.com with CPU only (no GPU).
"""

import modal
import os

# ─── Image: proto-tools + dependencies ───────────────────────────────
proto_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("proto-tools")  # Install proto-tools from GitHub
    .pip_install("requests", "aiohttp")
    .apt_install("blast+", "mmseqs2", "foldseek")  # System tools
)

# ─── Modal App ────────────────────────────────────────────────────────
app = modal.App(
    name="helixmind-proto-lite",
    image=proto_image,
)

# ─── CPU Container ────────────────────────────────────────────────────
@app.function(
    cpu=2,
    memory=4096,  # 4GB RAM
    timeout=120,  # 2 minute timeout
    min_containers=0,  # Scale to zero
)
@modal.fastapi_endpoint(method="POST")
async def run_tool(request: dict):
    """
    Execute a proto-tool on CPU.

    Request body:
    {
        "tool_key": "spliceai_score",
        "input": { "sequence": "ATCG...", "variant": "43094169A>C" },
        "config": {}  # optional tool configuration
    }

    Response:
    {
        "tool_key": "spliceai_score",
        "status": "completed",
        "result": { "delta_score": 0.82, ... },
        "execution_time_ms": 12340
    }
    """
    from proto_tools.api import run_tool as proto_run_tool

    tool_key = request.get("tool_key")
    tool_input = request.get("input", {})
    tool_config = request.get("config", {})

    try:
        result = proto_run_tool(
            tool_key=tool_key,
            input=tool_input,
            config=tool_config,
        )
        return {
            "tool_key": tool_key,
            "status": "completed",
            "result": result,
            "execution_time_ms": 0,  # Filled by wrapper
        }
    except Exception as e:
        return {
            "tool_key": tool_key,
            "status": "failed",
            "error": str(e),
        }
```

```python
# backend/proto_tools/modal_deploy_gpu.py
"""
Proto-Tools GPU — H100 container for structure prediction + scoring.
"""

import modal

proto_gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("proto-tools")
    .pip_install("torch", "esm", "openfold")
    .apt_install("cuda-toolkit-12-0")
)

app = modal.App(
    name="helixmind-proto-gpu",
    image=proto_gpu_image,
)

@app.function(
    gpu="H100",
    memory=16384,  # 16GB RAM
    timeout=600,  # 10 minute timeout for long predictions
    min_containers=0,
)
@modal.fastapi_endpoint(method="POST")
async def run_tool_gpu(request: dict):
    """
    Execute a GPU-intensive proto-tool.

    Same interface as the CPU container, but with GPU access.
    """
    from proto_tools.api import run_tool as proto_run_tool

    tool_key = request.get("tool_key")
    tool_input = request.get("input", {})
    tool_config = request.get("config", {})

    try:
        result = proto_run_tool(
            tool_key=tool_key,
            input=tool_input,
            config=tool_config,
        )
        return {
            "tool_key": tool_key,
            "status": "completed",
            "result": result,
        }
    except Exception as e:
        return {
            "tool_key": tool_key,
            "status": "failed",
            "error": str(e),
        }
```

### 4.3 Deployment Commands

```bash
# Deploy CPU container (proto-tools-lite)
cd backend/proto_tools
modal deploy modal_deploy_lite.py

# Deploy GPU container (proto-tools-gpu)
modal deploy modal_deploy_gpu.py

# Set environment variables in frontend/.env.local
PROTO_TOOLS_LITE_URL=https://<your-modal-workspace>--helixmind-proto-lite-run-tool.modal.run
PROTO_TOOLS_GPU_URL=https://<your-modal-workspace>--helixmind-proto-gpu-run-tool-gpu.modal.run
```

### 4.4 Tool Routing (CPU vs GPU)

The Next.js API route routes tool calls to the correct Modal container:

```typescript
// frontend/src/lib/proto-tools-router.ts

const CPU_TOOLS = [
  "uniprot_fetch", "alphafold_db_fetch", "alphamissense_db_fetch",
  "ensembl_vep", "ensembl_lookup", "ensembl_sequence",
  "pdb_fetch_entry", "pdb_fetch_fasta",
  "ncbi_esearch", "ncbi_efetch", "ncbi_esummary",
  "pubchem_fetch",
  "spliceai_score", "spliceai_predict",
  "pangolin_score", "pangolin_predict",
  "splice_transformer",
  "interproscan_fetch",
  "blast_search", "foldseek_search",
  "mmseqs2_search_proteins", "mafft_align",
  "meme_fimo_scan", "viennarna_prediction",
  "dssp_secondary_structure", "segmasker_score",
  "structure_metrics", "ccd_lookup",
];

const GPU_TOOLS = [
  "esmfold_prediction",
  "esm2_score", "esm2_embedding", "esm2_gradient",
  "alphafold2_prediction", "alphafold3_prediction",
  "boltz2_prediction", "boltz2_affinity",
  "proteinmpnn_score", "proteinmpnn_sample",
  "ligandmpnn_score",
  "esm_if1_score",
  "chai1_prediction", "protenix_prediction",
  "rf3_prediction",
  "bioemu_sample", "rfdiffusion3_design",
  "evo2_score", "evo2_sample", "evo1_score",
  "borzoi_prediction", "enformer_prediction",
  "alphagenome_score_variants",
];

export function getToolEndpoint(toolKey: string): string {
  if (GPU_TOOLS.includes(toolKey)) {
    return process.env.PROTO_TOOLS_GPU_URL!;
  }
  return process.env.PROTO_TOOLS_LITE_URL!;
}
```

---

## 5. Tool-Calling Loop — How It Works

### 5.1 The Loop (in `/api/chat/route.ts`)

```
┌─────────────────────────────────────────────────────────────────────┐
│  TOOL-CALLING LOOP (max 3 iterations)                                │
│                                                                     │
│  Iteration 1:                                                       │
│  ┌──────────────┐                                                  │
│  │ Call Nemotron │ ← messages + tools + report context              │
│  │ with stream   │                                                  │
│  └──────┬───────┘                                                  │
│         │                                                           │
│         ▼                                                           │
│  Parse SSE chunks:                                                  │
│  ├── reasoning_delta → stream to frontend                          │
│  ├── content_delta → stream to frontend                            │
│  ├── tool_calls → execute tool, stream status to frontend          │
│  └── finish_reason: "tool_calls" → continue loop                   │
│         │                                                           │
│         ▼                                                           │
│  Execute tool:                                                      │
│  ├── Determine CPU vs GPU endpoint                                 │
│  ├── fetch(MODAL_ENDPOINT, { tool_key, input })                    │
│  ├── Stream "calling" status to frontend                           │
│  ├── Wait for result (5s–60s)                                      │
│  ├── Stream "completed" + result to frontend                       │
│  └── Add tool result to messages as role="tool"                    │
│         │                                                           │
│         ▼                                                           │
│  Iteration 2: (if tool was called)                                  │
│  ┌──────────────┐                                                  │
│  │ Call Nemotron │ ← messages (now includes tool result)            │
│  │ again         │                                                  │
│  └──────┬───────┘                                                  │
│         │                                                           │
│         ▼                                                           │
│  Parse SSE chunks:                                                  │
│  ├── reasoning_delta → stream to frontend                          │
│  ├── content_delta → stream to frontend                            │
│  ├── tool_calls → execute another tool (if needed)                 │
│  └── finish_reason: "stop" → exit loop, send "done"               │
│                                                                     │
│  Max 3 iterations. After that, force stop.                          │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.2 Pseudocode

```typescript
async function chatWithToolCalling(messages, tools, systemPrompt) {
  const MAX_TOOL_CALLS = 3;
  let toolCallCount = 0;
  let allMessages = [{ role: "system", content: systemPrompt }, ...messages];

  while (toolCallCount < MAX_TOOL_CALLS) {
    // 1. Call Nemotron with streaming
    const nvidiaStream = await callNemotron(allMessages, tools);

    // 2. Parse the stream
    const { reasoning, content, toolCalls, finishReason } = await parseStream(nvidiaStream);

    // 3. Stream reasoning + content to frontend (already done during parsing)

    // 4. If no tool calls, we're done
    if (finishReason === "stop" || toolCalls.length === 0) {
      break;
    }

    // 5. Execute tool calls
    for (const toolCall of toolCalls) {
      // Stream "calling" status to frontend
      streamToClient({ event: "tool_call", data: { toolName, status: "calling" } });

      // Execute the tool
      const result = await executeTool(toolCall.name, toolCall.arguments);

      // Stream "completed" status to frontend
      streamToClient({ event: "tool_call", data: { toolName, status: "completed", result } });

      // Add tool result to messages for next Nemotron call
      allMessages.push({ role: "tool", tool_call_id: toolCall.id, content: JSON.stringify(result) });
    }

    toolCallCount++;
  }

  // Persist to database
  await persistMessages(allMessages);
}
```

---

## 6. Nemotron Function Definitions

### 6.1 Tool Schema Format (OpenAI-compatible)

Nemotron uses the OpenAI function-calling format. Each tool is defined as a
"function" with a name, description, and JSON schema for parameters.

```typescript
// frontend/src/lib/chat-tools.ts

export const CHAT_TOOLS = [
  // ─── Tier 1: Database Retrieval (fast, <1s) ───────────────────────
  {
    type: "function",
    function: {
      name: "fetch_uniprot",
      description: "Fetch protein entry from UniProt by accession. Returns protein name, domains, function, disease associations, and subcellular location. Use this when you need protein context, domain information, or disease associations for a gene.",
      parameters: {
        type: "object",
        properties: {
          accession: {
            type: "string",
            description: "UniProt accession (e.g. P38398 for BRCA1, P04637 for TP53)",
          },
        },
        required: ["accession"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "fetch_alphafold_db",
      description: "Fetch predicted 3D structure from AlphaFold Protein Structure Database. Returns PDB structure, per-residue pLDDT confidence scores, and PAE matrix. Use this when you need structural context for a protein.",
      parameters: {
        type: "object",
        properties: {
          uniprot_accession: {
            type: "string",
            description: "UniProt accession (e.g. P38398)",
          },
          structure_format: {
            type: "string",
            enum: ["pdb", "mmcif"],
            description: "Structure file format. Default: pdb",
          },
        },
        required: ["uniprot_accession"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "fetch_alphamissense",
      description: "Fetch per-residue AlphaMissense pathogenicity scores from the AlphaFold DB. Returns pathogenicity scores for all possible substitutions at each residue. Use this for missense variant pathogenicity assessment.",
      parameters: {
        type: "object",
        properties: {
          uniprot_accession: {
            type: "string",
            description: "UniProt accession",
          },
        },
        required: ["uniprot_accession"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "run_ensembl_vep",
      description: "Predict variant consequences using Ensembl VEP. Returns molecular consequence, impact level, amino acid change, and transcript information. Use this when you need to annotate a variant's molecular effect.",
      parameters: {
        type: "object",
        properties: {
          hgvs_notation: {
            type: "string",
            description: "HGVS notation (e.g. '17:g.43094169A>C' or 'ENST00000357654:c.5074G>A')",
          },
          species: {
            type: "string",
            description: "Species slug. Default: homo_sapiens",
          },
          assembly: {
            type: "string",
            enum: ["GRCh38", "GRCh37"],
            description: "Genome assembly. Default: GRCh38",
          },
        },
        required: ["hgvs_notation"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "fetch_ensembl_sequence",
      description: "Fetch DNA/cDNA/protein sequence for an Ensembl ID. Use this when you need the reference sequence for a gene or transcript.",
      parameters: {
        type: "object",
        properties: {
          ensembl_id: {
            type: "string",
            description: "Ensembl ID (e.g. ENST00000357654 for transcript, ENSP00000350183 for protein)",
          },
          sequence_type: {
            type: "string",
            enum: ["genomic", "cds", "cdna", "protein"],
            description: "Type of sequence to fetch. Default: genomic",
          },
        },
        required: ["ensembl_id"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "fetch_pdb_entry",
      description: "Fetch structure metadata from RCSB PDB. Returns title, experimental method, resolution. Use this when checking for experimental structures.",
      parameters: {
        type: "object",
        properties: {
          pdb_id: {
            type: "string",
            description: "PDB ID (e.g. 1BRCA, 1TUP)",
          },
        },
        required: ["pdb_id"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "search_ncbi",
      description: "Search NCBI Entrez databases (protein, nucleotide, pubmed, clinvar). Returns matching IDs. Use this for literature or sequence searches.",
      parameters: {
        type: "object",
        properties: {
          database: {
            type: "string",
            enum: ["protein", "nuccore", "pubmed", "clinvar"],
            description: "NCBI database to search",
          },
          query: {
            type: "string",
            description: "Search query (e.g. 'BRCA1 missense variant')",
          },
          retmax: {
            type: "integer",
            description: "Maximum results. Default: 10",
          },
        },
        required: ["database", "query"],
      },
    },
  },

  // ─── Tier 2: CPU ML Tools (5–30s) ────────────────────────────────
  {
    type: "function",
    function: {
      name: "run_spliceai_score",
      description: "Score a variant for splice-altering effects using SpliceAI deep learning model. Returns delta scores for acceptor/donor gain/loss. Delta >0.2 is high confidence splice-altering. Use this when the user asks about splicing effects or when the variant is near an exon boundary (±50bp).",
      parameters: {
        type: "object",
        properties: {
          sequence: {
            type: "string",
            description: "DNA sequence (at least 5000bp surrounding the variant for accurate prediction)",
          },
          variant: {
            type: "string",
            description: "Variant in format 'positionRefAlt' (e.g. '43094169A>C')",
          },
        },
        required: ["sequence", "variant"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "run_pangolin_score",
      description: "Score variant for tissue-specific splice effects using Pangolin. Returns tissue-specific splice scores. Use as alternative to SpliceAI when tissue-specific predictions are needed.",
      parameters: {
        type: "object",
        properties: {
          sequence: {
            type: "string",
            description: "DNA sequence surrounding the variant",
          },
          variant: {
            type: "string",
            description: "Variant notation",
          },
        },
        required: ["sequence", "variant"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "run_interproscan",
      description: "Fetch domain annotations from InterPro. More comprehensive than UniProt for domain detection. Use when UniProt doesn't have domain data or when searching by raw sequence.",
      parameters: {
        type: "object",
        properties: {
          accession: {
            type: "string",
            description: "UniProt accession (for direct lookup)",
          },
          sequence: {
            type: "string",
            description: "Raw protein sequence (for de novo search)",
          },
        },
      },
    },
  },
  {
    type: "function",
    function: {
      name: "run_blast_search",
      description: "Search for homologous sequences using BLAST. Returns hits with E-values, percent identity, and alignments. Use this when looking for similar proteins or genes in other organisms.",
      parameters: {
        type: "object",
        properties: {
          sequence: {
            type: "string",
            description: "Query sequence (protein or DNA)",
          },
          program: {
            type: "string",
            enum: ["blastp", "blastn", "tblastn"],
            description: "BLAST program. Default: blastp",
          },
          database: {
            type: "string",
            description: "Database to search (e.g. 'nr', 'refseq_protein'). Default: nr",
          },
          max_results: {
            type: "integer",
            description: "Maximum hits. Default: 10",
          },
        },
        required: ["sequence"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "run_foldseek_search",
      description: "Search for structurally similar proteins using Foldseek. Input is a PDB structure. Returns structurally related proteins. Use when searching by 3D structure rather than sequence.",
      parameters: {
        type: "object",
        properties: {
          pdb_structure: {
            type: "string",
            description: "PDB format structure string",
          },
          database: {
            type: "string",
            description: "Database (e.g. 'alphafolddb', 'pdb100'). Default: alphafolddb",
          },
        },
        required: ["pdb_structure"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "run_viennarna",
      description: "Predict RNA secondary structure using ViennaRNA MFE folding. Returns minimum free energy structure. Use when analyzing RNA folding effects of a variant.",
      parameters: {
        type: "object",
        properties: {
          sequence: {
            type: "string",
            description: "RNA sequence (will be converted from DNA if needed)",
          },
        },
        required: ["sequence"],
      },
    },
  },

  // ─── Tier 3: GPU Tools (30s–5min) ────────────────────────────────
  {
    type: "function",
    function: {
      name: "run_esmfold",
      description: "Predict protein 3D structure from sequence using ESMFold. Returns PDB structure and pLDDT scores. Takes ~60 seconds on GPU. Use this when no AlphaFold DB structure exists or when you need to predict a mutant structure.",
      parameters: {
        type: "object",
        properties: {
          sequence: {
            type: "string",
            description: "Protein sequence (single-letter amino acids)",
          },
        },
        required: ["sequence"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "run_esm2_score",
      description: "Score a protein sequence using ESM2 language model. Returns log-likelihood and pseudo-perplexity. Use this to assess how evolutionarily constrained a protein region is. Takes 30s–2min on GPU.",
      parameters: {
        type: "object",
        properties: {
          sequence: {
            type: "string",
            description: "Protein sequence",
          },
        },
        required: ["sequence"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "run_alphafold2",
      description: "Predict protein 3D structure using AlphaFold2 with MSA. Higher accuracy than ESMFold but slower (2–5 min on GPU). Use when ESMFold confidence is low or when high accuracy is needed.",
      parameters: {
        type: "object",
        properties: {
          sequence: {
            type: "string",
            description: "Protein sequence",
          },
        },
        required: ["sequence"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "run_proteinmpnn_score",
      description: "Score a protein sequence against its structure using ProteinMPNN. Returns structure-conditioned perplexity. Use to assess how well a sequence fits its structure. Takes 30s–1min on GPU.",
      parameters: {
        type: "object",
        properties: {
          sequence: {
            type: "string",
            description: "Protein sequence",
          },
          structure_pdb: {
            type: "string",
            description: "PDB structure string",
          },
        },
        required: ["sequence", "structure_pdb"],
      },
    },
  },
];
```

### 6.2 Tool Selection Logic

Not all tools are sent to Nemotron on every call. The tools array is filtered
based on the mode and variant context:

```typescript
export function getToolsForMode(mode: "report" | "general", report?: Report) {
  if (mode === "general") {
    // General mode: only database retrieval tools (no variant-specific tools)
    return CHAT_TOOLS.filter(t =>
      ["fetch_uniprot", "fetch_alphafold_db", "search_ncbi"].includes(t.function.name)
    );
  }

  // Report mode: all tools, but prioritize based on variant type
  const tools = [...CHAT_TOOLS];

  // If variant is near splice site, ensure SpliceAI is included
  if (isNearSpliceSite(report)) {
    // SpliceAI is already in the list
  }

  // If variant is missense, ensure AlphaMissense is included
  if (isMissense(report)) {
    // AlphaMissense is already in the list
  }

  return tools;
}
```

---

## 7. SSE Event Protocol for Tool Calls

### 7.1 New Event Type: `tool_call`

The existing SSE protocol has: `reasoning_delta`, `content_delta`, `done`, `error`.

We add: `tool_call`, `tool_result`.

| Event | Payload | When |
|-------|---------|------|
| `reasoning_delta` | `{ delta: string }` | Nemotron reasoning token |
| `content_delta` | `{ delta: string }` | Nemotron content token |
| `tool_call` | `{ toolCallId: string, toolName: string, status: "calling", args: object }` | Nemotron requests a tool |
| `tool_result` | `{ toolCallId: string, toolName: string, status: "completed" \| "failed", result?: object, error?: string, executionTimeMs: number }` | Tool execution finished |
| `done` | `{ sessionId: string, latencyMs: number, tokens: object }` | Response complete |
| `error` | `{ error: string }` | Error occurred |

### 7.2 Example SSE Stream with Tool Call

```
event: reasoning_delta
data: {"delta":"Let me check if this variant affects splicing. I'll run SpliceAI..."}

event: tool_call
data: {"toolCallId":"call_abc123","toolName":"run_spliceai_score","status":"calling","args":{"sequence":"ATCG...","variant":"43094169A>C"}}

event: tool_result
data: {"toolCallId":"call_abc123","toolName":"run_spliceai_score","status":"completed","result":{"delta_score_acceptor_gain":0.82,"delta_score_donor_loss":0.03},"executionTimeMs":12340}

event: reasoning_delta
data: {"delta":"SpliceAI shows a delta score of 0.82 for acceptor gain, which is above the 0.2 threshold..."}

event: content_delta
data: {"delta":"Yes, SpliceAI predicts this variant creates a cryptic acceptor site..."}

event: content_delta
data: {"delta":" with a delta score of 0.82 (high confidence). This suggests..."}

event: done
data: {"sessionId":"sess_xyz","latencyMs":45000,"tokens":{"total":850}}
```

---

## 8. Frontend UI — Tool Call Display

### 8.1 ChatToolCallBlock Component

A new component that renders between the chain-of-thought and the answer:

```
┌──────────────────────────────────────────────────────────────┐
│  🧠 Chain of Thought                                          │
│  "Let me check if this variant affects splicing. I'll        │
│   run SpliceAI to get splice-site predictions..."            │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│  🔧 Tool Call: SpliceAI Score                          12.3s  │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ Input:                                                 │  │
│  │ • Sequence: ATCG...5000bp...CGAT                      │  │
│  │ • Variant: 43094169A>C                                 │  │
│  │                                                        │  │
│  │ Result: ✅ Completed                                   │  │
│  │ • Acceptor gain Δ: 0.82 (high confidence)              │  │
│  │ • Donor loss Δ: 0.03 (negligible)                      │  │
│  │ • Interpretation: Cryptic acceptor site created        │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│  💬 Answer                                                    │
│  "Yes, SpliceAI predicts this variant creates a cryptic      │
│   acceptor site with Δ score 0.82..."                        │
└──────────────────────────────────────────────────────────────┘
```

### 8.2 Component Design

```typescript
// frontend/src/components/chat/chat-tool-call-block.tsx

interface ToolCall {
  toolCallId: string;
  toolName: string;
  status: "calling" | "completed" | "failed";
  args: Record<string, unknown>;
  result?: Record<string, unknown>;
  error?: string;
  executionTimeMs?: number;
}

function ChatToolCallBlock({ toolCall }: { toolCall: ToolCall }) {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <div className="rounded-lg border border-blue-200 bg-blue-50 p-2.5">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1.5 text-[11px] font-semibold text-blue-700 uppercase">
          <Wrench className="h-3 w-3" />
          {toolCall.toolName}
        </div>
        <div className="flex items-center gap-2">
          {toolCall.status === "calling" && (
            <span className="flex items-center gap-1 text-[10px] text-blue-600">
              <Loader2 className="h-3 w-3 animate-spin" />
              Running...
            </span>
          )}
          {toolCall.status === "completed" && (
            <span className="text-[10px] text-green-600">
              ✅ {toolCall.executionTimeMs ? `${(toolCall.executionTimeMs / 1000).toFixed(1)}s` : ""}
            </span>
          )}
          {toolCall.status === "failed" && (
            <span className="text-[10px] text-red-600">❌ Failed</span>
          )}
        </div>
      </div>

      {/* Expandable details */}
      <button onClick={() => setIsExpanded(!isExpanded)}>
        {isExpanded ? "Hide details" : "Show details"}
      </button>
      {isExpanded && (
        <div className="mt-2 text-xs">
          <div className="mb-1 font-medium">Input:</div>
          <pre className="bg-white p-2 rounded">{JSON.stringify(toolCall.args, null, 2)}</pre>
          {toolCall.result && (
            <>
              <div className="mb-1 font-medium">Result:</div>
              <pre className="bg-white p-2 rounded">{JSON.stringify(toolCall.result, null, 2)}</pre>
            </>
          )}
          {toolCall.error && (
            <div className="text-red-600">{toolCall.error}</div>
          )}
        </div>
      )}
    </div>
  );
}
```

### 8.3 Updated ChatMessageData Interface

```typescript
export interface ChatMessageData {
  id: string;
  role: "user" | "assistant" | "system" | "tool";
  content: string;
  reasoning?: string;
  toolCalls?: ToolCall[];  // NEW: tool calls made during this message
  createdAt: string;
  isStreaming?: boolean;
}
```

---

## 9. Implementation Steps

### Phase 13a: Modal Deployment (3–4 days)

| Step | Task | Files | Est. Time |
|------|------|-------|-----------|
| 1 | Install proto-tools locally and test | — | 2 hours |
| 2 | Create `backend/proto_tools/` directory | — | 10 min |
| 3 | Write `modal_deploy_lite.py` (CPU container) | `backend/proto_tools/modal_deploy_lite.py` | 3 hours |
| 4 | Write `modal_deploy_gpu.py` (GPU container) | `backend/proto_tools/modal_deploy_gpu.py` | 3 hours |
| 5 | Deploy CPU container to Modal | — | 1 hour |
| 6 | Test CPU tools (UniProt, SpliceAI, BLAST) | — | 2 hours |
| 7 | Deploy GPU container to Modal | — | 1 hour |
| 8 | Test GPU tools (ESMFold, ESM2) | — | 2 hours |
| 9 | Add environment variables to `.env.local` | `frontend/.env.local` | 10 min |

### Phase 13b: Tool Definitions + Routing (1–2 days)

| Step | Task | Files | Est. Time |
|------|------|-------|-----------|
| 10 | Create `chat-tools.ts` with all tool definitions | `frontend/src/lib/chat-tools.ts` | 3 hours |
| 11 | Create `proto-tools-router.ts` for CPU/GPU routing | `frontend/src/lib/proto-tools-router.ts` | 1 hour |
| 12 | Create `/api/tools/proto` API route (proxy to Modal) | `frontend/src/app/api/tools/proto/route.ts` | 2 hours |
| 13 | Test tool execution from Next.js | — | 2 hours |

### Phase 13c: Chat API Tool-Calling Loop (2–3 days)

| Step | Task | Files | Est. Time |
|------|------|-------|-----------|
| 14 | Add `tool_call` and `tool_result` SSE event types | `frontend/src/app/api/chat/route.ts` | 1 hour |
| 15 | Add tool definitions to Nemotron API call | `frontend/src/app/api/chat/route.ts` | 2 hours |
| 16 | Implement tool-calling loop (max 3 iterations) | `frontend/src/app/api/chat/route.ts` | 4 hours |
| 17 | Parse `tool_calls` from Nemotron SSE stream | `frontend/src/app/api/chat/route.ts` | 3 hours |
| 18 | Execute tools via Modal endpoints | `frontend/src/app/api/chat/route.ts` | 2 hours |
| 19 | Feed tool results back to Nemotron | `frontend/src/app/api/chat/route.ts` | 2 hours |
| 20 | Persist tool calls to database (metadata field) | `frontend/src/app/api/chat/route.ts` | 1 hour |
| 21 | End-to-end test: ask splicing question → SpliceAI called | — | 2 hours |

### Phase 13d: Frontend Tool Call UI (1–2 days)

| Step | Task | Files | Est. Time |
|------|------|-------|-----------|
| 22 | Create `ChatToolCallBlock` component | `frontend/src/components/chat/chat-tool-call-block.tsx` | 2 hours |
| 23 | Update `ChatMessageData` interface with `toolCalls` | `frontend/src/components/chat/chat-message.tsx` | 30 min |
| 24 | Render tool call blocks in `ChatMessage` | `frontend/src/components/chat/chat-message.tsx` | 2 hours |
| 25 | Update `ChatSidePanel` SSE parsing for `tool_call` events | `frontend/src/components/chat/chat-side-panel.tsx` | 3 hours |
| 26 | Add tool call styling (blue card, expandable details) | `frontend/src/styles/globals.css` | 1 hour |
| 27 | End-to-end test: tool call renders in chat UI | — | 1 hour |

### Phase 13e: Polish + Testing (1 day)

| Step | Task | Files | Est. Time |
|------|------|-------|-----------|
| 28 | Add error handling for tool failures | `frontend/src/app/api/chat/route.ts` | 2 hours |
| 29 | Add timeout handling (60s for CPU, 5min for GPU) | `frontend/src/app/api/chat/route.ts` | 1 hour |
| 30 | Add tool call count limit (max 3 per turn) | `frontend/src/app/api/chat/route.ts` | 30 min |
| 31 | Test all 5 scenarios from §3.2 | — | 3 hours |
| 32 | Update `implementation.md` with tool-calling section | `backend/design_assistant/implementation.md` | 1 hour |

**Total estimated time: 8–12 days**

---

## 10. File Structure

```
backend/
├── proto_tools/                    ← NEW directory
│   ├── __init__.py
│   ├── modal_deploy_lite.py        ← CPU container (Tier 1 + 2 tools)
│   ├── modal_deploy_gpu.py         ← GPU container (Tier 3 + 4 tools)
│   ├── tool_definitions.py         ← Python tool schemas for proto-tools
│   └── README.md                   ← Setup + deployment instructions

frontend/
├── src/
│   ├── lib/
│   │   ├── chat-tools.ts           ← NEW: Nemotron function definitions
│   │   └── proto-tools-router.ts   ← NEW: CPU/GPU endpoint routing
│   ├── app/
│   │   └── api/
│   │       ├── chat/
│   │       │   └── route.ts        ← MODIFY: add tool-calling loop
│   │       └── tools/
│   │           └── proto/
│   │               └── route.ts    ← NEW: proxy to Modal endpoints
│   └── components/
│       └── chat/
│           ├── chat-message.tsx           ← MODIFY: render tool calls
│           ├── chat-tool-call-block.tsx   ← NEW: tool call display
│           └── chat-side-panel.tsx        ← MODIFY: parse tool_call events
```

---

## 11. Cost Analysis

### 11.1 Modal CPU Container (`proto-tools-lite`)

| Metric | Value |
|--------|-------|
| CPU cost | $0.000031/sec |
| Memory (4GB) | $0.0000035/sec |
| SpliceAI call (10s) | $0.000345 |
| UniProt call (1s) | $0.000035 |
| BLAST call (15s) | $0.000517 |
| 1000 CPU tool calls/month | ~$0.35 |
| **Monthly estimate** | **~$1–3** |

### 11.2 Modal GPU Container (`proto-tools-gpu`)

| Metric | Value |
|--------|-------|
| H100 GPU cost | $0.001/sec |
| Memory (16GB) | $0.000014/sec |
| ESMFold call (60s) | $0.060 |
| ESM2 call (90s) | $0.090 |
| AlphaFold2 call (180s) | $0.180 |
| 100 GPU tool calls/month | ~$9 |
| **Monthly estimate** | **~$5–15** |

### 11.3 Total Monthly Cost

| Service | Cost |
|---------|------|
| proto-tools-lite (CPU) | ~$1–3 |
| proto-tools-gpu (H100) | ~$5–15 |
| NVIDIA API (Nemotron) | Free (research access) |
| **Total** | **~$6–18/month** |

---

## 12. Testing Plan

### 12.1 Unit Tests

| Test | What It Verifies |
|------|-----------------|
| `test_uniprot_fetch` | UniProt API returns protein domains |
| `test_spliceai_score` | SpliceAI returns delta scores for a known variant |
| `test_esmfold` | ESMFold returns PDB structure for a protein sequence |
| `test_tool_routing` | CPU tools go to CPU endpoint, GPU tools go to GPU endpoint |
| `test_tool_definitions` | All tool definitions have valid JSON schema |

### 12.2 Integration Tests

| Test | What It Verifies |
|------|-----------------|
| `test_chat_with_spliceai` | User asks about splicing → Nemotron calls SpliceAI → result in response |
| `test_chat_with_uniprot` | User asks about domains → Nemotron calls UniProt → result in response |
| `test_chat_with_esmfold` | User asks for structure → Nemotron calls ESMFold → result in response |
| `test_tool_failure` | Tool fails → Nemotron handles gracefully |
| `test_tool_budget` | Nemotron tries >3 tools → loop stops at 3 |
| `test_no_tool_needed` | Question answerable from report → no tool called |

### 12.3 End-to-End Scenarios

| Scenario | User Question | Expected Tool Call | Expected Response |
|----------|--------------|--------------------|--------------------|
| Splicing | "Does this variant affect splicing?" | SpliceAI | Delta scores + interpretation |
| Domains | "What domains does BRCA1 have?" | UniProt | Domain list + variant impact |
| Structure | "Predict the mutant structure" | ESMFold | pLDDT + structural change |
| Homology | "Find similar variants in other genes" | BLAST | Homolog list + conservation |
| Constraint | "How constrained is this protein region?" | ESM2 | Log-likelihood + interpretation |
| No tool | "What is a VUS?" | None | Direct answer from knowledge |

---

## 13. Acceptance Criteria

### Phase 13 — Tool Calling

- [ ] proto-tools installed and tested locally
- [ ] `proto-tools-lite` (CPU) deployed on Modal
- [ ] `proto-tools-gpu` (H100) deployed on Modal
- [ ] At least 5 tools accessible via Modal endpoints:
  - [ ] UniProt Fetch
  - [ ] SpliceAI Score
  - [ ] ESMFold Prediction
  - [ ] BLAST Search
  - [ ] AlphaFold DB Fetch
- [ ] Tool definitions created for Nemotron (`chat-tools.ts`)
- [ ] `/api/chat` route supports tool-calling loop (max 3 calls)
- [ ] `tool_call` and `tool_result` SSE events implemented
- [ ] `ChatToolCallBlock` component renders tool calls in chat UI
- [ ] Tool call status shows: calling → completed/failed
- [ ] Tool results are expandable (show input + output)
- [ ] Tool failures handled gracefully (no crash, error message shown)
- [ ] Tool budget enforced (max 3 tool calls per turn)
- [ ] CPU tools route to CPU endpoint, GPU tools route to GPU endpoint
- [ ] End-to-end test: "Does this variant affect splicing?" → SpliceAI called
- [ ] End-to-end test: "What domains does this protein have?" → UniProt called
- [ ] TypeScript compilation passes
- [ ] No lint errors in new files
- [ ] Cost < $20/month for typical usage

---

> **Document maintained by Soham Kar**
> GitHub: [https://github.com/soham-kar/cancer-detection_evo2](https://github.com/soham-kar/cancer-detection_evo2)
> Created: 2026-07-05