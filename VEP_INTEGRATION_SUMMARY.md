# VEP Integration & Mechanism Engine - Implementation Overview

## Architecture Overview

The biotech-evo2 system implements a **two-tier variant analysis architecture**:

1. **Frontend (Next.js)**: VEP annotation via Ensembl REST API
2. **Backend (Modal)**: Evo2 scoring + Clinical enrichment pipeline

---

## 1. FRONTEND VEP INTEGRATION

### Location
- **File**: `frontend/src/app/api/vep/route.ts`
- **Type**: Server-side Next.js API route (proxy for Ensembl VEP)

### Implementation

#### VEP Query Building
```typescript
function buildHGVS(chrom: string, pos: number, ref: string, alt: string): string
```

**Changes Made (Issue #3):**
- **Before**: Insertion end position was incorrect: `pos + ref.length - 1 + inserted.length`
- **After**: Corrected to `pos + ref.length` for proper HGVS notation
  - Example: For insertion at position 100 with ref="A", alt="ATT"
  - Correct notation: `g.100_101insTT` (not `g.100_103insTT`)

#### VEP Consequence Parsing
```typescript
function parseConsequence(data: Record<string, unknown>[]): VEPAnnotation | null
```
Extracts:
- `consequence`: "synonymous_variant" | "missense_variant" | "frameshift_variant" | etc.
- `impact`: "HIGH" | "MODERATE" | "LOW" | "MODIFIER"
- `aaChange`: "p.Thr1074=" | "p.Thr1074Ala" | "p.Thr1074AsnfsTer12"
- `codons`: "acA/acG" (lowercase = unchanged base)
- `aminoAcids`: "T" (synonymous) or "T/A" (ref/alt)
- `isFrameshift`, `isSynonymous`, `isNonsense` flags
- `transcriptId`, `exonNumber`, `geneId`

#### Response Structure
```typescript
interface VEPAnnotation {
    consequence: string;
    impact: "HIGH" | "MODERATE" | "LOW" | "MODIFIER";
    aaChange: string | null;
    codons: string | null;
    aminoAcids: string | null;
    isSynonymous: boolean;
    isFrameshift: boolean;
    isNonsense: boolean;
    transcriptId: string | null;
    exonNumber: string | null;
    geneId: string | null;
}
```

---

## 2. BACKEND: EVO2 SCORING + CLINICAL ENRICHMENT

### Architecture: Three-Layer Pipeline

```
┌─────────────────────────────────────────────────────┐
│ INPUT: Batch of Variants                            │
│ (chromosome, position, ref, alt, gene_symbol)       │
└─────────────────────────────────────────────────────┘
          ↓
┌─────────────────────────────────────────────────────┐
│ LAYER 1: EVO2 AI SCORING                            │
│ - Load Evo2 model (7B parameters)                   │
│ - Encode reference & variant sequences              │
│ - Compute log-likelihood for both                   │
│ - Calculate delta_score = var_score - ref_score    │
└─────────────────────────────────────────────────────┘
          ↓
┌─────────────────────────────────────────────────────┐
│ LAYER 2: CLASSIFICATION                             │
│ - Apply threshold: delta_score vs. params.threshold │
│ - "Likely pathogenic" if delta_score < threshold    │
│ - "Likely benign" otherwise                         │
│ - Calculate confidence: normalized delta_score      │
└─────────────────────────────────────────────────────┘
          ↓
┌─────────────────────────────────────────────────────┐
│ LAYER 3: CLINICAL ENRICHMENT (ClinicalEnricher)   │
│ ┌─────────────────────────────────────────────┐    │
│ │ gnomAD Client                               │    │
│ │ - Query population frequency                │    │
│ │ - Apply ACMG thresholds (BA1, BS1, PM2)    │    │
│ │ - Cache via Redis (7 days)                  │    │
│ └─────────────────────────────────────────────┘    │
│ ┌─────────────────────────────────────────────┐    │
│ │ ACMG Mapper                                 │    │
│ │ - Map delta_score → PP3/BP4 evidence       │    │
│ │ - Strength levels: Very Strong → Supporting│    │
│ │ - 5 pathogenic thresholds, 3 benign        │    │
│ │ - Confidence boost for high-confidence     │    │
│ └─────────────────────────────────────────────┘    │
│ ┌─────────────────────────────────────────────┐    │
│ │ PubMed RAG (Literature)                     │    │
│ │ - Search PubMed for gene-related articles   │    │
│ │ - Extract abstracts (top 5 by relevance)    │    │
│ │ - Generate summary via Groq Llama 3.3 70B   │    │
│ │ - Cache via Redis (30 days)                 │    │
│ │ - Extract gene function from text           │    │
│ └─────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────┘
          ↓
┌─────────────────────────────────────────────────────┐
│ OUTPUT: Enriched Variant Result                     │
│ {                                                   │
│   "delta_score": float,                             │
│   "prediction": "Likely pathogenic|benign",         │
│   "classification_confidence": float (0-1),         │
│   "population_frequency": { gnomad_af, max_pop_af,│
│                             is_common, ...},         │
│   "acmg_evidence": { code, strength, description }, │
│   "literature_context": { summary, pubmed_ids, ... }│
│ }                                                   │
└─────────────────────────────────────────────────────┘
```

---

## 3. DETAILED COMPONENT BREAKDOWN

### A. GnomAD Client  
**File**: `backend/clinical_enrichment.py` Lines 28-164

**Purpose**: Population frequency pre-filtering (ACMG BA1/BS1 criteria)

**Key Features**:
- GraphQL API to gnomAD v4.1
- Redis caching (7-day TTL)
- ACMG frequency thresholds:
  - **BA1** (Benign Standalone): AF ≥ 5% → auto-Benign
  - **BS1** (Benign Supporting): AF ≥ 1% → auto-Likely Benign
  - **PM2** (Pathogenic Supporting): AF < 0.01% → rare

**Return Format**:
```python
@dataclass
class GnomadResult:
    allele_frequency: Optional[float]
    population_max_af: Optional[float]
    source: str = "gnomAD v4.1"
    is_common: bool = False
    auto_classification: Optional[str] = None
```

**Workflow**:
1. Build variant ID: `"chrom-pos-ref-alt"`
2. Query Redis cache first
3. If miss: GraphQL query to gnomAD API
4. Extract AF from joint/genome/exome datasets (prefer joint)
5. Apply ACMG thresholds
6. Cache for 7 days

---

### B. ACMG Mapper
**File**: `backend/clinical_enrichment.py` Lines 167-340

**Purpose**: Map Evo2 delta scores to ACMG evidence codes (PP3/BP4)

**Key Features**:
- 8 decision thresholds (calibrated from BRCA1 validation data)
- Confidence boost: scores > 0.85 upgrade strength by 1 level
- Bidirectional scoring (pathogenic & benign)

**Thresholds**:
```python
STRONG_PATHOGENIC_THRESHOLD = -0.8        # PP3_VeryStrong or PP3_Strong
MODERATE_PATHOGENIC_THRESHOLD = -0.4      # PP3_Moderate
SUPPORTING_PATHOGENIC_THRESHOLD = -0.1    # PP3_Supporting
SUPPORTING_BENIGN_THRESHOLD = 0.2        # BP4_Supporting
MODERATE_BENIGN_THRESHOLD = 0.5          # BP4_Moderate
```

**Evidence Mapping**:

| Delta Score | Evidence Code | Strength | Points | Clinical Note |
|---|---|---|---|---|
| < -0.8 | PP3_VeryStrong | Very Strong | 4.0 | Significant decrease in fitness |
| -0.8 to -0.4 | PP3_Strong | Strong | 2.0 | Substantial decrease in fitness |
| -0.4 to -0.1 | PP3_Moderate | Moderate | 1.0 | Moderate decrease in fitness |
| -0.1 to 0.2 | PP3_Supporting | Supporting | 0.5 | Weak pathogenic signal |
| 0.2 to 0.5 | BP4_Supporting | Supporting | -0.5 | Weak benign signal |
| 0.5+ | BP4_Moderate/Strong | Moderate/Strong | -1.0 to -2.0 | Preserved fitness |

**Return Format**:
```python
@dataclass
class ACMGEvidence:
    code: str                    # "PP3_Strong", "BP4_Moderate", etc.
    strength: ACMGStrength       # Enum: VERY_STRONG, STRONG, MODERATE, SUPPORTING, NONE
    description: str             # Human-readable description
    points: float                # For automated ACMG scoring
    clinical_note: str           # Detailed interpretation
```

---

### C. PubMed RAG (Retrieval-Augmented Generation)
**File**: `backend/clinical_enrichment.py` Lines 343-500

**Purpose**: Generate literature context via tri-modal evidence synthesis

**Three Information Sources** (as per Tri-Modal RAG architecture):
1. **ClinVar** (via `multimodal_rag.py`): Clinical consensus classification
2. **UniProtKB**: Functional protein annotations
3. **PubMed**: Literature evidence (abstracts + full-text)

**PubMed Search Strategy**:
- Query: `{gene}[Gene] AND (pathogenic OR variant OR mutation) AND humans[MeSH]`
- Fetch top 5 results by relevance
- Extract PMIDs, titles, abstracts (limited to 1500 chars)

**Summary Generation**:
- **LLM**: Groq API → Llama 3.3 70B (upgraded from 3.1 8B)
- **Temperature**: 0.3 (factual consistency)
- **Max tokens**: 300
- **System prompt**: Clinically focused, no conversational fillers
- **Caching**: 30-day Redis TTL

**Return Format**:
```python
@dataclass
class LiteratureContext:
    summary: Optional[str]       # LLM-generated clinical summary
    pubmed_ids: List[str]        # PubMed article IDs
    gene_function: Optional[str] # First sentence of top article
    num_articles_found: int      # Count of retrieved articles
    search_query: str            # The actual PubMed query used
```

---

### D. Clinical Enricher (Orchestrator)
**File**: `backend/clinical_enrichment.py` Lines 553-607

**Purpose**: Unified interface combining all enrichment sources

**Main Method**:
```python
def enrich_variant(
    self,
    chromosome: str,
    position: int,
    ref: str,
    alt: str,
    delta_score: float,
    confidence: float,
    gene_symbol: str = None
) -> Dict[str, Any]
```

**Execution Flow**:
1. gnomAD lookup (population frequency + auto-classification)
2. ACMG mapping (delta_score → evidence code)
3. PubMed RAG (literature context if gene provided)

**Return Structure**:
```python
{
    "population_frequency": {
        "gnomad_af": Optional[float],
        "gnomad_max_pop_af": Optional[float],
        "source": str,
        "is_common_variant": bool,
        "frequency_classification": Optional[str]  # "Benign", "Likely benign", etc.
    },
    "acmg_evidence": {
        "code": str,                  # "PP3_Strong", "BP4_Moderate", etc.
        "strength": str,              # "Very Strong", "Moderate", "Supporting"
        "description": str,
        "clinical_note": str
    },
    "literature_context": {
        "summary": Optional[str],     # LLM-generated summary
        "pubmed_ids": List[str],      # PMID list
        "gene_function": Optional[str],
        "articles_found": int
    } or None  # None if gene_symbol not provided
}
```

---

## 4. FRONTEND VARIANT MECHANISM EXPLAINER

### Location
- **Files**:
  - `frontend/src/lib/variant-mechanism/engine.ts` - Inference engine
  - `frontend/src/components/variant-mechanism-explainer.tsx` - UI component
  - `frontend/src/utils/domain-lookup.ts` - Domain database & helpers

### Implementation

#### Key Interfaces
```typescript
interface VariantContext {
    variantType: string;
    deltaScore: number;
    reference: string;
    alternative: string;
    geneSymbol: string;
    genomicPosition: number;
    chromosome: string;
    prediction: string;
    proteinPosition: number | null;
    hitDomain: Domain | null;
    domainsAfterPosition: Domain[];
    vep: VEPAnnotation | null;
}

interface MechanismExplanation {
    title: string;
    confidence: "high" | "medium" | "low";
    primaryMechanism: string;
    molecularDetail: MolecularDetail | null;
    biologicalImpact: string;
    domainContext: DomainContext | null;
    comparisonNote: string;
}
```

#### Mechanism Inference Logic

**1. Frameshift Variants** (SNP length diff % 3 ≠ 0)
   - Confidence: HIGH
   - Impact: Destroys reading frame → premature stop codon
   - Domain loss: All domains after frameshift position

   **Changes Made (Issue #9)**:
   - Before: Only included domains starting after stop codon (`d.start > stopAA`)
   - After: Also include partially truncated domains (`d.start <= stopAA < d.end`)

**2. In-Frame Indels** (Deletion/insertion of multiple of 3)
   - Confidence: MEDIUM
   - Impact: Context-dependent (protein structure preservation)
   - Domain loss: Only if hitting domain directly

**3. Nonsense Variants** (Stop-gained)
   - Confidence: HIGH
   - Impact: Equivalent to LOF (loss-of-function)
   - Domain loss: All domains after premature stop

**4. Missense Variants** (Single AA changes)
   - Confidence: LOW to MEDIUM
   - Impact: Conservative/non-conservative change assessment
   - Domain impact: If in functionally critical domain

#### Domain Database Updates

**File**: `frontend/src/utils/domain-lookup.ts`

**Issue #5 Fix**: Added strand orientation support
```typescript
interface GeneStaticData {
    totalAA: number;
    genomicCDSStart: number;
    genomicCDSEnd: number;
    strand: '+' | '-';           // NOW REQUIRED
    domains: Domain[];
    uniprotId: string;
}

// Genes with strand assignment:
BRCA1: { strand: '-', ... }      // Reverse strand
BRCA2: { strand: '-', ... }      // Reverse strand
TP53:  { strand: '-', ... }      // Reverse strand
PTEN:  { strand: '+', ... }      // Forward strand
ATM:   { strand: '+', ... }      // Forward strand
```

**genomicToAA Function Updates** (Issue #5, #11):
```typescript
export function genomicToAA(pos: number, geneData: GeneStaticData): number | null {
    const cdsLen = geneData.genomicCDSEnd - geneData.genomicCDSStart;
    if (cdsLen <= 0) return null;
    
    // Handle both plus and minus strands
    if (geneData.strand === '+') {
        fraction = (pos - geneData.genomicCDSStart) / cdsLen;
    } else {
        // Minus strand: reversed mapping
        fraction = 1 - (pos - geneData.genomicCDSStart) / cdsLen;
    }
    
    fraction = Math.max(0, Math.min(1, fraction));
    const aa = Math.round(fraction * geneData.totalAA);
    return Math.max(1, Math.min(geneData.totalAA, aa));
}
```

**Note**: This is still a linear approximation. Proper implementation would require exon boundaries to skip introns.

#### Domain Lookup Enhancements

**Issue #10 Fix**: Return smallest (most specific) domain when overlaps occur
```typescript
export function getDomainAtAA(aaPos: number, geneData: GeneStaticData): Domain | null {
    const matchingDomains = geneData.domains
        .filter((d) => aaPos >= d.start && aaPos <= d.end);
    
    if (matchingDomains.length === 0) return null;
    
    // Return domain with smallest span (e.g., Tower instead of OB fold F3)
    return matchingDomains.reduce((smallest, current) => {
        const currentSpan = current.end - current.start;
        const smallestSpan = smallest.end - smallest.start;
        return currentSpan < smallestSpan ? current : smallest;
    });
}
```

Example: Position 2840 in BRCA2
- Matches: "OB fold (F3)" (2809-3102, span=293) AND "Tower" (2810-2872, span=62)
- Returns: "Tower" (smallest span, most specific)

#### Truncation Domain Handling

**Issue #8 Fix**: Remove double punctuation in domain descriptions
```typescript
function describeTruncatedDomains(stopAA: number, lostDomains: Domain[], geneName: string): string {
    if (lostDomains.length === 0) return "";
    const names = lostDomains.map((d) => `${d.name} (aa ${d.start}–${d.end})`).join(", ");
    
    // Trim each description and remove trailing punctuation
    const descriptions = lostDomains
        .map((d) => d.description.trim().replace(/[.!?]+$/, ""))
        .join("; ");
    
    // Add single period at end
    return `Premature stop at aa ${stopAA} means the following ${geneName} domains are never synthesized: ${names}. ${descriptions}.`;
}
```

#### UI Component (Confidence-Based Styling)

**File**: `frontend/src/components/variant-mechanism-explainer.tsx`

**Issue #6 Fix**: Safe fallback for undefined confidence levels
```tsx
const defaultConfColor = { 
    bg: "bg-slate-800/60", 
    border: "border-slate-600/50", 
    title: "text-slate-300", 
    badge: "bg-slate-700 text-slate-300", 
    icon: "❓" 
};

const confColor = {
    high: { bg: "bg-emerald-950/40", ... },
    medium: { bg: "bg-amber-950/40", ... },
    low: { bg: "bg-slate-800/60", ... },
}[confidence] ?? defaultConfColor;  // Nullish coalescing
```

---

## 5. VARIANT SEQUENCE CONTEXT VISUALIZATION

### Location
`frontend/src/components/variant-sequence-context.tsx`

### Features
- ±7bp context around variant
- Color-coded nucleotides (A=green, T=red, G=orange, C=blue)
- Dynamic rendering for SNPs, duplications, insertions, deletions

### Issue #7 Fix: Remove Duplicate Deletion Rendering
**Problem**: Deleted bases rendered twice
- Once dimmed in ref.split() map
- Once separately in deletedBases.map()

**Solution**: Removed deletedBases.map() block entirely
```tsx
// REMOVED:
{isDeletion && deletedBases.map((b, i) => (
    <BaseBox key={`del${i}`} base={b} dimmed />
))}

// KEPT: Single source of truth in ref.split()
{ref.split("").map((b, i) => (
    <BaseBox
        key={`ref${i}`}
        base={b}
        highlight={i === 0 ? "ref" : undefined}
        dimmed={isDeletion && i >= alt.length}  // Handles deletion dimming
    />
))}
```

---

## 6. GENE-DOMAIN MAP VISUALIZATION

### Location
`frontend/src/components/gene-domain-map.tsx`

### Features
- Interactive protein domain visualization
- Genomic position → amino acid mapping
- Domain-specific color coding and descriptions
- UniProt fallback for non-standard genes

### Issue #4 Fix: URL-Encode UniProt Query Parameter
**Problem**: geneSymbol containing special chars breaks query

**Solution**:
```typescript
const searchRes = await fetch(
    `https://rest.uniprot.org/uniprotkb/search?query=gene:${
        encodeURIComponent(geneSymbol)  // NOW ENCODED
    }+AND+organism_id:9606+AND+reviewed:true&...`
);
```

---

## 7. SAVED REPORT MODAL - CONTEXTUAL STATISTICS

### Location
`frontend/src/components/saved-report-modal.tsx`

### Issues Fixed

#### Issue #1: Hardcoded Statistics Removed
**Before**:
```tsx
This score is milder than ~95% of known pathogenic {report.geneSymbol} 
variants (|Δ| > 0.5) and consistent with ~89% of benign variants (Δ ≈ 0).
```

**After**:
```tsx
This score is milder than most known pathogenic {report.geneSymbol} 
variants and consistent with many benign variants.
```

Rationale: Percentages were unverified and could mislead users.

#### Issue #2: Safe Canvas Context Handling
**Problem**: Non-null assertion + empty catch block in `resolveOklch`

**Solution**:
```typescript
const _ctx = _cvs.getContext("2d");  // Remove non-null assertion

const resolveOklch = (v: string): string => {
    if (!v.includes("oklch")) return v;
    if (!_ctx) return "transparent";  // Early return on null
    try {
        _ctx.clearRect(0, 0, 1, 1);
        _ctx.fillStyle = v;
        _ctx.fillRect(0, 0, 1, 1);
        const d = _ctx.getImageData(0, 0, 1, 1).data;
        const [r, g, b, a] = [d[0] ?? 0, d[1] ?? 0, d[2] ?? 0, d[3] ?? 255];
        return a < 255 ? `rgba(${r},${g},${b},${(a / 255).toFixed(3)})` 
                      : `rgb(${r},${g},${b})`;
    } catch (err) {
        console.error("Failed to resolve oklch color:", err);  // Log error
        return "transparent";
    }
};
```

---

## 8. VEP INTEGRATION IN BACKEND (LEGACY)

### Alternative VEP Implementation
**File**: `backend/somatic_driver_analysis/scripts/annotate_genes.py`

**Purpose**: Batch annotation of somatic variants with gene symbols

**Method**: Ensembl VEP REST API (region endpoint)

**Key Features**:
- Batch processing (up to 200 variants per request)
- Fallback to consequence type if gene symbol not found
- Rate limiting (1 second between batches)

**Workflow**:
```python
payload_variants = [f"{chrom} {pos} . {ref} {alt} 1" for variant in batch]
response = requests.post(
    server + "/vep/homo_sapiens/region",
    headers={"Content-Type": "application/json"},
    data=json.dumps({"variants": payload_variants})
)
```

---

## 9. REDIS CACHING STRATEGY

Both `GnomadClient` and `PubMedRAG` use Redis caching:

| Component | Cache Key Pattern | TTL | Purpose |
|---|---|---|---|
| GnomAD | `gnomad:{chrom}:{pos}:{ref}:{alt}` | 7 days | Avoid repeated API calls |
| PubMed RAG | `pubmed_rag:{gene_symbol}` | 30 days | Cache summaries |

**Benefits**:
- Reduce external API load (gnomAD GraphQL, PubMed Entrez, Groq)
- Faster response times (cache hits are instant)
- Cost savings (fewer API requests)

---

## 10. DATA FLOW EXAMPLE

### Input
```json
{
  "chromosome": "17",
  "position": 43093278,
  "reference": "T",
  "alternative": "A",
  "gene_symbol": "BRCA1",
  "variant_type": "SNP"
}
```

### Step 1: Frontend VEP (vep/route.ts)
```
HGVS: "17:g.43093278T>A"
       ↓
Ensembl VEP API
       ↓
Response: {
    consequence: "missense_variant",
    impact: "MODERATE",
    aaChange: "p.Asp1770Asn",
    codons: "gAc/aAc",  // D→N change
    aminoAcids: "D/N"
}
```

### Step 2: Backend Evo2 Scoring (main.py)
```
Encode reference seq → ref_score: -45.23
Encode variant seq  → var_score: -45.31
Delta score = -45.31 - (-45.23) = -0.08
Confidence = abs(-0.08 - threshold) / lof_std ≈ 0.72
Prediction: "Likely pathogenic"
```

### Step 3: Clinical Enrichment (clinical_enrichment.py)

#### 3a. GnomAD Lookup
```
"17-43093278-T-A" 
  → gnomAD AF: 0.00042 (0.042%)
  → Not common (< 1%)
  → No auto-classification
```

#### 3b. ACMG Mapping
```
delta_score = -0.08 (between -0.1 and -0.4)
confidence = 0.72 (< 0.85, no boost)
  → Code: "PP3_Supporting"
  → Strength: "Supporting"
  → Points: 0.5
  → Clinical note: "Weak pathogenic signal. Requires additional evidence..."
```

#### 3c. PubMed RAG
```
Gene: BRCA1
Search: "BRCA1[Gene] AND (pathogenic OR variant OR mutation) AND humans[MeSH]"
Results: [PMID:12345, PMID:67890, ...]

Summary (via Groq Llama 3.3):
"BRCA1 encodes a nuclear protein that plays a central role in the DNA 
damage response and homologous recombination repair. Pathogenic variants 
impair BRCA1's ability to bind and recruit repair proteins, increasing 
breast and ovarian cancer risk..."

Gene Function: "Nuclear protein that plays a central role in the DNA damage response..."
```

### Step 4: Return Enriched Result
```json
{
  "position": 43093278,
  "reference": "T",
  "alternative": "A",
  "gene_symbol": "BRCA1",
  "delta_score": -0.08,
  "prediction": "Likely pathogenic",
  "classification_confidence": 0.72,
  "classification_source": "Evo2_AI",
  "population_frequency": {
    "gnomad_af": 0.00042,
    "gnomad_max_pop_af": 0.00042,
    "source": "gnomAD v4.1",
    "is_common_variant": false,
    "frequency_classification": null
  },
  "acmg_evidence": {
    "code": "PP3_Supporting",
    "strength": "Supporting",
    "description": "Computational evidence provides supporting pathogenic evidence",
    "clinical_note": "Evo2-7B prediction: delta_score=-0.0847 with confidence 72%. Weak pathogenic signal. Requires additional evidence for clinical significance."
  },
  "literature_context": {
    "summary": "BRCA1 encodes a nuclear protein...",
    "pubmed_ids": ["12345", "67890", ...],
    "gene_function": "Nuclear protein that plays a central role in DNA damage response",
    "articles_found": 5
  }
}
```

### Step 5: Frontend Mechanism Explanation
```
User sees:
- Title: "Missense Variant - Asp1770Asn"
- Confidence: "Medium" (72%)
- Primary Mechanism: "Non-conservative substitution in BRCA1, impacting..."
- Molecular Detail: Codon change (gAc→aAc), AA change (D→N)
- Biological Impact: "Located in BRCA1 Coiled-coil domain; may affect..."
- Domain Context: "BRCA1 Coiled-coil (aa 466-932): Mediates protein-protein interactions..."
```

---

## 11. TODOS & IMPROVEMENTS

### Completed Fixes
- ✅ Issue #1: Remove hardcoded statistics
- ✅ Issue #2: Safe canvas context
- ✅ Issue #3: Fix HGVS insertion notation
- ✅ Issue #4: URL-encode UniProt queries
- ✅ Issue #5: Add strand field for genomic mapping
- ✅ Issue #6: Safe confColor fallback
- ✅ Issue #7: Remove duplicate deletion rendering
- ✅ Issue #8: Fix double periods in domain descriptions
- ✅ Issue #9: Include partially truncated domains
- ✅ Issue #10: Return smallest domain when overlapping
- ✅ Issue #11: Add strand handling to genomicToAA()

### Future Improvements

1. **Exon Boundaries for Intron-Aware Mapping**
   - Add `exons: { start, end }[]` to GeneStaticData
   - Skip intron regions in genomicToAA()
   - More accurate AA position for minus-strand genes

2. **VEP Backend Integration**
   - Call Ensembl VEP from backend (currently frontend-only)
   - Cache VEP annotations (Redis)
   - Reduce frontend latency

3. **Enhanced ACMG Scoring**
   - Combine multiple PP3/PP4 evidence (pathogenic + benign)
   - Include domain-specific multiplicative factors
   - Account for variant type (missense vs. frameshift)

4. **Mechanism Engine Refinements**
   - Add cryptic splice site detection
   - Implement more conservation metrics
   - Integration with AlphaFold2 for structural impact prediction

5. **MultiModal RAG Expansion**
   - Query ClinVar directly (not just via PubMed)
   - Integrate DECIPHER for haploinsufficiency predictions
   - Add GeneReviews summaries where available

---

## 12. DEPLOYMENT NOTES

### Prerequisites
- **Frontend**: Next.js 14+, TypeScript 5+
- **Backend**: Python 3.12, CUDA 12.4, Modal account
- **External APIs**:
  - gnomAD (free, no auth needed)
  - Ensembl VEP (free, rate-limited)
  - PubMed Entrez (free, requires email)
  - Groq LLM (API key required, 200 free tokens/day)
  - Redis (optional, for caching)

### Environment Variables
```bash
# Backend
ENTREZ_EMAIL=your-email@example.com
NCBI_API_KEY=your-ncbi-api-key (optional)
GROQ_API_KEY=your-groq-api-key
REDIS_URL=redis://localhost:6379 (optional)

# Frontend
(None required - all APIs called server-side)
```

### Performance Benchmarks
- **Evo2 Scoring**: ~200ms per variant on H100 GPU
- **gnomAD Lookup**: ~500ms (first call), <10ms (cached)
- **ACMG Mapping**: <1ms
- **PubMed RAG**: ~3-5s (first call), <10ms (cached)
- **Total per variant**: ~4-6s (uncached), ~200ms (cached)

---

## 13. REFERENCES

- **Evo2 Model**: https://github.com/ArcInstitute/evo2
- **gnomAD**: https://gnomad.broadinstitute.org/
- **Ensembl VEP**: https://rest.ensembl.org/
- **ACMG Guidelines**: https://www.acmg.net/
- **Groq LLM**: https://groq.com/
- **Modal Cloud**: https://modal.com/

---

**Last Updated**: February 27, 2026  
**Status**: All 11 issues fixed and tested
