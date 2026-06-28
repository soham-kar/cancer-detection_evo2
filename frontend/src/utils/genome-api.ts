import { Viaoda_Libre } from "next/font/google";
import { env } from "~/env";

export interface GenomeAssemblyFromSearch {
  id: string;
  name: string;
  sourceName: string;
  active: boolean;
}

export interface ChromosomeFromSeach {
  name: string;
  size: number;
}

export interface GeneFromSearch {
  symbol: string;
  name: string;
  chrom: string;
  description: string;
  gene_id?: string;
}

export interface GeneDetailsFromSearch {
  genomicinfo?: {
    chrstart: number;
    chrstop: number;
    strand?: string;
  }[];
  summary?: string;
  organism?: {
    scientificname: string;
    commonname: string;
  };
}

export interface GeneBounds {
  min: number;
  max: number;
}

export interface ClinvarVariant {
  clinvar_id: string;
  title: string;
  variation_type: string;
  classification: string;
  gene_sort: string;
  chromosome: string;
  location: string;
  evo2Result?: {
    prediction: string;
    delta_score: number;
    classification_confidence: number;
    // Clinical enrichment fields
    population_frequency?: PopulationFrequency;
    acmg_evidence?: ACMGEvidence;
    literature_context?: LiteratureContext;
  };
  isAnalyzing?: boolean;
  evo2Error?: string;
}

export interface PopulationFrequency {
  gnomad_af: number | null;
  gnomad_max_pop_af: number | null;
  source: string;
  is_common_variant: boolean;
}

export interface ACMGEvidence {
  code: string | null;
  strength: string | null;
  description: string;
  clinical_note: string;
}

export interface LiteratureContext {
  summary: string | null;
  pubmed_ids: string[];
  gene_function: string | null;
  articles_found: number;
}

export interface EvidenceConfidence {
  vep: { available: boolean; confidence: string; note: string };
  evo2: { available: boolean; confidence: string; note: string };
  gnomad: { available: boolean; confidence: string; note: string };
  clinvar: { available: boolean; confidence: string; note: string };
  uniprot: { available: boolean; confidence: string; note: string };
  pubmed: { available: boolean; confidence: string; note: string };
  overall: {
    level: string;
    sources_available: string;
    high_confidence_sources: string;
  };
}

// ─── ISM (In-Silico Mutagenesis) Scan Types ─────────────────────────────

export interface ISMAlternativeScore {
  delta: number;
  direction: "pathogenic" | "benign" | "neutral";
  magnitude: number;
}

export interface ISMPositionData {
  genomic_position: number | null;
  relative_position: number;
  reference: string;
  alternatives: Record<string, ISMAlternativeScore>;
  max_delta: number;
  is_constrained: boolean;
}

export interface ISMScanSummary {
  constrained_positions: number;
  total_positions_scanned: number;
  constraint_zone: "high" | "moderate" | "low";
  peak_constraint_position: number | null;
  peak_constraint_magnitude: number;
  constraint_boundaries: number[];
  scan_duration_ms: number;
}

export interface ISMScanResult {
  scan_radius: number;
  stride: number;
  window_size: number;
  reference_score: number;
  positions: Record<string, ISMPositionData>;
  summary: ISMScanSummary;
}

// ─── XAI Factor Types ──────────────────────────────────────────────────

export interface XAIFactor {
  label: string;
  contribution: number;
  color: string;
  detail: string;
}

export interface XAIFactors {
  factors: XAIFactor[];
  total: number;
}

// ─── Counterfactual Types ──────────────────────────────────────────────

export interface CounterfactualAllele {
  delta: number;
  prediction: string;
  direction: "pathogenic" | "benign" | "neutral";
  magnitude: number;
}

export interface Counterfactuals {
  reference: string;
  alternatives: Record<string, CounterfactualAllele>;
  tolerated_alleles: string[];
  pathogenic_alleles: string[];
  neutral_alleles: string[];
  summary: string;
}

// ─── ACMG Criteria Types ───────────────────────────────────────────────

export interface ACMGCriterion {
  met: boolean;
  strength: string | null;
  rationale: string;
}

export interface ACMGCriteriaResult {
  criteria: Record<string, ACMGCriterion>;
  met_count: number;
  total_evaluated: number;
  strength_counts: {
    very_strong: number;
    strong: number;
    moderate: number;
    supporting: number;
    standalone: number;
  };
  acmg_classification: string;
  classification_rationale: string;
}

// ─── Knowledge Graph Types ──────────────────────────────────────────────

export interface KnowledgeGraphDisease {
  id: string;
  name: string;
  score: number;
}

export interface KnowledgeGraphDrug {
  name: string;
  type: string;
  phase: string;
  mechanism: string;
}

export interface KnowledgeGraph {
  gene: string;
  ensembl_id?: string;
  diseases: KnowledgeGraphDisease[];
  drugs: KnowledgeGraphDrug[];
  clinical_actionability: string | null;
}

// ─── External Scores Types ──────────────────────────────────────────────

export interface CADDScore {
  phred: number;
  raw: number;
  interpretation: string;
}

export interface AlphaMissenseScore {
  score: number;
  confidence: string;
  classification: string;
}

export interface REVELScore {
  score: number;
  interpretation: string;
}

export interface ExternalScores {
  cadd: CADDScore | null;
  revel: REVELScore | null;
  alphamissense: AlphaMissenseScore | null;
  concordance_note: string | null;
}

// ─── ACMG Refined Types ─────────────────────────────────────────────────

export interface ACMGRefinedCriterion {
  met: boolean;
  strength: string | null;
  justification: string;
}

export interface ACMGRefinedResult {
  criteria: Record<string, ACMGRefinedCriterion>;
  acmg_classification: string;
  classification_confidence: string;
  narrative: string;
}

export interface AnalysisResult {
  position: number;
  reference: string;
  alternative: string;
  delta_score: number;
  prediction: string;
  classification_confidence: number;
  classification_source: string;
  // Clinical enrichment fields
  population_frequency?: PopulationFrequency;
  acmg_evidence?: ACMGEvidence;
  literature_context?: LiteratureContext;
  // Multi-modal RAG output
  clinical_summary?: string | null;
  evidence_confidence?: EvidenceConfidence;
  // In-Silico Mutagenesis scan
  ism_scan?: ISMScanResult | null;
  // XAI confidence decomposition (backend-computed)
  xai_factors?: XAIFactors | null;
  // Counterfactual analysis (from ISM position 0)
  counterfactuals?: Counterfactuals | null;
  // ACMG/AMP criteria mapping
  acmg_criteria?: ACMGCriteriaResult | null;
  // LLM-refined ACMG criteria
  acmg_criteria_refined?: ACMGRefinedResult | null;
  // Knowledge graph (gene-disease-drug)
  knowledge_graph?: KnowledgeGraph | null;
  // External scores (CADD, REVEL)
  external_scores?: ExternalScores | null;
  // Legacy raw evidence (for power users)
  clinvar_evidence?: {
    status: string | null;
    review_status: string | null;
    variation_id: string | null;
    num_submitters: number;
    conflicting: boolean;
    summary: string | null;
  } | null;
  protein_context?: {
    accession: string | null;
    protein_name: string | null;
    function: string | null;
    domains: Array<{ name: string; start: number; end: number }>;
    subcellular_location: string | null;
    disease_associations: string[];
  } | null;
}

export async function getAvailableGenomes() {
  const apiUrl = "https://api.genome.ucsc.edu/list/ucscGenomes";
  const response = await fetch(apiUrl);
  if (!response.ok) {
    throw new Error("Failed to fetch genome list from UCSC API");
  }

  const genomeData = await response.json();
  if (!genomeData.ucscGenomes) {
    throw new Error("UCSC API error: missing ucscGenomes");
  }

  const genomes = genomeData.ucscGenomes;
  const structuredGenomes: Record<string, GenomeAssemblyFromSearch[]> = {};

  for (const genomeId in genomes) {
    const genomeInfo = genomes[genomeId];
    const organism = genomeInfo.organism || "Other";

    if (!structuredGenomes[organism]) structuredGenomes[organism] = [];
    structuredGenomes[organism].push({
      id: genomeId,
      name: genomeInfo.description || genomeId,
      sourceName: genomeInfo.sourceName || genomeId,
      active: !!genomeInfo.active,
    });
  }

  return { genomes: structuredGenomes };
}

export async function getGenomeChromosomes(genomeId: string) {
  const apiUrl = `https://api.genome.ucsc.edu/list/chromosomes?genome=${genomeId}`;
  const response = await fetch(apiUrl);
  if (!response.ok) {
    throw new Error("Failed to fetch chromosome list from UCSC API");
  }

  const chromosomeData = await response.json();
  if (!chromosomeData.chromosomes) {
    throw new Error("UCSC API error: missing chromosomes");
  }

  const chromosomes: ChromosomeFromSeach[] = [];
  for (const chromId in chromosomeData.chromosomes) {
    if (
      chromId.includes("_") ||
      chromId.includes("Un") ||
      chromId.includes("random")
    )
      continue;
    chromosomes.push({
      name: chromId,
      size: chromosomeData.chromosomes[chromId],
    });
  }

  // chr1, chr2, ... chrX, chrY
  chromosomes.sort((a, b) => {
    const anum = a.name.replace("chr", "");
    const bnum = b.name.replace("chr", "");
    const isNumA = /^\d+$/.test(anum);
    const isNumB = /^\d+$/.test(bnum);
    if (isNumA && isNumB) return Number(anum) - Number(bnum);
    if (isNumA) return -1;
    if (isNumB) return 1;
    return anum.localeCompare(bnum);
  });

  return { chromosomes };
}

export async function searchGenes(query: string, genome: string) {
  const url = "https://clinicaltables.nlm.nih.gov/api/ncbi_genes/v3/search";
  const params = new URLSearchParams({
    terms: query,
    df: "chromosome,Symbol,description,map_location,type_of_gene",
    ef: "chromosome,Symbol,description,map_location,type_of_gene,GenomicInfo,GeneID",
  });
  const response = await fetch(`${url}?${params}`);
  if (!response.ok) {
    throw new Error("NCBI API Error");
  }

  const data = await response.json();
  const results: GeneFromSearch[] = [];

  if (data[0] > 0) {
    const fieldMap = data[2];
    const geneIds = fieldMap.GeneID || [];
    for (let i = 0; i < Math.min(10, data[0]); ++i) {
      if (i < data[3].length) {
        try {
          const display = data[3][i];
          let chrom = display[0];
          if (chrom && !chrom.startsWith("chr")) {
            chrom = `chr${chrom}`;
          }
          results.push({
            symbol: display[1], // Symbol is at index 1
            name: display[2], // description is at index 2
            chrom,
            description: display[2],
            gene_id: geneIds[i] || "",
          });
        } catch {
          continue;
        }
      }
    }
  }

  return { query, genome, results };
}

export async function fetchGeneDetails(geneId: string): Promise<{
  geneDetails: GeneDetailsFromSearch | null;
  geneBounds: GeneBounds | null;
  initialRange: { start: number; end: number } | null;
}> {
  try {
    const detailUrl = `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=gene&id=${geneId}&retmode=json`;
    const detailsResponse = await fetch(detailUrl);

    if (!detailsResponse.ok) {
      console.error(
        `Failed to fetch gene details: ${detailsResponse.statusText}`,
      );
      return { geneDetails: null, geneBounds: null, initialRange: null };
    }

    const detailData = await detailsResponse.json();

    if (detailData.result && detailData.result[geneId]) {
      const detail = detailData.result[geneId];

      if (detail.genomicinfo && detail.genomicinfo.length > 0) {
        const info = detail.genomicinfo[0];

        const minPos = Math.min(info.chrstart, info.chrstop);
        const maxPos = Math.max(info.chrstart, info.chrstop);
        const bounds = { min: minPos, max: maxPos };

        const geneSize = maxPos - minPos;
        const seqStart = minPos;
        const seqEnd = geneSize > 10000 ? minPos + 10000 : maxPos;
        const range = { start: seqStart, end: seqEnd };

        return { geneDetails: detail, geneBounds: bounds, initialRange: range };
      }
    }

    return { geneDetails: null, geneBounds: null, initialRange: null };
  } catch (err) {
    return { geneDetails: null, geneBounds: null, initialRange: null };
  }
}

export async function fetchGeneSequence(
  chrom: string,
  start: number,
  end: number,
  genomeId: string,
): Promise<{
  sequence: string;
  actualRange: { start: number; end: number };
  error?: string;
}> {
  try {
    const chromosome = chrom.startsWith("chr") ? chrom : `chr${chrom}`;

    const apiStart = start - 1;
    const apiEnd = end;

    const apiUrl = `https://api.genome.ucsc.edu/getData/sequence?genome=${genomeId};chrom=${chromosome};start=${apiStart};end=${apiEnd}`;
    const response = await fetch(apiUrl);
    const data = await response.json();

    const actualRange = { start, end };

    if (data.error || !data.dna) {
      return { sequence: "", actualRange, error: data.error };
    }

    const sequence = data.dna.toUpperCase();

    return { sequence, actualRange };
  } catch (err) {
    return {
      sequence: "",
      actualRange: { start, end },
      error: "Internal error in fetch gene sequence",
    };
  }
}

/**
 * Fetch the single reference nucleotide at a genomic position.
 * Used to resolve "N>A" → "T>A" when user types a position manually.
 */
export async function fetchSingleBase(
  chrom: string,
  position: number,
  genomeId: string,
): Promise<string> {
  try {
    const chromosome = chrom.startsWith("chr") ? chrom : `chr${chrom}`;
    // UCSC API uses 0-based half-open intervals
    const apiUrl = `https://api.genome.ucsc.edu/getData/sequence?genome=${genomeId};chrom=${chromosome};start=${position - 1};end=${position}`;
    const response = await fetch(apiUrl);
    if (!response.ok) return "";
    const data = await response.json();
    if (data.error || !data.dna) return "";
    return data.dna.toUpperCase();
  } catch {
    return "";
  }
}

export async function fetchClinvarVariants(
  chrom: string,
  geneBound: GeneBounds,
  genomeId: string,
): Promise<ClinvarVariant[]> {
  const chromFormatted = chrom.replace(/^chr/i, "");

  const minBound = Math.min(geneBound.min, geneBound.max);
  const maxBound = Math.max(geneBound.min, geneBound.max);

  const positionField = genomeId === "hg19" ? "chrpos37" : "chrpos38";
  const searchTerm = `${chromFormatted}[chromosome] AND ${minBound}:${maxBound}[${positionField}]`;

  const searchUrl =
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi";
  const searchParams = new URLSearchParams({
    db: "clinvar",
    term: searchTerm,
    retmode: "json",
    retmax: "20",
  });

  const searchResponse = await fetch(`${searchUrl}?${searchParams.toString()}`);

  if (!searchResponse.ok) {
    throw new Error("ClinVar search failed: " + searchResponse.statusText);
  }

  const searchData = await searchResponse.json();

  if (
    !searchData.esearchresult ||
    !searchData.esearchresult.idlist ||
    searchData.esearchresult.idlist.length === 0
  ) {
    console.log("No ClinVar variants found");
    return [];
  }

  const variantIds = searchData.esearchresult.idlist;

  const summaryUrl =
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi";
  const summaryParams = new URLSearchParams({
    db: "clinvar",
    id: variantIds.join(","),
    retmode: "json",
  });

  const summaryResponse = await fetch(
    `${summaryUrl}?${summaryParams.toString()}`,
  );

  if (!summaryResponse.ok) {
    throw new Error(
      "Failed to fetch variant details: " + summaryResponse.statusText,
    );
  }

  const summaryData = await summaryResponse.json();
  const variants: ClinvarVariant[] = [];

  if (summaryData.result && summaryData.result.uids) {
    for (const id of summaryData.result.uids) {
      const variant = summaryData.result[id];
      variants.push({
        clinvar_id: id,
        title: variant.title,
        variation_type: (variant.obj_type || "Unknown")
          .split(" ")
          .map(
            (word: string) =>
              word.charAt(0).toUpperCase() + word.slice(1).toLowerCase(),
          )
          .join(" "),
        classification:
          variant.germline_classification.description || "Unknown",
        gene_sort: variant.gene_sort || "",
        chromosome: chromFormatted,
        location: variant.location_sort
          ? parseInt(variant.location_sort).toLocaleString()
          : "Unknown",
      });
    }
  }

  return variants;
}

export async function analyzeVariantWithAPI({
  position,
  alternative,
  genomeId,
  chromosome,
  geneSymbol,
  runISMScan = false,
  ismScanRadius = 20,
  ismScanStride = 1,
}: {
  position: number;
  alternative: string;
  genomeId: string;
  chromosome: string;
  geneSymbol?: string;
  runISMScan?: boolean;
  ismScanRadius?: number;
  ismScanStride?: number;
}): Promise<AnalysisResult> {
  // Route through our credit-protected API
  const url = "/api/analyze";

  console.log("🔬 Calling Credit-Protected Analyze API:", {
    url,
    position,
    alternative,
    genomeId,
    chromosome,
    geneSymbol,
  });

  const requestBody = {
    variant_position: position,
    alternative: alternative,
    genome: genomeId,
    chromosome: chromosome,
    gene_symbol: geneSymbol,
    run_ism_scan: runISMScan,
    ism_scan_radius: ismScanRadius,
    ism_scan_stride: ismScanStride,
  };

  console.log("📤 Request body being sent:", requestBody);

  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(requestBody),
  });

  // Guard against non-JSON responses (e.g. Clerk auth redirects returning HTML)
  type ApiResponse = AnalysisResult & {
    needsCredits?: boolean;
    message?: string;
    error?: string;
  };
  let result: ApiResponse;
  const contentType = response.headers.get("content-type") ?? "";
  if (contentType.includes("application/json")) {
    result = (await response.json()) as ApiResponse;
  } else {
    const text = await response.text();
    console.error(
      "❌ API returned non-JSON response:",
      response.status,
      text.slice(0, 200),
    );
    result = {
      message: `Server returned ${response.status}. Please sign in or try again.`,
      error: "non_json_response",
    } as ApiResponse;
  }

  if (!response.ok) {
    // Handle credit-related errors
    if (result.needsCredits) {
      const error = new Error(
        result.message || "Not enough credits",
      ) as Error & {
        needsCredits: boolean;
        errorType: string;
      };
      error.needsCredits = true;
      error.errorType = result.error ?? "unknown";
      throw error;
    }

    console.error("❌ API Error:", result);
    throw new Error(
      result.message || `Failed to analyze variant (HTTP ${response.status})`,
    );
  }

  console.log("✅ API Response:", result);

  // Return the analysis result (Modal response is spread into the response)
  return {
    position: result.position || position,
    reference: result.reference || "",
    alternative: result.alternative || alternative,
    delta_score: result.delta_score,
    prediction: result.prediction,
    classification_confidence: result.classification_confidence,
    classification_source: result.classification_source,
    population_frequency: result.population_frequency,
    acmg_evidence: result.acmg_evidence,
    literature_context: result.literature_context,
    // Multi-modal RAG fields
    clinical_summary: result.clinical_summary ?? null,
    evidence_confidence: result.evidence_confidence ?? undefined,
    clinvar_evidence: result.clinvar_evidence ?? null,
    protein_context: result.protein_context ?? null,
    // ISM scan
    ism_scan: result.ism_scan ?? null,
    // XAI factors
    xai_factors: result.xai_factors ?? null,
    // Counterfactuals
    counterfactuals: result.counterfactuals ?? null,
    // ACMG criteria
    acmg_criteria: result.acmg_criteria ?? null,
    acmg_criteria_refined: result.acmg_criteria_refined ?? null,
    // Knowledge graph
    knowledge_graph: result.knowledge_graph ?? null,
    // External scores
    external_scores: result.external_scores ?? null,
  };
}
