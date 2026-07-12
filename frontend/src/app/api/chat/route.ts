// =============================================================================
// Chat API Route - Report-Aware AI Chatbot
// =============================================================================
// This endpoint powers a chatbot that has full context of the variant analysis
// report, including:
//   - VEP molecular consequence annotation
//   - Evo2-7B prediction, delta score, confidence
//   - gnomAD population frequency data
//   - ClinVar clinical classification
//   - UniProt protein structure & function
//   - PubMed literature evidence
//   - ACMG/AMP criteria (rule-based + LLM-refined)
//   - AlphaMissense, CADD, REVEL external scores
//   - Multi-tool concordance analysis
//   - ISM scan, XAI factors, counterfactuals
//   - Knowledge graph (gene-disease-drug)
//   - Clinical summary (multi-modal RAG)
//
// Powered by NVIDIA Nemotron-3 550B via NVIDIA API.
// =============================================================================

import { type NextRequest, NextResponse } from "next/server";
import { currentUser } from "@clerk/nextjs/server";
import { db } from "~/lib/db";
import { getToolsForMode } from "~/lib/chat-tools";
import { executeProtoTool } from "~/lib/proto-tools-router";
import { sanitizeToolResponse } from "~/lib/tool-response-sanitizer";

// =============================================================================
// Dynamic Tool Budget Calculator (Fix 2)
// =============================================================================
// Replaces hardcoded MAX_TOOL_CALLS = 3 with query-aware budgeting.
// Complex mechanistic questions (docking, structure prediction) need more
// tool calls than simple lookups.
// =============================================================================

function calculateToolBudget(query: string, mode: string): number {
  if (mode === "general") return 2; // General mode rarely needs deep ML tools

  const q = query.toLowerCase();

  // Deep mechanistic questions need more budget
  if (/docking|bind|affinity|resistance|inhibitor|drug/.test(q)) return 6;
  if (/predict.*structure|esmfold|mutant.*structure|rmsd|fold/.test(q)) return 5;
  if (/splic|exon.*skip|cryptic|splice/.test(q)) return 4;
  if (/homolog|blast|align|mafft|conserv|mmseqs/.test(q)) return 4;
  if (/design|inverse.*fold|proteinmpnn/.test(q)) return 5;

  // Simple lookups or report clarifications
  if (/what is|tell me about|explain.*report|why.*classified|summarize/.test(q)) return 2;

  // Default safe budget for standard variant questions
  return 4;
}

// =============================================================================
// Smart Input Resolver (Fix 3)
// =============================================================================
// Intercepts tool calls BEFORE they go to Modal. If a tool is missing a
// prerequisite (e.g., ESMFold needs a sequence but only got a gene symbol),
// fetch the prerequisite silently in the background without consuming a
// Nemotron tool-call iteration.
// =============================================================================

// =============================================================================
// Known Drug-Target Co-Crystal Structures (experimentally determined)
// =============================================================================
// Instead of running 10-minute Boltz2 diffusion predictions, look up experimentally
// determined PDB co-crystal structures for famous drug-target pairs. These are
// X-ray crystallography structures with sub-2Å resolution — far better than any
// computational prediction. The 3D viewer renders them instantly.

const KNOWN_COCRYSTALS: Record<string, { pdb_id: string; drug: string; target: string; ic50_nM?: number; binding_type: string; mechanism?: string }> = {
  // --- Kinase Inhibitors ---
  "ABL1_IMATINIB": { pdb_id: "1IEP", drug: "Imatinib", target: "ABL1 kinase", ic50_nM: 0.6, binding_type: "Type II inhibitor (DFG-out)" },
  "ABL1_DASATINIB": { pdb_id: "2GQG", drug: "Dasatinib", target: "ABL1 kinase", ic50_nM: 0.8, binding_type: "Type I inhibitor (active conformation)" },
  "ABL1_NILOTINIB": { pdb_id: "3CS9", drug: "Nilotinib", target: "ABL1 kinase", ic50_nM: 20, binding_type: "Type II inhibitor (DFG-out)" },
  "EGFR_ERLOTINIB": { pdb_id: "1M17", drug: "Erlotinib", target: "EGFR kinase", ic50_nM: 2.1, binding_type: "Type I inhibitor" },
  "EGFR_GEFITINIB": { pdb_id: "2ITY", drug: "Gefitinib", target: "EGFR kinase", ic50_nM: 23, binding_type: "Type I inhibitor" },
  "EGFR_OSIMERTINIB": { pdb_id: "4ZAU", drug: "Osimertinib", target: "EGFR kinase", ic50_nM: 12, binding_type: "Type I inhibitor (mutant-selective)" },
  "BRAF_VEMURAFENIB": { pdb_id: "3OG7", drug: "Vemurafenib", target: "BRAF V600E", ic50_nM: 31, binding_type: "Type I inhibitor" },
  "CDK4_PALBOCICLIB": { pdb_id: "2EUF", drug: "Palbociclib", target: "CDK4", ic50_nM: 11, binding_type: "Type I inhibitor" },
  "KIT_IMATINIB": { pdb_id: "1T46", drug: "Imatinib", target: "KIT kinase", ic50_nM: 100, binding_type: "Type II inhibitor" },
  "PDGFR_IMATINIB": { pdb_id: "6GJN", drug: "Imatinib", target: "PDGFRα", ic50_nM: 100, binding_type: "Type II inhibitor" },
  "VEGFR2_SUNITINIB": { pdb_id: "4AGD", drug: "Sunitinib", target: "VEGFR2", ic50_nM: 10, binding_type: "Type I inhibitor" },
  "ALK_CRIZOTINIB": { pdb_id: "2XP2", drug: "Crizotinib", target: "ALK kinase", ic50_nM: 24, binding_type: "Type I inhibitor" },
  "BCL2_VENETOCLAX": { pdb_id: "6O0K", drug: "Venetoclax", target: "BCL2", ic50_nM: 0.01, binding_type: "BH3 mimetic" },
  "IDH2_ENASIDENIB": { pdb_id: "5X08", drug: "Enasidenib", target: "IDH2 R140Q", ic50_nM: 100, binding_type: "Allosteric inhibitor" },

  // --- PARP Inhibitors (Synthetic Lethality for BRCA1/2 mutations) ---
  "PARP1_OLAPARIB": { pdb_id: "5DS3", drug: "Olaparib (Lynparza)", target: "PARP1 (DNA repair enzyme)", ic50_nM: 5.0, binding_type: "PARP1/2 Trapper", mechanism: "Synthetic Lethal for BRCA1/2 mutants — traps PARP1 on DNA, causing double-strand breaks that BRCA-deficient cells cannot repair" },
  "PARP1_RUCAPARIB": { pdb_id: "4R6E", drug: "Rucaparib (Rubraca)", target: "PARP1/2", ic50_nM: 1.4, binding_type: "PARP1/2 Inhibitor", mechanism: "Synthetic Lethal for BRCA1/2 mutants" },
  "PARP1_NIRAPARIB": { pdb_id: "6W0O", drug: "Niraparib (Zejula)", target: "PARP1/2", ic50_nM: 2.0, binding_type: "PARP1/2 Trapper", mechanism: "Synthetic Lethal for BRCA1/2 mutants" },
  "PARP1_TALAZOPARIB": { pdb_id: "7KKK", drug: "Talazoparib (Talzenna)", target: "PARP1/2", ic50_nM: 0.57, binding_type: "PARP1/2 Trapper", mechanism: "Synthetic Lethal for BRCA1/2 mutants — most potent PARP trapper" },

  // --- Brand-name aliases ---
  "LYNPARZA": { pdb_id: "5DS3", drug: "Olaparib (Lynparza)", target: "PARP1", ic50_nM: 5.0, binding_type: "PARP1/2 Trapper", mechanism: "Synthetic Lethal for BRCA1/2 mutants" },
  "RUBRACA": { pdb_id: "4R6E", drug: "Rucaparib (Rubraca)", target: "PARP1/2", ic50_nM: 1.4, binding_type: "PARP1/2 Inhibitor", mechanism: "Synthetic Lethal for BRCA1/2 mutants" },
  "ZEJULA": { pdb_id: "6W0O", drug: "Niraparib (Zejula)", target: "PARP1/2", ic50_nM: 2.0, binding_type: "PARP1/2 Trapper", mechanism: "Synthetic Lethal for BRCA1/2 mutants" },
  "TALZENNA": { pdb_id: "7KKK", drug: "Talazoparib (Talzenna)", target: "PARP1/2", ic50_nM: 0.57, binding_type: "PARP1/2 Trapper", mechanism: "Synthetic Lethal for BRCA1/2 mutants" },
};

/**
 * Check if a user's query matches a known drug-target co-crystal structure.
 * Returns the PDB ID and experimental data if found, null otherwise.
 * Also handles Synthetic Lethality: BRCA1/BRCA2 queries map to PARP1 inhibitors.
 */
function lookupKnownCocrystal(userMessage: string): { pdb_id: string; drug: string; target: string; ic50_nM?: number; binding_type: string; mechanism?: string } | null {
  const msg = userMessage.toUpperCase().replace(/[^A-Z0-9 ]/g, "");
  const lowerMsg = userMessage.toLowerCase();

  // ── BLOCK LIST: Negative keywords that should prevent interception ──
  // If any of these appear, let Nemotron handle the query with proper nuance
  const negativeKeywords = [
    "synonymous", "non-coding", "non coding", "utr", "5'utr", "3'utr",
    "ring domain", "brct domain", "coiled-coil", "scye domain",
    "specifically target", "binds to brca1 directly",
    "target the", "targeting the",
    "vus", "benign", "not pathogenic", "uncertain significance",
    "tp53", "p53",
  ];
  const isBlocked = negativeKeywords.some(keyword => lowerMsg.includes(keyword));

  // ── Synthetic Lethality: BRCA1/BRCA2 → PARP1 inhibitors ──
  // When a user asks for a drug for BRCA1/BRCA2, map to PARP1 (Olaparib by default)
  // But only if the query is NOT blocked by negative keywords
  const brcaMatch = msg.includes("BRCA1") || msg.includes("BRCA2") || msg.includes("BRCA");
  const drugMatch = msg.includes("DRUG") || msg.includes("SUGGEST") || msg.includes("THERAPY") || msg.includes("TREAT") || msg.includes("INHIBITOR") || msg.includes("MEDICINE") || msg.includes("TARGETED");
  if (brcaMatch && drugMatch && !isBlocked) {
    // Check if user named a specific PARP inhibitor
    if (msg.includes("RUCAPARIB") || msg.includes("RUBRACA")) return KNOWN_COCRYSTALS["PARP1_RUCAPARIB"];
    if (msg.includes("NIRAPARIB") || msg.includes("ZEJULA")) return KNOWN_COCRYSTALS["PARP1_NIRAPARIB"];
    if (msg.includes("TALAZOPARIB") || msg.includes("TALZENNA")) return KNOWN_COCRYSTALS["PARP1_TALAZOPARIB"];
    // Default: Olaparib (most commonly prescribed)
    return KNOWN_COCRYSTALS["PARP1_OLAPARIB"];
  }

  // ── Direct drug-target matching ──
  for (const [key, value] of Object.entries(KNOWN_COCRYSTALS)) {
    const [target, drug] = key.split("_");
    // Match if both drug and target appear in the message
    if (msg.includes(drug) && msg.includes(target)) {
      return value;
    }
  }

  // ── Brand-name only matching (e.g., "Lynparza" without mentioning PARP1) ──
  for (const [key, value] of Object.entries(KNOWN_COCRYSTALS)) {
    if (key === "LYNPARZA" || key === "RUBRACA" || key === "ZEJULA" || key === "TALZENNA") {
      if (msg.includes(key)) return value;
    }
  }

  return null;
}

async function resolveToolInputs(
  toolName: string,
  args: Record<string, unknown>,
): Promise<Record<string, unknown>> {
  const resolved = { ...args };

  // ── fetch_alphafold_db: If gene symbol given instead of UniProt accession, resolve it ──
  // UniProt accessions match ^[A-Z]\d{5}$ or ^[OPQ]\d{3}[A-Z]\d{2}$ (e.g., P38398, Q13362)
  // Gene symbols are short uppercase strings that don't match this pattern (e.g., BRCA1, TP53)
  if (
    toolName === "fetch_alphafold_db" &&
    resolved.uniprot_id &&
    typeof resolved.uniprot_id === "string"
  ) {
    const id = resolved.uniprot_id as string;
    // Check if it looks like a UniProt accession (not a gene symbol)
    const isUniprotAccession = /^[OPQ]\d{3}[A-Z]\d{2}$|^[A-NR-Z]\d{5}$/.test(id);
    if (!isUniprotAccession && id.length < 15 && /^[A-Z0-9]+$/.test(id)) {
      console.log(`[SmartResolver] fetch_alphafold_db: '${id}' looks like a gene symbol, resolving to UniProt accession`);
      try {
        // Query UniProt by gene symbol to get the canonical accession
        const uniData = await executeProtoTool("fetch_uniprot", {
          uniprot_id: id,
        });
        const uniResult = uniData.result as Record<string, unknown> | undefined;
        const accession = uniResult?.accession as string | undefined;
        if (accession) {
          resolved.uniprot_id = accession;
          console.log(`[SmartResolver] Resolved gene '${id}' → UniProt accession '${accession}'`);
        }
      } catch (e) {
        console.warn(`[SmartResolver] Failed to resolve gene symbol '${id}' to UniProt accession:`, e);
      }
    }
  }

  // ── ESMFold: If only gene_symbol given, fetch UniProt sequence ──
  if (
    toolName === "run_esmfold_prediction" &&
    !resolved.complexes &&
    (resolved.gene_symbol || resolved.gene)
  ) {
    const geneSymbol = String(resolved.gene_symbol || resolved.gene);
    console.log(`[SmartResolver] ESMFold missing sequence, fetching UniProt for ${geneSymbol}`);
    const uniData = await executeProtoTool("fetch_uniprot", {
      uniprot_id: geneSymbol,
    });
    const sequence = (uniData.result as Record<string, unknown>)?.sequence as string | undefined;
    if (sequence) {
      resolved.complexes = [sequence];
      delete resolved.gene_symbol;
      delete resolved.gene;
    }
  }

  // ── Boltz2 Affinity: Resolve both protein gene symbols and drug name ligands ──
  if (
    (toolName === "run_boltz2_affinity" || toolName === "run_boltz2_prediction") &&
    resolved.complexes &&
    !resolved._smiles_resolved
  ) {
    // Ensure complexes is an array of arrays — handle various formats Nemotron might produce
    let complexes: Array<unknown>[] = [];
    if (Array.isArray(resolved.complexes)) {
      complexes = resolved.complexes as Array<unknown>[];
    } else if (typeof resolved.complexes === "string") {
      // Nemotron passed a string instead of an array — try to parse
      try {
        complexes = JSON.parse(resolved.complexes) as Array<unknown>[];
      } catch {
        // Can't parse, skip resolution
        resolved._smiles_resolved = true;
        return resolved;
      }
    } else {
      // Unknown format, skip resolution
      resolved._smiles_resolved = true;
      return resolved;
    }

    // Check if any complex has a drug_name instead of SMILES
    const resolvedComplexes = await Promise.all(
      complexes.map(async (c) => {
        if (!Array.isArray(c)) return c;
        const [protein, ligand] = c as string[];

        // ── Resolve protein: if it looks like a gene symbol or UniProt ID (not a sequence), fetch UniProt ──
        let resolvedProtein = protein;
        // Strip "protein:" prefix if Nemotron added it
        if (typeof resolvedProtein === "string" && resolvedProtein.startsWith("protein:")) {
          resolvedProtein = resolvedProtein.substring(8);
        }
        // Check if it's a gene symbol (short, uppercase alphanumeric) or UniProt accession (e.g., P00519)
        if (resolvedProtein && typeof resolvedProtein === "string" && resolvedProtein.length < 15 && /^[A-Z0-9]+$/.test(resolvedProtein)) {
          console.log(`[SmartResolver] Boltz2 protein '${resolvedProtein}' looks like a gene symbol/UniProt ID, fetching sequence`);
          let sequence: string | undefined;
          try {
            const uniData = await executeProtoTool("fetch_uniprot", {
              uniprot_id: resolvedProtein,
            });
            sequence = (uniData.result as Record<string, unknown>)?.sequence as string | undefined;
          } catch (e) {
            console.log(`[SmartResolver] ERROR: UniProt fetch failed for '${resolvedProtein}':`, e);
          }
          if (sequence && sequence.length > 20) {
            resolvedProtein = sequence;
            console.log(`[SmartResolver] Resolved protein to ${sequence.length} aa sequence`);
          } else {
            // HARD STOP: Don't send a gene symbol to a 10-minute GPU tool — fail fast
            throw new Error(
              `SmartResolver: Failed to resolve protein '${resolvedProtein}' to a sequence (UniProt returned no data). ` +
              `Aborting GPU call to prevent wasting compute. Please retry the request.`
            );
          }
        }

        // ── Resolve ligand: if it looks like a drug name (not a SMILES string), fetch PubChem ──
        // Strip "ligand:" prefix if Nemotron added it
        let resolvedLigand = ligand;
        if (typeof resolvedLigand === "string" && resolvedLigand.startsWith("ligand:")) {
          resolvedLigand = resolvedLigand.substring(7);
        }
        // SMILES strings contain special chars: (, ), =, #, [, ], @, /, \, +, -
        // Drug names are alphanumeric only (letters + maybe digits), typically < 30 chars
        if (resolvedLigand && typeof resolvedLigand === "string" && !/[()=#\[\]@/\\+\-]/.test(resolvedLigand) && resolvedLigand.length < 30 && /^[A-Za-z0-9\s]+$/.test(resolvedLigand)) {
          console.log(`[SmartResolver] Boltz2 ligand '${resolvedLigand}' looks like a drug name, fetching PubChem`);
          const pubData = await executeProtoTool("fetch_pubchem", {
            name: resolvedLigand,
          });
          const pubResult = pubData.result as Record<string, unknown> | undefined;
          // PubChem returns SMILES in the 'smiles' field (not 'canonical_smiles')
          const smiles = pubResult?.smiles as string | undefined ||
            pubResult?.canonical_smiles as string | undefined ||
            pubResult?.connectivity_smiles as string | undefined;
          if (smiles) {
            console.log(`[SmartResolver] Resolved ligand to SMILES: ${smiles.substring(0, 50)}...`);
            resolvedLigand = smiles;
          } else {
            console.log(`[SmartResolver] WARNING: PubChem returned no SMILES for '${resolvedLigand}'. Result keys:`, pubResult ? Object.keys(pubResult) : 'null');
          }
        }

        return [resolvedProtein, resolvedLigand];
      }),
    );
    resolved.complexes = resolvedComplexes;
    resolved._smiles_resolved = true;
    // CRITICAL: Delete _smiles_resolved before sending to Modal — Boltz2's Pydantic
    // model rejects extra fields. This flag is only for our internal use.
    delete resolved._smiles_resolved;
  }

  // ── BLAST: If UniProt ID given instead of sequence, fetch sequence ──
  if (
    toolName === "run_blast_search" &&
    resolved.query &&
    typeof resolved.query === "string" &&
    /^[A-Z]\d{5}$/.test(resolved.query) // Looks like a UniProt accession
  ) {
    console.log(`[SmartResolver] BLAST query '${resolved.query}' looks like UniProt ID, fetching sequence`);
    const uniData = await executeProtoTool("fetch_uniprot", {
      uniprot_id: resolved.query,
    });
    const sequence = (uniData.result as Record<string, unknown>)?.sequence as string | undefined;
    if (sequence) {
      resolved.query = sequence;
    }
  }

  // ── ESM2 Score: If UniProt ID given instead of sequence, fetch sequence ──
  if (
    toolName === "run_esm2_score" &&
    resolved.sequences &&
    Array.isArray(resolved.sequences)
  ) {
    const sequences = resolved.sequences as string[];
    const resolvedSeqs = await Promise.all(
      sequences.map(async (s) => {
        if (typeof s === "string" && /^[A-Z]\d{5}$/.test(s)) {
          console.log(`[SmartResolver] ESM2 sequence '${s}' looks like UniProt ID, fetching`);
          const uniData = await executeProtoTool("fetch_uniprot", { uniprot_id: s });
          const seq = (uniData.result as Record<string, unknown>)?.sequence as string | undefined;
          return seq || s;
        }
        return s;
      }),
    );
    resolved.sequences = resolvedSeqs;
  }

  return resolved;
}

// =============================================================================
// Configuration
// =============================================================================

const NVIDIA_API_KEY = process.env.NVIDIA_API_KEY;
const NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions";
const MODEL = "nvidia/nemotron-3-ultra-550b-a55b";

// =============================================================================
// Types
// =============================================================================

interface ChatRequestBody {
  reportId?: string | null;
  message: string;
  sessionId?: string | null;
  mode?: "report" | "general";
}

interface StreamEvent {
  event: "reasoning_delta" | "content_delta" | "content_clear" | "tool_call" | "tool_result" | "3d_structure_payload" | "done" | "error";
  data: Record<string, unknown>;
}

function encodeEvent(event: StreamEvent): string {
  return `event: ${event.event}\ndata: ${JSON.stringify(event.data)}\n\n`;
}

// =============================================================================
// System Prompt Builder
// =============================================================================

function buildGeneralSystemPrompt(): string {
  return `You are HelixMind Chat, an expert AI assistant for clinical variant interpretation and genomic research.

## CRITICAL INSTRUCTION: TOOL USAGE
You have access to a set of bioinformatics tools (functions). Before answering any structural, sequence, or database query, you MUST review your available tool schemas. Do NOT claim a tool is unavailable until you have thoroughly checked your function list. Common tools include: fetch_uniprot, fetch_alphafold_db, fetch_alphamissense, run_ensembl_vep, search_ncbi, run_interproscan_fetch, run_foldseek_search, run_viennarna_prediction, run_blast_search, run_segmasker_score, and more. Always prefer calling a tool over guessing an answer.

## STRICT EXECUTION RULES (OBEY OR FAIL):
1. PARALLEL EXECUTION: If you need to call multiple independent tools (e.g., fetch_uniprot, fetch_alphafold_db, and fetch_alphamissense), you MUST emit them as a single tool_calls array containing all objects. DO NOT call them one by one — batch them in one response.
2. CONTEXT AWARENESS: If the user's question is about a variant already discussed, or data already present in the report context above, DO NOT re-fetch it. Use the data already available.
3. ARGUMENT STRICTNESS: Never guess a UniProt accession (e.g., do not guess "P38398" for BRCA1 unless you are certain). If you only have a gene symbol, pass the gene symbol to the tool — the backend will resolve it automatically.
4. SEQUENTIAL DEPENDENCY: If Tool B depends on Tool A's output (e.g., BLAST needs a sequence from UniProt), call Tool A first, wait for the result, then call Tool B. Do NOT try to call both simultaneously if there is a data dependency.
5. IN-CHAT 3D VIEWER: If the user says "show 3D", "view structure", "render structure", "show structure", or asks to see a protein structure visually, you MUST call fetch_alphafold_db (or run_esmfold_prediction for mutant sequences). Never tell the user the structure is "already available" or "look at the report panel" — you must explicitly invoke the tool to render the interactive 3D viewer inside this chat window.
6. GPU EXCLUSIVITY: When calling a GPU tool (run_esmfold_prediction, run_esm2_score, run_proteinmpnn_sample, run_proteinmpnn_score, run_pymol_rmsd_alignment), DO NOT call CPU-heavy tools (run_mmseqs2_search_proteins, run_blast_search, run_mafft_align) in the same parallel block. GPU tools need all available network and container resources. Call GPU tools alone or only with lightweight fetch tools (fetch_uniprot, fetch_pubchem).
7. DRUG-TARGET STRUCTURE PRIORITY: Before predicting a drug-protein structure, check if an experimentally determined PDB co-crystal structure exists. Call fetch_pdb_entry with a known co-crystal PDB ID (e.g., 1IEP for ABL1+Imatinib, 1M17 for EGFR+Erlotinib, 5DS3 for PARP1+Olaparib). These X-ray crystallography structures are far more accurate than any computational prediction and render instantly in the 3D viewer. Only use run_esmfold_prediction for novel proteins with no known structure. Do NOT use Boltz2 for drug docking in chat — it takes 10+ minutes.
8. SYNTHETIC LETHALITY: If the user asks for a drug for a BRCA1 or BRCA2 mutation, do NOT try to find a drug that binds to BRCA1/2 directly — there are none. Instead, explain that BRCA1/2 mutations cause Homologous Recombination Deficiency (HRD), making the cell dependent on PARP1/2 for DNA repair. The clinical strategy is Synthetic Lethality: inhibit PARP1/2 with Olaparib, Rucaparib, Niraparib, or Talazoparib. The backend will automatically fetch the PARP1 co-crystal structure (PDB: 5DS3 for Olaparib) when you mention these drugs.

## YOUR CAPABILITIES
- Explain variant pathogenicity concepts (VUS, pathogenic, benign, likely pathogenic)
- Interpret scores such as Evo2 delta likelihood, CADD PHRED, REVEL, and AlphaMissense
- Describe ACMG/AMP criteria and how they are applied
- Discuss population frequency data (gnomAD) and its clinical significance
- Summarize how to read a HelixMind variant analysis report
- Suggest therapeutic investigation strategies for variants
- Help researchers design follow-up experiments

## GUIDELINES
1. Answer using established genomics knowledge. Do not fabricate patient-specific facts.
2. If a question requires a specific variant report, say "Open a variant report and I can analyze it with full context."
3. Use plain language when explaining complex concepts. Assume the user is a researcher or clinician.
4. Always include a disclaimer when discussing clinical implications: "This is a computational prediction for research purposes. Clinical decisions require professional genetic counseling."
5. Be concise but thorough. Use bullet points for clarity when listing multiple items.`;
}

function buildSystemPrompt(report: Record<string, unknown>): string {
  // Extract all report sections for context
  const vep = report.vepAnnotation as Record<string, unknown> | null;
  // NOTE: CRITICAL INSTRUCTION about tool usage is prepended below
  const popFreq = report.populationFrequency as Record<string, unknown> | null;
  const acmg = report.acmgEvidence as Record<string, unknown> | null;
  const lit = report.literatureContext as Record<string, unknown> | null;
  const evidence = report.evidenceConfidence as Record<string, unknown> | null;
  const acmgCriteria = report.acmgCriteria as Record<string, unknown> | null;
  const acmgRefined = report.acmgCriteriaRefined as Record<string, unknown> | null;
  const externalScores = report.externalScores as Record<string, unknown> | null;
  const xai = report.xaiFactors as Record<string, unknown> | null;
  const counterfactuals = report.counterfactuals as Record<string, unknown> | null;
  const kg = report.knowledgeGraph as Record<string, unknown> | null;
  const ism = report.ismScanData as Record<string, unknown> | null;

  return `You are HelixMind Chat, an expert AI assistant for clinical variant interpretation. You have access to the complete analysis report for a genetic variant and can answer questions about any aspect of it.

## CRITICAL INSTRUCTION: TOOL USAGE
You have access to a set of bioinformatics tools (functions). Before answering any structural, sequence, or database query, you MUST review your available tool schemas. Do NOT claim a tool is unavailable until you have thoroughly checked your function list. Common tools include: fetch_uniprot, fetch_alphafold_db, fetch_alphamissense, run_ensembl_vep, search_ncbi, run_interproscan_fetch, run_foldseek_search, run_viennarna_prediction, run_blast_search, run_segmasker_score, run_mafft_align, run_mmseqs2_search_proteins, and more. Always prefer calling a tool over guessing an answer.

## STRICT EXECUTION RULES (OBEY OR FAIL):
1. PARALLEL EXECUTION: If you need to call multiple independent tools (e.g., fetch_uniprot, fetch_alphafold_db, and fetch_alphamissense), you MUST emit them as a single tool_calls array containing all objects. DO NOT call them one by one — batch them in one response.
2. CONTEXT AWARENESS: If the user's question is about a variant already discussed, or data already present in the report context above, DO NOT re-fetch it. Use the data already available.
3. ARGUMENT STRICTNESS: Never guess a UniProt accession (e.g., do not guess "P38398" for BRCA1 unless you are certain). If you only have a gene symbol, pass the gene symbol to the tool — the backend will resolve it automatically.
4. SEQUENTIAL DEPENDENCY: If Tool B depends on Tool A's output (e.g., BLAST needs a sequence from UniProt), call Tool A first, wait for the result, then call Tool B. Do NOT try to call both simultaneously if there is a data dependency.
5. IN-CHAT 3D VIEWER: If the user says "show 3D", "view structure", "render structure", "show structure", or asks to see a protein structure visually, you MUST call fetch_alphafold_db (or run_esmfold_prediction for mutant sequences). Never tell the user the structure is "already available" or "look at the report panel" — you must explicitly invoke the tool to render the interactive 3D viewer inside this chat window.
6. GPU EXCLUSIVITY: When calling a GPU tool (run_esmfold_prediction, run_esm2_score, run_proteinmpnn_sample, run_proteinmpnn_score, run_pymol_rmsd_alignment), DO NOT call CPU-heavy tools (run_mmseqs2_search_proteins, run_blast_search, run_mafft_align) in the same parallel block. GPU tools need all available network and container resources. Call GPU tools alone or only with lightweight fetch tools (fetch_uniprot, fetch_pubchem).
7. DRUG-TARGET STRUCTURE PRIORITY: Before predicting a drug-protein structure, check if an experimentally determined PDB co-crystal structure exists. Call fetch_pdb_entry with a known co-crystal PDB ID (e.g., 1IEP for ABL1+Imatinib, 1M17 for EGFR+Erlotinib, 5DS3 for PARP1+Olaparib). These X-ray crystallography structures are far more accurate than any computational prediction and render instantly in the 3D viewer. Only use run_esmfold_prediction for novel proteins with no known structure. Do NOT use Boltz2 for drug docking in chat — it takes 10+ minutes.
8. SYNTHETIC LETHALITY: If the user asks for a drug for a BRCA1 or BRCA2 mutation, do NOT try to find a drug that binds to BRCA1/2 directly — there are none. Instead, explain that BRCA1/2 mutations cause Homologous Recombination Deficiency (HRD), making the cell dependent on PARP1/2 for DNA repair. The clinical strategy is Synthetic Lethality: inhibit PARP1/2 with Olaparib, Rucaparib, Niraparib, or Talazoparib. The backend will automatically fetch the PARP1 co-crystal structure (PDB: 5DS3 for Olaparib) when you mention these drugs.

## YOUR CAPABILITIES
- Explain variant pathogenicity predictions in plain language
- Interpret ACMG/AMP criteria and classification rationale
- Discuss population frequency data (gnomAD) and its clinical significance
- Explain protein structural consequences using UniProt domain data
- Summarize literature evidence from PubMed
- Compare Evo2 AI predictions with ClinVar, AlphaMissense, CADD, and REVEL
- Explain VEP molecular consequences (missense, frameshift, splice, etc.)
- Discuss therapeutic strategies and clinical actionability
- Help researchers understand confidence metrics and evidence quality
- Explain ISM (In-Silico Mutagenesis) scan results and XAI factors

## THE VARIANT REPORT

### Basic Information
- Gene: ${report.geneSymbol}
- Chromosome: ${report.chromosome}
- Position: ${report.position}
- Reference Allele: ${report.reference}
- Alternative Allele: ${report.alternative}
- Genome Build: ${report.genomeId}

### Evo2-7B AI Prediction
- Prediction: ${report.prediction}
- Delta Likelihood Score: ${report.deltaScore}
- Classification Confidence: ${((report.classificationConfidence as number) * 100).toFixed(1)}%
- Classification Source: ${report.classificationSource || "Evo2_AI"}

### VEP Molecular Consequence
${vep ? `- Consequence: ${vep.consequence || "N/A"}
- Impact: ${vep.impact || "N/A"}
- Amino Acid Change: ${vep.aaChange || "N/A"} (${vep.aminoAcids || "N/A"})
- Codons: ${vep.codons || "N/A"}
- Transcript: ${vep.transcriptId || "N/A"}
- Is Synonymous: ${vep.isSynonymous}
- Is Frameshift: ${vep.isFrameshift}
- Is Nonsense: ${vep.isNonsense}` : "- VEP annotation not available"}

### Population Frequency (gnomAD)
${popFreq ? `- gnomAD Allele Frequency: ${popFreq.gnomad_af ?? "Not found"}
- Max Population AF: ${popFreq.gnomad_max_pop_af ?? "N/A"}
- Source: ${popFreq.source || "N/A"}
- Is Common Variant: ${popFreq.is_common_variant}` : "- gnomAD data not available"}

### ClinVar Classification
- ClinVar Status: ${report.clinvarClassification || "Not curated"}
- Variation Type: ${report.variationType || "N/A"}
- ClinVar ID: ${report.clinvarId || "N/A"}

### ACMG Evidence
${acmg ? `- Code: ${acmg.code || "None"}
- Strength: ${acmg.strength || "None"}
- Description: ${acmg.description}
- Clinical Note: ${acmg.clinical_note}` : "- ACMG evidence not available"}

### Literature Context (PubMed)
${lit ? `- Summary: ${lit.summary || "N/A"}
- Articles Found: ${lit.articles_found || 0}
- PubMed IDs: ${Array.isArray(lit.pubmed_ids) ? (lit.pubmed_ids as string[]).join(", ") : "N/A"}
- Gene Function: ${lit.gene_function || "N/A"}` : "- Literature data not available"}

### Evidence Confidence
${evidence ? `- VEP: ${(evidence.vep as Record<string,unknown>)?.confidence || "N/A"} (${(evidence.vep as Record<string,unknown>)?.available ? "Available" : "Missing"})
- Evo2: ${(evidence.evo2 as Record<string,unknown>)?.confidence || "N/A"} (${(evidence.evo2 as Record<string,unknown>)?.available ? "Available" : "Missing"})
- gnomAD: ${(evidence.gnomad as Record<string,unknown>)?.confidence || "N/A"} (${(evidence.gnomad as Record<string,unknown>)?.available ? "Available" : "Missing"})
- ClinVar: ${(evidence.clinvar as Record<string,unknown>)?.confidence || "N/A"} (${(evidence.clinvar as Record<string,unknown>)?.available ? "Available" : "Missing"})
- UniProt: ${(evidence.uniprot as Record<string,unknown>)?.confidence || "N/A"} (${(evidence.uniprot as Record<string,unknown>)?.available ? "Available" : "Missing"})
- PubMed: ${(evidence.pubmed as Record<string,unknown>)?.confidence || "N/A"} (${(evidence.pubmed as Record<string,unknown>)?.available ? "Available" : "Missing"})
- Overall: ${(evidence.overall as Record<string,unknown>)?.level || "N/A"} (${(evidence.overall as Record<string,unknown>)?.sources_available || "N/A"})` : "- Evidence confidence not available"}

### ACMG/AMP Criteria (Rule-Based)
${acmgCriteria ? formatACMGCriteria(acmgCriteria) : "- ACMG criteria not available"}

### LLM-Refined ACMG Classification
${acmgRefined ? `- ACMG Classification: ${acmgRefined.acmg_classification || "N/A"}
- Confidence: ${acmgRefined.classification_confidence || "N/A"}
- Narrative: ${acmgRefined.narrative || "N/A"}
${formatRefinedCriteria(acmgRefined)}` : "- Refined ACMG not available"}

### External Scores
${externalScores ? formatExternalScores(externalScores) : "- External scores not available"}

### Multi-Tool Concordance
${formatConcordance(report)}

### Clinical Summary (Multi-Modal RAG)
${report.clinicalSummary || "Not available"}

### XAI Confidence Factors
${xai ? formatXAI(xai) : "- XAI factors not available"}

### Counterfactual Analysis
${counterfactuals ? formatCounterfactuals(counterfactuals) : "- Counterfactuals not available"}

### Knowledge Graph
${kg ? formatKnowledgeGraph(kg) : "- Knowledge graph not available"}

### ISM Scan
${ism ? formatISM(ism) : "- ISM scan not available"}

## GUIDELINES
1. Answer questions using ONLY the data provided above. Do not fabricate information.
2. If asked about data not in the report, say "This information is not available in the current analysis report."
3. Use plain language when explaining complex concepts. Assume the user is a researcher or clinician.
4. Always include a disclaimer when discussing clinical implications: "This is a computational prediction for research purposes. Clinical decisions require professional genetic counseling."
5. Be concise but thorough. Use bullet points for clarity when listing multiple items.
6. When discussing pathogenicity, always reference the specific evidence sources (Evo2 score, ClinVar status, ACMG criteria, population data).
7. For therapeutic questions, note that recommendations are computational hypotheses requiring experimental validation.`;
}

// =============================================================================
// Formatting Helpers
// =============================================================================

function formatACMGCriteria(acmg: Record<string, unknown>): string {
  const criteria = acmg.criteria as Record<string, Record<string, unknown>> | undefined;
  if (!criteria) return "- ACMG criteria not available";
  
  const lines = Object.entries(criteria).map(([code, detail]) => {
    return `  - ${code}: ${detail.met ? "MET" : "Not Met"} (Strength: ${detail.strength || "N/A"}) — ${detail.rationale || ""}`;
  });
  
  return `- Met Count: ${acmg.met_count}/${acmg.total_evaluated}
- Classification: ${acmg.acmg_classification || "N/A"}
- Criteria:\n${lines.join("\n")}`;
}

function formatRefinedCriteria(acmg: Record<string, unknown>): string {
  const criteria = acmg.criteria as Record<string, Record<string, unknown>> | undefined;
  if (!criteria) return "";
  
  const lines = Object.entries(criteria).map(([code, detail]) => {
    return `  - ${code}: ${detail.met ? "MET" : "Not Met"} (Strength: ${detail.strength || "N/A"}) — ${detail.justification || ""}`;
  });
  
  return `- Refined Criteria:\n${lines.join("\n")}`;
}

function formatExternalScores(scores: Record<string, unknown>): string {
  const parts: string[] = [];
  
  const cadd = scores.cadd as Record<string, unknown> | null;
  if (cadd) {
    parts.push(`- CADD: PHRED ${cadd.phred} (Raw: ${cadd.raw}) — ${cadd.interpretation || "N/A"}`);
  }
  
  const revel = scores.revel as Record<string, unknown> | null;
  if (revel) {
    parts.push(`- REVEL: ${revel.score} — ${revel.interpretation || "N/A"}`);
  }
  
  const am = scores.alphamissense as Record<string, unknown> | null;
  if (am) {
    parts.push(`- AlphaMissense: ${am.score} (${am.classification || "N/A"}) — Confidence: ${am.confidence || "N/A"}`);
  }
  
  if (scores.concordance_note) {
    parts.push(`- Concordance Note: ${scores.concordance_note}`);
  }
  
  return parts.join("\n") || "- External scores not available";
}

function formatConcordance(report: Record<string, unknown>): string {
  const parts: string[] = [];
  parts.push(`- Evo2-7B: ${report.prediction} (Δ = ${report.deltaScore})`);
  
  const extScores = report.externalScores as Record<string, unknown> | null;
  if (extScores) {
    const am = extScores.alphamissense as Record<string, unknown> | null;
    if (am) parts.push(`- AlphaMissense: ${am.classification || "N/A"} (Score: ${am.score})`);
    
    const cadd = extScores.cadd as Record<string, unknown> | null;
    if (cadd) parts.push(`- CADD: ${cadd.interpretation || "N/A"} (PHRED: ${cadd.phred})`);
  }
  
  parts.push(`- ClinVar: ${report.clinvarClassification || "Not curated"}`);
  
  return parts.join("\n");
}

function formatXAI(xai: Record<string, unknown>): string {
  const factors = xai.factors as Array<Record<string, unknown>> | undefined;
  if (!factors) return "- XAI factors not available";
  
  const lines = factors.map(f => 
    `  - ${f.label}: Contribution ${f.contribution} — ${f.detail || ""}`
  );
  return `- Total: ${xai.total}\n- Factors:\n${lines.join("\n")}`;
}

function formatCounterfactuals(cf: Record<string, unknown>): string {
  return `- Reference: ${cf.reference}
- Tolerated Alleles: ${Array.isArray(cf.tolerated_alleles) ? (cf.tolerated_alleles as string[]).join(", ") : "N/A"}
- Pathogenic Alleles: ${Array.isArray(cf.pathogenic_alleles) ? (cf.pathogenic_alleles as string[]).join(", ") : "N/A"}
- Summary: ${cf.summary || "N/A"}`;
}

function formatKnowledgeGraph(kg: Record<string, unknown>): string {
  const diseases = kg.diseases as Array<Record<string, unknown>> | undefined;
  const drugs = kg.drugs as Array<Record<string, unknown>> | undefined;
  
  const diseaseLines = diseases?.map(d => `  - ${d.name} (Score: ${d.score})`).join("\n") || "  None";
  const drugLines = drugs?.map(d => `  - ${d.name} (${d.type}, Phase ${d.phase}): ${d.mechanism}`).join("\n") || "  None";
  
  return `- Gene: ${kg.gene}
- Clinical Actionability: ${kg.clinical_actionability || "N/A"}
- Associated Diseases:\n${diseaseLines}
- Relevant Drugs:\n${drugLines}`;
}

function formatISM(ism: Record<string, unknown>): string {
  const summary = ism.summary as Record<string, unknown> | undefined;
  if (!summary) return "- ISM scan data not available";
  
  return `- Constrained Positions: ${summary.constrained_positions}
- Total Scanned: ${summary.total_positions_scanned}
- Constraint Zone: ${summary.constraint_zone}
- Peak Constraint Position: ${summary.peak_constraint_position || "N/A"}
- Peak Magnitude: ${summary.peak_constraint_magnitude || "N/A"}`;
}

// =============================================================================
// POST /api/chat
// =============================================================================

export async function POST(request: NextRequest) {
  try {
    // --- Auth ---
    const user = await currentUser();
    if (!user) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const body = (await request.json()) as ChatRequestBody;
    const { reportId, message, sessionId, mode = "report" } = body;

    if (!message) {
      return NextResponse.json(
        { error: "message is required" },
        { status: 400 },
      );
    }

    const isReportMode = mode === "report" && reportId;

    // --- Load the full report if in report mode ---
    let report: Awaited<ReturnType<typeof db.analysisReport.findFirst>> = null;
    if (isReportMode) {
      report = await db.analysisReport.findFirst({
        where: { id: reportId!, clerkUserId: user.id },
      });

      if (!report) {
        return NextResponse.json(
          { error: "Report not found or access denied" },
          { status: 404 },
        );
      }
    }

    // --- Get or create chat session ---
    let chatSession;
    if (sessionId) {
      chatSession = await db.chatSession.findFirst({
        where: { id: sessionId, clerkUserId: user.id },
      });
      if (!chatSession) {
        return NextResponse.json(
          { error: "Chat session not found" },
          { status: 404 },
        );
      }
    } else {
      chatSession = await db.chatSession.create({
        data: {
          clerkUserId: user.id,
          analysisReportId: isReportMode ? reportId! : null,
          title: isReportMode
            ? `Chat about ${report!.geneSymbol} ${report!.reference}>${report!.alternative}`
            : "General genomics chat",
        },
      });
    }

    // --- Load previous messages for context (limited to prevent context bleed) ---
    const previousMessages = await db.chatMessage.findMany({
      where: { sessionId: chatSession.id },
      orderBy: { createdAt: "asc" },
      take: 6, // last 6 messages (3 turns) — enough for follow-up questions without context bleed
    });

    // --- Build the system prompt ---
    const systemPrompt = isReportMode
      ? buildSystemPrompt(report as unknown as Record<string, unknown>)
      : buildGeneralSystemPrompt();

    // --- Build messages array for NVIDIA API ---
    const nvidiaMessages: Array<{ role: string; content: string }> = [
      { role: "system", content: systemPrompt },
    ];

    // Add previous conversation history
    for (const msg of previousMessages) {
      nvidiaMessages.push({ role: msg.role, content: msg.content });
    }

    // Add the new user message
    nvidiaMessages.push({ role: "user", content: message });

    // --- Save user message ---
    await db.chatMessage.create({
      data: {
        sessionId: chatSession.id,
        role: "user",
        content: message,
      },
    });

    // --- Call NVIDIA Nemotron API (streaming) ---
    if (!NVIDIA_API_KEY) {
      throw new Error("NVIDIA_API_KEY is not configured");
    }

    console.log("[Chat] Calling NVIDIA Nemotron API (streaming)...");
    console.log(`[Chat] Last user message: "${message.substring(0, 100)}"`);
    console.log(`[Chat] Context messages: ${previousMessages.length} previous messages loaded`);
    const startTime = Date.now();

    // --- Get tool definitions for this mode ---
    const tools = getToolsForMode(mode);
    // Fix 2: Dynamic tool budget based on query complexity
    const MAX_TOOL_CALLS = calculateToolBudget(message, mode);
    console.log(`[Chat] Tool Budget: ${MAX_TOOL_CALLS} (mode=${mode}, query="${message.substring(0, 60)}...")`);

    // ── Smart Drug Lookup Interceptor ──
    // Before calling Nemotron, check if the user's message matches a known
    // drug-target co-crystal. If so, fetch the PDB directly and stream the
    // result — zero GPU time, instant response, experimentally determined structure.
    const knownCocrystal = lookupKnownCocrystal(message);
    if (knownCocrystal) {
      console.log(`[SmartDrugLookup] Match found: ${knownCocrystal.drug} + ${knownCocrystal.target} → PDB ${knownCocrystal.pdb_id}`);

      const stream = new ReadableStream({
        async start(controller) {
          const encoder = new TextEncoder();
          let streamClosed = false;
          const send = (event: StreamEvent) => {
            if (streamClosed) return;
            try {
              controller.enqueue(encoder.encode(encodeEvent(event)));
            } catch (e) {
              streamClosed = true;
            }
          };

          try {
            // 0. Send a tool_call event so the frontend creates an assistant message entry
            //    The frontend's SSE parser expects tool_call events to create the message
            const toolCallId = "smart-drug-lookup-1";
            send({
              event: "tool_call",
              data: {
                toolCallId,
                toolName: "fetch_pdb_entry",
                status: "calling",
                args: { pdb_id: knownCocrystal.pdb_id },
              },
            });

            // 1. Stream reasoning (chain of thought)
            send({
              event: "reasoning_delta",
              data: {
                delta: `User is asking about ${knownCocrystal.drug} and ${knownCocrystal.target}. I found an experimentally determined co-crystal structure in PDB: ${knownCocrystal.pdb_id}. Fetching the structure directly from RCSB instead of running slow GPU prediction.`,
              },
            });

            // 2. Fetch the PDB coordinates directly from RCSB
            //    The proto-tools pdb_fetch_entry only returns metadata, not coordinates.
            //    We fetch the actual PDB file from the RCSB download URL.
            const pdbUrl = `https://files.rcsb.org/download/${knownCocrystal.pdb_id}.pdb`;
            const pdbResponse = await fetch(pdbUrl, {
              signal: AbortSignal.timeout(30000),
            });

            if (!pdbResponse.ok) {
              throw new Error(`Failed to fetch PDB: ${pdbResponse.status}`);
            }

            const pdbString = await pdbResponse.text();

            if (pdbString && pdbString.length > 100 && pdbString.startsWith("HEADER")) {
              // 3. Send tool_result so the frontend marks the tool as completed
              send({
                event: "tool_result",
                data: {
                  toolCallId,
                  toolName: "fetch_pdb_entry",
                  status: "completed",
                  result: { pdb_id: knownCocrystal.pdb_id, title: knownCocrystal.target },
                  executionTimeMs: 500,
                },
              });

              // 4. Send the 3D structure payload for the in-chat viewer
              //    The frontend matches this to the toolCallId we sent above
              send({
                event: "3d_structure_payload",
                data: {
                  toolCallId,
                  pdbString,
                  title: `${knownCocrystal.target} + ${knownCocrystal.drug} (PDB: ${knownCocrystal.pdb_id})`,
                  geneSymbol: knownCocrystal.target,
                  source: "pdb_experimental",
                },
              });

              // 5. Stream the explanation with experimental data
              const ic50Text = knownCocrystal.ic50_nM
                ? `**Experimental IC50:** ${knownCocrystal.ic50_nM < 1 ? knownCocrystal.ic50_nM.toFixed(2) : knownCocrystal.ic50_nM} nM`
                : "IC50 data not available";
              const explanation = `I found an experimentally determined co-crystal structure for **${knownCocrystal.drug}** bound to **${knownCocrystal.target}** (PDB: ${knownCocrystal.pdb_id}).

### Experimentally Determined Structure

**PDB ID:** ${knownCocrystal.pdb_id}
**Drug:** ${knownCocrystal.drug}
**Target:** ${knownCocrystal.target}
**Binding Mode:** ${knownCocrystal.binding_type}
${ic50Text}
${knownCocrystal.mechanism ? `\n**Mechanism:** ${knownCocrystal.mechanism}` : ""}

This is an **X-ray crystallography structure** — far more accurate than any computational prediction. The 3D viewer above shows the protein (ribbons) with ${knownCocrystal.drug} (sticks) bound in the active site.

${knownCocrystal.mechanism && knownCocrystal.mechanism.includes("Synthetic Lethal") ? "**Why this drug works for BRCA mutations:** BRCA1/2 mutations disable Homologous Recombination (HR) repair. The cell becomes entirely dependent on PARP-mediated repair. By trapping PARP1 on DNA, this drug causes lethal double-strand breaks that only the BRCA-deficient (HR-impaired) cell cannot fix. Healthy cells with functional BRCA1/2 survive. This is the principle of **Synthetic Lethality** — the basis of PARP inhibitor therapy.\n\n" : ""}**Why this matters:** Using experimentally determined structures avoids the 10+ minute wait of de novo structure prediction (Boltz2) and provides atomic-resolution accuracy (typically 1.5–2.5 Å).

> ⚠️ This is a research tool. Clinical decisions require professional pharmacological assessment.`;

              send({
                event: "content_delta",
                data: { delta: explanation },
              });
            } else {
              send({
                event: "content_delta",
                data: { delta: `I found a known co-crystal structure (PDB: ${knownCocrystal.pdb_id}), but encountered an issue fetching the PDB file. You can view it directly at https://www.rcsb.org/structure/${knownCocrystal.pdb_id}` },
              });
            }

            send({ event: "done", data: {} });
          } catch (error) {
            console.error("[SmartDrugLookup] Error:", error);
            send({
              event: "content_delta",
              data: { delta: `I found a known co-crystal structure for ${knownCocrystal.drug} + ${knownCocrystal.target} (PDB: ${knownCocrystal.pdb_id}), but encountered an error fetching it. You can view it at https://www.rcsb.org/structure/${knownCocrystal.pdb_id}` },
            });
            send({ event: "done", data: {} });
          }

          controller.close();
        },
      });

      return new Response(stream, {
        headers: {
          "Content-Type": "text/event-stream",
          "Cache-Control": "no-cache",
          Connection: "keep-alive",
        },
      });
    }

    const stream = new ReadableStream({
      async start(controller) {
        const encoder = new TextEncoder();
        let streamClosed = false;
        const send = (event: StreamEvent) => {
          if (streamClosed) return;
          try {
            controller.enqueue(encoder.encode(encodeEvent(event)));
          } catch (e) {
            streamClosed = true;
            console.log("[Chat] Stream closed, stopping sends");
          }
        };

        try {
          let reasoningContent = "";
          let assistantContent = "";
          let toolCallCount = 0;
          let usage:
            | { total_tokens?: number; prompt_tokens?: number; completion_tokens?: number }
            | undefined;
          // Track accumulated tool calls for this message
          const messageToolCalls: Array<{
            toolCallId: string;
            toolName: string;
            status: "calling" | "completed" | "failed";
            args: Record<string, unknown>;
            result?: Record<string, unknown>;
            error?: string;
            executionTimeMs?: number;
          }> = [];

          // ─── Tool-calling loop (max MAX_TOOL_CALLS iterations) ───────────
          while (toolCallCount <= MAX_TOOL_CALLS) {
            // Call NVIDIA Nemotron API
            const nvidiaBody: Record<string, unknown> = {
              model: MODEL,
              messages: nvidiaMessages,
              temperature: 0.3,
              top_p: 0.9,
              max_tokens: 4096,
              stream: true,
            };

            // Add tools if we have any
            if (tools.length > 0) {
              nvidiaBody.tools = tools;
              nvidiaBody.tool_choice = "auto";
            }

            const nvidiaResponse = await fetch(NVIDIA_API_URL, {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
                Authorization: `Bearer ${NVIDIA_API_KEY}`,
              },
              body: JSON.stringify(nvidiaBody),
              signal: AbortSignal.timeout(900_000), // 15 min — Boltz2 can take 5+ min
            });

            if (!nvidiaResponse.ok) {
              const errorText = await nvidiaResponse.text();
              console.error("[Chat] NVIDIA API error:", errorText);
              throw new Error(`NVIDIA API error: ${nvidiaResponse.status}`);
            }

            if (!nvidiaResponse.body) {
              throw new Error("NVIDIA returned an empty response body");
            }

            // Parse the SSE stream from NVIDIA
            const reader = nvidiaResponse.body.getReader();
            const decoder = new TextDecoder();
            let buffer = "";
            let finishReason: string | null = null;
            // Accumulate tool calls from this iteration
            const iterationToolCalls: Array<{
              id: string;
              name: string;
              arguments: string;
            }> = [];

            while (true) {
              const { done, value } = await reader.read();
              if (done) break;

              buffer += decoder.decode(value, { stream: true });
              const lines = buffer.split("\n");
              buffer = lines.pop() || "";

              for (const line of lines) {
                const trimmed = line.trim();
                if (!trimmed || !trimmed.startsWith("data:")) continue;

                const dataStr = trimmed.slice(5).trim();
                if (dataStr === "[DONE]") continue;

                try {
                  const parsed = JSON.parse(dataStr) as {
                    choices?: Array<{
                      delta?: {
                        content?: string;
                        reasoning_content?: string;
                        tool_calls?: Array<{
                          id: string;
                          function: { name: string; arguments: string };
                        }>;
                      };
                      finish_reason?: string | null;
                    }>;
                    usage?: {
                      total_tokens?: number;
                      prompt_tokens?: number;
                      completion_tokens?: number;
                    };
                  };

                  const delta = parsed.choices?.[0]?.delta;
                  if (delta?.reasoning_content) {
                    reasoningContent += delta.reasoning_content;
                    send({
                      event: "reasoning_delta",
                      data: { delta: delta.reasoning_content },
                    });
                  }
                  if (delta?.content) {
                    assistantContent += delta.content;
                    send({
                      event: "content_delta",
                      data: { delta: delta.content },
                    });
                  }
                  if (delta?.tool_calls) {
                    for (const tc of delta.tool_calls) {
                      // Accumulate tool call (arguments may arrive in chunks)
                      const existing = iterationToolCalls.find(
                        (t) => t.id === tc.id,
                      );
                      if (existing) {
                        existing.arguments += tc.function.arguments;
                      } else {
                        iterationToolCalls.push({
                          id: tc.id,
                          name: tc.function.name,
                          arguments: tc.function.arguments,
                        });
                      }
                    }
                  }
                  if (parsed.choices?.[0]?.finish_reason) {
                    finishReason = parsed.choices[0].finish_reason;
                  }
                  if (parsed.usage) {
                    usage = parsed.usage;
                  }
                } catch {
                  // Ignore malformed SSE chunks
                }
              }
            }

            // ─── FALLBACK: Detect tool calls emitted as text (not structured) ──
            // Nemotron-3 sometimes outputs tool calls as JSON in reasoning_content
            // or content instead of using the structured tool_calls field. This
            // fallback parser scans the accumulated text for tool call patterns
            // and extracts them for execution.
            if (
              iterationToolCalls.length === 0 &&
              toolCallCount < MAX_TOOL_CALLS
            ) {
              // Patterns to match:
              // {"tool": "fetch_alphafold_db", "arguments": {"uniprot_id": "P38398"}}
              // {"tool_call": {"name": "fetch_alphafold_db", "arguments": {...}}}
              // ```json\n{"tool": "fetch_alphafold_db", ...}\n```
              const toolCallPattern =
                /\{[\s]*"tool"[\s]*:[\s]*"([a-zA-Z0-9_]+)"[\s]*,[\s]*"arguments"[\s]*:[\s]*(\{[^}]*\})[\s]*\}/g;
              const altPattern =
                /\{[\s]*"tool_call"[\s]*:[\s]*\{[\s]*"name"[\s]*:[\s]*"([a-zA-Z0-9_]+)"[\s]*,[\s]*"arguments"[\s]*:[\s]*(\{[^}]*\})[\s]*\}[\s]*\}/g;

              const combinedText = reasoningContent + "\n" + assistantContent;

              let match: RegExpExecArray | null;
              // Try primary pattern
              while ((match = toolCallPattern.exec(combinedText)) !== null) {
                const toolName = match[1];
                const argsStr = match[2];
                try {
                  JSON.parse(argsStr); // Validate JSON
                  iterationToolCalls.push({
                    id: `fallback-${toolCallCount}-${iterationToolCalls.length}`,
                    name: toolName,
                    arguments: argsStr,
                  });
                  console.log(`[Chat] Fallback parser found tool call: ${toolName} with args: ${argsStr}`);
                } catch {
                  console.warn(`[Chat] Fallback parser found ${toolName} but args JSON is invalid: ${argsStr}`);
                }
              }
              // Try alternate pattern if primary found nothing
              if (iterationToolCalls.length === 0) {
                while ((match = altPattern.exec(combinedText)) !== null) {
                  const toolName = match[1];
                  const argsStr = match[2];
                  try {
                    JSON.parse(argsStr);
                    iterationToolCalls.push({
                      id: `fallback-${toolCallCount}-${iterationToolCalls.length}`,
                      name: toolName,
                      arguments: argsStr,
                    });
                    console.log(`[Chat] Fallback parser (alt) found tool call: ${toolName} with args: ${argsStr}`);
                  } catch {
                    console.warn(`[Chat] Fallback parser (alt) found ${toolName} but args JSON is invalid: ${argsStr}`);
                  }
                }
              }

              // If we found tool calls via fallback, override finishReason
              if (iterationToolCalls.length > 0) {
                finishReason = "tool_calls";
                // Clear the assistant content that was streamed — it contained
                // the raw JSON which we don't want the user to see.
                if (assistantContent) {
                  send({
                    event: "content_clear",
                    data: {},
                  });
                  // Reset assistantContent so it doesn't get persisted with JSON junk
                  assistantContent = "";
                }
              }
            }

            // ─── Check if Nemotron wants to call tools ────────────────────
            if (
              iterationToolCalls.length > 0 &&
              finishReason === "tool_calls" &&
              toolCallCount < MAX_TOOL_CALLS
            ) {
              // Execute each tool call
              for (const tc of iterationToolCalls) {
                let parsedArgs: Record<string, unknown> = {};
                try {
                  parsedArgs = JSON.parse(tc.arguments);
                } catch {
                  parsedArgs = { _raw: tc.arguments };
                }

                // Send "calling" status to frontend
                send({
                  event: "tool_call",
                  data: {
                    toolCallId: tc.id,
                    toolName: tc.name,
                    status: "calling",
                    args: parsedArgs,
                  },
                });

                const messageToolCall: {
                  toolCallId: string;
                  toolName: string;
                  status: "calling" | "completed" | "failed";
                  args: Record<string, unknown>;
                  result?: Record<string, unknown>;
                  error?: string;
                  executionTimeMs?: number;
                } = {
                  toolCallId: tc.id,
                  toolName: tc.name,
                  status: "calling",
                  args: parsedArgs,
                };
                messageToolCalls.push(messageToolCall);

                // Fix 3: Smart Input Resolution — resolve missing prerequisites silently
                parsedArgs = await resolveToolInputs(tc.name, parsedArgs);

                // Execute the tool via Modal
                console.log(`[Chat] Executing tool: ${tc.name} with args:`, parsedArgs);
                const toolResult = await executeProtoTool(tc.name, parsedArgs);

                // Update status
                messageToolCall.status = toolResult.status;
                messageToolCall.result = toolResult.result;
                messageToolCall.error = toolResult.error;
                messageToolCall.executionTimeMs = toolResult.executionTimeMs;

                // ── SPLIT STREAM: Send raw PDB to browser for 3D rendering ──
                // The sanitizer strips PDB coordinates from Nemotron's context (LLMs can't read 3D coords),
                // but the browser CAN render them. We intercept the raw PDB here and send it via a
                // separate SSE event so the frontend can render the 3D viewer inline in the chat.
                if (toolResult.status === "completed" && toolResult.result) {
                  const rawResult = toolResult.result as Record<string, unknown>;
                  // Debug: log the keys of the result to find where PDB data is stored
                  console.log(`[Chat] Tool result keys for ${tc.name}:`, Object.keys(rawResult));
                  // Check for PDB string in various possible field names
                  let pdbString: string | null = null;
                  
                  // Try direct string fields
                  if (typeof rawResult.structure === 'string') pdbString = rawResult.structure as string;
                  else if (typeof rawResult.pdb_string === 'string') pdbString = rawResult.pdb_string as string;
                  else if (typeof rawResult.pdb === 'string') pdbString = rawResult.pdb as string;
                  else if (typeof rawResult.pdb_data === 'string') pdbString = rawResult.pdb_data as string;
                  else if (typeof rawResult.coordinates === 'string') pdbString = rawResult.coordinates as string;
                  else if (rawResult.results && Array.isArray(rawResult.results)) {
                    const firstResult = (rawResult.results as Array<Record<string, unknown>>)[0];
                    if (firstResult) {
                      if (typeof firstResult.pdb_string === 'string') pdbString = firstResult.pdb_string as string;
                      else if (typeof firstResult.pdb === 'string') pdbString = firstResult.pdb as string;
                    }
                  }

                  // ── FALLBACK 1: If tool returned a pdb_url, fetch the PDB directly ──
                  if (!pdbString && tc.name === "fetch_alphafold_db" && rawResult.pdb_url) {
                    const pdbUrl = rawResult.pdb_url as string;
                    console.log(`[Chat] Fetching PDB from tool-provided URL: ${pdbUrl}`);
                    try {
                      const pdbResponse = await fetch(pdbUrl, { signal: AbortSignal.timeout(15000) });
                      if (pdbResponse.ok) {
                        pdbString = await pdbResponse.text();
                        console.log(`[Chat] Fetched PDB from URL: ${pdbString.length} chars`);
                      }
                    } catch (fetchErr) {
                      console.error(`[Chat] Failed to fetch PDB from URL:`, fetchErr);
                    }
                  }

                  // ── FALLBACK 2: If still no PDB, construct URL from UniProt ID ──
                  if (!pdbString && tc.name === "fetch_alphafold_db" && parsedArgs.uniprot_id) {
                    const uniprotId = String(parsedArgs.uniprot_id);
                    console.log(`[Chat] No PDB in tool result, fetching directly from AlphaFold API for ${uniprotId}`);
                    try {
                      // Try multiple model versions (v4, v5, v6) — AlphaFold updates incrementally
                      const modelVersions = ["v4", "v5", "v6"];
                      for (const ver of modelVersions) {
                        // Try PDB format first
                        const pdbUrl = `https://alphafold.ebi.ac.uk/files/AF-${uniprotId}-F1-model_${ver}.pdb`;
                        const pdbResponse = await fetch(pdbUrl, { signal: AbortSignal.timeout(10000) });
                        if (pdbResponse.ok) {
                          pdbString = await pdbResponse.text();
                          console.log(`[Chat] Fetched PDB from AlphaFold (${ver}): ${pdbString.length} chars`);
                          break;
                        }
                        // Try mmCIF format
                        const cifUrl = `https://alphafold.ebi.ac.uk/files/AF-${uniprotId}-F1-model_${ver}.cif`;
                        const cifResponse = await fetch(cifUrl, { signal: AbortSignal.timeout(10000) });
                        if (cifResponse.ok) {
                          pdbString = await cifResponse.text();
                          console.log(`[Chat] Fetched mmCIF from AlphaFold (${ver}): ${pdbString.length} chars`);
                          break;
                        }
                      }
                    } catch (fetchErr) {
                      console.error(`[Chat] Failed to fetch PDB directly:`, fetchErr);
                    }
                  }

                  if (pdbString && (tc.name === "fetch_alphafold_db" || tc.name === "run_esmfold_prediction")) {
                    const geneSymbol = (report as Record<string, unknown> | null)?.geneSymbol as string || "Protein";
                    send({
                      event: "3d_structure_payload",
                      data: {
                        toolCallId: tc.id,
                        pdbString: pdbString,
                        title: tc.name === "fetch_alphafold_db" ? "AlphaFold DB Structure" : "ESMFold Prediction",
                        geneSymbol: geneSymbol,
                      },
                    });
                    console.log(`[Chat] Sent 3D structure payload for ${tc.name} (${pdbString.length} chars)`);
                  }
                }

                // Send result to frontend
                // Send a COMPACT version to the frontend UI to prevent SSE parsing issues
                // (large raw results can exceed the SSE chunk buffer and cause the
                // tool_result event to be silently dropped, leaving the spinner stuck)
                const compactResultStr = toolResult.status === "completed"
                  ? sanitizeToolResponse(tc.name, toolResult.result)
                  : JSON.stringify({ error: toolResult.error });
                let compactResultParsed: Record<string, unknown> = {};
                try {
                  compactResultParsed = JSON.parse(compactResultStr);
                } catch {
                  compactResultParsed = { _raw: compactResultStr.substring(0, 500) };
                }
                send({
                  event: "tool_result",
                  data: {
                    toolCallId: tc.id,
                    toolName: tc.name,
                    status: toolResult.status,
                    result: compactResultParsed,
                    error: toolResult.error,
                    executionTimeMs: toolResult.executionTimeMs,
                  },
                });

                // Add tool result to messages for next Nemotron call
                // The assistant message must include the tool_calls array
                nvidiaMessages.push({
                  role: "assistant",
                  content: "",
                  tool_calls: [
                    {
                      id: tc.id,
                      type: "function",
                      function: {
                        name: tc.name,
                        arguments: tc.arguments,
                      },
                    },
                  ],
                } as { role: string; content: string; tool_calls?: unknown });
                // Fix 1: Sanitize tool response before feeding to Nemotron
                // Replaces naive 50K truncation with tool-specific extraction
                const toolResultData = toolResult.status === "completed"
                  ? toolResult.result
                  : { error: toolResult.error };
                const toolResultStr = sanitizeToolResponse(tc.name, toolResultData);
                nvidiaMessages.push({
                  role: "tool",
                  tool_call_id: tc.id,
                  content: toolResultStr,
                } as { role: string; content: string; tool_call_id?: string });
              }

              toolCallCount++;
              // Continue the loop — Nemotron will process tool results
              continue;
            }

            // No tool calls or max reached — we're done
            break;
          }

          const latency = Date.now() - startTime;

          // --- Persist assistant message ---
          await db.chatMessage.create({
            data: {
              sessionId: chatSession.id,
              role: "assistant",
              content: assistantContent || "(no response)",
              metadata: {
                latency_ms: latency,
                model: MODEL,
                tokens: usage,
                reasoning_content: reasoningContent || undefined,
                tool_calls:
                  messageToolCalls.length > 0
                    ? JSON.parse(JSON.stringify(messageToolCalls))
                    : undefined,
              },
            },
          });

          // --- Update session timestamp ---
          await db.chatSession.update({
            where: { id: chatSession.id },
            data: { updatedAt: new Date() },
          });

          send({
            event: "done",
            data: {
              sessionId: chatSession.id,
              latencyMs: latency,
              tokens: usage,
            },
          });
          controller.close();
        } catch (err) {
          console.error("[Chat] Streaming error:", err);
          send({
            event: "error",
            data: {
              error: err instanceof Error ? err.message : "Streaming failed",
            },
          });
          controller.close();
        }
      },
    });

    return new Response(stream, {
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
      },
    });
  } catch (error) {
    console.error("[Chat] Error:", error);
    return NextResponse.json(
      {
        error: error instanceof Error ? error.message : "Internal server error",
      },
      { status: 500 },
    );
  }
}

// =============================================================================
// GET /api/chat — Load chat history for a session
// =============================================================================

export async function GET(request: NextRequest) {
  try {
    const user = await currentUser();
    if (!user) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const { searchParams } = new URL(request.url);
    const sessionId = searchParams.get("sessionId");
    const reportId = searchParams.get("reportId");

    if (sessionId) {
      // Load specific session
      const session = await db.chatSession.findFirst({
        where: { id: sessionId, clerkUserId: user.id },
        include: {
          messages: { orderBy: { createdAt: "asc" } },
        },
      });

      if (!session) {
        return NextResponse.json({ error: "Session not found" }, { status: 404 });
      }

      return NextResponse.json({ session });
    }

    if (reportId) {
      // Load all sessions for a report
      const sessions = await db.chatSession.findMany({
        where: { analysisReportId: reportId, clerkUserId: user.id },
        orderBy: { updatedAt: "desc" },
        include: {
          messages: { orderBy: { createdAt: "asc" }, take: 1 },
        },
      });

      return NextResponse.json({ sessions });
    }

    // Load all user's chat sessions
    const sessions = await db.chatSession.findMany({
      where: { clerkUserId: user.id },
      orderBy: { updatedAt: "desc" },
      take: 20,
      include: {
        messages: { orderBy: { createdAt: "asc" }, take: 1 },
      },
    });

    return NextResponse.json({ sessions });
  } catch (error) {
    console.error("[Chat] Error loading history:", error);
    return NextResponse.json(
      { error: "Failed to load chat history" },
      { status: 500 },
    );
  }
}
