// =============================================================================
// Chat Tools — Nemotron Function Definitions for Tool Calling
// =============================================================================
// These definitions are sent to the NVIDIA Nemotron API alongside messages.
// When Nemotron decides it needs external data, it returns a tool_call with
// the function name and arguments. The /api/chat route executes the tool
// via the Modal proto-tools endpoint and feeds the result back.
//
// Tools are grouped by tier:
//   Tier 1: Database retrieval (fast, <1s) — always available
//   Tier 2: CPU ML tools (5-30s) — available when needed
// =============================================================================

interface ToolParameter {
  type: string;
  description: string;
  enum?: string[];
}

interface ToolFunction {
  name: string;
  description: string;
  parameters: {
    type: string;
    properties: Record<string, ToolParameter>;
    required: string[];
  };
}

interface ToolDefinition {
  type: "function";
  function: ToolFunction;
}

function defineTool(
  name: string,
  description: string,
  properties: Record<string, ToolParameter>,
  required: string[],
): ToolDefinition {
  return {
    type: "function",
    function: {
      name,
      description,
      parameters: {
        type: "object",
        properties,
        required,
      },
    },
  };
}

// =============================================================================
// Tier 1: Database Retrieval Tools (<1s, always available)
// =============================================================================

const TIER1_TOOLS: ToolDefinition[] = [
  defineTool(
    "fetch_uniprot",
    "USE THIS when the user asks about protein domains, function, disease associations, or subcellular location. Do NOT use for structural analysis — use fetch_alphafold_db instead. Do NOT use for sequence retrieval — use fetch_ensembl_sequence instead. Returns protein name, domains, function, disease associations, and subcellular location for a UniProt accession.",
    {
      uniprot_id: {
        type: "string",
        description: "UniProt accession (e.g. P38398 for BRCA1, P04637 for TP53, O15105 for MSH2)",
      },
    },
    ["uniprot_id"],
  ),
  defineTool(
    "fetch_alphafold_db",
    "USE THIS when the user asks about 3D protein structure, structural context, pLDDT confidence scores, or PAE matrix. Do NOT use for domain annotations — use run_interproscan_fetch instead. Returns predicted 3D structure from AlphaFold Protein Structure Database.",
    {
      uniprot_id: {
        type: "string",
        description: "UniProt accession (e.g. P38398)",
      },
    },
    ["uniprot_id"],
  ),
  defineTool(
    "fetch_alphamissense",
    "USE THIS when the user asks about missense variant pathogenicity, clinical significance of a specific amino acid substitution, or per-residue pathogenicity scores. Do NOT use for indels, frameshifts, or non-coding variants. Returns AlphaMissense pathogenicity scores for all possible substitutions at each residue.",
    {
      uniprot_id: {
        type: "string",
        description: "UniProt accession",
      },
    },
    ["uniprot_id"],
  ),
  defineTool(
    "run_ensembl_vep",
    "USE THIS when the user asks to annotate a variant, determine its molecular consequence, or translate HGVS notation into protein impact. Returns molecular consequence, impact level, amino acid change, and transcript information.",
    {
      hgvs: {
        type: "string",
        description: "HGVS notation (e.g. '17:g.43094169A>C' or 'ENST00000357654:c.5074G>A')",
      },
    },
    ["hgvs"],
  ),
  defineTool(
    "fetch_ensembl_sequence",
    "USE THIS when the user needs a reference DNA/cDNA/protein sequence for a gene or transcript. Do NOT use for protein domain info — use fetch_uniprot instead. Returns the nucleotide or amino acid sequence for an Ensembl ID.",
    {
      ensembl_id: {
        type: "string",
        description: "Ensembl ID (e.g. ENST00000357654 for transcript, ENSP00000350183 for protein)",
      },
    },
    ["ensembl_id"],
  ),
  defineTool(
    "fetch_pdb_entry",
    "USE THIS when the user asks about experimental protein structures, PDB entries, or crystallographic resolution. Returns structure metadata from RCSB PDB including title, experimental method, and resolution.",
    {
      pdb_id: {
        type: "string",
        description: "PDB ID (e.g. 1BRCA, 1TUP)",
      },
    },
    ["pdb_id"],
  ),
  defineTool(
    "search_ncbi",
    "USE THIS when the user wants to search for literature (PubMed), protein sequences, or ClinVar records. Do NOT use to fetch full sequences — use fetch_ncbi_efetch instead. Returns matching IDs from NCBI Entrez databases.",
    {
      db: {
        type: "string",
        enum: ["protein", "nuccore", "pubmed", "clinvar"],
        description: "NCBI database to search (e.g. 'pubmed' for literature, 'protein' for sequences)",
      },
      search_term: {
        type: "string",
        description: "Search query (e.g. 'BRCA1 missense variant')",
      },
    },
    ["db", "search_term"],
  ),
  defineTool(
    "fetch_ncbi_efetch",
    "USE THIS when the user wants to retrieve actual FASTA sequences from NCBI after a search. Do NOT use for searching — use search_ncbi first. Returns FASTA records from NCBI protein or nucleotide databases.",
    {
      db: {
        type: "string",
        enum: ["protein", "nuccore"],
        description: "NCBI database: protein or nucleotide",
      },
      identifier: {
        type: "string",
        description: "Accession or GI number (e.g. 'NP_009225.1' for BRCA1 protein)",
      },
    },
    ["db", "identifier"],
  ),
  defineTool(
    "fetch_ncbi_esummary",
    "USE THIS after running search_ncbi to get metadata about matching records. Returns title, organism, length, and other summary data from NCBI Entrez.",
    {
      db: {
        type: "string",
        description: "NCBI database (e.g. 'protein', 'pubmed', 'clinvar')",
      },
      identifier: {
        type: "string",
        description: "Entrez UID (comma-separated for multiple, e.g. '12345,67890')",
      },
    },
    ["db", "identifier"],
  ),
  defineTool(
    "fetch_ensembl_lookup",
    "USE THIS when the user needs gene metadata, genomic coordinates, or biotype information. Returns chromosome, start/end position, biotype, and description for an Ensembl gene ID.",
    {
      ensembl_id: {
        type: "string",
        description: "Ensembl gene ID (e.g. 'ENSG00000012048') or gene symbol (e.g. 'BRCA1')",
      },
      species: {
        type: "string",
        description: "Species slug (e.g. 'homo_sapiens'). Default: homo_sapiens",
      },
    },
    ["ensembl_id"],
  ),
  defineTool(
    "fetch_pdb_fasta",
    "USE THIS when the user needs the actual amino acid or nucleotide sequence of a PDB structure entry. Returns chain sequences from RCSB PDB with protein/nucleotide classification.",
    {
      pdb_id: {
        type: "string",
        description: "PDB ID (e.g. '1JM7' for BRCA1 RING domain, '1T29' for BRCT)",
      },
    },
    ["pdb_id"],
  ),
  defineTool(
    "fetch_pubchem",
    "USE THIS when the user asks about drug compounds, small molecules, or chemical structures. Do NOT use for proteins — use fetch_uniprot instead. Returns canonical structure data, synonyms, and molecular formula from PubChem.",
    {
      identifier: {
        type: "string",
        description: "Compound identifier: CID number, name (e.g. 'aspirin'), SMILES, or InChIKey",
      },
      namespace: {
        type: "string",
        enum: ["cid", "name", "smiles", "inchikey"],
        description: "Type of identifier. Default: name",
      },
    },
    ["identifier"],
  ),
];

// =============================================================================
// Tier 2: CPU ML Tools (5-30s, available when needed)
// =============================================================================

const TIER2_TOOLS: ToolDefinition[] = [
  // ── Foldseek moved to TOP of array to combat 'lost in the middle' syndrome ──
  defineTool(
    "run_foldseek_search",
    "USE THIS when the user asks to search for structurally similar proteins, find structural homologs, or compare 3D structures. Input is a PDB structure string. Returns structurally related proteins with TM-scores. Do NOT use for sequence similarity — use run_blast_search instead.",
    {
      structure: {
        type: "string",
        description: "PDB format structure string",
      },
      database: {
        type: "string",
        description: "Database (e.g. 'alphafolddb', 'pdb100'). Default: alphafolddb",
      },
    },
    ["structure"],
  ),
  defineTool(
    "run_spliceai_predict",
    "USE THIS when the user asks about splice sites, splice junctions, or splice-site probabilities from a DNA sequence. Do NOT use for tissue-specific splicing — use run_pangolin_predict instead. Returns per-position [neither, acceptor, donor] probabilities using SpliceAI deep learning model.",
    {
      sequences: {
        type: "string",
        description: "DNA sequence(s) as a single string or comma-separated list. Each sequence should be at least 400bp for accurate prediction.",
      },
    },
    ["sequences"],
  ),
  defineTool(
    "run_pangolin_predict",
    "USE THIS when the user asks about tissue-specific splicing or splice probabilities across different tissues. Do NOT use for generic splice prediction — use run_spliceai_predict instead. Returns per-position splice probabilities across tissues using Pangolin.",
    {
      sequences: {
        type: "string",
        description: "DNA sequence(s). Each sequence should be at least 5000bp for accurate prediction.",
      },
    },
    ["sequences"],
  ),
  defineTool(
    "run_pangolin_score_variants",
    "USE THIS when the user asks about splice-altering effects of a specific variant. Returns gain/loss scores for acceptor/donor splice sites. Delta >0.2 suggests splice disruption. Requires the DNA sequence (>=5000bp flank on each side of the variant).",
    {
      variants: {
        type: "array",
        description: "Array of variant objects to score",
        items: {
          type: "object",
          properties: {
            sequence: {
              type: "string",
              description: "Full DNA sequence including at least 5000bp on each side of the variant position",
            },
            variant_position: {
              type: "number",
              description: "0-based position of the variant within the sequence",
            },
            reference_bases: {
              type: "string",
              description: "Reference allele at the variant position (e.g. 'A')",
            },
            alternate_bases: {
              type: "string",
              description: "Alternate allele at the variant position (e.g. 'C')",
            },
          },
        },
      },
    },
    ["variants"],
  ),
  defineTool(
    "run_dssp_secondary_structure",
    "USE THIS when the user asks about helix/sheet/loop percentages or secondary structure composition of a protein. Do NOT use for overall structure quality — use run_structure_metrics instead. Assigns secondary structure from a PDB structure file using DSSP.",
    {
      inputs: {
        type: "string",
        description: "PDB format structure string",
      },
    },
    ["inputs"],
  ),
  defineTool(
    "run_interproscan_fetch",
    "USE THIS when the user asks for comprehensive domain annotations, Pfam domains, SMART domains, or InterPro entries. Do NOT use for basic protein info — use fetch_uniprot instead. Returns domain hits across Pfam, SMART, PROSITE, Gene3D, Panther, and all InterPro member databases.",
    {
      uniprot_id: {
        type: "string",
        description: "UniProt accession for direct lookup (e.g. 'P38398' for BRCA1, 'P04637' for TP53)",
      },
      sequence: {
        type: "string",
        description: "Raw protein sequence for submit-and-scan path (requires email in config)",
      },
    },
    [],
  ),
  defineTool(
    "run_structure_metrics",
    "USE THIS when the user asks about structure quality, gyration radius, longest helix, or overall structural composition. Do NOT use for secondary structure percentages — use run_dssp_secondary_structure instead. Computes structural quality metrics from a PDB file.",
    {
      structures: {
        type: "string",
        description: "PDB format structure string",
      },
    },
    ["structures"],
  ),
  defineTool(
    "run_blast_search",
    "USE THIS when the user asks to search for homologous sequences, find similar proteins or genes in other organisms, or run BLAST. Returns hits with E-values, percent identity, and alignments. Do NOT use for structural similarity — use run_foldseek_search instead.",
    {
      query: {
        type: "string",
        description: "Query sequence (protein or DNA) or path to a FASTA file",
      },
      program: {
        type: "string",
        enum: ["blastp", "blastn", "blastx", "tblastn", "tblastx"],
        description: "BLAST program. Default: blastn",
      },
      database: {
        type: "string",
        enum: ["nt", "nr", "refseq_rna", "refseq_protein", "swissprot", "pdb"],
        description: "NCBI database to search (online mode). Default: nt",
      },
    },
    ["query"],
  ),
  defineTool(
    "run_mmseqs2_search_proteins",
    "USE THIS when the user needs fast protein sequence search or rapid homology detection. Faster than BLAST with similar sensitivity. Do NOT use for structural homology — use run_foldseek_search instead. Returns per-sequence results with E-values and alignment info.",
    {
      query_sequences: {
        type: "array",
        description: "Array of protein query sequences to search",
        items: { type: "string" },
      },
    },
    ["query_sequences"],
  ),
  defineTool(
    "run_mafft_align",
    "USE THIS when the user asks to align sequences, perform multiple sequence alignment, or compare homologous sequences for conservation analysis. Returns aligned sequences using MAFFT.",
    {
      sequences: {
        type: "array",
        description: "Array of sequences to align (minimum 2 required)",
        items: { type: "string" },
      },
    },
    ["sequences"],
  ),
  defineTool(
    "run_segmasker_score",
    "USE THIS when the user asks about low-complexity regions, compositionally biased sequences, or SEG masking before homology searches. Returns per-sequence low-complexity fractions and counts.",
    {
      sequences: {
        type: "array",
        description: "Array of protein sequences to analyze for low-complexity regions",
        items: { type: "string" },
      },
    },
    ["sequences"],
  ),
  // ── ViennaRNA moved to BOTTOM of array to combat 'lost in the middle' syndrome ──
  defineTool(
    "run_viennarna_prediction",
    "USE THIS when the user asks about RNA secondary structure, RNA folding, MFE (minimum free energy), or hairpin/stem-loop prediction. Do NOT use for protein structure — use fetch_alphafold_db instead. Returns the MFE structure in dot-bracket notation and the free energy.",
    {
      sequences: {
        type: "array",
        description: "Array of RNA sequences (DNA will be converted: T→U). At least 50bp recommended.",
        items: { type: "string" },
      },
    },
    ["sequences"],
  ),
];

// =============================================================================
// Tier 3: GPU Tools (30s–5min, available in report mode)
// =============================================================================

const TIER3_TOOLS: ToolDefinition[] = [
  defineTool(
    "run_esmfold_prediction",
    "USE THIS when the user asks to predict or fold a 3D protein structure from an amino acid sequence. Do NOT use for experimental structures — use fetch_pdb_entry instead. Do NOT use for structure comparison — use run_pymol_rmsd_alignment instead. Returns predicted 3D coordinates and per-residue pLDDT confidence scores. Requires a protein sequence as input.",
    {
      complexes: {
        type: "array",
        description: "Array of protein sequences (amino acid strings) to fold. Each sequence should be ≤ 2400 residues.",
        items: { type: "string" },
      },
    },
    ["complexes"],
  ),
  defineTool(
    "run_pymol_rmsd_alignment",
    "USE THIS when the user asks to compare two 3D protein structures, calculate RMSD, or measure structural differences. Do NOT use for sequence comparison — use run_blast_search instead. Returns post-alignment RMSD in Angstroms and alignment statistics. Requires two PDB structure strings as input.",
    {
      target_structure: {
        type: "string",
        description: "Target/reference PDB structure string",
      },
      mobile_structure: {
        type: "string",
        description: "Mobile/query PDB structure string to align against the target",
      },
    },
    ["target_structure", "mobile_structure"],
  ),
];

// =============================================================================
// Tool Selection — which tools to send to Nemotron
// =============================================================================

/**
 * Get the tools to send to Nemotron based on the chat mode.
 *
 * - General mode: only Tier 1 database retrieval tools
 * - Report mode: all tools (Tier 1 + Tier 2 + Tier 3)
 */
export function getToolsForMode(mode: "report" | "general"): ToolDefinition[] {
  if (mode === "general") {
    return TIER1_TOOLS;
  }
  return [...TIER1_TOOLS, ...TIER2_TOOLS, ...TIER3_TOOLS];
}

/**
 * All available tools (for health check / debugging).
 */
export const ALL_TOOLS: ToolDefinition[] = [...TIER1_TOOLS, ...TIER2_TOOLS, ...TIER3_TOOLS];

/**
 * Get tool definition by name (for validation).
 */
export function getToolByName(name: string): ToolDefinition | undefined {
  return ALL_TOOLS.find((t) => t.function.name === name);
}