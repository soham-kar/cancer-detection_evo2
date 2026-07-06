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
    "Fetch protein entry from UniProt by accession. Returns protein name, domains, function, disease associations, and subcellular location. Use this when you need protein context, domain information, or disease associations for a gene.",
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
    "Fetch predicted 3D structure from AlphaFold Protein Structure Database. Returns PDB structure URL, per-residue pLDDT confidence scores, and PAE matrix. Use this when you need structural context for a protein.",
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
    "Fetch per-residue AlphaMissense pathogenicity scores from the AlphaFold DB. Returns pathogenicity scores for all possible substitutions at each residue. Use this for missense variant pathogenicity assessment.",
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
    "Predict variant consequences using Ensembl VEP. Returns molecular consequence, impact level, amino acid change, and transcript information. Use this when you need to annotate a variant's molecular effect.",
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
    "Fetch DNA/cDNA/protein sequence for an Ensembl ID. Use this when you need the reference sequence for a gene or transcript.",
    {
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
    ["ensembl_id"],
  ),
  defineTool(
    "fetch_pdb_entry",
    "Fetch structure metadata from RCSB PDB. Returns title, experimental method, resolution. Use this when checking for experimental structures.",
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
    "Search NCBI Entrez databases (protein, nucleotide, pubmed, clinvar). Returns matching IDs. Use this for literature or sequence searches.",
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
    "Fetch FASTA records from NCBI sequence databases (protein or nucleotide) by accession or ID. Use this to retrieve actual sequences from NCBI after a search.",
    {
      db: {
        type: "string",
        enum: ["protein", "nuccore"],
        description: "NCBI database: protein or nucleotide",
      },
      id: {
        type: "string",
        description: "Accession or GI number (e.g. 'NP_009225.1' for BRCA1 protein)",
      },
    },
    ["db", "id"],
  ),
  defineTool(
    "fetch_ncbi_esummary",
    "Retrieve record summary metadata from NCBI Entrez by ID. Returns title, organism, length, and other metadata. Use this after an NCBI search to get details about matching records.",
    {
      db: {
        type: "string",
        description: "NCBI database (e.g. 'protein', 'pubmed', 'clinvar')",
      },
      id: {
        type: "string",
        description: "Entrez UID (comma-separated for multiple, e.g. '12345,67890')",
      },
    },
    ["db", "id"],
  ),
  defineTool(
    "fetch_ensembl_lookup",
    "Look up an Ensembl gene record by Ensembl gene ID or gene symbol. Returns chromosome, start/end position, biotype, and description. Use this when you need gene metadata or genomic coordinates.",
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
    "Fetch chain sequences from RCSB PDB with protein/nucleotide classification. Use this when you need the actual amino acid or nucleotide sequence of a PDB structure entry.",
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
    "Resolve small-molecule identifiers (CID, name, SMILES, InChIKey) against PubChem. Returns canonical structure data, synonyms, and molecular formula. Use this when looking up drug compounds or small molecules.",
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
  defineTool(
    "run_spliceai_predict",
    "Predict per-position splice-site probabilities (acceptor/donor) from a DNA sequence using SpliceAI deep learning model. Returns per-position [neither, acceptor, donor] probabilities. Use this when you need the splice profile of a DNA sequence.",
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
    "Predict tissue-specific splice-site probabilities from a DNA sequence using Pangolin. Returns per-position splice probabilities across tissues. Use this when tissue-specific splicing is needed.",
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
    "Score a variant for splice-altering effects using Pangolin deep learning model. Returns gain/loss scores for acceptor/donor splice sites. Delta >0.2 suggests splice disruption. Use this when the user asks about splicing effects of a specific variant. Requires the DNA sequence (>=5000bp flank on each side of the variant).",
    {
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
    ["sequence", "variant_position", "reference_bases", "alternate_bases"],
  ),
  defineTool(
    "run_dssp_secondary_structure",
    "Assign secondary structure (helix/sheet/loop percentages) from a PDB structure file using DSSP. Use this when analyzing the structural composition of a protein.",
    {
      structure: {
        type: "string",
        description: "PDB format structure string",
      },
    },
    ["structure"],
  ),
  defineTool(
    "run_interproscan_fetch",
    "Fetch InterPro domain annotations by UniProt accession (direct REST lookup) or by raw protein sequence (submit-and-scan). Returns domain hits across Pfam, SMART, PROSITE, Gene3D, Panther, and all InterPro member databases. Use this for comprehensive domain annotation when UniProt data is insufficient.",
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
    "Compute structural quality metrics (secondary structure percentages, longest helix, gyration radius) from a PDB file. Use this to evaluate the overall quality and composition of a protein structure.",
    {
      structure: {
        type: "string",
        description: "PDB format structure string",
      },
    },
    ["structure"],
  ),
  defineTool(
    "run_viennarna_prediction",
    "Predict RNA secondary structure using ViennaRNA MFE (minimum free energy) folding. Returns the MFE structure in dot-bracket notation and the free energy. Use this when analyzing RNA folding effects of a variant.",
    {
      sequence: {
        type: "string",
        description: "RNA sequence (DNA will be converted: T→U). At least 50bp recommended.",
      },
    },
    ["sequence"],
  ),
  defineTool(
    "run_blast_search",
    "Search for homologous sequences using BLAST against NCBI databases. Returns hits with E-values, percent identity, and alignments. Use this when looking for similar proteins or genes in other organisms.",
    {
      sequence: {
        type: "string",
        description: "Query sequence (protein or DNA)",
      },
      program: {
        type: "string",
        enum: ["blastp", "blastn", "tblastn"],
        description: "BLAST program. Default: blastp (protein query vs protein db)",
      },
      database: {
        type: "string",
        description: "Database to search (e.g. 'nr', 'refseq_protein', 'swissprot'). Default: nr",
      },
    },
    ["sequence"],
  ),
  defineTool(
    "run_mmseqs2_search_proteins",
    "Fast protein sequence search using MMseqs2. Faster than BLAST with similar sensitivity. Returns per-sequence results with E-values and alignment info. Use this for rapid homology detection.",
    {
      sequence: {
        type: "string",
        description: "Protein query sequence",
      },
      database: {
        type: "string",
        description: "MMseqs2 database path or name. Default: uniref50",
      },
    },
    ["sequence"],
  ),
  defineTool(
    "run_mafft_align",
    "Multiple sequence alignment using MAFFT (Multiple Alignment using Fast Fourier Transform). Returns aligned sequences. Use this when aligning homologous sequences for conservation analysis.",
    {
      sequences: {
        type: "string",
        description: "Sequences in FASTA format (e.g. '>seq1\\nMKTI...\\n>seq2\\nMVLSP...')",
      },
    },
    ["sequences"],
  ),
  defineTool(
    "run_foldseek_search",
    "Search for structurally similar proteins using Foldseek. Input is a PDB structure. Returns structurally related proteins with TM-scores. Use this when searching by 3D structure rather than sequence.",
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
];

// =============================================================================
// Tool Selection — which tools to send to Nemotron
// =============================================================================

/**
 * Get the tools to send to Nemotron based on the chat mode.
 *
 * - General mode: only Tier 1 database retrieval tools
 * - Report mode: all tools (Tier 1 + Tier 2)
 */
export function getToolsForMode(mode: "report" | "general"): ToolDefinition[] {
  if (mode === "general") {
    return TIER1_TOOLS;
  }
  return [...TIER1_TOOLS, ...TIER2_TOOLS];
}

/**
 * All available tools (for health check / debugging).
 */
export const ALL_TOOLS: ToolDefinition[] = [...TIER1_TOOLS, ...TIER2_TOOLS];

/**
 * Get tool definition by name (for validation).
 */
export function getToolByName(name: string): ToolDefinition | undefined {
  return ALL_TOOLS.find((t) => t.function.name === name);
}