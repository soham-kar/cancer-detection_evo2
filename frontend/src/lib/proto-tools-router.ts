// =============================================================================
// Proto-Tools Router — Routes tool calls to the correct Modal endpoint
// =============================================================================
// CPU tools go to proto-tools-lite (fast, <30s)
// GPU tools go to proto-tools-gpu (slow, 30s-5min) — not yet deployed
// =============================================================================

const PROTO_TOOLS_LITE_URL =
  process.env.PROTO_TOOLS_LITE_URL ||
  "https://sohamkar45--helixmind-proto-lite-run-tool.modal.run";

const PROTO_TOOLS_GPU_URL = process.env.PROTO_TOOLS_GPU_URL || "";

// Mapping from Nemotron function names → Modal proto-tools keys
// Nemotron uses descriptive names (fetch_uniprot), Modal uses proto-tools keys (uniprot_fetch)
const TOOL_NAME_MAP: Record<string, string> = {
  fetch_uniprot: "uniprot_fetch",
  fetch_alphafold_db: "alphafold_db_fetch",
  fetch_alphamissense: "alphamissense_fetch",
  run_ensembl_vep: "ensembl_vep",
  fetch_ensembl_lookup: "ensembl_lookup",
  fetch_ensembl_sequence: "ensembl_sequence",
  fetch_pdb_entry: "pdb_fetch_entry",
  fetch_pdb_fasta: "pdb_fetch_fasta",
  search_ncbi: "ncbi_esearch",
  fetch_ncbi_efetch: "ncbi_efetch",
  fetch_ncbi_esummary: "ncbi_esummary",
  fetch_pubchem: "pubchem_fetch",
  run_spliceai_predict: "spliceai_predict",
  run_pangolin_predict: "pangolin_predict",
  run_pangolin_score_variants: "pangolin_score_variants",
  run_dssp_secondary_structure: "dssp_secondary_structure",
  run_interproscan_fetch: "interproscan_fetch",
  run_structure_metrics: "structure_metrics",
};

// Tools that run on CPU (proto-tools-lite)
const CPU_TOOLS = new Set(Object.values(TOOL_NAME_MAP));

// Tools that require GPU (proto-tools-gpu) — will be added later
const GPU_TOOLS = new Set([
  "esmfold_prediction",
  "esm2_score",
  "esm2_embedding",
  "alphafold2_prediction",
  "proteinmpnn_score",
]);

interface ToolExecutionResult {
  toolKey: string;
  status: "completed" | "failed";
  result?: Record<string, unknown>;
  error?: string;
  executionTimeMs: number;
}

/**
 * Execute a proto-tool via the appropriate Modal endpoint.
 *
 * @param toolKey - The proto-tools tool identifier (e.g. "uniprot_fetch")
 * @param input - Tool input parameters
 * @param config - Optional tool configuration
 * @returns Tool execution result with status, result, and timing
 */
export async function executeProtoTool(
  nemotronToolName: string,
  input: Record<string, unknown>,
  config?: Record<string, unknown>,
): Promise<ToolExecutionResult> {
  const startTime = Date.now();

  // Map Nemotron function name → Modal proto-tools key
  const toolKey = TOOL_NAME_MAP[nemotronToolName] || nemotronToolName;

  // Determine endpoint
  const endpoint = GPU_TOOLS.has(toolKey)
    ? PROTO_TOOLS_GPU_URL
    : PROTO_TOOLS_LITE_URL;

  if (!endpoint) {
    return {
      toolKey,
      status: "failed",
      error: `No endpoint configured for tool '${toolKey}'`,
      executionTimeMs: 0,
    };
  }

  try {
    const response = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        tool_key: toolKey,
        input,
        config: config || {},
      }),
      signal: AbortSignal.timeout(120_000), // 2 min timeout
    });

    if (!response.ok) {
      const errorText = await response.text();
      return {
        toolKey,
        status: "failed",
        error: `Modal endpoint returned ${response.status}: ${errorText}`,
        executionTimeMs: Date.now() - startTime,
      };
    }

    const data = (await response.json()) as {
      tool_key: string;
      status: string;
      result?: Record<string, unknown>;
      error?: string;
      execution_time_ms: number;
    };

    return {
      toolKey: data.tool_key || toolKey,
      status: data.status === "completed" ? "completed" : "failed",
      result: data.result,
      error: data.error,
      executionTimeMs: data.execution_time_ms || Date.now() - startTime,
    };
  } catch (err) {
    return {
      toolKey,
      status: "failed",
      error: err instanceof Error ? err.message : "Tool execution failed",
      executionTimeMs: Date.now() - startTime,
    };
  }
}

/**
 * Check if a tool is a CPU tool (fast).
 */
export function isCpuTool(toolKey: string): boolean {
  return CPU_TOOLS.has(toolKey);
}

/**
 * Check if a tool is a GPU tool (slow).
 */
export function isGpuTool(toolKey: string): boolean {
  return GPU_TOOLS.has(toolKey);
}