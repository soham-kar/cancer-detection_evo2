"use client";

import { useState } from "react";
import { Wrench, ChevronDown, ChevronUp, Loader2, CheckCircle, XCircle } from "lucide-react";
import { cn } from "~/lib/utils";

// =============================================================================
// ChatToolCallBlock — Displays a tool call made by the AI chatbot
// =============================================================================

export interface ToolCallData {
  toolCallId: string;
  toolName: string;
  status: "calling" | "completed" | "failed";
  args: Record<string, unknown>;
  result?: Record<string, unknown>;
  error?: string;
  executionTimeMs?: number;
}

interface ChatToolCallBlockProps {
  toolCall: ToolCallData;
}

// Human-readable tool name mapping
const TOOL_LABELS: Record<string, string> = {
  fetch_uniprot: "UniProt Fetch",
  fetch_alphafold_db: "AlphaFold DB Fetch",
  fetch_alphamissense: "AlphaMissense Fetch",
  run_ensembl_vep: "Ensembl VEP",
  fetch_ensembl_lookup: "Ensembl Lookup",
  fetch_ensembl_sequence: "Ensembl Sequence",
  fetch_pdb_entry: "PDB Fetch Entry",
  fetch_pdb_fasta: "PDB Fetch FASTA",
  search_ncbi: "NCBI Search",
  fetch_ncbi_efetch: "NCBI EFetch",
  fetch_ncbi_esummary: "NCBI ESummary",
  fetch_pubchem: "PubChem Fetch",
  run_spliceai_predict: "SpliceAI Predict",
  run_pangolin_predict: "Pangolin Predict",
  run_pangolin_score_variants: "Pangolin Score Variants",
  run_dssp_secondary_structure: "DSSP Secondary Structure",
  run_interproscan_fetch: "InterProScan Fetch",
  run_structure_metrics: "Structure Metrics",
  run_viennarna_prediction: "ViennaRNA Prediction",
  run_blast_search: "BLAST Search",
  run_mmseqs2_search_proteins: "MMseqs2 Search",
  run_mafft_align: "MAFFT Alignment",
  run_foldseek_search: "Foldseek Search",
  run_segmasker_score: "Segmasker",
  run_esmfold_prediction: "ESMFold Prediction",
  run_pymol_rmsd_alignment: "PyMOL RMSD Alignment",
  run_boltz2_affinity: "Boltz2 Affinity",
};

export function ChatToolCallBlock({ toolCall }: ChatToolCallBlockProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  const label = TOOL_LABELS[toolCall.toolName] || toolCall.toolName;
  const timeStr =
    toolCall.executionTimeMs != null
      ? `${(toolCall.executionTimeMs / 1000).toFixed(1)}s`
      : "";

  return (
    <div className="rounded-lg border border-blue-200 bg-blue-50/50 p-2.5">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1.5 text-[11px] font-semibold tracking-wide text-blue-700 uppercase">
          <Wrench className="h-3 w-3" />
          {label}
        </div>
        <div className="flex items-center gap-2">
          {toolCall.status === "calling" && (
            <span className="flex items-center gap-1 text-[10px] text-blue-600">
              <Loader2 className="h-3 w-3 animate-spin" />
              Running...
            </span>
          )}
          {toolCall.status === "completed" && (
            <span className="flex items-center gap-1 text-[10px] text-green-600">
              <CheckCircle className="h-3 w-3" />
              {timeStr}
            </span>
          )}
          {toolCall.status === "failed" && (
            <span className="flex items-center gap-1 text-[10px] text-red-600">
              <XCircle className="h-3 w-3" />
              Failed
            </span>
          )}
        </div>
      </div>

      {/* Expandable details */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="mt-1 flex items-center gap-1 text-[10px] text-blue-600/70 hover:text-blue-700"
      >
        {isExpanded ? (
          <ChevronUp className="h-2.5 w-2.5" />
        ) : (
          <ChevronDown className="h-2.5 w-2.5" />
        )}
        {isExpanded ? "Hide details" : "Show details"}
      </button>

      {isExpanded && (
        <div className="mt-2 space-y-2 text-xs">
          {/* Input */}
          <div>
            <div className="mb-0.5 font-medium text-blue-700">Input:</div>
            <pre className="max-h-40 overflow-auto rounded bg-white p-2 text-[10px] leading-relaxed">
              {JSON.stringify(toolCall.args, null, 2)}
            </pre>
          </div>

          {/* Result */}
          {toolCall.status === "completed" && toolCall.result && (
            <div>
              <div className="mb-0.5 font-medium text-green-700">Result:</div>
              <pre className="max-h-60 overflow-auto rounded bg-white p-2 text-[10px] leading-relaxed">
                {JSON.stringify(toolCall.result, null, 2)}
              </pre>
            </div>
          )}

          {/* Error */}
          {toolCall.status === "failed" && toolCall.error && (
            <div>
              <div className="mb-0.5 font-medium text-red-700">Error:</div>
              <pre className="rounded bg-red-50 p-2 text-[10px] text-red-600">
                {toolCall.error}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
}