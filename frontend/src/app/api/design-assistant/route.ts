// =============================================================================
// Design Assistant API Route
// =============================================================================
// Calls the HelixDesign backend (Python) to analyze a variant report and
// return therapeutic strategy recommendations.
//
// The Python backend lives in backend/design_assistant/ and uses:
//   - Rule-based strategy selection (fast, always available)
//   - Nemotron-3 Ultra 550B for LLM refinement (medium/low confidence cases)
//
// This route is protected by Clerk authentication (inherits from middleware).
// =============================================================================

import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";
import { auth } from "@clerk/nextjs/server";
import { spawn } from "child_process";
import path from "path";
import fs from "fs";

// =============================================================================
// POST /api/design-assistant
// =============================================================================

export async function POST(request: NextRequest) {
  // --- Auth check ---
  const { userId } = await auth();
  if (!userId) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  try {
    const body = await request.json();
    const { report } = body as { report: Record<string, unknown> };

    if (!report) {
      return NextResponse.json(
        { error: "Missing 'report' in request body" },
        { status: 400 },
      );
    }

    // --- Normalize camelCase → snake_case for Python backend ---
    const normalized = normalizeReport(report);

    // --- Call Python backend ---
    const result = await callDesignAssistant(normalized);

    return NextResponse.json(result);
  } catch (error) {
    console.error("Design assistant error:", error);
    return NextResponse.json(
      {
        error:
          error instanceof Error ? error.message : "Internal server error",
      },
      { status: 500 },
    );
  }
}

// =============================================================================
// Python backend call
// =============================================================================

async function callDesignAssistant(
  report: Record<string, unknown>,
): Promise<Record<string, unknown>> {
  return new Promise((resolve, reject) => {
    const scriptPath = path.join(
      process.cwd(),
      "..",
      "backend",
      "design_assistant",
      "run_api.py",
    );

    const python = spawn("python", [scriptPath], {
      cwd: path.join(process.cwd(), "..", "backend"),
      env: {
        ...process.env,
        PYTHONPATH: path.join(process.cwd(), "..", "backend"),
        PYTHONUNBUFFERED: "1",
        NVIDIA_API_KEY: loadNvidiaKey(),
      },
      stdio: ["pipe", "pipe", "pipe"],
    });

    let stdout = "";
    let stderr = "";

    python.stdout.on("data", (data: Buffer) => {
      stdout += data.toString();
    });

    python.stderr.on("data", (data: Buffer) => {
      stderr += data.toString();
    });

    python.on("close", (code: number | null) => {
      if (code !== 0) {
        console.error("Design assistant stderr:", stderr);
        reject(
          new Error(`Python process exited with code ${code}: ${stderr}`),
        );
        return;
      }

      try {
        const result = JSON.parse(stdout.trim());
        resolve(result);
      } catch {
        reject(
          new Error(
            `Failed to parse Python output as JSON. stdout: ${stdout.slice(0, 500)}`,
          ),
        );
      }
    });

    python.on("error", (err: Error) => {
      reject(new Error(`Failed to spawn Python process: ${err.message}`));
    });

    // Send the report as JSON via stdin
    python.stdin.write(JSON.stringify(report));
    python.stdin.end();
  });
}

// =============================================================================
// Field name normalization: camelCase → snake_case
// =============================================================================

function camelToSnake(key: string): string {
  return key.replace(/([A-Z])/g, "_$1").toLowerCase();
}

function normalizeReport(
  report: Record<string, unknown>,
): Record<string, unknown> {
  const normalized: Record<string, unknown> = {};

  // Direct camelCase → snake_case mappings
  const fieldMap: Record<string, string> = {
    geneSymbol: "gene_symbol",
    deltaScore: "delta_score",
    classificationConfidence: "classification_confidence",
    externalScores: "external_scores",
    clinvarClassification: "clinvar_classification",
    variationType: "variation_type",
    populationFrequency: "population_frequency",
    acmgEvidence: "acmg_evidence",
    literatureContext: "literature_context",
    clinicalSummary: "clinical_summary",
    evidenceConfidence: "evidence_confidence",
    vepAnnotation: "vep_annotation",
    ismScanData: "ism_scan",
    xaiFactors: "xai_factors",
    counterfactuals: "counterfactuals",
    acmgCriteria: "acmg_criteria",
    knowledgeGraph: "knowledge_graph",
    acmgCriteriaRefined: "acmg_criteria_refined",
    multiModelConsensus: "multi_model_consensus",
    proteinContext: "protein_context",
    clinvarEvidence: "clinvar_evidence",
    clinvarId: "clinvar_id",
    genomeId: "genome_id",
    classificationSource: "classification_source",
  };

  for (const [key, value] of Object.entries(report)) {
    const mappedKey = fieldMap[key] ?? camelToSnake(key);

    // Recursively normalize nested objects
    if (value && typeof value === "object" && !Array.isArray(value)) {
      normalized[mappedKey] = normalizeReport(
        value as Record<string, unknown>,
      );
    } else {
      normalized[mappedKey] = value;
    }
  }

  // --- Reconstruct missing structured fields from flat frontend data ---

  // clinvar_evidence: check multiple possible sources
  if (!normalized["clinvar_evidence"]) {
    const clinvarStatus =
      report["clinvarClassification"] ||
      (report["clinvarEvidence"] as Record<string, unknown> | undefined)?.["status"] ||
      ((report["evidenceConfidence"] as Record<string, unknown> | undefined)?.["clinvar"] as Record<string, unknown> | undefined)?.["available"]
        ? "Unknown significance"
        : null;

    if (clinvarStatus) {
      const statusText = String(clinvarStatus);
      if (statusText !== "Not Found" && statusText !== "Error") {
        normalized["clinvar_evidence"] = {
          status: statusText,
          num_submitters: 0,
          review_status: "unknown",
        };
      }
    }
  }

  // gnomAD: if population_frequency exists but gnomad_af is null, still mark as available (absent from database)
  if (normalized["population_frequency"]) {
    const pop = normalized["population_frequency"] as Record<string, unknown>;
    // If the field exists but AF is null, it means "not found" which is still a data point
    if (pop["gnomad_af"] === null || pop["gnomad_af"] === undefined) {
      pop["gnomad_af"] = null; // Ensure it's explicitly null
    }
  }

  // protein_context: frontend doesn't have this; infer from evidenceConfidence
  if (!normalized["protein_context"]) {
    const evConf = report["evidenceConfidence"] as Record<string, unknown> | undefined;
    const uniprot = evConf?.["uniprot"] as Record<string, unknown> | undefined;
    if (uniprot?.["available"]) {
      const geneSymbol = report["geneSymbol"];
      normalized["protein_context"] = {
        function: `${typeof geneSymbol === "string" ? geneSymbol : "Unknown"} protein`,
        domains: [],
      };
    }
  }

  // multi_model_consensus: build from externalScores + prediction
  if (!normalized["multi_model_consensus"]) {
    const extScores = report["externalScores"] as Record<string, unknown> | undefined;
    const models = [];
    if (report["prediction"]) models.push("Evo2");
    if (extScores?.["alphamissense"]) models.push("AlphaMissense");
    if (extScores?.["cadd"]) models.push("CADD");
    if (models.length > 0) {
      normalized["multi_model_consensus"] = {
        consensus_classification: report["prediction"] ?? "Uncertain",
        consensus_confidence: "Low",
        models_agree: 1,
        models_total: models.length,
      };
    }
  }

  return normalized;
}

// =============================================================================
// NVIDIA API key loader
// =============================================================================

function loadNvidiaKey(): string {
  // Try process.env first (Next.js loads .env.local automatically)
  if (process.env.NVIDIA_API_KEY) {
    return process.env.NVIDIA_API_KEY;
  }

  // Fallback: read directly from .env.local
  const envPath = path.join(process.cwd(), ".env.local");
  try {
    const content = fs.readFileSync(envPath, "utf-8");
    for (const line of content.split("\n")) {
      const trimmed = line.trim();
      if (trimmed.startsWith("NVIDIA_API_KEY=")) {
        const value = trimmed.slice("NVIDIA_API_KEY=".length).trim().replace(/^["']|["']$/g, "");
        if (value) return value;
      }
    }
  } catch {
    // File not found or unreadable
  }

  return "";
}
