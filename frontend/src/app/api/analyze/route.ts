// =============================================================================
// Variant Analysis API Route - Main Analysis Orchestration
// =============================================================================
// This endpoint orchestrates the complete variant analysis pipeline:
//   1. User authentication and credit management (free tier + paid credits)
//   2. Reference allele fetching from UCSC Genome Browser
//   3. VEP molecular consequence annotation from Ensembl
//   4. Evo2 AI prediction via Modal backend
//   5. Database persistence for analysis history
//
// Credit System:
//   - New users: 5 paid credits + 10 free analyses per day
//   - Free tier: 10 analyses/day, then 48h cooldown
//   - Paid credits: No cooldown, unlimited analyses
// =============================================================================

import type { NextRequest } from "next/server";
import { NextResponse } from "next/server";
import {
  getOrCreateUser,
  deductCredit,
  incrementFreeRuns,
  setCooldown,
} from "~/lib/user-utils";
import { db } from "~/lib/db";
type InputJsonValue =
  | string
  | number
  | boolean
  | { [key: string]: InputJsonValue }
  | InputJsonValue[];

// Credit system configuration
const FREE_LIMIT_PER_DAY = 10; // Maximum free analyses per day
const COOLDOWN_DAYS = 2; // Cooldown period after free limit exhausted

// Modal backend API endpoint for GPU-accelerated Evo2 inference
const MODAL_API_URL = process.env.NEXT_PUBLIC_ANALYZE_SINGLE_VARIANT_BASE_URL;

// =============================================================================
// Type Definitions
// =============================================================================

/**
 * Analysis request payload sent from frontend to this API route.
 *
 * Required fields:
 *   - variant_position: 1-based genomic position
 *   - alternative: Alternative allele sequence
 *   - genome: Genome build (hg19, hg38, mm10, mm39)
 *   - chromosome: Chromosome identifier (chr1-chr22, chrX, chrY, chrM)
 *
 * Optional fields:
 *   - reference: Reference allele (fetched from UCSC if not provided)
 *   - gene_symbol: HGNC gene symbol for literature context and thresholds
 *   - clinvar_classification: Known ClinVar classification for comparison
 *   - variation_type: Variant type from ClinVar (e.g., "single nucleotide variant")
 *   - clinvar_id: ClinVar accession ID
 *   - analysis_source: Context flag indicating analysis origin
 *   - vep_annotation: Pre-computed VEP annotation (populated by this endpoint)
 */
interface AnalysisRequestBody {
  variant_position: number;
  alternative: string;
  genome: string;
  chromosome: string;
  reference?: string;
  gene_symbol?: string;
  clinvar_classification?: string;
  variation_type?: string;
  clinvar_id?: string;
  analysis_source?: "clinvar" | "custom";
  vep_annotation?: unknown;
  run_ism_scan?: boolean;
  ism_scan_radius?: number;
  ism_scan_stride?: number;
}

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * Call Modal backend API for GPU-accelerated Evo2 variant analysis.
 *
 * This function sends the variant data to the deployed Modal container running
 * the Evo2-7B model on H100 GPU. The backend performs:
 *   - Genome sequence fetching and variant construction
 *   - Evo2 likelihood scoring (reference vs variant)
 *   - Gene-specific threshold application
 *   - VEP override logic for high-confidence variants
 *   - Clinical enrichment (gnomAD, ACMG, PubMed)
 *
 * @param body - Variant analysis request with all required fields
 * @returns Promise resolving to Modal API response with prediction results
 * @throws Error if Modal API URL not configured or request fails
 */
async function callModalAnalysis(body: AnalysisRequestBody) {
  if (!MODAL_API_URL) {
    throw new Error("MODAL_API_URL not configured");
  }

  console.log("[MODAL] Calling Modal API:", { url: MODAL_API_URL, body });

  const response = await fetch(MODAL_API_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    const errorText = await response.text();
    console.error("[MODAL] API Error:", errorText);
    throw new Error(`Modal API failed: ${errorText}`);
  }

  const result = await response.json();
  console.log("[MODAL] API Response:", result);
  return result;
}

// =============================================================================
// Main Analysis Endpoint
// =============================================================================

/**
 * POST /api/analyze - Main variant analysis endpoint
 *
 * Pipeline stages:
 * 1. Authentication & Credit Check
 *    - Verify user authentication via Clerk
 *    - Check cooldown status (48h cooldown after free limit)
 *    - Validate credit availability (paid or free)
 *
 * 2. Credit Deduction
 *    - Prefer paid credits over free credits
 *    - Increment free run counter if using free tier
 *
 * 3. Reference Allele Fetch (UCSC)
 *    - Fetch reference allele from UCSC genome API if not provided
 *    - Required for VEP annotation accuracy
 *
 * 4. VEP Annotation (Ensembl)
 *    - Fetch molecular consequence annotation
 *    - Provides override logic for high-confidence variants
 *    - Enriches RAG context with molecular mechanism
 *
 * 5. Modal Backend Analysis
 *    - Send variant + VEP data to Evo2 GPU backend
 *    - Receive prediction, confidence, and clinical enrichment
 *
 * 6. Database Persistence
 *    - Save analysis results to PostgreSQL via Prisma
 *    - Enable analysis history and reporting features
 *
 * @param request - Next.js request object with variant data in body
 * @returns JSON response with analysis results and updated credit balance
 */
export async function POST(request: NextRequest) {
  try {
    // ===== Stage 1: Authentication (Clerk only, DB optional) =====
    let user: {
      clerkId: string;
      credits: number;
      freeRunsToday: number;
      cooldownUntil: Date | null;
    } | null = null;
    let dbAvailable = true;

    try {
      user = await getOrCreateUser();
    } catch (dbError) {
      console.warn(
        "[DB] Database unavailable, running in stateless mode:",
        (dbError as Error).message,
      );
      dbAvailable = false;
      // Fallback: authenticate via Clerk directly without DB
      const { currentUser } = await import("@clerk/nextjs/server");
      const clerkUser = await currentUser();
      if (!clerkUser) {
        return NextResponse.json(
          { error: "unauthorized", message: "Please sign in to continue" },
          { status: 401 },
        );
      }
      user = {
        clerkId: clerkUser.id,
        credits: 999,
        freeRunsToday: 0,
        cooldownUntil: null,
      };
    }

    // ===== Stage 2: Credit Check (skip if DB unavailable) =====
    let creditUsed: "paid" | "free" | "none" = "none";

    if (dbAvailable && user) {
      // Check if user is in cooldown period
      if (user.cooldownUntil && new Date(user.cooldownUntil) > new Date()) {
        const remainingTime = Math.ceil(
          (new Date(user.cooldownUntil).getTime() - Date.now()) /
            (1000 * 60 * 60),
        );
        return NextResponse.json(
          {
            error: "cooldown_active",
            message: `You've used all free credits. Please wait ${remainingTime} hours or purchase credits.`,
            cooldownUntil: user.cooldownUntil,
            needsCredits: true,
          },
          { status: 429 },
        );
      }

      // Check if free tier limit reached
      if (user.freeRunsToday >= FREE_LIMIT_PER_DAY && user.credits <= 0) {
        const cooldownUntil = new Date();
        cooldownUntil.setDate(cooldownUntil.getDate() + COOLDOWN_DAYS);
        try {
          await setCooldown(user.clerkId, cooldownUntil);
        } catch {}
        return NextResponse.json(
          {
            error: "free_limit_reached",
            message: `You've used your ${FREE_LIMIT_PER_DAY} free analyses today. Purchase credits or wait 2 days.`,
            cooldownUntil,
            needsCredits: true,
          },
          { status: 429 },
        );
      }

      // Deduct credit
      if (user.credits > 0) {
        try {
          await deductCredit(user.clerkId);
        } catch {}
        creditUsed = "paid";
      } else {
        try {
          await incrementFreeRuns(user.clerkId);
        } catch {}
        creditUsed = "free";
      }
    }

    // Parse request body
    const body = (await request.json()) as AnalysisRequestBody;

    // ===== Stage 3: Reference Allele Fetch =====
    // Fetch reference allele from UCSC Genome Browser if not provided by user
    // This is required for accurate VEP annotation and validation
    let referenceAllele = body.reference;
    if (!referenceAllele) {
      try {
        console.log(
          "[UCSC] Fetching reference allele from UCSC Genome Browser...",
        );
        const ucscUrl = `https://api.genome.ucsc.edu/getData/sequence?genome=${body.genome || "hg38"};chrom=${body.chromosome};start=${body.variant_position - 1};end=${body.variant_position}`;
        const ucscRes = await fetch(ucscUrl, {
          signal: AbortSignal.timeout(5000),
        });
        if (ucscRes.ok) {
          const ucscData = (await ucscRes.json()) as { dna?: string };
          referenceAllele = ucscData.dna?.toUpperCase();
          console.log("[UCSC] Reference allele fetched:", referenceAllele);
        }
      } catch (error) {
        console.warn("[UCSC] Failed to fetch reference from UCSC:", error);
      }
    }

    // ===== Stage 4: VEP Molecular Consequence Annotation =====
    // Fetch VEP annotation from Ensembl BEFORE Modal analysis to:
    //   1. Provide molecular mechanism context for RAG generation
    //   2. Enable VEP override logic for high-confidence variants
    //   3. Enrich XAI explanations with structural consequences
    let vepAnnotation: unknown = null;
    if (referenceAllele && !/[NRYWSKMBDHV]/i.test(referenceAllele)) {
      console.log("[VEP] Starting VEP fetch for:", {
        chromosome: body.chromosome,
        position: body.variant_position,
        reference: referenceAllele,
        alternative: body.alternative,
      });
      try {
        const vepRes = await fetch(
          `${process.env.NEXT_PUBLIC_APP_URL ?? "http://localhost:3000"}/api/vep`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              chromosome: body.chromosome,
              position: body.variant_position,
              reference: referenceAllele,
              alternative: body.alternative,
            }),
            signal: AbortSignal.timeout(10000), // 10 second timeout for Ensembl API
          },
        );
        if (vepRes.ok) {
          const vepData = (await vepRes.json()) as { vep: unknown };
          vepAnnotation = vepData.vep ?? null;
          console.log("[VEP] Annotation fetched successfully:", vepAnnotation);
        } else {
          console.warn("[VEP] Endpoint returned non-OK status:", vepRes.status);
        }
      } catch (error) {
        // VEP failures are non-fatal - analysis continues without molecular consequence
        console.warn("[VEP] Annotation failed, continuing without it:", error);
      }
    } else {
      console.warn(
        "[VEP] Skipping VEP fetch - reference allele unavailable or contains ambiguous bases:",
        referenceAllele,
      );
    }

    // ===== Stage 5: Modal Backend Analysis =====
    // Send complete variant data + VEP annotation to Evo2 backend for:
    //   - GPU-accelerated sequence scoring
    //   - Gene-specific threshold application
    //   - VEP override logic
    //   - Clinical enrichment (gnomAD, ACMG, PubMed)
    const analysisResult = await callModalAnalysis({
      ...body,
      vep_annotation: vepAnnotation ?? undefined,
    });

    // ===== Stage 6: Database Persistence (skip if DB unavailable) =====
    if (dbAvailable) {
      try {
        // Store structured fields as native PostgreSQL JSON
        const toJson = (val: unknown): InputJsonValue | undefined =>
          val !== null && val !== undefined
            ? (val as InputJsonValue)
            : undefined;
        await db.analysisReport.create({
          data: {
            clerkUserId: user!.clerkId,
            geneSymbol: body.gene_symbol || "Unknown",
            chromosome: body.chromosome,
            position: body.variant_position,
            reference: analysisResult.reference || body.reference || "",
            alternative: body.alternative,
            genomeId: body.genome,
            prediction: analysisResult.prediction || "",
            deltaScore: analysisResult.delta_score || 0,
            classificationConfidence:
              analysisResult.classification_confidence || 0,
            classificationSource: analysisResult.classification_source || null,
            clinvarClassification: body.clinvar_classification || null,
            variationType: body.variation_type || null,
            clinvarId: body.clinvar_id || null,
            populationFrequency: toJson(analysisResult.population_frequency),
            acmgEvidence: toJson(analysisResult.acmg_evidence),
            literatureContext: toJson(analysisResult.literature_context),
            clinicalSummary: analysisResult.clinical_summary ?? null,
            evidenceConfidence: toJson(analysisResult.evidence_confidence),
            ismScanData: toJson(analysisResult.ism_scan),
            xaiFactors: toJson(analysisResult.xai_factors),
            counterfactuals: toJson(analysisResult.counterfactuals),
            acmgCriteria: toJson(analysisResult.acmg_criteria),
            knowledgeGraph: toJson(analysisResult.knowledge_graph),
            externalScores: toJson(analysisResult.external_scores),
            acmgCriteriaRefined: toJson(analysisResult.acmg_criteria_refined),
            analysisSource: body.analysis_source || null,
            vepAnnotation: toJson(vepAnnotation),
          },
        });
        console.log("[DATABASE] Analysis saved successfully");
        // Notify any open UI that a new report was saved
        // (Frontend event is dispatched client-side after a successful response)
      } catch (saveError) {
        console.error("[DATABASE] Failed to save analysis:", saveError);
      }
    }

    // Calculate updated credit balance for response
    const updatedCredits =
      creditUsed === "paid" ? (user?.credits ?? 0) - 1 : (user?.credits ?? 0);
    const updatedFreeRuns =
      creditUsed === "free"
        ? (user?.freeRunsToday ?? 0) + 1
        : (user?.freeRunsToday ?? 0);

    // Return success with analysis result and credit info
    return NextResponse.json({
      success: true,
      creditUsed,
      remainingCredits: updatedCredits,
      freeRunsToday: updatedFreeRuns,
      freeRunsRemaining: FREE_LIMIT_PER_DAY - updatedFreeRuns,
      dbAvailable,
      ...analysisResult, // Spread the Modal API response
    });
  } catch (error) {
    console.error("Analysis API error:", error);
    // Log full error details for debugging
    if (error instanceof Error) {
      console.error("Error name:", error.name);
      console.error("Error message:", error.message);
      console.error("Error stack:", error.stack);
    }

    if (
      error instanceof Error &&
      error.message === "Unauthorized: No user found"
    ) {
      return NextResponse.json(
        { error: "unauthorized", message: "Please sign in to continue" },
        { status: 401 },
      );
    }

    if (error instanceof Error && error.message.includes("Modal API failed")) {
      return NextResponse.json(
        { error: "analysis_failed", message: error.message },
        { status: 502 },
      );
    }

    // Return the actual error message for debugging
    const errorMessage =
      error instanceof Error ? error.message : "Internal server error";
    return NextResponse.json(
      { error: "internal_error", message: errorMessage },
      { status: 500 },
    );
  }
}

// GET endpoint to check user's credit status
export async function GET() {
  try {
    const user = await getOrCreateUser();

    return NextResponse.json({
      credits: user.credits,
      freeRunsToday: user.freeRunsToday,
      freeRunsRemaining: Math.max(0, FREE_LIMIT_PER_DAY - user.freeRunsToday),
      cooldownUntil: user.cooldownUntil,
      isOnCooldown: user.cooldownUntil
        ? new Date(user.cooldownUntil) > new Date()
        : false,
    });
  } catch (error) {
    if (
      error instanceof Error &&
      error.message === "Unauthorized: No user found"
    ) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 },
    );
  }
}
