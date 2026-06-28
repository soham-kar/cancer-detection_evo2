"use client";

/**
 * Design Therapeutics Panel
 * =========================
 * Proto-inspired therapeutic reasoning assistant.
 * Consumes a variant analysis report and returns:
 * - Recommended therapeutic strategy
 * - Evidence-backed reasoning
 * - Ranked next steps
 * - Limitations and caveats
 *
 * Powered by Nemotron-3 Ultra 550B (NVIDIA) with rule-based fallback.
 */

import React, { useState } from "react";

// =============================================================================
// TYPES
// =============================================================================

interface DesignAssistantResponse {
  strategy_class: string;
  confidence: string;
  reasoning_summary: string;
  evidence_bullets: Array<{
    source: string;
    finding: string;
    signal: string;
    confidence: string;
  }>;
  recommended_next_steps: Array<{
    priority: string;
    action: string;
    detail: string;
  }>;
  limitations: string[];
  generated_at: string;
  evidence_summary: string;
  rule_based: boolean;
}

interface DesignTherapeuticsProps {
  report: {
    geneSymbol: string;
    chromosome: string;
    position: number;
    reference: string;
    alternative: string;
    prediction: string;
    deltaScore: number;
    classificationConfidence: number;
    clinvarClassification?: string;
    variationType?: string;
    populationFrequency?: {
      gnomad_af: number | null;
      is_common_variant: boolean;
    };
    acmgEvidence?: {
      code: string | null;
      strength: string | null;
      description: string;
      clinical_note: string;
    };
    literatureContext?: {
      articles_found: number;
    };
    externalScores?: {
      cadd?: { phred: number; interpretation: string } | null;
      alphamissense?: {
        score: number;
        classification: string;
      } | null;
    };
    vepAnnotation?: {
      consequence?: string;
      impact?: string;
      aaChange?: string;
      aminoAcids?: string;
    } | null;
    ismScanData?: {
      summary?: {
        constrained_positions: number;
        total_positions_scanned: number;
        constraint_zone: string;
      };
    } | null;
    proteinContext?: {
      function?: string;
      domains?: Array<{ name: string }>;
    } | null;
    multiModelConsensus?: {
      consensus_classification?: string;
      consensus_confidence?: string;
      models_agree?: number;
      models_total?: number;
    } | null;
  };
  className?: string;
}

// =============================================================================
// HELPERS
// =============================================================================

function getStrategyIcon(strategy: string): string {
  switch (strategy) {
    case "structural_rescue":
      return "🔬";
    case "splice_rescue":
      return "🧬";
    case "allele_specific_targeting":
      return "🎯";
    case "protein_binder_design":
      return "💊";
    case "literature_and_evidence_review":
      return "📚";
    case "observe_and_reassess":
      return "👁️";
    case "functional_validation_first":
      return "🧪";
    default:
      return "🔍";
  }
}

function getStrategyLabel(strategy: string): string {
  switch (strategy) {
    case "structural_rescue":
      return "Structural Rescue";
    case "splice_rescue":
      return "Splice Rescue";
    case "allele_specific_targeting":
      return "Allele-Specific Targeting";
    case "protein_binder_design":
      return "Protein Binder Design";
    case "literature_and_evidence_review":
      return "Literature & Evidence Review";
    case "observe_and_reassess":
      return "Observe & Reassess";
    case "functional_validation_first":
      return "Functional Validation First";
    default:
      return strategy.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
  }
}

function getConfidenceColor(confidence: string): string {
  switch (confidence) {
    case "high":
      return "bg-emerald-100 text-emerald-700 border-emerald-200";
    case "medium":
      return "bg-amber-100 text-amber-700 border-amber-200";
    case "low":
      return "bg-red-100 text-red-700 border-red-200";
    default:
      return "bg-slate-100 text-slate-600 border-slate-200";
  }
}

function getPriorityColor(priority: string): string {
  switch (priority) {
    case "high":
      return "bg-red-50 border-red-200 text-red-700";
    case "medium":
      return "bg-amber-50 border-amber-200 text-amber-700";
    case "low":
      return "bg-slate-50 border-slate-200 text-slate-600";
    default:
      return "bg-slate-50 border-slate-200 text-slate-600";
  }
}

function getSignalIcon(signal: string): string {
  switch (signal) {
    case "pathogenic":
      return "🔴";
    case "benign":
      return "🟢";
    case "uncertain":
      return "🟡";
    default:
      return "⚪";
  }
}

/**
 * Format reasoning text with highlighted clinical terms, numbers, and conclusions.
 * Wraps key phrases in styled spans for visual scanning.
 */
function formatReasoning(text: string): React.ReactNode {
  if (!text) return text;

  // Split into sentences for processing
  const sentences = text.split(/(?<=[.!?])\s+/);

  return (
    <span>
      {sentences.map((sentence, i) => {
        // Highlight key clinical terms
        let formatted = sentence;

        // Bold clinical classifications
        formatted = formatted.replace(
          /\b(Variant of Uncertain Significance|VUS|Likely Pathogenic|Likely Benign|Pathogenic|Benign|Uncertain Significance)\b/gi,
          (match) =>
            `<strong class="font-semibold text-slate-800">${match}</strong>`,
        );

        // Highlight gene names
        formatted = formatted.replace(
          /\b(BRCA1|BRCA2|TP53|PTEN|MLH1|MSH2|MSH6|PMS2|APC|RET|VHL|RB1|CDH1|STK11|SMAD4|BMPR1A|NF1|NF2|MEN1|PTCH1|SUFU)\b/g,
          (match) =>
            `<span class="font-medium text-indigo-700">${match}</span>`,
        );

        // Highlight numbers and percentages
        formatted = formatted.replace(
          /\b(\d+[-/]\d+)\b/g,
          (match) =>
            `<span class="font-mono font-semibold text-slate-700">${match}</span>`,
        );

        // Highlight clinical actions
        formatted = formatted.replace(
          /\b(re-evaluat\w+|functional validation|segregation analysis|genetic counsel\w+|clinical trial|surveillance|prophylactic)\b/gi,
          (match) =>
            `<span class="font-medium text-emerald-700">${match}</span>`,
        );

        // Highlight guideline references
        formatted = formatted.replace(
          /\b(ACMG|AMP|NCCN|ESMO|ASCO|ClinGen)\b/g,
          (match) =>
            `<span class="font-semibold text-amber-700">${match}</span>`,
        );

        // Highlight key conclusions (sentences starting with key phrases)
        const conclusionPrefixes = [
          "This variant",
          "Current guidelines",
          "Management should",
          "The evidence",
          "No therapeutic",
          "Functional studies",
        ];
        const isConclusion = conclusionPrefixes.some((prefix) =>
          sentence.startsWith(prefix),
        );

        return (
          <span key={i}>
            <span
              dangerouslySetInnerHTML={{ __html: formatted }}
              className={isConclusion ? "font-medium text-slate-700" : ""}
            />
            {i < sentences.length - 1 ? " " : ""}
          </span>
        );
      })}
    </span>
  );
}

// =============================================================================
// COMPONENT
// =============================================================================

export function DesignTherapeutics({
  report,
  className = "",
}: DesignTherapeuticsProps) {
  const [result, setResult] = useState<DesignAssistantResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showEvidence, setShowEvidence] = useState(true);

  const handleAnalyze = async () => {
    setLoading(true);
    setError(null);
    try {
      const resp = await fetch("/api/design-assistant", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ report }),
      });
      if (!resp.ok) {
        const errData = await resp.json().catch(() => ({}));
        throw new Error(
          (errData as { error?: string }).error ?? `API returned ${resp.status}`,
        );
      }
      const data = (await resp.json()) as DesignAssistantResponse;
      setResult(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Analysis failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={`rounded-lg border border-slate-200 bg-white ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between border-b border-slate-100 px-4 py-3">
        <div className="flex items-center gap-2">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className="h-4 w-4 text-indigo-600"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M13 10V3L4 14h7v7l9-11h-7z"
            />
          </svg>
          <h4 className="text-sm font-semibold text-slate-800">
            Design Therapeutics
          </h4>
          <span className="rounded bg-indigo-50 px-1.5 py-0.5 text-[10px] font-medium text-indigo-600">
            Nemotron-3 550B
          </span>
        </div>
        {!result && !loading && (
          <button
            onClick={handleAnalyze}
            className="inline-flex items-center gap-1.5 rounded-md bg-indigo-600 px-3 py-1.5 text-xs font-semibold text-white shadow-sm transition-colors hover:bg-indigo-700"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="h-3.5 w-3.5"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09zM18.259 8.715L18 9.75l-.259-1.035a3.375 3.375 0 00-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 002.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 002.455 2.456L21.75 6l-1.036.259a3.375 3.375 0 00-2.455 2.456zM16.894 20.567L16.5 21.75l-.394-1.183a2.25 2.25 0 00-1.423-1.423L13.5 18.75l1.183-.394a2.25 2.25 0 001.423-1.423l.394-1.183.394 1.183a2.25 2.25 0 001.423 1.423l1.183.394-1.183.394a2.25 2.25 0 00-1.423 1.423z"
              />
            </svg>
            Analyze Therapeutic Options
          </button>
        )}
      </div>

      <div className="p-4">
        {/* Loading state */}
        {loading && (
          <div className="flex items-center justify-center py-8">
            <div className="flex items-center gap-3">
              <div className="h-5 w-5 animate-spin rounded-full border-2 border-indigo-600 border-t-transparent" />
              <span className="text-sm text-slate-500">
                Nemotron is reasoning about therapeutic strategies…
              </span>
            </div>
          </div>
        )}

        {/* Error state */}
        {error && (
          <div className="rounded-md border border-red-200 bg-red-50 p-4">
            <div className="flex items-start gap-2">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                className="mt-0.5 h-4 w-4 flex-shrink-0 text-red-500"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
                />
              </svg>
              <div>
                <p className="text-sm font-medium text-red-800">
                  Analysis unavailable
                </p>
                <p className="mt-1 text-xs text-red-600">{error}</p>
                <button
                  onClick={handleAnalyze}
                  className="mt-2 text-xs font-medium text-red-700 underline hover:text-red-800"
                >
                  Retry
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Empty state (before first analysis) */}
        {!result && !loading && !error && (
          <div className="py-6 text-center">
            <div className="mx-auto mb-3 flex h-12 w-12 items-center justify-center rounded-full bg-indigo-50">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                className="h-6 w-6 text-indigo-500"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={1.5}
                  d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09z"
                />
              </svg>
            </div>
            <p className="text-sm text-slate-600">
              Nemotron-3 550B can reason about therapeutic strategies
            </p>
            <p className="mt-1 text-xs text-slate-400">
              Click the button above to analyze this variant
            </p>
          </div>
        )}

        {/* Results */}
        {result && (
          <div className="space-y-4">
            {/* Strategy header */}
            <div className="flex items-start gap-3">
              <span className="mt-0.5 text-2xl">
                {getStrategyIcon(result.strategy_class)}
              </span>
              <div className="flex-1">
                <div className="flex items-center gap-2">
                  <h5 className="text-sm font-semibold text-slate-800">
                    {getStrategyLabel(result.strategy_class)}
                  </h5>
                  <span
                    className={`inline-flex items-center rounded-full border px-2 py-0.5 text-[10px] font-semibold ${getConfidenceColor(result.confidence)}`}
                  >
                    {result.confidence.toUpperCase()} confidence
                  </span>
                  {result.rule_based && (
                    <span className="rounded bg-slate-100 px-1.5 py-0.5 text-[10px] text-slate-500">
                      rule-based
                    </span>
                  )}
                </div>
                <div className="mt-2 text-xs leading-relaxed text-slate-600">
                  {formatReasoning(result.reasoning_summary)}
                </div>
              </div>
            </div>

            {/* Evidence summary */}
            <div className="rounded-md bg-slate-50 p-3">
              <p className="text-[10px] font-medium text-slate-500 uppercase tracking-wide">
                Evidence Summary
              </p>
              <p className="mt-1 text-xs text-slate-600">
                {result.evidence_summary}
              </p>
            </div>

            {/* Evidence bullets (collapsible) */}
            <div>
              <button
                onClick={() => setShowEvidence(!showEvidence)}
                className="flex w-full items-center justify-between text-xs font-medium text-slate-500 hover:text-slate-700"
              >
                <span>
                  Evidence Sources ({result.evidence_bullets.length})
                </span>
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  className={`h-3.5 w-3.5 transition-transform ${showEvidence ? "rotate-180" : ""}`}
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M19 9l-7 7-7-7"
                  />
                </svg>
              </button>
              {showEvidence && (
                <div className="mt-2 space-y-1.5">
                  {result.evidence_bullets.map((b, i) => (
                    <div
                      key={i}
                      className="flex items-start gap-2 rounded bg-white px-2.5 py-1.5 text-xs"
                    >
                      <span className="mt-0.5 flex-shrink-0">
                        {getSignalIcon(b.signal)}
                      </span>
                      <div>
                        <span className="font-medium text-slate-700">
                          {b.source}:
                        </span>{" "}
                        <span className="text-slate-500">{b.finding}</span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Recommended next steps */}
            <div>
              <p className="mb-2 text-[10px] font-medium text-slate-500 uppercase tracking-wide">
                Recommended Next Steps
              </p>
              <div className="space-y-2">
                {result.recommended_next_steps.map((step, i) => (
                  <div
                    key={i}
                    className={`rounded-md border p-2.5 ${getPriorityColor(step.priority)}`}
                  >
                    <div className="flex items-center gap-2">
                      <span className="text-[10px] font-semibold uppercase">
                        {step.priority}
                      </span>
                      <span className="text-xs font-medium">
                        {step.action}
                      </span>
                    </div>
                    <p className="mt-1 text-[11px] leading-relaxed opacity-80">
                      {step.detail}
                    </p>
                  </div>
                ))}
              </div>
            </div>

            {/* Limitations */}
            <div className="rounded-md border border-amber-200 bg-amber-50 p-3">
              <p className="mb-1.5 text-[10px] font-medium text-amber-700 uppercase tracking-wide">
                Limitations & Caveats
              </p>
              <ul className="space-y-1">
                {result.limitations.map((lim, i) => (
                  <li
                    key={i}
                    className="flex items-start gap-1.5 text-[11px] text-amber-800"
                  >
                    <span className="mt-0.5 flex-shrink-0 text-amber-500">
                      •
                    </span>
                    <span>{lim}</span>
                  </li>
                ))}
              </ul>
            </div>

            {/* Disclaimer */}
            <div className="rounded border border-slate-200 bg-slate-50 p-2.5">
              <p className="text-[10px] leading-relaxed text-slate-400">
                <strong>Research tool only.</strong> Therapeutic strategies are
                computational hypotheses generated by Nemotron-3 550B. They
                require experimental validation and clinical review. Not for
                clinical decision-making.
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
