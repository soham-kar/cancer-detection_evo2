"use client";

/**
 * Phase 1 / Day 5: Multi-Model Consensus Panel
 * ==============================================
 * Visual comparison of Evo2-7B, AlphaMissense, CADD, and REVEL predictions
 * with weighted voting, agreement visualization, and LLM-powered
 * disagreement explanation.
 *
 * Papers:
 * - Cheng et al. (2023) — Science: AlphaMissense
 * - Nguyen et al. (2024) — bioRxiv: Evo2
 * - Kircher et al. (2014) — Nature Genetics: CADD
 * - Ioannidis et al. (2016) — AJHG: REVEL
 */

import React from "react";

// =============================================================================
// TYPES
// =============================================================================

interface ModelPrediction {
  model: string;
  score: number;
  classification: string;
  confidence: string;
  weight: number;
  detail: string;
}

interface ConsensusData {
  predictions: ModelPrediction[];
  consensus_classification: string;
  consensus_confidence: string;
  agreement_level: string;
  models_agree: number;
  models_total: number;
  weighted_score: number;
  disagreement_explanation?: string | null;
  clinical_note: string;
}

interface MultiModelConsensusProps {
  data: ConsensusData;
  className?: string;
}

// =============================================================================
// HELPERS
// =============================================================================

function getClassificationColor(classification: string): string {
  const lower = classification.toLowerCase();
  if (lower.includes("pathogenic"))
    return "text-red-600 bg-red-50 border-red-200";
  if (lower.includes("benign"))
    return "text-green-600 bg-green-50 border-green-200";
  return "text-amber-600 bg-amber-50 border-amber-200";
}

function getConfidenceColor(confidence: string): string {
  switch (confidence) {
    case "High":
      return "bg-emerald-500";
    case "Medium":
      return "bg-amber-500";
    case "Low":
      return "bg-red-400";
    default:
      return "bg-slate-300";
  }
}

function getModelIcon(model: string): string {
  switch (model) {
    case "Evo2-7B":
      return "🧬";
    case "AlphaMissense":
      return "🔬";
    case "CADD":
      return "📊";
    case "REVEL":
      return "🧪";
    default:
      return "🔍";
  }
}

function getAgreementBadge(
  level: string,
  modelsAgree: number,
  modelsTotal: number,
): {
  label: string;
  color: string;
} {
  switch (level) {
    case "Full":
      return {
        label: `${modelsAgree}/${modelsTotal} Full Agreement`,
        color: "bg-emerald-100 text-emerald-700",
      };
    case "Strong":
      return {
        label: `${modelsAgree}/${modelsTotal} Strong Consensus`,
        color: "bg-blue-100 text-blue-700",
      };
    case "Split":
      return {
        label: `${modelsAgree}/${modelsTotal} Split Decision`,
        color: "bg-amber-100 text-amber-700",
      };
    case "Weak":
      return {
        label: `${modelsAgree}/${modelsTotal} Weak Consensus`,
        color: "bg-red-100 text-red-700",
      };
    default:
      return {
        label: `${modelsAgree}/${modelsTotal} Models`,
        color: "bg-slate-100 text-slate-600",
      };
  }
}

// =============================================================================
// COMPONENT
// =============================================================================

export function MultiModelConsensus({
  data,
  className = "",
}: MultiModelConsensusProps) {
  if (!data || !data.predictions || data.predictions.length === 0) {
    return null;
  }

  const agreement = getAgreementBadge(
    data.agreement_level,
    data.models_agree,
    data.models_total,
  );
  const consensusColor = getClassificationColor(data.consensus_classification);

  return (
    <div
      className={`overflow-hidden rounded-xl border border-slate-200 bg-white ${className}`}
    >
      {/* Header */}
      <div className="border-b border-slate-100 bg-gradient-to-r from-indigo-50 to-purple-50 px-4 py-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-lg">🔬</span>
            <h3 className="text-sm font-semibold text-slate-700">
              Multi-Model Consensus
            </h3>
          </div>
          <span
            className={`rounded-full px-2 py-0.5 text-[10px] font-medium ${agreement.color}`}
          >
            {agreement.label}
          </span>
        </div>
      </div>

      {/* Consensus Verdict */}
      <div className="border-b border-slate-100 px-4 py-3">
        <div className="flex items-center gap-3">
          <div
            className={`rounded-lg border px-3 py-1.5 text-sm font-bold ${consensusColor}`}
          >
            {data.consensus_classification}
          </div>
          <div className="flex items-center gap-1.5">
            <div
              className={`h-2 w-2 rounded-full ${getConfidenceColor(data.consensus_confidence)}`}
            />
            <span className="text-xs text-slate-500">
              {data.consensus_confidence} confidence
            </span>
          </div>
        </div>
        <p className="mt-1.5 text-xs text-slate-500">{data.clinical_note}</p>
      </div>

      {/* Model Predictions */}
      <div className="space-y-2.5 px-4 py-3">
        {data.predictions.map((pred) => (
          <div key={pred.model} className="flex items-center gap-3">
            {/* Model name + icon */}
            <div className="flex w-32 flex-shrink-0 items-center gap-1.5">
              <span className="text-sm">{getModelIcon(pred.model)}</span>
              <span className="text-xs font-medium text-slate-600">
                {pred.model}
              </span>
            </div>

            {/* Score bar */}
            <div className="flex flex-1 items-center gap-2">
              <div className="h-2 flex-1 overflow-hidden rounded-full bg-slate-100">
                <div
                  className={`h-full rounded-full transition-all duration-500 ${
                    pred.classification.includes("Pathogenic")
                      ? "bg-red-400"
                      : pred.classification.includes("Benign")
                        ? "bg-green-400"
                        : "bg-amber-400"
                  }`}
                  style={{ width: `${Math.min(100, pred.score * 100)}%` }}
                />
              </div>
              <span className="w-10 text-right font-mono text-[10px] text-slate-400">
                {pred.score.toFixed(2)}
              </span>
            </div>

            {/* Classification badge */}
            <span
              className={`flex-shrink-0 rounded border px-1.5 py-0.5 text-[10px] font-medium ${getClassificationColor(pred.classification)}`}
            >
              {pred.classification}
            </span>

            {/* Confidence dot */}
            <div
              className={`h-1.5 w-1.5 flex-shrink-0 rounded-full ${getConfidenceColor(pred.confidence)}`}
              title={`${pred.confidence} confidence`}
            />
          </div>
        ))}
      </div>

      {/* Weight Legend */}
      <div className="border-t border-slate-100 bg-slate-50/50 px-4 py-2">
        <div className="flex items-center gap-4 text-[10px] text-slate-400">
          <span>Weights:</span>
          {data.predictions.map((pred) => (
            <span key={pred.model} className="flex items-center gap-1">
              {getModelIcon(pred.model)} {(pred.weight * 100).toFixed(0)}%
            </span>
          ))}
        </div>
      </div>

      {/* Disagreement Explanation */}
      {data.disagreement_explanation && (
        <div className="border-t border-amber-100 bg-amber-50/50 px-4 py-3">
          <div className="flex items-start gap-2">
            <span className="mt-0.5 text-sm">💡</span>
            <div>
              <p className="mb-0.5 text-[11px] font-medium text-amber-700">
                Why models disagree
              </p>
              <p className="text-[11px] leading-relaxed text-amber-600">
                {data.disagreement_explanation}
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// =============================================================================
// EXPORT HELPER: Build consensus data from API response
// =============================================================================

export function buildConsensusData(apiResult: {
  prediction?: string;
  classification_confidence?: number;
  external_scores?: {
    alphamissense?: {
      score: number;
      confidence: string;
      classification: string;
    } | null;
    cadd?: { phred: number; interpretation: string } | null;
    revel?: { score: number } | null;
  };
}): ConsensusData | null {
  // Need at least Evo2 prediction
  if (!apiResult.prediction) return null;

  const predictions: ModelPrediction[] = [];

  // Evo2-7B
  predictions.push({
    model: "Evo2-7B",
    score: apiResult.classification_confidence || 0.5,
    classification: apiResult.prediction,
    confidence:
      (apiResult.classification_confidence || 0) >= 0.8
        ? "High"
        : (apiResult.classification_confidence || 0) >= 0.5
          ? "Medium"
          : "Low",
    weight: 0.35,
    detail: "DNA foundation model (7B params, 1Mbp context)",
  });

  const scores = apiResult.external_scores || {};

  // AlphaMissense
  if (scores.alphamissense) {
    predictions.push({
      model: "AlphaMissense",
      score: scores.alphamissense.score,
      classification: scores.alphamissense.classification,
      confidence: scores.alphamissense.confidence || "Medium",
      weight: 0.3,
      detail: "Structure-aware variant effect predictor (Cheng et al. 2023)",
    });
  }

  // CADD
  if (scores.cadd) {
    predictions.push({
      model: "CADD",
      score: Math.min(1, scores.cadd.phred / 40),
      classification: scores.cadd.interpretation || "Uncertain",
      confidence:
        scores.cadd.phred >= 20 || scores.cadd.phred < 10 ? "High" : "Medium",
      weight: 0.2,
      detail: "Combined Annotation Dependent Depletion (Kircher et al. 2014)",
    });
  }

  // REVEL
  if (scores.revel) {
    const rScore = scores.revel.score;
    predictions.push({
      model: "REVEL",
      score: rScore,
      classification:
        rScore > 0.75
          ? "Likely Pathogenic"
          : rScore < 0.25
            ? "Likely Benign"
            : "Uncertain",
      confidence: rScore > 0.75 || rScore < 0.25 ? "High" : "Medium",
      weight: 0.15,
      detail: "Rare Exome Variant Ensemble Learner (Ioannidis et al. 2016)",
    });
  }

  if (predictions.length < 2) return null;

  // Compute consensus
  const pathogenicCount = predictions.filter((p) =>
    p.classification.includes("Pathogenic"),
  ).length;
  const benignCount = predictions.filter((p) =>
    p.classification.includes("Benign"),
  ).length;
  const n = predictions.length;

  let consensusClassification: string;
  let agreementLevel: string;
  let modelsAgree: number;

  if (pathogenicCount >= n - 1) {
    consensusClassification = "Likely Pathogenic";
    agreementLevel = pathogenicCount === n ? "Full" : "Strong";
    modelsAgree = pathogenicCount;
  } else if (benignCount >= n - 1) {
    consensusClassification = "Likely Benign";
    agreementLevel = benignCount === n ? "Full" : "Strong";
    modelsAgree = benignCount;
  } else if (pathogenicCount >= n / 2) {
    consensusClassification = "Likely Pathogenic";
    agreementLevel = benignCount >= 2 ? "Split" : "Weak";
    modelsAgree = pathogenicCount;
  } else if (benignCount >= n / 2) {
    consensusClassification = "Likely Benign";
    agreementLevel = pathogenicCount >= 2 ? "Split" : "Weak";
    modelsAgree = benignCount;
  } else {
    consensusClassification = "Uncertain Significance";
    agreementLevel = "Split";
    modelsAgree = Math.max(pathogenicCount, benignCount);
  }

  const agreeRatio = modelsAgree / n;
  const consensusConfidence =
    agreeRatio >= 0.75 ? "High" : agreeRatio >= 0.5 ? "Medium" : "Low";

  return {
    predictions,
    consensus_classification: consensusClassification,
    consensus_confidence: consensusConfidence,
    agreement_level: agreementLevel,
    models_agree: modelsAgree,
    models_total: n,
    weighted_score: 0,
    clinical_note: `${modelsAgree}/${n} models agree on ${consensusClassification}. Confidence: ${consensusConfidence}.`,
  };
}
