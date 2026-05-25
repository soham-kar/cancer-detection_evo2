"use client";

import { Brain, ChevronUp, ChevronDown, Minus, Sparkles } from "lucide-react";
import type { ACMGRefinedResult, ACMGCriteriaResult } from "~/utils/genome-api";

interface ACMGRefinedCardProps {
  refined: ACMGRefinedResult;
  ruleBased?: ACMGCriteriaResult | null;
}

const CLASSIFICATION_COLORS: Record<string, string> = {
  pathogenic: "bg-red-100 text-red-800 border-red-300",
  "likely pathogenic": "bg-orange-100 text-orange-800 border-orange-300",
  vus: "bg-slate-100 text-slate-700 border-slate-300",
  "uncertain significance": "bg-slate-100 text-slate-700 border-slate-300",
  "likely benign": "bg-green-100 text-green-700 border-green-300",
  benign: "bg-green-100 text-green-800 border-green-300",
};

const CONFIDENCE_COLORS: Record<string, string> = {
  high: "bg-green-100 text-green-700",
  medium: "bg-yellow-100 text-yellow-700",
  low: "bg-red-100 text-red-700",
};

function getClassificationColor(classification: string): string {
  const key = classification.toLowerCase().trim();
  return CLASSIFICATION_COLORS[key] ?? "bg-slate-100 text-slate-700 border-slate-300";
}

function getConfidenceColor(confidence: string): string {
  const key = confidence.toLowerCase().trim();
  return CONFIDENCE_COLORS[key] ?? "bg-slate-100 text-slate-700";
}

function compareStrength(
  code: string,
  refined: ACMGRefinedResult,
  ruleBased?: ACMGCriteriaResult | null
): { direction: "up" | "down" | "same"; ruleStrength: string; refinedStrength: string } {
  const ruleCriterion = ruleBased?.criteria?.[code];
  const refinedCriterion = refined.criteria?.[code];

  const ruleStr = ruleCriterion?.strength ?? (ruleCriterion?.met ? "Met" : "Not Met");
  const refinedStr = refinedCriterion?.strength ?? (refinedCriterion?.met ? "Met" : "Not Met");

  const strengthOrder = ["not met", "supporting", "moderate", "strong", "very strong"];

  const ruleIdx = strengthOrder.indexOf(ruleStr.toLowerCase());
  const refinedIdx = strengthOrder.indexOf(refinedStr.toLowerCase());

  if (refinedIdx > ruleIdx) return { direction: "up", ruleStrength: ruleStr, refinedStrength: refinedStr };
  if (refinedIdx < ruleIdx) return { direction: "down", ruleStrength: ruleStr, refinedStrength: refinedStr };
  return { direction: "same", ruleStrength: ruleStr, refinedStrength: refinedStr };
}

export function ACMGRefinedCard({ refined, ruleBased }: ACMGRefinedCardProps) {
  const classificationColor = getClassificationColor(refined.acmg_classification);
  const confidenceColor = getConfidenceColor(refined.classification_confidence);

  // Collect all criteria codes from both rule-based and refined
  const allCodes = new Set<string>();
  if (ruleBased?.criteria) Object.keys(ruleBased.criteria).forEach((k) => allCodes.add(k));
  if (refined.criteria) Object.keys(refined.criteria).forEach((k) => allCodes.add(k));

  const criteriaEntries = Array.from(allCodes).sort();

  return (
    <div className="rounded-xl border border-indigo-200 bg-white overflow-hidden shadow-sm">
      {/* Header */}
      <div className="border-b border-indigo-100 bg-gradient-to-r from-indigo-50 to-purple-50 px-5 py-4">
        <div className="flex items-center gap-2.5">
          <span className="flex h-8 w-8 items-center justify-center rounded-full bg-indigo-100">
            <Brain className="h-4 w-4 text-indigo-600" />
          </span>
          <div>
            <h4 className="text-sm font-semibold text-indigo-900">
              LLM-Refined ACMG Classification
            </h4>
            <p className="text-[11px] text-indigo-500">
              Llama 3.3 70B clinical review of rule-based criteria
            </p>
          </div>
          <span className="ml-auto rounded-full bg-indigo-100 px-2.5 py-0.5 text-[10px] font-medium text-indigo-600">
            <Sparkles className="inline h-3 w-3 mr-1" />
            AI-Refined
          </span>
        </div>
      </div>

      <div className="p-5 space-y-4">
        {/* Classification & Confidence Badges */}
        <div className="flex flex-wrap items-center gap-3">
          <span className={`inline-flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-xs font-semibold ${classificationColor}`}>
            {refined.acmg_classification}
          </span>
          <span className={`inline-flex items-center gap-1 rounded-full px-2.5 py-1 text-[11px] font-medium ${confidenceColor}`}>
            {refined.classification_confidence} Confidence
          </span>
        </div>

        {/* Narrative */}
        {refined.narrative && (
          <blockquote className="border-l-3 border-indigo-300 bg-indigo-50/50 rounded-r-lg px-4 py-3 text-sm text-slate-700 leading-relaxed italic">
            &ldquo;{refined.narrative}&rdquo;
          </blockquote>
        )}

        {/* Per-Criterion Adjustments Table */}
        {criteriaEntries.length > 0 && (
          <div>
            <h5 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">
              Criterion Adjustments
            </h5>
            <div className="overflow-x-auto rounded-lg border border-slate-200">
              <table className="w-full text-xs">
                <thead>
                  <tr className="bg-slate-50 border-b border-slate-200">
                    <th className="px-3 py-2 text-left font-semibold text-slate-600">Code</th>
                    <th className="px-3 py-2 text-left font-semibold text-slate-600">Rule-Based</th>
                    <th className="px-3 py-2 text-center font-semibold text-slate-600 w-8"></th>
                    <th className="px-3 py-2 text-left font-semibold text-slate-600">LLM-Refined</th>
                    <th className="px-3 py-2 text-left font-semibold text-slate-600">Justification</th>
                  </tr>
                </thead>
                <tbody>
                  {criteriaEntries.map((code, i) => {
                    const comparison = compareStrength(code, refined, ruleBased);
                    const refinedCriterion = refined.criteria?.[code];
                    const justification = refinedCriterion?.justification ?? "—";

                    const directionIcon =
                      comparison.direction === "up" ? (
                        <ChevronUp className="h-3.5 w-3.5 text-green-600" />
                      ) : comparison.direction === "down" ? (
                        <ChevronDown className="h-3.5 w-3.5 text-red-600" />
                      ) : (
                        <Minus className="h-3.5 w-3.5 text-slate-400" />
                      );

                    const rowBg =
                      comparison.direction === "up"
                        ? "bg-green-50/30"
                        : comparison.direction === "down"
                          ? "bg-red-50/30"
                          : i % 2 === 0
                            ? "bg-white"
                            : "bg-slate-50/50";

                    return (
                      <tr key={code} className={`border-b border-slate-100 ${rowBg}`}>
                        <td className="px-3 py-2.5 font-mono font-semibold text-slate-700">
                          {code}
                        </td>
                        <td className="px-3 py-2.5 text-slate-600">
                          {comparison.ruleStrength}
                        </td>
                        <td className="px-1 py-2.5 text-center">
                          {directionIcon}
                        </td>
                        <td className="px-3 py-2.5">
                          <span
                            className={
                              comparison.direction === "up"
                                ? "text-green-700 font-medium"
                                : comparison.direction === "down"
                                  ? "text-red-700 font-medium"
                                  : "text-slate-600"
                            }
                          >
                            {comparison.refinedStrength}
                          </span>
                        </td>
                        <td className="px-3 py-2.5 text-slate-500 leading-relaxed max-w-[250px]">
                          {justification}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            {/* Legend */}
            <div className="mt-2 flex items-center gap-4 text-[10px] text-slate-400">
              <span className="flex items-center gap-1">
                <ChevronUp className="h-3 w-3 text-green-500" /> Upgraded
              </span>
              <span className="flex items-center gap-1">
                <ChevronDown className="h-3 w-3 text-red-500" /> Downgraded
              </span>
              <span className="flex items-center gap-1">
                <Minus className="h-3 w-3 text-slate-400" /> Unchanged
              </span>
            </div>
          </div>
        )}

        {/* Disclaimer */}
        <div className="rounded-md bg-amber-50 border border-amber-200 px-3 py-2 text-[10px] text-amber-700 leading-relaxed">
          <strong>AI-Assisted Review:</strong> This refinement is generated by an LLM for research purposes.
          Always validate ACMG classifications with a board-certified clinical geneticist before clinical use.
        </div>
      </div>
    </div>
  );
}
