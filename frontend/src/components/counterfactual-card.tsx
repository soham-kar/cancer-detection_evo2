"use client";

import React from "react";
import type { Counterfactuals } from "~/utils/genome-api";

const NUCLEOTIDE_COLORS: Record<string, string> = {
  A: "#22c55e",
  T: "#ef4444",
  G: "#f59e0b",
  C: "#3b82f6",
};

interface CounterfactualCardProps {
  data: Counterfactuals;
  observedAlternative?: string;
  className?: string;
}

export function CounterfactualCard({
  data,
  observedAlternative,
  className = "",
}: CounterfactualCardProps) {
  const altEntries = Object.entries(data.alternatives).sort(
    ([, a], [, b]) => b.magnitude - a.magnitude
  );

  return (
    <div className={`rounded-lg border border-slate-200 bg-white ${className}`}>
      <div className="border-b border-slate-100 px-4 py-3">
        <div className="flex items-center gap-2">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className="h-4 w-4 text-amber-500"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
            />
          </svg>
          <h4 className="text-sm font-semibold text-slate-800">
            Counterfactual Analysis
          </h4>
          <span className="text-[10px] text-slate-400">
            What if this were a different mutation?
          </span>
        </div>
      </div>

      <div className="p-4 space-y-3">
        {/* Summary */}
        <p className="text-xs text-slate-600 leading-relaxed">{data.summary}</p>

        {/* Allele cards */}
        <div className="grid grid-cols-3 gap-2">
          {altEntries.map(([alt, scores]) => {
            const isObserved = observedAlternative === alt;
            const isTolerated = scores.direction === "benign";
            const isPathogenic = scores.direction === "pathogenic";

            return (
              <div
                key={alt}
                className={`rounded-md border p-2.5 text-center transition-all ${
                  isObserved
                    ? "ring-2 ring-indigo-400 border-indigo-300 bg-indigo-50/50"
                    : isTolerated
                      ? "border-green-200 bg-green-50/50"
                      : isPathogenic
                        ? "border-red-200 bg-red-50/50"
                        : "border-slate-200 bg-slate-50"
                }`}
              >
                <div className="flex items-center justify-center gap-1.5 mb-1.5">
                  <span
                    className="text-sm font-bold"
                    style={{ color: NUCLEOTIDE_COLORS[data.reference] || "#64748b" }}
                  >
                    {data.reference}
                  </span>
                  <span className="text-[10px] text-slate-400">→</span>
                  <span
                    className="text-sm font-bold"
                    style={{ color: NUCLEOTIDE_COLORS[alt] || "#64748b" }}
                  >
                    {alt}
                  </span>
                  {isObserved && (
                    <span className="text-[9px] bg-indigo-100 text-indigo-700 px-1 rounded font-medium">
                      observed
                    </span>
                  )}
                </div>

                <div
                  className={`text-xs font-semibold ${
                    isTolerated
                      ? "text-green-700"
                      : isPathogenic
                        ? "text-red-700"
                        : "text-slate-500"
                  }`}
                >
                  {scores.prediction}
                </div>

                <div className="mt-1 text-[10px] font-mono text-slate-500">
                  Δ = {scores.delta >= 0 ? "+" : ""}
                  {scores.delta.toFixed(6)}
                </div>

                <div className="mt-1">
                  <div className="h-1 w-full rounded-full bg-slate-100 overflow-hidden">
                    <div
                      className={`h-full rounded-full ${
                        isTolerated
                          ? "bg-green-400"
                          : isPathogenic
                            ? "bg-red-400"
                            : "bg-slate-300"
                      }`}
                      style={{
                        width: `${Math.min(100, scores.magnitude * 5000)}%`,
                      }}
                    />
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        {/* Clinical note */}
        <div className="rounded-md bg-slate-50 border border-slate-100 p-2.5">
          <p className="text-[10px] text-slate-600 leading-relaxed">
            <strong>Clinical relevance:</strong>{" "}
            {data.pathogenic_alleles.length === 3
              ? "All three alternative nucleotides are predicted pathogenic — this position is under absolute evolutionary constraint. Any mutation at this site is likely deleterious."
              : data.tolerated_alleles.length === 3
                ? "All alternative nucleotides are tolerated — this position is evolutionarily neutral. The specific nucleotide identity does not affect protein function."
              : data.neutral_alleles && data.neutral_alleles.length === 3
                ? "All three alternatives have uncertain effect — Evo2 cannot distinguish any nucleotide change from background at this position. The delta scores are too small to classify, suggesting this position is in a low-information genomic region."
              : data.tolerated_alleles.length > 0 && data.pathogenic_alleles.length > 0
                ? `Only ${data.tolerated_alleles.join(", ")} ${
                    data.tolerated_alleles.length === 1 ? "is" : "are"
                  } tolerated at this position. The observed change to ${observedAlternative || "?"} ${
                    data.alternatives[observedAlternative || ""]?.direction === "benign"
                      ? "is consistent with evolutionary tolerance."
                      : "may be functionally significant."
                  }`
                : data.summary}
          </p>
        </div>
      </div>
    </div>
  );
}
