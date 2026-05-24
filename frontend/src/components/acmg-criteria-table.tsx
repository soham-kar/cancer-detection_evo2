"use client";

import React from "react";
import type { ACMGCriteriaResult } from "~/utils/genome-api";

interface ACMGCriteriaTableProps {
  data: ACMGCriteriaResult;
  className?: string;
  compact?: boolean;
}

const STRENGTH_COLORS: Record<string, string> = {
  "Very Strong": "bg-purple-100 text-purple-800 border-purple-200",
  Strong: "bg-red-100 text-red-800 border-red-200",
  Moderate: "bg-amber-100 text-amber-800 border-amber-200",
  Supporting: "bg-blue-100 text-blue-800 border-blue-200",
  Standalone: "bg-green-100 text-green-800 border-green-200",
};

export function ACMGCriteriaTable({
  data,
  className = "",
  compact = false,
}: ACMGCriteriaTableProps) {
  const criteriaEntries = Object.entries(data.criteria);
  const metEntries = criteriaEntries.filter(([, c]) => c.met);
  const notMetEntries = criteriaEntries.filter(([, c]) => !c.met);

  return (
    <div className={`rounded-lg border border-slate-200 bg-white ${className}`}>
      <div className="border-b border-slate-100 px-4 py-3">
        <div className="flex items-center justify-between">
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
                d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"
              />
            </svg>
            <h4 className="text-sm font-semibold text-slate-800">
              ACMG/AMP Criteria Mapping
            </h4>
          </div>
          <span
            className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-[10px] font-semibold ${
              data.acmg_classification.includes("Pathogenic")
                ? "bg-red-100 text-red-800"
                : data.acmg_classification.includes("Benign")
                  ? "bg-green-100 text-green-800"
                  : "bg-yellow-100 text-yellow-800"
            }`}
          >
            {data.acmg_classification}
          </span>
        </div>
        <p className="mt-1 text-[10px] text-slate-500">
          {data.classification_rationale}
        </p>
      </div>

      <div className="p-4 space-y-3">
        {/* Met criteria */}
        {metEntries.length > 0 && (
          <div className="space-y-1.5">
            <div className="text-[10px] font-semibold text-green-700 uppercase tracking-wider">
              Met Criteria ({metEntries.length})
            </div>
            {metEntries.map(([code, criterion]) => (
              <div
                key={code}
                className="flex items-start gap-2 rounded-md border border-green-100 bg-green-50/50 p-2"
              >
                <span className="flex-shrink-0 mt-0.5 text-green-500">✓</span>
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-mono font-bold text-slate-700">
                      {code}
                    </span>
                    {criterion.strength && (
                      <span
                        className={`inline-flex items-center rounded px-1.5 py-0.5 text-[9px] font-semibold border ${
                          STRENGTH_COLORS[criterion.strength] ||
                          "bg-slate-100 text-slate-600 border-slate-200"
                        }`}
                      >
                        {criterion.strength}
                      </span>
                    )}
                  </div>
                  <p className="mt-0.5 text-[10px] text-slate-600 leading-relaxed">
                    {criterion.rationale}
                  </p>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Not met criteria (collapsed in compact mode) */}
        {!compact && notMetEntries.length > 0 && (
          <div className="space-y-1.5">
            <div className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider">
              Not Met ({notMetEntries.length})
            </div>
            {notMetEntries.map(([code, criterion]) => (
              <div
                key={code}
                className="flex items-start gap-2 rounded-md border border-slate-100 bg-slate-50/50 p-2"
              >
                <span className="flex-shrink-0 mt-0.5 text-slate-300">—</span>
                <div className="min-w-0 flex-1">
                  <span className="text-xs font-mono text-slate-400">
                    {code}
                  </span>
                  <p className="mt-0.5 text-[10px] text-slate-400 leading-relaxed">
                    {criterion.rationale}
                  </p>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Strength summary */}
        <div className="flex flex-wrap gap-1.5 pt-1 border-t border-slate-100">
          {data.strength_counts.very_strong > 0 && (
            <span className="inline-flex items-center rounded px-1.5 py-0.5 text-[9px] font-semibold bg-purple-100 text-purple-800">
              {data.strength_counts.very_strong} Very Strong
            </span>
          )}
          {data.strength_counts.strong > 0 && (
            <span className="inline-flex items-center rounded px-1.5 py-0.5 text-[9px] font-semibold bg-red-100 text-red-800">
              {data.strength_counts.strong} Strong
            </span>
          )}
          {data.strength_counts.moderate > 0 && (
            <span className="inline-flex items-center rounded px-1.5 py-0.5 text-[9px] font-semibold bg-amber-100 text-amber-800">
              {data.strength_counts.moderate} Moderate
            </span>
          )}
          {data.strength_counts.supporting > 0 && (
            <span className="inline-flex items-center rounded px-1.5 py-0.5 text-[9px] font-semibold bg-blue-100 text-blue-800">
              {data.strength_counts.supporting} Supporting
            </span>
          )}
          {data.strength_counts.standalone > 0 && (
            <span className="inline-flex items-center rounded px-1.5 py-0.5 text-[9px] font-semibold bg-green-100 text-green-800">
              {data.strength_counts.standalone} Standalone
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
