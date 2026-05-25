"use client";

import React, { useState } from "react";
import type { ACMGCriteriaResult } from "~/utils/genome-api";

interface ACMGCriteriaTableProps {
  data: ACMGCriteriaResult;
  className?: string;
  compact?: boolean;
}

const STRENGTH_COLORS: Record<string, { bg: string; text: string; border: string; dot: string }> = {
  "Very Strong": { bg: "bg-purple-50", text: "text-purple-700", border: "border-purple-200", dot: "bg-purple-500" },
  Strong: { bg: "bg-rose-50", text: "text-rose-700", border: "border-rose-200", dot: "bg-rose-500" },
  Moderate: { bg: "bg-amber-50", text: "text-amber-700", border: "border-amber-200", dot: "bg-amber-500" },
  Supporting: { bg: "bg-sky-50", text: "text-sky-700", border: "border-sky-200", dot: "bg-sky-500" },
  Standalone: { bg: "bg-emerald-50", text: "text-emerald-700", border: "border-emerald-200", dot: "bg-emerald-500" },
};

const CLASSIFICATION_CONFIG: Record<string, { label: string; gradient: string; badge: string; icon: string }> = {
  Pathogenic: {
    label: "Pathogenic",
    gradient: "from-red-500 to-red-600",
    badge: "bg-red-100 text-red-800 border-red-200",
    icon: "M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z",
  },
  "Likely Pathogenic": {
    label: "Likely Pathogenic",
    gradient: "from-orange-500 to-red-500",
    badge: "bg-orange-100 text-orange-800 border-orange-200",
    icon: "M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z",
  },
  "Uncertain Significance (VUS)": {
    label: "Variant of Uncertain Significance",
    gradient: "from-yellow-400 to-amber-500",
    badge: "bg-yellow-100 text-yellow-800 border-yellow-200",
    icon: "M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z",
  },
  "Likely Benign": {
    label: "Likely Benign",
    gradient: "from-teal-400 to-emerald-500",
    badge: "bg-teal-100 text-teal-800 border-teal-200",
    icon: "M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z",
  },
  Benign: {
    label: "Benign",
    gradient: "from-green-500 to-emerald-600",
    badge: "bg-green-100 text-green-800 border-green-200",
    icon: "M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z",
  },
};

function getClassificationConfig(classification: string) {
  const keys = ["Pathogenic", "Likely Pathogenic", "Uncertain Significance (VUS)", "Likely Benign", "Benign"] as const;
  for (const key of keys) {
    if (classification.includes(key)) return CLASSIFICATION_CONFIG[key];
  }
  return CLASSIFICATION_CONFIG["Uncertain Significance (VUS)"];
}

export function ACMGCriteriaTable({
  data,
  className = "",
  compact = false,
}: ACMGCriteriaTableProps) {
  const [showUnmet, setShowUnmet] = useState(false);
  const criteriaEntries = Object.entries(data.criteria);
  const metEntries = criteriaEntries.filter(([, c]) => c.met);
  const notMetEntries = criteriaEntries.filter(([, c]) => !c.met);

  const config = getClassificationConfig(data.acmg_classification)!;
  const totalStrength =
    data.strength_counts.very_strong * 8 +
    data.strength_counts.strong * 4 +
    data.strength_counts.moderate * 2 +
    data.strength_counts.supporting * 1 +
    data.strength_counts.standalone * 8;

  return (
    <div className={`rounded-xl border border-slate-200 bg-white shadow-sm overflow-hidden ${className}`}>
      {/* ── Header with classification badge ── */}
      <div className="relative overflow-hidden">
        <div className={`absolute inset-0 bg-gradient-to-r ${config.gradient} opacity-10`} />
        <div className="relative px-5 py-4 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className={`flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br ${config.gradient} shadow-sm`}>
              <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d={config.icon} />
              </svg>
            </div>
            <div>
              <h4 className="text-sm font-bold text-slate-800">ACMG/AMP Criteria</h4>
              <p className="text-[10px] text-slate-500">{data.classification_rationale}</p>
            </div>
          </div>
          <div className="flex flex-col items-end gap-1">
            <span className={`inline-flex items-center rounded-full px-3 py-1 text-xs font-bold border ${config.badge}`}>
              {config.label}
            </span>
            <span className="text-[9px] text-slate-400">{data.met_count}/{data.total_evaluated} criteria met</span>
          </div>
        </div>
      </div>

      <div className="px-5 py-4 space-y-4">
        {/* ── Evidence Balance Bar ── */}
        <div>
          <div className="flex justify-between text-[9px] font-semibold text-slate-500 uppercase tracking-wider mb-1.5">
            <span>Evidence Balance</span>
            <span>Strength Score: {totalStrength}</span>
          </div>
          <div className="h-2.5 w-full rounded-full bg-slate-100 overflow-hidden flex">
            {data.strength_counts.very_strong > 0 && (
              <div className="bg-purple-500 h-full" style={{ width: `${(data.strength_counts.very_strong * 8 / Math.max(1, totalStrength)) * 100}%` }} />
            )}
            {data.strength_counts.strong > 0 && (
              <div className="bg-rose-500 h-full" style={{ width: `${(data.strength_counts.strong * 4 / Math.max(1, totalStrength)) * 100}%` }} />
            )}
            {data.strength_counts.moderate > 0 && (
              <div className="bg-amber-500 h-full" style={{ width: `${(data.strength_counts.moderate * 2 / Math.max(1, totalStrength)) * 100}%` }} />
            )}
            {data.strength_counts.supporting > 0 && (
              <div className="bg-sky-500 h-full" style={{ width: `${(data.strength_counts.supporting * 1 / Math.max(1, totalStrength)) * 100}%` }} />
            )}
            {data.strength_counts.standalone > 0 && (
              <div className="bg-emerald-500 h-full" style={{ width: `${(data.strength_counts.standalone * 8 / Math.max(1, totalStrength)) * 100}%` }} />
            )}
          </div>
          <div className="flex flex-wrap gap-x-3 gap-y-0.5 mt-1.5">
            {data.strength_counts.very_strong > 0 && (
              <span className="text-[9px] text-slate-500"><span className="inline-block w-1.5 h-1.5 rounded-full bg-purple-500 mr-1" />{data.strength_counts.very_strong} Very Strong</span>
            )}
            {data.strength_counts.strong > 0 && (
              <span className="text-[9px] text-slate-500"><span className="inline-block w-1.5 h-1.5 rounded-full bg-rose-500 mr-1" />{data.strength_counts.strong} Strong</span>
            )}
            {data.strength_counts.moderate > 0 && (
              <span className="text-[9px] text-slate-500"><span className="inline-block w-1.5 h-1.5 rounded-full bg-amber-500 mr-1" />{data.strength_counts.moderate} Moderate</span>
            )}
            {data.strength_counts.supporting > 0 && (
              <span className="text-[9px] text-slate-500"><span className="inline-block w-1.5 h-1.5 rounded-full bg-sky-500 mr-1" />{data.strength_counts.supporting} Supporting</span>
            )}
            {data.strength_counts.standalone > 0 && (
              <span className="text-[9px] text-slate-500"><span className="inline-block w-1.5 h-1.5 rounded-full bg-emerald-500 mr-1" />{data.strength_counts.standalone} Standalone</span>
            )}
          </div>
        </div>

        {/* ── Met Criteria Table ── */}
        {metEntries.length > 0 && (
          <div>
            <div className="text-[10px] font-bold text-slate-700 uppercase tracking-wider mb-2 flex items-center gap-1.5">
              <span className="flex h-5 w-5 items-center justify-center rounded-full bg-green-100 text-green-600 text-[10px] font-bold">
                {metEntries.length}
              </span>
              Criteria Met — Evidence for Pathogenicity
            </div>
            <div className="rounded-lg border border-slate-200 overflow-hidden">
              <table className="w-full text-left">
                <thead>
                  <tr className="bg-slate-50 border-b border-slate-200">
                    <th className="px-3 py-2 text-[10px] font-semibold text-slate-500 uppercase tracking-wider w-16">Code</th>
                    <th className="px-3 py-2 text-[10px] font-semibold text-slate-500 uppercase tracking-wider w-24">Strength</th>
                    <th className="px-3 py-2 text-[10px] font-semibold text-slate-500 uppercase tracking-wider">Rationale</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {metEntries.map(([code, criterion]) => {
                    const colors = criterion.strength ? STRENGTH_COLORS[criterion.strength] : null;
                    return (
                      <tr key={code} className={colors ? `${colors.bg}` : ""}>
                        <td className="px-3 py-2.5">
                          <div className="flex items-center gap-2">
                            <svg className="h-3.5 w-3.5 text-green-500 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3}>
                              <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                            </svg>
                            <span className="text-xs font-mono font-bold text-slate-700">{code}</span>
                          </div>
                        </td>
                        <td className="px-3 py-2.5">
                          {criterion.strength && (
                            <span className={`inline-flex items-center rounded-md px-2 py-0.5 text-[10px] font-bold border ${colors?.bg} ${colors?.text} ${colors?.border}`}>
                              <span className={`inline-block w-1.5 h-1.5 rounded-full ${colors?.dot} mr-1`} />
                              {criterion.strength}
                            </span>
                          )}
                        </td>
                        <td className="px-3 py-2.5 text-[11px] text-slate-600 leading-relaxed">
                          {criterion.rationale}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* ── Unmet Criteria (collapsible) ── */}
        {notMetEntries.length > 0 && (
          <div>
            <button
              onClick={() => setShowUnmet(!showUnmet)}
              className="flex items-center gap-2 text-[10px] font-semibold text-slate-400 uppercase tracking-wider hover:text-slate-600 transition-colors"
            >
              <svg
                className={`h-3 w-3 transition-transform ${showUnmet ? "rotate-90" : ""}`}
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={2}
              >
                <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
              </svg>
              {notMetEntries.length} Criteria Not Met
              {!showUnmet && <span className="text-slate-300">(click to expand)</span>}
            </button>

            {showUnmet && (
              <div className="mt-2 rounded-lg border border-slate-100 bg-slate-50/50 overflow-hidden">
                <table className="w-full text-left">
                  <tbody className="divide-y divide-slate-100/50">
                    {notMetEntries.map(([code, criterion]) => (
                      <tr key={code}>
                        <td className="px-3 py-2 w-16">
                          <span className="text-xs font-mono text-slate-400">{code}</span>
                        </td>
                        <td className="px-3 py-2 text-[11px] text-slate-400 leading-relaxed">
                          {criterion.rationale}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
