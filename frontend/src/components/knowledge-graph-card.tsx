"use client";

import React from "react";
import type { KnowledgeGraph } from "~/utils/genome-api";

interface KnowledgeGraphCardProps {
  data: KnowledgeGraph;
  className?: string;
}

export function KnowledgeGraphCard({ data, className = "" }: KnowledgeGraphCardProps) {
  if (!data.diseases.length && !data.drugs.length) return null;

  return (
    <div className={`rounded-lg border border-slate-200 bg-white ${className}`}>
      <div className="border-b border-slate-100 px-4 py-3">
        <div className="flex items-center gap-2">
          <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 text-teal-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
          </svg>
          <h4 className="text-sm font-semibold text-slate-800">Knowledge Graph</h4>
          <span className="text-[10px] text-slate-400">Gene → Disease → Drug</span>
        </div>
      </div>

      <div className="p-4 space-y-3">
        {/* Clinical Actionability */}
        {data.clinical_actionability && (
          <div className="rounded-md bg-teal-50 border border-teal-100 p-3">
            <p className="text-xs text-teal-800 leading-relaxed">{data.clinical_actionability}</p>
          </div>
        )}

        <div className="grid gap-3 md:grid-cols-2">
          {/* Diseases */}
          {data.diseases.length > 0 && (
            <div>
              <div className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider mb-2">
                Associated Diseases
              </div>
              <div className="space-y-1.5">
                {data.diseases.map((d) => (
                  <a
                    key={d.id}
                    href={`https://platform.opentargets.org/disease/${d.id}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center justify-between rounded-md border border-slate-100 bg-slate-50 px-2.5 py-1.5 hover:bg-teal-50 hover:border-teal-200 transition-colors"
                  >
                    <span className="text-xs text-slate-700 truncate mr-2">{d.name}</span>
                    <span className="flex-shrink-0 text-[10px] font-mono text-teal-600 bg-teal-50 px-1.5 py-0.5 rounded">
                      {Math.round(d.score * 100)}%
                    </span>
                  </a>
                ))}
              </div>
            </div>
          )}

          {/* Drugs */}
          {data.drugs.length > 0 && (
            <div>
              <div className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider mb-2">
                Known Drugs
              </div>
              <div className="space-y-1.5">
                {data.drugs.slice(0, 5).map((d) => (
                  <div
                    key={d.name}
                    className="rounded-md border border-slate-100 bg-slate-50 px-2.5 py-1.5"
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-medium text-slate-700">{d.name}</span>
                      <span className={`text-[9px] font-semibold px-1.5 py-0.5 rounded ${
                        d.phase === "Approved"
                          ? "bg-green-100 text-green-700"
                          : d.phase.startsWith("Phase")
                            ? "bg-blue-100 text-blue-700"
                            : "bg-slate-100 text-slate-500"
                      }`}>
                        {d.phase}
                      </span>
                    </div>
                    <div className="mt-0.5 text-[10px] text-slate-500">
                      {d.type}{d.mechanism !== "Unknown" ? ` • ${d.mechanism}` : ""}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Open Targets link */}
        {data.ensembl_id && (
          <div className="text-right">
            <a
              href={`https://platform.opentargets.org/target/${data.ensembl_id}`}
              target="_blank"
              rel="noopener noreferrer"
              className="text-[10px] text-teal-600 hover:text-teal-800 font-medium"
            >
              View on Open Targets →
            </a>
          </div>
        )}
      </div>
    </div>
  );
}
