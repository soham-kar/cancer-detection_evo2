"use client";

import React from "react";
import type { ExternalScores } from "~/utils/genome-api";

interface ToolConcordanceProps {
  data: ExternalScores;
  evo2Prediction: string;
  evo2Delta: number;
  clinvarClassification?: string | null;
  className?: string;
}

export function ToolConcordance({
  data,
  evo2Prediction,
  evo2Delta,
  clinvarClassification,
  className = "",
}: ToolConcordanceProps) {
  const rows: { tool: string; score: string; prediction: string; agrees: boolean | null }[] = [
    {
      tool: "Evo2-7B",
      score: `Δ = ${evo2Delta >= 0 ? "+" : ""}${evo2Delta.toFixed(6)}`,
      prediction: evo2Prediction,
      agrees: null, // reference
    },
  ];

  if (data.cadd) {
    rows.push({
      tool: "CADD",
      score: `PHRED = ${data.cadd.phred}`,
      prediction: data.cadd.interpretation,
      agrees: data.cadd.interpretation.toLowerCase().includes(
        evo2Prediction.toLowerCase().includes("pathogenic") ? "pathogenic" : "benign"
      ),
    });
  }

  if (data.alphamissense) {
    rows.push({
      tool: "AlphaMissense",
      score: `Score = ${data.alphamissense.score}`,
      prediction: data.alphamissense.classification,
      agrees: data.alphamissense.classification.toLowerCase().includes(
        evo2Prediction.toLowerCase().includes("pathogenic") ? "pathogenic" : "benign"
      ),
    });
  }

  if (clinvarClassification && clinvarClassification !== "Unknown") {
    rows.push({
      tool: "ClinVar",
      score: clinvarClassification,
      prediction: clinvarClassification,
      agrees: clinvarClassification.toLowerCase().includes(
        evo2Prediction.toLowerCase().includes("pathogenic") ? "pathogenic" : "benign"
      ),
    });
  }

  const agreeCount = rows.filter((r) => r.agrees === true).length;
  const totalComparable = rows.filter((r) => r.agrees !== null).length;

  return (
    <div className={`rounded-lg border border-slate-200 bg-white ${className}`}>
      <div className="border-b border-slate-100 px-4 py-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
            <h4 className="text-sm font-semibold text-slate-800">Multi-Tool Concordance</h4>
          </div>
          {totalComparable > 0 && (
            <span className={`inline-flex items-center rounded-full px-2 py-0.5 text-[10px] font-semibold ${
              agreeCount === totalComparable
                ? "bg-green-100 text-green-800"
                : agreeCount > 0
                  ? "bg-yellow-100 text-yellow-800"
                  : "bg-red-100 text-red-800"
            }`}>
              {agreeCount}/{totalComparable} tools agree
            </span>
          )}
        </div>
      </div>

      <div className="p-4">
        <div className="overflow-x-auto">
          <table className="w-full text-xs border-separate" style={{ borderSpacing: "0 2px" }}>
            <thead>
              <tr className="text-slate-400">
                <th className="text-left font-medium px-2 py-1">Tool</th>
                <th className="text-left font-medium px-2 py-1">Score</th>
                <th className="text-left font-medium px-2 py-1">Prediction</th>
                <th className="text-center font-medium px-2 py-1 w-16">Concordance</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.tool} className="bg-slate-50 rounded-md">
                  <td className="px-2 py-1.5 font-medium text-slate-700 rounded-l-md">
                    {row.tool}
                  </td>
                  <td className="px-2 py-1.5 font-mono text-[10px] text-slate-600">
                    {row.score}
                  </td>
                  <td className="px-2 py-1.5">
                    <span className={`inline-flex items-center rounded px-1.5 py-0.5 text-[10px] font-semibold ${
                      row.prediction.toLowerCase().includes("pathogenic")
                        ? "bg-red-100 text-red-700"
                        : row.prediction.toLowerCase().includes("benign")
                          ? "bg-green-100 text-green-700"
                          : "bg-yellow-100 text-yellow-700"
                    }`}>
                      {row.prediction}
                    </span>
                  </td>
                  <td className="px-2 py-1.5 text-center rounded-r-md">
                    {row.agrees === null ? (
                      <span className="text-[10px] text-slate-400">—</span>
                    ) : row.agrees ? (
                      <span className="text-green-600 font-semibold text-[10px]">✅ Agrees</span>
                    ) : (
                      <span className="text-red-500 font-semibold text-[10px]">⚠ Differs</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {totalComparable > 0 && (
          <div className="mt-3 rounded-md bg-slate-50 border border-slate-100 p-2.5">
            <p className="text-[10px] text-slate-600 leading-relaxed">
              <strong>Interpretation:</strong>{" "}
              {agreeCount === totalComparable
                ? `All ${totalComparable} external tools agree with Evo2's ${evo2Prediction.toLowerCase()} classification. This concordance strengthens confidence in the prediction.`
                : agreeCount > 0
                  ? `${agreeCount}/${totalComparable} tools agree with Evo2. Discordance may reflect different methodological approaches — consider expert review.`
                  : `No external tools agree with Evo2's classification. This discordance warrants careful review and may indicate a borderline or difficult-to-classify variant.`}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
