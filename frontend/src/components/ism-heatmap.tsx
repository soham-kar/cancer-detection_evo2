"use client";

import React, { useState, useMemo } from "react";
import type { ISMScanResult, ISMPositionData } from "~/utils/genome-api";

// ─── Types ───────────────────────────────────────────────────────────────

interface ISMHeatmapProps {
  data: ISMScanResult;
  geneSymbol?: string;
  variantPosition: number;
  chromosome?: string;
  className?: string;
  compact?: boolean;
}

// ─── Constants ───────────────────────────────────────────────────────────

const NUCLEOTIDE_COLORS: Record<string, string> = {
  A: "#22c55e",
  T: "#ef4444",
  G: "#f59e0b",
  C: "#3b82f6",
};

const CONSTRAINT_COLORS = {
  high: { bg: "#dc2626", text: "#7f1d1d", label: "Highly Constrained" },
  moderate: { bg: "#f59e0b", text: "#78350f", label: "Moderately Constrained" },
  low: { bg: "#e5e7eb", text: "#6b7280", label: "Neutral / Tolerated" },
};

// ─── Helpers ─────────────────────────────────────────────────────────────

/** Map |Δ| magnitude to a color intensity (0–1 → light to dark) */
function constraintColor(magnitude: number, maxMagnitude: number): string {
  if (magnitude < 0.0001) return "rgba(229, 231, 235, 0.6)"; // neutral
  const intensity = Math.min(1, magnitude / Math.max(maxMagnitude, 0.001));
  // Interpolate from amber-100 to red-600
  const r = Math.round(229 + (220 - 229) * intensity);
  const g = Math.round(231 + (38 - 231) * intensity);
  const b = Math.round(235 + (38 - 235) * intensity);
  return `rgba(${r}, ${g}, ${b}, ${0.3 + intensity * 0.7})`;
}

function formatDelta(d: number): string {
  const sign = d >= 0 ? "+" : "";
  if (Math.abs(d) < 0.0001) return `${sign}${d.toExponential(2)}`;
  if (Math.abs(d) < 0.001) return `${sign}${d.toFixed(6)}`;
  return `${sign}${d.toFixed(4)}`;
}

// ─── Component ───────────────────────────────────────────────────────────

export function ISMHeatmap({
  data,
  geneSymbol,
  variantPosition,
  chromosome,
  className = "",
  compact = false,
}: ISMHeatmapProps) {
  const [hoveredCell, setHoveredCell] = useState<{
    pos: string;
    alt: string;
  } | null>(null);
  const [showAllPositions, setShowAllPositions] = useState(false);

  // Parse position keys into sorted numeric array
  const positionEntries = useMemo(() => {
    const entries = Object.entries(data.positions)
      .map(([key, val]) => ({ relPos: parseInt(key, 10), ...val }))
      .sort((a, b) => a.relPos - b.relPos);
    return entries;
  }, [data.positions]);

  // Compute global max magnitude for color scaling
  const globalMaxMag = useMemo(
    () => Math.max(0.0001, ...positionEntries.map((p) => p.max_delta)),
    [positionEntries]
  );

  // Determine which alternative nucleotides to show as rows
  const altNucleotides = useMemo(() => {
    const alts = new Set<string>();
    positionEntries.forEach((p) => {
      Object.keys(p.alternatives).forEach((a) => alts.add(a));
    });
    return Array.from(alts).sort();
  }, [positionEntries]);

  // For compact mode, show only constrained positions + boundaries
  const displayPositions = useMemo(() => {
    if (!compact || showAllPositions) return positionEntries;
    return positionEntries.filter(
      (p) =>
        p.is_constrained ||
        data.summary.constraint_boundaries.includes(p.relPos) ||
        p.relPos === 0
    );
  }, [positionEntries, compact, showAllPositions, data.summary.constraint_boundaries]);

  const { summary } = data;
  const zoneColor = CONSTRAINT_COLORS[summary.constraint_zone];

  if (positionEntries.length === 0) {
    return (
      <div className={`rounded-md border border-slate-200 bg-white p-4 ${className}`}>
        <p className="text-sm text-slate-500">No ISM scan data available.</p>
      </div>
    );
  }

  return (
    <div className={`rounded-lg border border-slate-200 bg-white ${className}`}>
      {/* ── Header ── */}
      <div className="border-b border-slate-100 px-5 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="flex h-8 w-8 items-center justify-center rounded-full bg-indigo-50">
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
                  d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
                />
              </svg>
            </div>
            <div>
              <h4 className="text-sm font-semibold text-slate-800">
                In-Silico Mutagenesis Scan
              </h4>
              <p className="text-xs text-slate-500">
                ±{summary.total_positions_scanned > 0 ? data.scan_radius : 0}bp positional constraint mapping
                {geneSymbol && ` • ${geneSymbol}`}
                {chromosome && ` • ${chromosome}:${variantPosition.toLocaleString()}`}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <span
              className="inline-flex items-center rounded-full px-2.5 py-0.5 text-[10px] font-semibold"
              style={{
                backgroundColor: zoneColor.bg + "18",
                color: zoneColor.text,
              }}
            >
              {zoneColor.label}
            </span>
            <span className="text-xs text-slate-400">
              {summary.constrained_positions}/{summary.total_positions_scanned} constrained
            </span>
          </div>
        </div>
      </div>

      {/* ── Heatmap Grid ── */}
      <div className="overflow-x-auto">
        <div className="inline-block min-w-full p-4">
          <table className="border-separate" style={{ borderSpacing: "1px" }}>
            {/* Column headers: position numbers */}
            <thead>
              <tr>
                <th className="px-2 py-1 text-left text-[10px] font-medium text-slate-400 uppercase tracking-wider">
                  Ref
                </th>
                {displayPositions.map((p) => (
                  <th
                    key={p.relPos}
                    className={`px-1.5 py-1 text-center text-[10px] font-medium ${
                      p.relPos === 0
                        ? "text-indigo-700 bg-indigo-50 rounded-t"
                        : "text-slate-400"
                    }`}
                  >
                    {p.relPos === 0 ? "★" : p.relPos > 0 ? `+${p.relPos}` : p.relPos}
                  </th>
                ))}
              </tr>
            </thead>

            <tbody>
              {/* Reference track */}
              <tr>
                <td className="px-2 py-0.5 text-[10px] font-semibold text-slate-500 uppercase">
                  Ref
                </td>
                {displayPositions.map((p) => (
                  <td
                    key={`ref-${p.relPos}`}
                    className={`px-1.5 py-0.5 text-center text-[11px] font-bold ${
                      p.relPos === 0 ? "bg-indigo-50" : ""
                    }`}
                    style={{ color: NUCLEOTIDE_COLORS[p.reference] || "#64748b" }}
                  >
                    {p.reference}
                  </td>
                ))}
              </tr>

              {/* Alternative nucleotide rows */}
              {altNucleotides.map((alt) => (
                <tr key={alt}>
                  <td
                    className="px-2 py-0.5 text-[10px] font-semibold uppercase"
                    style={{ color: NUCLEOTIDE_COLORS[alt] || "#64748b" }}
                  >
                    →{alt}
                  </td>
                  {displayPositions.map((p) => {
                    const score = p.alternatives[alt];
                    const isHovered =
                      hoveredCell?.pos === String(p.relPos) &&
                      hoveredCell?.alt === alt;
                    const isVariantPos = p.relPos === 0;

                    if (!score) {
                      return (
                        <td
                          key={`${alt}-${p.relPos}`}
                          className={`px-1.5 py-0.5 text-center ${
                            isVariantPos ? "bg-indigo-50" : ""
                          }`}
                        >
                          <div className="w-8 h-6 rounded bg-slate-100" />
                        </td>
                      );
                    }

                    const bgColor = constraintColor(score.magnitude, globalMaxMag);
                    const textColor =
                      score.magnitude > 0.001
                        ? "text-white font-semibold"
                        : "text-slate-500";

                    return (
                      <td
                        key={`${alt}-${p.relPos}`}
                        className={`relative px-1.5 py-0.5 text-center transition-colors ${
                          isVariantPos ? "bg-indigo-50/50" : ""
                        }`}
                        onMouseEnter={() =>
                          setHoveredCell({ pos: String(p.relPos), alt })
                        }
                        onMouseLeave={() => setHoveredCell(null)}
                      >
                        <div
                          className={`flex h-7 w-9 items-center justify-center rounded text-[10px] ${textColor} cursor-default transition-transform ${
                            isHovered ? "scale-110 ring-2 ring-indigo-400 z-10" : ""
                          }`}
                          style={{ backgroundColor: bgColor }}
                          title={`${p.reference}→${alt}: Δ = ${formatDelta(score.delta)} (${score.direction})`}
                        >
                          {score.magnitude > 0.005
                            ? "●"
                            : score.magnitude > 0.001
                              ? "◉"
                              : "·"}
                        </div>
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* ── Legend ── */}
      <div className="flex flex-wrap items-center gap-4 px-5 py-2 border-t border-slate-50">
        <div className="flex items-center gap-1.5">
          <div className="h-3 w-3 rounded bg-red-500/70" />
          <span className="text-[10px] text-slate-500">|Δ| &gt; 0.005</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="h-3 w-3 rounded bg-amber-400/70" />
          <span className="text-[10px] text-slate-500">|Δ| &gt; 0.001</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="h-3 w-3 rounded bg-slate-200" />
          <span className="text-[10px] text-slate-500">|Δ| &lt; 0.001</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="text-[10px] font-semibold text-indigo-600">★</span>
          <span className="text-[10px] text-slate-500">Query variant</span>
        </div>
        {compact && !showAllPositions && (
          <button
            onClick={() => setShowAllPositions(true)}
            className="ml-auto text-[10px] text-indigo-600 hover:text-indigo-800 font-medium"
          >
            Show all {positionEntries.length} positions →
          </button>
        )}
      </div>

      {/* ── Hover Detail Tooltip ── */}
      {hoveredCell && (() => {
        const posData = data.positions[hoveredCell.pos];
        const score = posData?.alternatives[hoveredCell.alt];
        if (!posData || !score) return null;
        return (
          <div className="border-t border-slate-100 bg-slate-50 px-5 py-3">
            <div className="flex items-center gap-3">
              <span className="text-xs font-mono font-semibold text-slate-700">
                Position {hoveredCell.pos}
              </span>
              <span className="text-xs text-slate-500">
                {posData.reference}
                <span className="mx-1 text-slate-300">→</span>
                <span style={{ color: NUCLEOTIDE_COLORS[hoveredCell.alt] }}>
                  {hoveredCell.alt}
                </span>
              </span>
              <span className="text-xs font-mono text-slate-700">
                Δ = {formatDelta(score.delta)}
              </span>
              <span
                className={`inline-flex items-center rounded-full px-2 py-0.5 text-[10px] font-semibold ${
                  score.direction === "pathogenic"
                    ? "bg-red-100 text-red-700"
                    : score.direction === "benign"
                      ? "bg-green-100 text-green-700"
                      : "bg-slate-100 text-slate-600"
                }`}
              >
                {score.direction}
              </span>
              {posData.is_constrained && (
                <span className="text-[10px] text-amber-600 font-medium">
                  Constrained position
                </span>
              )}
            </div>
          </div>
        );
      })()}

      {/* ── Summary Interpretation ── */}
      <div className="border-t border-slate-100 px-5 py-4 space-y-3">
        <h5 className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
          Interpretation
        </h5>

        {/* Constraint zone description */}
        <div className="flex items-start gap-2">
          <div
            className="mt-0.5 h-2.5 w-2.5 rounded-full flex-shrink-0"
            style={{ backgroundColor: zoneColor.bg }}
          />
          <p className="text-sm text-slate-700 leading-relaxed">
            <strong>
              {summary.constraint_zone === "high"
                ? "Highly constrained region"
                : summary.constraint_zone === "moderate"
                  ? "Moderately constrained region"
                  : "Low constraint region"}
            </strong>
            {" — "}
            {summary.constrained_positions} of {summary.total_positions_scanned} positions
            show evolutionary constraint (|Δ| &gt; 0.001).
            {summary.constraint_zone === "high"
              ? " This region is under strong purifying selection — most mutations are likely deleterious."
              : summary.constraint_zone === "moderate"
                ? " This region shows moderate evolutionary pressure — some positions are functionally important while others are tolerant."
                : " Most positions in this region tolerate mutations — the sequence is evolutionarily plastic."}
          </p>
        </div>

        {/* Peak constraint */}
        {summary.peak_constraint_position !== null && (
          <div className="flex items-start gap-2">
            <div className="mt-0.5 h-2.5 w-2.5 rounded-full flex-shrink-0 bg-indigo-500" />
            <p className="text-sm text-slate-700 leading-relaxed">
              <strong>Peak constraint at position {summary.peak_constraint_position}</strong>
              {" (|Δ| = "}
              {summary.peak_constraint_magnitude.toFixed(6)}
              {"). "}
              {summary.peak_constraint_position === 0
                ? "The query variant itself is at the most constrained position — this nucleotide is under strongest evolutionary pressure."
                : `The highest constraint is ${Math.abs(summary.peak_constraint_position)}bp ${summary.peak_constraint_position > 0 ? "downstream" : "upstream"} from the variant — this may indicate a nearby functional element.`}
            </p>
          </div>
        )}

        {/* Constraint boundaries */}
        {summary.constraint_boundaries.length > 0 && (
          <div className="flex items-start gap-2">
            <div className="mt-0.5 h-2.5 w-2.5 rounded-full flex-shrink-0 bg-amber-500" />
            <p className="text-sm text-slate-700 leading-relaxed">
              <strong>Constraint boundaries detected at positions: </strong>
              {summary.constraint_boundaries.map((b, i) => (
                <span key={b}>
                  {i > 0 && ", "}
                  <span className="font-mono text-xs">{b > 0 ? `+${b}` : b}</span>
                </span>
              ))}
              {". "}
              These sharp transitions in evolutionary constraint may indicate functional domain boundaries, regulatory element edges, or splice site positions.
            </p>
          </div>
        )}

        {/* Clinical note */}
        <div className="rounded-md bg-indigo-50 border border-indigo-100 p-3">
          <p className="text-xs text-indigo-800 leading-relaxed">
            <strong>💡 Clinical Relevance: </strong>
            {summary.constraint_zone === "high" && summary.constrained_positions > 5
              ? `The variant falls within a ${summary.constrained_positions}bp constrained microdomain. Mutations throughout this region are predicted to disrupt function, providing orthogonal evidence for pathogenicity. This spatial constraint pattern is not captured by single-position scoring methods (CADD, REVEL, AlphaMissense).`
              : summary.constraint_zone === "moderate"
                ? "The variant sits in a region with mixed constraint — some neighboring positions are sensitive while others are tolerant. This pattern is consistent with a partially constrained functional element. Consider correlation with protein domain annotations for additional context."
                : "The low constraint across this region suggests the variant is in an evolutionarily plastic area. However, absence of constraint does not rule out pathogenicity — some functional elements (e.g., disordered regions, post-translational modification sites) may not show strong evolutionary signal."}
          </p>
        </div>

        {/* Performance note */}
        <p className="text-[10px] text-slate-400">
          ISM scan completed in {summary.scan_duration_ms < 1000
            ? `${Math.round(summary.scan_duration_ms)}ms`
            : `${(summary.scan_duration_ms / 1000).toFixed(1)}s`}
          {" • "}Evo2-7B • {data.window_size.toLocaleString()}bp context window
        </p>
      </div>
    </div>
  );
}
