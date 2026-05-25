"use client";

import React from "react";
import { Brain } from "lucide-react";
import { VariantSequenceContext } from "./variant-sequence-context";
import { GeneDomainMap } from "./gene-domain-map";
import { VariantMechanismExplainer } from "./variant-mechanism-explainer";
import type { XAIFactor } from "~/utils/genome-api";
import type { VEPAnnotation } from "~/app/api/vep/route";

// ─── Types ───────────────────────────────────────────────────────────────

interface XAIPanelProps {
  // Core variant data
  geneSymbol: string;
  chromosome: string;
  position: number;
  reference: string;
  alternative: string;
  prediction: string;
  deltaScore: number;
  classificationConfidence: number;
  variationType?: string | null;
  // Clinical data
  clinvarClassification?: string | null;
  populationFrequency?: {
    gnomad_af: number | null;
    gnomad_max_pop_af?: number | null;
  } | null;
  acmgEvidence?: {
    code: string | null;
    strength?: string | null;
    description?: string;
    clinical_note?: string | null;
  } | null;
  // XAI factors (from backend or client-side)
  xaiFactors?: XAIFactor[] | null;
  // VEP annotation
  vepAnnotation?: VEPAnnotation | null;
  // Controls
  defaultOpen?: boolean;
  className?: string;
}

// ─── Constants ───────────────────────────────────────────────────────────

const SCALE_MIN = -0.02;
const SCALE_MAX = 0.01;
const SCALE_RANGE = SCALE_MAX - SCALE_MIN;
const PATH_BOUNDARY = -0.001;
const PATH_PCT = Math.round(((PATH_BOUNDARY - SCALE_MIN) / SCALE_RANGE) * 100);
const ZERO_PCT = Math.round(((0 - SCALE_MIN) / SCALE_RANGE) * 100);

// ─── Component ───────────────────────────────────────────────────────────

export function XAIPanel({
  geneSymbol,
  chromosome,
  position,
  reference,
  alternative,
  prediction,
  deltaScore,
  classificationConfidence,
  variationType,
  clinvarClassification,
  populationFrequency,
  acmgEvidence,
  xaiFactors,
  vepAnnotation,
  defaultOpen = false,
  className = "",
}: XAIPanelProps) {
  const [showXAI, setShowXAI] = React.useState(defaultOpen);

  const delta = deltaScore ?? 0;
  const isPathogenic = prediction.toLowerCase().includes("pathogenic");
  const isBenign = prediction.toLowerCase().includes("benign");
  const confidencePct = Math.round((classificationConfidence ?? 0) * 100);
  const af = populationFrequency?.gnomad_af;

  // Discordance check
  const clinvarNorm = (clinvarClassification ?? "").toLowerCase().trim();
  const evo2Norm = prediction.toLowerCase().trim();
  const isDiscordant = clinvarNorm && clinvarNorm !== "unknown" && clinvarNorm !== evo2Norm;

  // Use provided XAI factors or compute fallback
  const factors: XAIFactor[] = xaiFactors ?? [];

  // Evidence consensus rows
  const clinvarPresent = !!clinvarClassification && clinvarClassification !== "Unknown";
  const clinvarAgrees = clinvarPresent && !isDiscordant;
  const gnomadPresent = af !== null && af !== undefined;
  const acmgPresent = !!acmgEvidence?.code && acmgEvidence.code !== "None";
  const acmgSupportsBenign = acmgEvidence?.code?.includes("BP") ?? false;
  const acmgSupportsPath = acmgEvidence?.code?.includes("PP") || acmgEvidence?.code?.includes("PS") || acmgEvidence?.code?.includes("PM");

  type RowStatus = "agree" | "warn" | "na";
  const consensusRows: { source: string; finding: string; status: RowStatus; note: string }[] = [
    {
      source: "ClinVar",
      finding: clinvarPresent ? clinvarClassification! : "Not curated",
      status: clinvarPresent ? (clinvarAgrees ? "agree" : "warn") : "na",
      note: clinvarPresent
        ? (clinvarAgrees ? "Independent curation agrees with Evo2" : "⚠ Discordant — may warrant reclassification")
        : "No ClinVar entry — variant not yet submitted to public databases",
    },
    {
      source: "gnomAD v4.1",
      finding: gnomadPresent ? `AF = ${(af! * 100).toFixed(4)}%` : "Not observed",
      status: gnomadPresent ? "agree" : "na",
      note: gnomadPresent
        ? (af! > 0.01 ? "Common — strong benign signal (BA1 criterion)" : "Ultra-rare — absence supports pathogenicity but is not diagnostic")
        : "Not observed in 800K+ individuals — absence is neutral (not diagnostic); many benign variants are simply not yet sampled",
    },
    {
      source: "ACMG Code",
      finding: acmgPresent ? `${acmgEvidence!.code} (${acmgEvidence!.strength ?? "N/A"})` : "No code triggered",
      status: acmgPresent ? (acmgSupportsBenign ? "agree" : acmgSupportsPath ? "warn" : "na") : "na",
      note: acmgPresent
        ? (acmgSupportsBenign ? "Benign criteria met" : acmgSupportsPath ? "Pathogenic criteria triggered" : "Neutral ACMG criterion")
        : "Delta score in uncertain range — between PP3 (pathogenic support) and BP4 (benign support) thresholds",
    },
    {
      source: "Evo2 AI",
      finding: `${prediction} (Evo2-7B: Δ = ${delta >= 0 ? "+" : ""}${delta.toFixed(6)})`,
      status: "agree",
      note: "Reference prediction — trained on 9.3T DNA tokens across 100K+ genomes",
    },
  ];
  const agreeCount = consensusRows.filter((r) => r.status === "agree").length;

  // Evidence strength
  const evidenceStrength = [
    (clinvarClassification && clinvarClassification !== "not_provided" && clinvarClassification !== "Unknown") ? 25 : 0,
    gnomadPresent ? 25 : 0,
    acmgPresent ? 25 : 0,
    Math.abs(delta) > 0.00001 ? 25 : 0,
  ].reduce((a, b) => a + b, 0);

  // Prediction detail
  const predictionDetail = delta < 0
    ? `The negative delta score (${delta.toFixed(4)}) means the mutant sequence has lower evolutionary likelihood than the reference — Evo2 has seen very few sequences like this in healthy genomes, suggesting functional disruption.`
    : `The positive delta score (${delta.toFixed(4)}) means the mutant sequence is well-tolerated by evolution — Evo2 has seen sequences like this in healthy genomes, suggesting the variant preserves protein function.`;

  const populationDetail = af !== null && af !== undefined
    ? af > 0.05
      ? `Common in gnomAD at ${(af * 100).toFixed(2)}% frequency (BA1 criterion) — strongly supports benign classification.`
      : `Observed in gnomAD at ${(af * 100).toExponential(2)} frequency — ultra-rare, which can suggest pathogenicity or recent mutation.`
    : `Not observed in gnomAD (v4.1, 800,000+ individuals). Absence alone is not diagnostic — many ultra-rare benign variants are simply not yet sampled.`;

  const acmgDetail = acmgEvidence?.clinical_note
    ?? `No specific ACMG evidence code triggered. The delta score falls in the uncertain range (between thresholds for PP3 and BP4).`;

  const clamp = Math.max(SCALE_MIN, Math.min(SCALE_MAX, delta));
  const needlePct = Math.round(((clamp - SCALE_MIN) / SCALE_RANGE) * 100);
  const zone = isPathogenic ? "Pathogenic" : isBenign ? "Benign" : "Uncertain";
  const zoneColor = isPathogenic ? "text-red-600" : isBenign ? "text-green-700" : "text-yellow-600";

  return (
    <div className={`rounded-md border border-indigo-200 bg-indigo-50/50 p-4 ${className}`}>
      <button
        onClick={() => setShowXAI((v) => !v)}
        className="w-full flex items-center justify-between text-sm font-medium text-indigo-900 hover:text-indigo-700 transition-colors"
      >
        <span className="flex items-center gap-2">
          <Brain className="h-4 w-4" />
          Why did Evo2 predict this? — XAI Confidence Breakdown
        </span>
        <span className="text-xs text-indigo-500">{showXAI ? "▲ hide" : "▼ show"}</span>
      </button>

      {showXAI && (
        <div className="mt-4 space-y-4">
          <p className="text-xs text-indigo-700/80">
            Evo2 is a DNA language model trained on millions of genomic sequences. It scores how &quot;normal&quot; a sequence looks to evolution — like a grammar checker for DNA. Here is what drove this prediction:
          </p>

          {/* ── Clinical Verdict Banner ── */}
          <div className={`rounded-lg border p-3 ${isPathogenic ? "bg-red-50 border-red-200" : isBenign ? "bg-green-50 border-green-200" : "bg-amber-50 border-amber-200"}`}>
            <div className="flex items-center gap-2 mb-2">
              <span className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-bold ${isPathogenic ? "bg-red-100 text-red-800" : isBenign ? "bg-green-100 text-green-800" : "bg-amber-100 text-amber-800"}`}>
                {isPathogenic ? "⚠️ PATHOGENIC" : isBenign ? "✓ BENIGN" : "◐ UNCERTAIN SIGNIFICANCE"}
              </span>
              <span className="text-[10px] text-slate-500">
                Confidence: {confidencePct}% · Evidence: {evidenceStrength}%
              </span>
            </div>
            {isPathogenic ? (
              <p className="text-xs text-red-700 leading-relaxed">
                <strong>Clinical action:</strong> Consider confirmatory testing, family segregation analysis, and genetic counseling. This variant is predicted to disrupt protein function.
              </p>
            ) : isBenign ? (
              <p className="text-xs text-green-700 leading-relaxed">
                <strong>Clinical action:</strong> Likely benign. No immediate action required unless conflicting family history or phenotype.
              </p>
            ) : (
              <p className="text-xs text-amber-700 leading-relaxed">
                <strong>Clinical action:</strong> Variant of Uncertain Significance. Consider: functional validation (splice assay, protein stability), family segregation analysis, or re-review in 12–24 months as new evidence accumulates.
              </p>
            )}
          </div>

          {/* ── Delta Score Context Scale ── */}
          <div className="rounded-md bg-white border border-indigo-100 p-3">
            <div className="text-xs font-medium text-indigo-800 mb-2">📊 Delta Score in Context</div>
            <div className="space-y-1.5">
              <div className="relative h-5 rounded-full overflow-hidden">
                <div className="absolute inset-0 flex">
                  <div className="h-full bg-red-200" style={{ width: `${PATH_PCT}%` }} title={`Pathogenic zone (delta < ${PATH_BOUNDARY})`} />
                  <div className="h-full bg-yellow-100" style={{ width: `${ZERO_PCT - PATH_PCT}%` }} title="Uncertain zone" />
                  <div className="h-full bg-green-200" style={{ width: `${100 - ZERO_PCT}%` }} title="Benign zone (delta > 0.0)" />
                </div>
                <div
                  className="absolute top-0 bottom-0 w-1 rounded-full bg-[#1e1b4b] shadow-md transition-all duration-700"
                  style={{ left: `calc(${needlePct}% - 2px)` }}
                  title={`Delta: ${delta.toFixed(6)}`}
                />
              </div>
              <div className="flex justify-between text-[9px] text-[#3c4f3d]/60">
                <span>◀ Pathogenic</span>
                <span className="text-yellow-600">Uncertain</span>
                <span>Benign ▶</span>
              </div>
              <div className="flex justify-between text-[8px] text-[#3c4f3d]/40">
                <span>{SCALE_MIN}</span>
                <span>{PATH_BOUNDARY}</span>
                <span>0.0</span>
                <span>+{SCALE_MAX}</span>
              </div>
              <div className={`text-[10px] font-medium mt-1 ${zoneColor}`}>
                Delta score <strong>{delta.toFixed(6)}</strong> → Evo2 classifies this as <strong>{zone}</strong>.{" "}
                {isPathogenic
                  ? "Negative delta means this sequence looks statistically unusual to Evo2 — it has rarely seen sequences like this in healthy genomes."
                  : isBenign
                    ? "Positive delta means Evo2 has seen similar sequences in healthy genomes — evolutionarily well-tolerated."
                    : "Delta score is near zero — Evo2 cannot distinguish this sequence from normal background."}
              </div>
              <div className="text-[10px] text-[#3c4f3d]/50 mt-1 italic">
                This score is milder than most known pathogenic {geneSymbol} variants and consistent with many benign variants.
              </div>
            </div>
          </div>

          {/* ── Evidence Consensus ── */}
          <div className="rounded-md bg-white border border-indigo-100 p-3 space-y-2">
            <div className="text-xs font-medium text-indigo-800">🛡️ Evidence Consensus</div>
            
            {/* Evidence Completeness Ring */}
            <div className="flex items-center gap-4">
              <div className="relative h-16 w-16 flex-shrink-0">
                <svg viewBox="0 0 36 36" className="h-full w-full -rotate-90">
                  <path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" fill="none" stroke="#e2e8f0" strokeWidth="3" />
                  <path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" fill="none" stroke={agreeCount >= 3 ? "#22c55e" : agreeCount === 2 ? "#f59e0b" : "#ef4444"} strokeWidth="3" strokeDasharray={`${(agreeCount / 4) * 100}, 100`} />
                </svg>
                <div className="absolute inset-0 flex flex-col items-center justify-center">
                  <span className="text-sm font-bold text-slate-700">{agreeCount}<span className="text-[9px] text-slate-400">/4</span></span>
                </div>
              </div>
              <div>
                <div className="text-sm font-semibold text-slate-700">
                  {agreeCount >= 3 ? "Strong Evidence Base" : agreeCount === 2 ? "Moderate Evidence" : "Limited Evidence"}
                </div>
                <div className="text-[11px] text-slate-500 mt-0.5">
                  {agreeCount} of 4 evidence sources support {isBenign ? "benign" : isPathogenic ? "pathogenic" : "uncertain"} classification
                </div>
              </div>
            </div>

            {/* Source Pills */}
            <div className="flex flex-wrap gap-2">
              {consensusRows.map((row) => (
                <div key={row.source} className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[10px] font-medium border ${row.status === "agree" ? "bg-green-50 text-green-700 border-green-200" : row.status === "warn" ? "bg-amber-50 text-amber-700 border-amber-200" : "bg-slate-50 text-slate-500 border-slate-200"}`}>
                  {row.status === "agree" ? "✓" : row.status === "warn" ? "⚠" : "○"} {row.source}
                </div>
              ))}
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-xs border-separate" style={{ borderSpacing: "0 3px" }}>
                <thead>
                  <tr className="text-[#3c4f3d]/50">
                    <th className="text-left font-medium px-2 py-1">Source</th>
                    <th className="text-left font-medium px-2 py-1">Finding</th>
                    <th className="text-left font-medium px-2 py-1">Status</th>
                    <th className="text-left font-medium px-2 py-1 hidden sm:table-cell">Note</th>
                  </tr>
                </thead>
                <tbody>
                  {consensusRows.map((row) => (
                    <tr key={row.source} className="bg-[#f9fafb] rounded-md">
                      <td className="px-2 py-1.5 font-medium text-[#3c4f3d] rounded-l-md">{row.source}</td>
                      <td className="px-2 py-1.5 font-mono text-[10px] text-[#3c4f3d]/80">{row.finding}</td>
                      <td className="px-2 py-1.5">
                        <span
                          className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-semibold ${
                            row.status === "agree"
                              ? "bg-green-100 text-green-700"
                              : row.status === "warn"
                                ? "bg-amber-100 text-amber-700"
                                : "bg-slate-100 text-slate-500"
                          }`}
                        >
                          {row.status === "agree" ? "✓ Available" : row.status === "warn" ? "⚠ Caution" : "— Missing"}
                        </span>
                      </td>
                      <td className="px-2 py-1.5 text-[#3c4f3d]/50 hidden sm:table-cell rounded-r-md text-[10px]">{row.note}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="space-y-2 text-sm text-[#3c4f3d] leading-relaxed border-t border-[#3c4f3d]/10 pt-3">
              <div className="flex items-start gap-2">
                <span className={`mt-1 h-2 w-2 rounded-full flex-shrink-0 ${isPathogenic ? "bg-red-500" : "bg-green-500"}`} />
                <p><strong>Evo2 predicts {prediction}</strong> with {confidencePct}% model confidence. {predictionDetail}</p>
              </div>
              <div className="flex items-start gap-2">
                <span className="mt-1 h-2 w-2 rounded-full flex-shrink-0 bg-amber-500" />
                <p><strong>Population data:</strong> {populationDetail}</p>
              </div>
              <div className="flex items-start gap-2">
                <span className="mt-1 h-2 w-2 rounded-full flex-shrink-0 bg-purple-500" />
                <p><strong>ACMG ({acmgEvidence?.code || "None"}):</strong> {acmgDetail}</p>
              </div>
            </div>
          </div>

          {/* ── Confidence Breakdown ── */}
          <div className="rounded-md bg-white border border-indigo-100 p-3 space-y-2">
            <div className="text-xs font-medium text-indigo-800">⚖️ Confidence Breakdown</div>
            <div>
              <div className="flex justify-between text-[10px] text-[#3c4f3d]/70 mb-0.5">
                <span>Model Confidence <span className="text-[9px] text-[#3c4f3d]/40">(Evo2 internal consistency)</span></span>
                <span className={`font-mono font-semibold ${confidencePct >= 70 ? "text-green-600" : confidencePct >= 40 ? "text-yellow-600" : "text-red-600"}`}>{confidencePct}%</span>
              </div>
              <div className="h-2 w-full rounded-full bg-gray-100 overflow-hidden">
                <div className={`h-full rounded-full transition-all duration-500 ${confidencePct >= 70 ? "bg-green-500" : confidencePct >= 40 ? "bg-yellow-400" : "bg-red-400"}`} style={{ width: `${confidencePct}%` }} />
              </div>
              <div className="text-[9px] text-slate-400 mt-0.5">
                {confidencePct >= 70 ? "High confidence — model is consistent across multiple forward passes" : confidencePct >= 40 ? "Moderate confidence — some uncertainty in model predictions" : "Low confidence — model predictions are inconsistent; interpret with caution"}
              </div>
            </div>
            <div>
              <div className="flex justify-between text-[10px] text-[#3c4f3d]/70 mb-0.5">
                <span>Evidence Strength <span className="text-[9px] text-[#3c4f3d]/40">(external data sources)</span></span>
                <span className={`font-mono font-semibold ${evidenceStrength >= 75 ? "text-green-600" : evidenceStrength >= 50 ? "text-yellow-600" : "text-red-600"}`}>{evidenceStrength}%</span>
              </div>
              <div className="h-2 w-full rounded-full bg-gray-100 overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all duration-500 ${evidenceStrength >= 75 ? "bg-green-500" : evidenceStrength >= 50 ? "bg-yellow-400" : "bg-red-400"}`}
                  style={{ width: `${evidenceStrength}%` }}
                />
              </div>
              <div className="text-[9px] text-slate-400 mt-0.5">
                {evidenceStrength >= 75 ? "Strong evidence — multiple independent sources corroborate the prediction" : evidenceStrength >= 50 ? "Moderate evidence — some sources missing or ambiguous" : "Weak evidence — insufficient external data for confident classification"}
              </div>
            </div>
            {evidenceStrength < 50 && (
              <div className={`text-[10px] rounded px-2 py-1.5 border ${isPathogenic ? "bg-red-50 text-red-700 border-red-200" : isBenign ? "bg-green-50 text-green-700 border-green-200" : "bg-amber-50 text-amber-700 border-amber-200"}`}>
                <strong>⚠ Limited evidence ({evidenceStrength}%):</strong> {isPathogenic ? "Despite pathogenic prediction, external evidence is sparse. Consider functional validation before clinical action." : isBenign ? "Limited evidence supports benign classification. Re-review if new data emerges." : "Insufficient evidence for definitive classification. This is a true Variant of Uncertain Significance."}
              </div>
            )}
          </div>

          {/* ── Factor Bars ── */}
          {factors.length > 0 && (
            <div className="space-y-3">
              {factors.map((f) => (
                <div key={f.label}>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="font-medium text-[#3c4f3d]">{f.label}</span>
                    <span className="font-mono text-[#3c4f3d]/70">{f.contribution}%</span>
                  </div>
                  <div className="h-2 w-full rounded-full bg-gray-200 overflow-hidden">
                    <div
                      className="h-full rounded-full transition-all duration-500"
                      style={{ width: `${(f.contribution / 50) * 100}%`, backgroundColor: f.color }}
                    />
                  </div>
                  <div className="mt-0.5 text-[10px] text-[#3c4f3d]/60">{f.detail}</div>
                </div>
              ))}
            </div>
          )}

          {/* ── Sequence Context ── */}
          <VariantSequenceContext
            chromosome={chromosome}
            position={position}
            reference={reference}
            alternative={alternative}
            geneSymbol={geneSymbol}
            variationType={variationType}
          />

          {/* ── Protein Domain Map ── */}
          <GeneDomainMap
            geneSymbol={geneSymbol}
            genomicPosition={position}
            chromosome={chromosome}
            prediction={prediction}
          />

          {/* ── Molecular Mechanism Card ── */}
          <VariantMechanismExplainer
            geneSymbol={geneSymbol}
            chromosome={chromosome}
            genomicPosition={position}
            reference={reference || "N"}
            alternative={alternative}
            variantType={variationType || "SNV"}
            deltaScore={deltaScore}
            prediction={prediction}
            vepAnnotation={vepAnnotation}
          />

          {/* ACMG clinical note */}
          {acmgEvidence?.clinical_note && (
            <div className="rounded-md bg-white border border-indigo-100 p-3">
              <div className="text-xs font-medium text-indigo-800 mb-1">🧬 Evo2 Model Note</div>
              <div className="text-xs text-[#3c4f3d]/80 leading-relaxed">{acmgEvidence.clinical_note}</div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
