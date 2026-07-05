"use client";

import { useEffect, useRef, useState } from "react";
import { Button } from "./ui/button";
import { cn } from "~/lib/utils";
import {
  X,
  ExternalLink,
  Download,
  Copy,
  Trash2,
  FileDown,
} from "lucide-react";
import { getClassificationColorClasses } from "~/utils/coloring-utils";
import type { VEPAnnotation } from "~/app/api/vep/route";
import { FormattedClinicalSummary } from "./formatted-clinical-summary";
import { ISMHeatmap } from "./ism-heatmap";
import { CounterfactualCard } from "./counterfactual-card";
import { ACMGCriteriaTable } from "./acmg-criteria-table";
import { XAIPanel } from "./xai-panel";
import { KnowledgeGraphCard } from "./knowledge-graph-card";
import { ToolConcordance } from "./tool-concordance";
import {
  MultiModelConsensus,
  buildConsensusData,
} from "./multi-model-consensus";
import { ACMGRefinedCard } from "./acmg-refined-card";
import { DesignTherapeutics } from "./design-therapeutics";
import { useActiveVariant } from "~/hooks/use-active-variant";
import type {
  ISMScanResult,
  XAIFactors,
  Counterfactuals,
  ACMGCriteriaResult,
  KnowledgeGraph,
  ExternalScores,
  ACMGRefinedResult,
} from "~/utils/genome-api";

interface SavedReport {
  id: string;
  geneSymbol: string;
  chromosome: string;
  position: number;
  reference: string;
  alternative: string;
  genomeId: string;
  prediction: string;
  deltaScore: number;
  classificationConfidence: number;
  classificationSource?: string;
  clinvarClassification?: string;
  variationType?: string;
  clinvarId?: string;
  populationFrequency?: {
    gnomad_af: number | null;
    gnomad_max_pop_af: number | null;
    source: string;
    is_common_variant: boolean;
    frequency_classification?: string | null;
  };
  acmgEvidence?: {
    code: string | null;
    strength: string | null;
    description: string;
    clinical_note: string;
  };
  literatureContext?: {
    summary: string | null;
    pubmed_ids: string[];
    gene_function: string | null;
    articles_found: number;
  };
  // Multi-modal RAG output
  clinicalSummary?: string | null;
  evidenceConfidence?: {
    vep: { available: boolean; confidence: string; note: string };
    evo2: { available: boolean; confidence: string; note: string };
    gnomad: { available: boolean; confidence: string; note: string };
    clinvar: { available: boolean; confidence: string; note: string };
    uniprot: { available: boolean; confidence: string; note: string };
    pubmed: { available: boolean; confidence: string; note: string };
    overall: {
      level: string;
      sources_available: string;
      high_confidence_sources: string;
    };
  } | null;
  vepAnnotation?: VEPAnnotation | null;
  ismScanData?: ISMScanResult | null;
  xaiFactors?: XAIFactors | null;
  counterfactuals?: Counterfactuals | null;
  acmgCriteria?: ACMGCriteriaResult | null;
  knowledgeGraph?: KnowledgeGraph | null;
  externalScores?: ExternalScores | null;
  acmgCriteriaRefined?: ACMGRefinedResult | null;
  createdAt: string;
}

// ─── XAI Helper: compute confidence factors from report data ───────────────
// NOTE: Evo2 delta scores are typically in the range 0.00001 – 0.01 for benign
// variants and -0.001 to -0.5 for pathogenic ones. Thresholds are calibrated
// to this real-world distribution, NOT a 0–1.0 range.
function computeConfidenceFactors(report: SavedReport) {
  const delta = report.deltaScore ?? 0;
  const absDelta = Math.abs(delta);
  const factors: {
    label: string;
    contribution: number;
    color: string;
    detail: string;
  }[] = [];

  // 1. Delta score signal strength (0–50%)
  // Evo2 typical range: |delta| < 0.0001 = almost nothing, 0.001 = weak, 0.01+ = moderate, 0.1+ = strong
  const EVO2_WEAK = 0.0001;
  const EVO2_MOD = 0.005;
  const EVO2_STRONG = 0.05;
  const EVO2_MAX = 0.5;

  const deltaContrib = Math.min(
    50,
    Math.round(
      (Math.log10(absDelta / EVO2_WEAK + 1) /
        Math.log10(EVO2_MAX / EVO2_WEAK + 1)) *
        50,
    ),
  );
  const signalLabel =
    absDelta >= EVO2_STRONG
      ? "strong"
      : absDelta >= EVO2_MOD
        ? "moderate"
        : absDelta >= EVO2_WEAK
          ? "weak"
          : "minimal";
  factors.push({
    label: "Evolutionary Signal",
    contribution: Math.max(1, deltaContrib),
    color: delta < 0 ? "#ef4444" : "#22c55e",
    detail: `|Δ| = ${absDelta.toFixed(6)} — ${signalLabel} evolutionary pressure signal`,
  });

  // 2. Score direction clarity (0–20%)
  // "Near zero" means the delta is so small Evo2 is effectively undecided.
  // Use Evo2-calibrated threshold: < 0.0001 is truly ambiguous.
  const nearZero = absDelta < EVO2_WEAK;
  const ambiguous = absDelta < EVO2_MOD;
  const directionContrib = nearZero
    ? 5
    : ambiguous
      ? 10
      : absDelta >= EVO2_STRONG
        ? 20
        : 14;
  const directionLabel = nearZero
    ? `Effect Size: Minimal`
    : ambiguous
      ? `Effect Size: Weak`
      : `Effect Size: Clear`;
  factors.push({
    label: directionLabel,
    contribution: directionContrib,
    color: nearZero ? "#f59e0b" : ambiguous ? "#fb923c" : "#6366f1",
    detail: nearZero
      ? `|Δ| = ${absDelta.toFixed(6)} — essentially zero, Evo2 sees this sequence as equally likely`
      : ambiguous
        ? `Weak ${delta < 0 ? "pathogenic" : "benign"} signal (|Δ| = ${absDelta.toFixed(6)}), interpret cautiously`
        : `Clear ${delta < 0 ? "pathogenic" : "benign"} direction (|Δ| = ${absDelta.toFixed(6)})`,
  });

  // 3. Population rarity signal (0–15%)
  const af = report.populationFrequency?.gnomad_af;
  let popContrib = 10;
  let popDetail =
    "Not observed in gnomAD (800k+ individuals) — novel or ultra-rare";
  if (af !== null && af !== undefined) {
    if (af > 0.05) {
      popContrib = 2;
      popDetail = `Common variant (AF=${(af * 100).toFixed(2)}%) — strong benign signal (BA1 criterion)`;
    } else if (af > 0.01) {
      popContrib = 5;
      popDetail = `Uncommon (AF=${(af * 100).toFixed(2)}%)`;
    } else if (af > 0.001) {
      popContrib = 10;
      popDetail = `Rare (AF=${(af * 100).toFixed(4)}%) — low population frequency`;
    } else {
      popContrib = 15;
      popDetail = `Ultra-rare (AF=${af.toExponential(2)}) — rarity supports pathogenic classification`;
    }
  }
  factors.push({
    label: "Population Rarity",
    contribution: popContrib,
    color: "#8b5cf6",
    detail: popDetail,
  });

  // 4. ACMG evidence (0–15%)
  const acmgCode = report.acmgEvidence?.code ?? "None";
  let acmgContrib = 5;
  let acmgDetail = "No ACMG code triggered — delta score in uncertain range";
  if (acmgCode && acmgCode !== "None") {
    if (acmgCode.includes("VeryStrong") || acmgCode.includes("PVS")) {
      acmgContrib = 15;
      acmgDetail = `${acmgCode} — very strong ACMG criterion met`;
    } else if (
      acmgCode.includes("Strong") ||
      acmgCode.startsWith("PS") ||
      acmgCode.startsWith("BS")
    ) {
      acmgContrib = 12;
      acmgDetail = `${acmgCode} — strong ACMG criterion`;
    } else if (
      acmgCode.includes("Moderate") ||
      acmgCode.startsWith("PM") ||
      acmgCode.startsWith("BP")
    ) {
      acmgContrib = 9;
      acmgDetail = `${acmgCode} — moderate ACMG criterion`;
    } else if (acmgCode.includes("Supporting") || acmgCode.startsWith("PP")) {
      acmgContrib = 6;
      acmgDetail = `${acmgCode} — supporting ACMG criterion`;
    } else {
      acmgContrib = 7;
      acmgDetail = `${acmgCode} — ACMG criterion triggered`;
    }
  }
  factors.push({
    label: "ACMG Evidence",
    contribution: acmgContrib,
    color: "#0ea5e9",
    detail: acmgDetail,
  });

  const total = factors.reduce((s, f) => s + f.contribution, 0);
  return { factors, total };
}

export function SavedReportModal({
  report,
  onClose,
  onDelete,
}: {
  report: SavedReport | null;
  onClose: () => void;
  onDelete?: (reportId: string) => void;
}) {
  const [isDeleting, setIsDeleting] = useState(false);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [successToast, setSuccessToast] = useState<{
    message: string;
    icon: string;
  } | null>(null);
  const [showXAI, setShowXAI] = useState(false);
  const [isPdfLoading, setIsPdfLoading] = useState(false);
  const contentRef = useRef<HTMLDivElement>(null);
  const { setActiveVariant, clearActiveVariant, isChatPanelOpen, isChatExpanded } = useActiveVariant();

  useEffect(() => {
    if (report) {
      document.body.style.overflow = "hidden";
      setActiveVariant({
        reportId: report.id,
        geneSymbol: report.geneSymbol,
        chromosome: report.chromosome,
        position: report.position,
        reference: report.reference,
        alternative: report.alternative,
        genomeId: report.genomeId,
        prediction: report.prediction,
        deltaScore: report.deltaScore,
        classificationConfidence: report.classificationConfidence,
        clinvarClassification: report.clinvarClassification,
        variationType: report.variationType,
        clinvarId: report.clinvarId,
        reportData: report as unknown as Record<string, unknown>,
      });
      return () => {
        document.body.style.overflow = "";
        clearActiveVariant();
      };
    }
  }, [report, setActiveVariant, clearActiveVariant]);

  if (!report) return null;

  const delta = report.deltaScore ?? 0;
  const af = report.populationFrequency?.gnomad_af;
  const isPathogenic = report.prediction.toLowerCase().includes("pathogenic");
  const isBenign = report.prediction.toLowerCase().includes("benign");
  const confidencePct = Math.round(
    (report.classificationConfidence ?? 0) * 100,
  );

  // ClinVar vs Evo2 discordance
  const clinvarNorm = (report.clinvarClassification ?? "").toLowerCase().trim();
  const evo2Norm = report.prediction.toLowerCase().trim();
  const isDiscordant =
    clinvarNorm && clinvarNorm !== "unknown" && clinvarNorm !== evo2Norm;

  // Use backend-computed XAI factors if available, fall back to client-side
  const factors =
    report.xaiFactors?.factors ?? computeConfidenceFactors(report).factors;
  const xaiTotal =
    report.xaiFactors?.total ?? computeConfidenceFactors(report).total;

  // ─── Dynamic Evidence Summary ───────────────────────────────────────────
  const predictionDetail =
    delta < 0
      ? `The negative delta score (${delta.toFixed(4)}) means the mutant sequence has lower evolutionary likelihood than the reference — Evo2 has seen very few sequences like this in healthy genomes, suggesting functional disruption.`
      : `The positive delta score (${delta.toFixed(4)}) means the mutant sequence is well-tolerated by evolution — Evo2 has seen sequences like this in healthy genomes, suggesting the variant preserves protein function.`;

  const populationDetail =
    af !== null && af !== undefined
      ? af > 0.05
        ? `Common in gnomAD at ${(af * 100).toFixed(2)}% frequency (BA1 criterion) — strongly supports benign classification.`
        : `Observed in gnomAD at ${(af * 100).toExponential(2)} frequency — ultra-rare, which can suggest pathogenicity or recent mutation.`
      : `Not observed in gnomAD (v4.1, 800,000+ individuals). Absence alone is not diagnostic — many ultra-rare benign variants are simply not yet sampled.`;

  const acmgDetail =
    report.acmgEvidence?.clinical_note ??
    `No specific ACMG evidence code triggered. The delta score falls in the uncertain range (between thresholds for PP3 and BP4).`;

  // ─── Report text generation ─────────────────────────────────────────────
  const generateReportText = () => {
    const date = new Date().toLocaleDateString("en-US", {
      month: "2-digit",
      day: "2-digit",
      year: "numeric",
    });
    const changeNotation = `${report.reference || "N"}>${report.alternative}`;
    const varSomeChrom = report.chromosome.startsWith("chr")
      ? report.chromosome
      : `chr${report.chromosome}`;

    return `HELIXMIND VARIANT ANALYSIS REPORT
Generated: ${date}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

VARIANT INFORMATION
• Gene:       ${report.geneSymbol}
• Position:   ${report.chromosome}:${report.position.toLocaleString()}
• Change:     ${changeNotation}
• Type:       ${report.variationType || "Single Nucleotide Variant"}
• ClinVar ID: ${report.clinvarId || "N/A"}

CLASSIFICATION COMPARISON  
• ClinVar Assessment:  ${report.clinvarClassification || "Unknown"}
• Evo2 AI Prediction:  ${report.prediction}
• Concordance:         ${isDiscordant ? "⚠ DISCORDANT" : "✓ Concordant"}

EVO2 AI SCORING
• Delta Score:  ${delta.toFixed(6)}
• Confidence:   ${confidencePct}%
• Interpretation: ${predictionDetail}

EXPLAINABLE AI — CONFIDENCE BREAKDOWN
${factors.map((f) => `• ${f.label.padEnd(25)} ${f.contribution}%  — ${f.detail}`).join("\n")}
  ─────────────────────────────────────
  Total XAI Score:            ~${xaiTotal}%

POPULATION FREQUENCY (gnomAD v4.1)
• ${populationDetail}

ACMG EVIDENCE
• Code:     ${report.acmgEvidence?.code || "None"}
• Strength: ${report.acmgEvidence?.strength || "N/A"}
• Note:     ${acmgDetail}

LITERATURE (PubMed)
• Articles found: ${report.literatureContext?.articles_found ?? 0}
${
  (report.literatureContext?.pubmed_ids ?? [])
    .slice(0, 5)
    .map((id) => `• https://pubmed.ncbi.nlm.nih.gov/${id}/`)
    .join("\n") || "• No references available"
}

AI CLINICAL SUMMARY
${report.literatureContext?.summary || "No LLM summary available. Configure groq-config Modal secret to enable."}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Cross-references:
• ClinVar:  https://www.ncbi.nlm.nih.gov/clinvar/?term=${report.chromosome}%5Bchr%5D+AND+${report.position}%5Bpos%5D
• VarSome:  https://varsome.com/position/${report.genomeId}/${varSomeChrom}-${report.position}
• gnomAD:   https://gnomad.broadinstitute.org/variant/${report.chromosome.replace("chr", "")}-${report.position}-${report.reference || "N"}-${report.alternative}
`;
  };

  const showSuccessToast = (message: string, icon: string) => {
    setSuccessToast({ message, icon });
    setTimeout(() => setSuccessToast(null), 3000);
  };
  const copyReport = async () => {
    try {
      await navigator.clipboard.writeText(generateReportText());
      showSuccessToast("Report copied!", "📋");
    } catch {
      showSuccessToast("Failed to copy", "❌");
    }
  };
  const downloadReport = () => {
    const blob = new Blob([generateReportText()], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `helixmind-${report.geneSymbol}-${report.position}.txt`;
    a.click();
    URL.revokeObjectURL(url);
    showSuccessToast("Report downloaded!", "📥");
  };

  const downloadPDF = async () => {
    if (!contentRef.current) return;
    setIsPdfLoading(true);
    showSuccessToast("Generating PDF…", "⏳");

    // Temporarily expand XAI panel so all content is captured
    const wasXAIOpen = showXAI;
    if (!wasXAIOpen) setShowXAI(true);

    // Wait for DOM to settle after state change
    await new Promise((r) => setTimeout(r, 600));

    // Variables hoisted so the finally block can always restore them
    let _clone: HTMLElement | null = null;
    let _origGCS: typeof window.getComputedStyle | null = null;

    try {
      const { default: html2canvas } = await import("html2canvas");
      const { default: jsPDF } = await import("jspdf");

      const el = contentRef.current;

      // ── 1×1 canvas: browser natively converts oklch → rgb for Canvas 2D ──
      const _cvs = document.createElement("canvas");
      _cvs.width = _cvs.height = 1;
      const _ctx = _cvs.getContext("2d");

      const resolveOklch = (v: string): string => {
        if (!v.includes("oklch")) return v;
        if (!_ctx) return "transparent";
        try {
          _ctx.clearRect(0, 0, 1, 1);
          _ctx.fillStyle = v;
          _ctx.fillRect(0, 0, 1, 1);
          const d = _ctx.getImageData(0, 0, 1, 1).data;
          const [r, g, b, a] = [d[0] ?? 0, d[1] ?? 0, d[2] ?? 0, d[3] ?? 255];
          return a < 255
            ? `rgba(${r},${g},${b},${(a / 255).toFixed(3)})`
            : `rgb(${r},${g},${b})`;
        } catch (err) {
          console.error("Failed to resolve oklch color:", err);
          return "transparent";
        }
      };

      // ── NUCLEAR FIX: proxy window.getComputedStyle ──────────────────────
      // html2canvas calls window.getComputedStyle() for every element's colors.
      // Turbopack injects styles via adoptedStyleSheets (not in document.styleSheets),
      // so CSS-rule patching can't reach them. Proxying getComputedStyle intercepts
      // oklch at the only level guaranteed to work regardless of injection mechanism.
      _origGCS = window.getComputedStyle;
      window.getComputedStyle = (
        element: Element,
        pseudoElt?: string | null,
      ): CSSStyleDeclaration => {
        const cs = _origGCS!(element, pseudoElt);
        return new Proxy(cs, {
          get(target, prop: string | symbol) {
            const raw = (target as unknown as Record<string | symbol, unknown>)[
              prop
            ];
            // Convert string properties
            if (typeof raw === "string" && raw.includes("oklch")) {
              return resolveOklch(raw);
            }
            // Wrap getPropertyValue
            if (prop === "getPropertyValue") {
              return (name: string) => {
                const v = target.getPropertyValue(name);
                return v.includes("oklch") ? resolveOklch(v) : v;
              };
            }
            // Bind methods to real target
            if (typeof raw === "function")
              return (raw as Function).bind(target);
            return raw;
          },
        });
      };

      // Clone element and remove scroll constraints so we capture full height
      _clone = el.cloneNode(true) as HTMLElement;
      _clone.style.position = "fixed";
      _clone.style.top = "0";
      _clone.style.left = "-9999px";
      _clone.style.width = `${el.scrollWidth}px`;
      _clone.style.maxHeight = "none";
      _clone.style.height = "auto";
      _clone.style.overflow = "visible";
      _clone.style.zIndex = "-1";
      _clone.style.backgroundColor = "#ffffff";
      document.body.appendChild(_clone);

      await new Promise((r) => requestAnimationFrame(r));

      const canvas = await html2canvas(_clone, {
        scale: 2,
        useCORS: true,
        backgroundColor: "#ffffff",
        width: _clone.scrollWidth,
        height: _clone.scrollHeight,
        windowWidth: _clone.scrollWidth,
        scrollX: 0,
        scrollY: 0,
      });

      // ── Restore everything ── (also done in finally for error safety)

      // A4 in mm: 210 × 297
      const pdf = new jsPDF({
        orientation: "portrait",
        unit: "mm",
        format: "a4",
      });
      const pageW = pdf.internal.pageSize.getWidth();
      const pageH = pdf.internal.pageSize.getHeight();
      const margin = 12;
      const contentW = pageW - margin * 2;

      // Scale image to fit page width
      const imgW = canvas.width;
      const imgH = canvas.height;
      const ratio = contentW / (imgW / 2); // /2 because scale:2
      const scaledH = (imgH / 2) * ratio;

      // Add header to every page
      const addHeader = (
        pdf: InstanceType<typeof jsPDF>,
        pageNum: number,
        totalPages: number,
      ) => {
        pdf.setFontSize(8);
        pdf.setTextColor(100);
        pdf.text(
          `HelixMind Variant Analysis Report  |  ${report.geneSymbol} ${report.chromosome}:${report.position}  |  ${new Date().toLocaleDateString("en-GB")}`,
          margin,
          8,
        );
        pdf.text(`Page ${pageNum} of ${totalPages}`, pageW - margin, 8, {
          align: "right",
        });
        pdf.setDrawColor(200);
        pdf.line(margin, 10, pageW - margin, 10);
      };

      const usableH = pageH - margin * 2 - 14; // subtract header+footer space
      const totalPages = Math.ceil(scaledH / usableH);

      let yOffset = 0;
      for (let page = 1; page <= totalPages; page++) {
        if (page > 1) pdf.addPage();
        addHeader(pdf, page, totalPages);

        // Crop the canvas slice for this page
        const srcY = Math.round((yOffset / ratio) * 2); // back to canvas pixels
        const sliceH = Math.round((usableH / ratio) * 2);

        const pageCanvas = document.createElement("canvas");
        pageCanvas.width = imgW;
        pageCanvas.height = Math.min(sliceH, imgH - srcY);
        const ctx = pageCanvas.getContext("2d")!;
        ctx.drawImage(
          canvas,
          0,
          srcY,
          imgW,
          pageCanvas.height,
          0,
          0,
          imgW,
          pageCanvas.height,
        );

        const pageImg = pageCanvas.toDataURL("image/png");
        const sliceRenderH = (pageCanvas.height / 2) * ratio;
        pdf.addImage(
          pageImg,
          "PNG",
          margin,
          margin + 12,
          contentW,
          sliceRenderH,
          undefined,
          "FAST",
        );

        yOffset += usableH;
      }

      // Footer disclaimer on last page
      pdf.setFontSize(7);
      pdf.setTextColor(150);
      pdf.text(
        "This report is generated by HelixMind for research purposes only. Not for clinical diagnostic use.",
        pageW / 2,
        pageH - 6,
        { align: "center" },
      );

      pdf.save(
        `helixmind-${report.geneSymbol}-${report.chromosome}-${report.position}.pdf`,
      );
      showSuccessToast("PDF downloaded!", "📄");
    } catch (err) {
      console.error("PDF export failed:", err);
      showSuccessToast("PDF export failed", "❌");
    } finally {
      // Restore window.getComputedStyle and remove clone
      if (_origGCS) window.getComputedStyle = _origGCS;
      if (_clone && _clone.parentNode) _clone.parentNode.removeChild(_clone);
      setIsPdfLoading(false);
      if (!wasXAIOpen) setShowXAI(false);
    }
  };
  const confirmDelete = async () => {
    setIsDeleting(true);
    try {
      const response = await fetch("/api/history/delete", {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ reportId: report.id }),
      });
      if (response.ok) {
        onDelete?.(report.id);
        onClose();
      } else {
        const d = await response.json();
        alert(d.error || "Failed to delete");
      }
    } catch {
      alert("Failed to delete");
    } finally {
      setIsDeleting(false);
      setShowDeleteConfirm(false);
    }
  };

  const varSomeChrom = report.chromosome.startsWith("chr")
    ? report.chromosome
    : `chr${report.chromosome}`;

  return (
    <div className={cn(
      "fixed inset-0 z-50 flex h-screen min-h-screen items-center justify-center overflow-y-auto bg-black/50 p-4 transition-all duration-300",
      isChatPanelOpen && (isChatExpanded ? "pr-[620px]" : "pr-[420px]"),
    )}>
      <div className="max-h-[90vh] w-full max-w-3xl overflow-hidden rounded-2xl bg-white shadow-xl">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-[#3c4f3d]/10 p-5">
          <h3 className="text-lg font-medium text-[#3c4f3d]">
            Variant Analysis Report
          </h3>
          <Button
            variant="ghost"
            size="sm"
            onClick={onClose}
            className="h-7 w-7 cursor-pointer p-0 text-[#3c4f3d]/70 hover:bg-[#e9eeea]/70"
          >
            <X className="h-5 w-5" />
          </Button>
        </div>

        {/* Scrollable content */}
        <div
          ref={contentRef}
          className="max-h-[calc(90vh-130px)] space-y-5 overflow-y-auto p-5"
        >
          {/* ── Variant Info ── */}
          <div className="rounded-md border border-[#3c4f3d]/10 bg-[#e9eeea]/30 p-4">
            <h4 className="mb-3 text-sm font-medium text-[#3c4f3d]">
              Variant Information
            </h4>
            <div className="grid gap-4 text-xs md:grid-cols-2">
              <div className="space-y-2">
                <div className="flex">
                  <span className="w-28 text-[#3c4f3d]/70">Position:</span>
                  <span>{report.position.toLocaleString()}</span>
                </div>
                <div className="flex">
                  <span className="w-28 text-[#3c4f3d]/70">Type:</span>
                  <span>
                    {report.variationType || "Single Nucleotide Variant"}
                  </span>
                </div>
              </div>
              <div className="space-y-2">
                <div className="flex">
                  <span className="w-28 text-[#3c4f3d]/70">Variant:</span>
                  <span className="font-mono">
                    <span className="text-green-600">
                      {report.reference || "N"}
                    </span>
                    <span className="text-orange-500">&gt;</span>
                    <span className="text-blue-600">{report.alternative}</span>
                  </span>
                </div>
                <div className="flex">
                  <span className="w-28 text-[#3c4f3d]/70">Gene:</span>
                  <span className="font-medium">{report.geneSymbol}</span>
                </div>
              </div>
            </div>
            <div className="mt-4 flex flex-wrap gap-3">
              <a
                href={
                  report.clinvarId
                    ? `https://www.ncbi.nlm.nih.gov/clinvar/variation/${report.clinvarId}/`
                    : `https://www.ncbi.nlm.nih.gov/clinvar/?term=${report.chromosome}%5Bchr%5D+AND+${report.position}%5Bpos%5D`
                }
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-1.5 rounded-md bg-[#de8246]/10 px-3 py-1.5 text-xs font-medium text-[#de8246] transition-colors hover:bg-[#de8246]/20"
              >
                <ExternalLink className="h-3 w-3" /> View in ClinVar
              </a>
              <a
                href={`https://varsome.com/position/${report.genomeId}/${varSomeChrom}-${report.position}`}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-1.5 rounded-md bg-teal-500/10 px-3 py-1.5 text-xs font-medium text-teal-700 transition-colors hover:bg-teal-500/20"
              >
                <ExternalLink className="h-3 w-3" /> VarSome
              </a>
              <a
                href={`https://gnomad.broadinstitute.org/variant/${report.chromosome.replace("chr", "")}-${report.position}-${report.reference || "N"}-${report.alternative}`}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-1.5 rounded-md bg-blue-500/10 px-3 py-1.5 text-xs font-medium text-blue-700 transition-colors hover:bg-blue-500/20"
              >
                <ExternalLink className="h-3 w-3" /> gnomAD
              </a>
            </div>
          </div>

          {/* ── Classification Comparison ── */}
          <div>
            <h4 className="mb-3 text-sm font-medium text-[#3c4f3d]">
              Analysis Comparison
            </h4>
            <div className="rounded-md border border-[#3c4f3d]/10 bg-white p-4">
              <div className="grid gap-4 md:grid-cols-2">
                {/* ClinVar */}
                <div className="rounded-md bg-[#e9eeea]/50 p-4">
                  <h5 className="mb-2 flex items-center gap-2 text-xs font-medium text-[#3c4f3d]">
                    <span className="flex h-5 w-5 items-center justify-center rounded-full bg-[#3c4f3d]/10">
                      <span className="h-3 w-3 rounded-full bg-[#3c4f3d]"></span>
                    </span>
                    ClinVar Assessment
                  </h5>
                  <div
                    className={`w-fit rounded-md px-2 py-1 text-xs font-normal ${getClassificationColorClasses(report.clinvarClassification || "Unknown")}`}
                  >
                    {report.clinvarClassification || "Unknown significance"}
                  </div>
                </div>
                {/* Evo2 */}
                <div className="rounded-md bg-[#e9eeea]/50 p-4">
                  <h5 className="mb-2 flex items-center gap-2 text-xs font-medium text-[#3c4f3d]">
                    <span className="flex h-5 w-5 items-center justify-center rounded-full bg-[#3c4f3d]/10">
                      <span className="h-3 w-3 rounded-full bg-[#de8246]"></span>
                    </span>
                    Evo2 Prediction
                  </h5>
                  <div
                    className={`flex w-fit items-center gap-1 rounded-md px-2 py-1 text-xs font-normal ${getClassificationColorClasses(report.prediction)}`}
                  >
                    {isPathogenic ? (
                      <svg
                        xmlns="http://www.w3.org/2000/svg"
                        className="h-3 w-3"
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
                    ) : (
                      <svg
                        xmlns="http://www.w3.org/2000/svg"
                        className="h-3 w-3"
                        fill="none"
                        viewBox="0 0 24 24"
                        stroke="currentColor"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                        />
                      </svg>
                    )}
                    {report.prediction}
                  </div>
                  <div className="mt-3">
                    <div className="mb-1 text-xs text-[#3c4f3d]/70">
                      Delta Likelihood Score:
                    </div>
                    <div
                      className={`font-mono text-sm font-semibold ${isPathogenic ? "text-red-800" : isBenign ? "text-green-800" : "text-[#3c4f3d]"}`}
                    >
                      {delta.toFixed(6)}
                    </div>
                    <div className="relative mt-2 h-3 rounded-full bg-gradient-to-r from-red-500 via-yellow-400 to-green-500">
                      <div className="absolute top-0 left-1/2 h-3 w-0.5 bg-white/70"></div>
                      <div
                        className="absolute top-1/2 h-4 w-4 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-white bg-[#3c4f3d] shadow-md"
                        style={{
                          left: `${Math.max(5, Math.min(95, ((Math.max(-0.01, Math.min(0.01, delta)) + 0.01) / 0.02) * 100))}%`,
                        }}
                      />
                    </div>
                    <div className="mt-1 flex justify-between text-[10px] text-[#3c4f3d]/60">
                      <span>Pathogenic</span>
                      <span>Neutral</span>
                      <span>Benign</span>
                    </div>
                  </div>
                  <div className="mt-3">
                    <div className="mb-1 text-xs text-[#3c4f3d]/70">
                      Confidence:
                    </div>
                    <div className="h-2 w-full overflow-hidden rounded-full bg-gray-200">
                      <div
                        className="h-full rounded-full bg-gradient-to-r from-green-400 to-green-600"
                        style={{ width: `${confidencePct}%` }}
                      />
                    </div>
                    <div className="mt-1 text-right text-xs font-medium text-[#3c4f3d]">
                      {confidencePct}%
                    </div>
                  </div>
                </div>
              </div>
              {isDiscordant && (
                <div className="mt-4 flex items-center gap-2 text-xs text-amber-700">
                  <span className="flex h-4 w-4 items-center justify-center font-bold text-amber-600">
                    !
                  </span>
                  <span>
                    ClinVar ({report.clinvarClassification}) differs from Evo2 (
                    {report.prediction})
                  </span>
                </div>
              )}
            </div>
          </div>

          {/* ── Clinical Annotations ── */}
          <div className="rounded-md border border-[#3c4f3d]/10 bg-[#e9eeea]/30 p-4">
            <h4 className="mb-3 text-sm font-medium text-[#3c4f3d]">
              Clinical Annotations
            </h4>
            <div className="grid gap-4 md:grid-cols-3">
              <div className="rounded-md bg-white p-3">
                <div className="mb-1 text-xs font-medium text-[#3c4f3d]/70">
                  Population Frequency
                </div>
                {af !== null && af !== undefined ? (
                  <div className="font-mono text-sm text-[#3c4f3d]">
                    {(af * 100).toFixed(6)}%
                  </div>
                ) : (
                  <div className="text-sm text-[#3c4f3d]/50">
                    Not found in gnomAD
                  </div>
                )}
                {report.populationFrequency?.frequency_classification && (
                  <div className="mt-1 text-xs font-medium text-green-700">
                    {report.populationFrequency.frequency_classification}
                  </div>
                )}
              </div>
              <div className="rounded-md bg-white p-3">
                <div className="mb-1 text-xs font-medium text-[#3c4f3d]/70">
                  ACMG Evidence
                </div>
                {report.acmgEvidence?.code &&
                report.acmgEvidence.code !== "None" ? (
                  <div className="mb-1 font-mono text-xs font-semibold text-indigo-700">
                    {report.acmgEvidence.code}
                  </div>
                ) : null}
                <div className="text-xs text-[#3c4f3d]/60">
                  {report.acmgEvidence?.description ||
                    "Computational evidence is inconclusive"}
                </div>
              </div>
              <div className="rounded-md bg-white p-3">
                <div className="mb-1 text-xs font-medium text-[#3c4f3d]/70">
                  Literature
                </div>
                <div className="mb-2 text-sm text-[#3c4f3d]/60">
                  {(report.literatureContext?.articles_found ?? 0) > 0 ? (
                    `${report.literatureContext!.articles_found} articles`
                  ) : (
                    <span className="text-amber-600">
                      0 articles{" "}
                      <span className="text-[10px]">(PubMed unavailable)</span>
                    </span>
                  )}
                </div>
                {(report.literatureContext?.pubmed_ids ?? []).length > 0 && (
                  <div className="flex flex-wrap gap-1">
                    {report
                      .literatureContext!.pubmed_ids.slice(0, 5)
                      .map((id) => (
                        <a
                          key={id}
                          href={`https://pubmed.ncbi.nlm.nih.gov/${id}/`}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="inline-flex items-center gap-1 rounded-full border border-slate-200 bg-slate-100 px-2 py-0.5 text-[10px] font-medium text-slate-600 transition-colors hover:bg-slate-200"
                        >
                          PMID:{id}
                        </a>
                      ))}
                  </div>
                )}
              </div>
            </div>
            {/* Multi-Modal RAG Clinical Summary */}
            {report.clinicalSummary && (
              <div className="mt-4 rounded-md border border-[#3c4f3d]/10 bg-white p-4">
                <div className="mb-3 flex items-center gap-2">
                  <span className="flex h-6 w-6 items-center justify-center rounded-full bg-[#de8246]/10">
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      className="h-4 w-4 text-[#de8246]"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M9 12h6m-6-4h6m-2 5.5c.5.3 1.2.2 1.4-.3.3-.5.2-1.2-.3-1.4-.5-.3-1.2-.2-1.4.3-.3.5-.2 1.2.3 1.4zM7 20h10v-2H7v2zM9 2h6v4H9V2z"
                      />
                    </svg>
                  </span>
                  <div className="text-xs font-medium text-[#3c4f3d]">
                    Clinical Summary (Multi-Modal RAG)
                  </div>
                  <span className="ml-auto rounded bg-[#e9eeea] px-2 py-0.5 text-[10px] font-medium text-[#3c4f3d]/70">
                    Llama 3.3 70B
                  </span>
                </div>
                <FormattedClinicalSummary summary={report.clinicalSummary} />

                {/* Evidence Confidence Matrix */}
                {report.evidenceConfidence && (
                  <div className="mt-4 border-t border-[#3c4f3d]/10 pt-3">
                    <div className="mb-2 text-[10px] font-medium tracking-wider text-[#3c4f3d]/50 uppercase">
                      Evidence Confidence
                    </div>
                    <div className="grid grid-cols-3 gap-2">
                      {Object.entries(report.evidenceConfidence)
                        .filter(([key]) => key !== "overall")
                        .map(([key, val]) => {
                          const v = val as {
                            available: boolean;
                            confidence: string;
                            note: string;
                          };
                          const color =
                            v.confidence === "High"
                              ? "bg-green-50 text-green-700 border-green-200"
                              : v.confidence === "Medium"
                                ? "bg-yellow-50 text-yellow-700 border-yellow-200"
                                : v.confidence === "Low"
                                  ? "bg-orange-50 text-orange-700 border-orange-200"
                                  : "bg-gray-50 text-gray-500 border-gray-200";
                          return (
                            <div
                              key={key}
                              className={`rounded border px-2 py-1.5 ${color}`}
                              title={v.note}
                            >
                              <div className="flex items-center justify-between">
                                <span className="text-[10px] font-semibold tracking-wide uppercase">
                                  {key}
                                </span>
                                <span className="text-[10px] font-medium">
                                  {v.confidence}
                                </span>
                              </div>
                              <div className="mt-0.5 text-[9px] leading-tight opacity-80">
                                {v.available ? "Available" : "Missing"}
                              </div>
                            </div>
                          );
                        })}
                    </div>
                    {report.evidenceConfidence.overall && (
                      <div className="mt-2 flex items-center gap-2">
                        <span className="text-[10px] text-[#3c4f3d]/60">
                          Overall:
                        </span>
                        <span
                          className={`inline-flex items-center rounded-full px-2 py-0.5 text-[10px] font-semibold ${
                            report.evidenceConfidence.overall.level === "High"
                              ? "bg-green-100 text-green-800"
                              : report.evidenceConfidence.overall.level ===
                                  "Medium"
                                ? "bg-yellow-100 text-yellow-800"
                                : "bg-red-100 text-red-800"
                          }`}
                        >
                          {report.evidenceConfidence.overall.level}
                        </span>
                        <span className="text-[10px] text-[#3c4f3d]/50">
                          {report.evidenceConfidence.overall.sources_available}{" "}
                          sources,{" "}
                          {
                            report.evidenceConfidence.overall
                              .high_confidence_sources
                          }{" "}
                          high-confidence
                        </span>
                      </div>
                    )}
                  </div>
                )}

                {/* Disclaimer */}
                <div className="mt-4 rounded border border-amber-200 bg-amber-50 p-3">
                  <div className="flex items-start gap-2">
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      className="mt-0.5 h-4 w-4 flex-shrink-0 text-amber-600"
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
                    <p className="text-[11px] leading-relaxed text-amber-800">
                      <strong>Computational Prediction Only:</strong> This
                      summary is generated by an AI model (Llama 3.3 70B) using
                      publicly available databases. It is intended for research
                      and educational purposes only and must not be used as a
                      substitute for professional clinical judgment, genetic
                      counseling, or laboratory validation. Always verify
                      critical findings with a board-certified clinical
                      geneticist or molecular pathologist before making patient
                      care decisions.
                    </p>
                  </div>
                </div>
              </div>
            )}

            {/* Fallback: Legacy AI Clinical Summary */}
            {!report.clinicalSummary && report.literatureContext?.summary && (
              <div className="mt-4 rounded-md bg-white p-3">
                <div className="mb-2 text-xs font-medium text-[#3c4f3d]/70">
                  AI Clinical Summary
                </div>
                <FormattedClinicalSummary
                  summary={report.literatureContext.summary}
                />
              </div>
            )}
          </div>

          {/* ── In-Silico Mutagenesis Scan ── */}
          {report.ismScanData && (
            <div className="mt-4">
              <ISMHeatmap
                data={report.ismScanData}
                geneSymbol={report.geneSymbol}
                variantPosition={report.position}
                chromosome={report.chromosome}
              />
            </div>
          )}

          {/* ── Counterfactual Analysis ── */}
          {report.counterfactuals && (
            <div className="mt-4">
              <CounterfactualCard
                data={report.counterfactuals}
                observedAlternative={report.alternative}
              />
            </div>
          )}

          {/* ── ACMG/AMP Criteria Mapping ── */}
          {report.acmgCriteria && (
            <div className="mt-4">
              <ACMGCriteriaTable data={report.acmgCriteria} />
            </div>
          )}

          {/* ── LLM-Refined ACMG Criteria ── */}
          {report.acmgCriteriaRefined && (
            <div className="mt-4">
              <ACMGRefinedCard
                refined={report.acmgCriteriaRefined}
                ruleBased={report.acmgCriteria ?? null}
              />
            </div>
          )}

          {/* ── XAI: Why did Evo2 say this? ── */}
          <XAIPanel
            geneSymbol={report.geneSymbol}
            chromosome={report.chromosome}
            position={report.position}
            reference={report.reference}
            alternative={report.alternative}
            prediction={report.prediction}
            deltaScore={report.deltaScore}
            classificationConfidence={report.classificationConfidence}
            variationType={report.variationType}
            clinvarClassification={report.clinvarClassification}
            populationFrequency={report.populationFrequency}
            acmgEvidence={report.acmgEvidence}
            xaiFactors={report.xaiFactors?.factors ?? null}
            vepAnnotation={report.vepAnnotation}
          />

          {/* ── Knowledge Graph ── */}
          {report.knowledgeGraph && (
            <div className="mt-4">
              <KnowledgeGraphCard data={report.knowledgeGraph} />
            </div>
          )}

          {/* ── Multi-Model Consensus Panel ── */}
          {(() => {
            const consensusData = buildConsensusData({
              prediction: report.prediction,
              classification_confidence: report.classificationConfidence,
              external_scores: report.externalScores ?? undefined,
            });
            return consensusData ? (
              <div className="mt-4">
                <MultiModelConsensus data={consensusData} />
              </div>
            ) : null;
          })()}

          {/* ── Multi-Tool Concordance ── */}
          {report.externalScores && (
            <div className="mt-4">
              <ToolConcordance
                data={report.externalScores}
                evo2Prediction={report.prediction}
                evo2Delta={report.deltaScore}
                clinvarClassification={report.clinvarClassification}
              />
            </div>
          )}

          {/* ── Design Therapeutics ── */}
          <div className="mt-4">
            <DesignTherapeutics report={report} />
          </div>

          {/* ── Evidence Consensus Table now lives inside XAI panel above ── */}
        </div>
        {/* end scrollable content */}

        {/* ── Footer action bar — inside the white card ── */}
        <div className="flex flex-wrap items-center justify-between gap-2 rounded-b-2xl border-t border-[#3c4f3d]/10 bg-[#f9fafb] px-5 py-3">
          <div className="flex items-center gap-2">
            <button
              onClick={copyReport}
              className="inline-flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-medium text-[#3c4f3d]/70 transition-colors hover:bg-[#3c4f3d]/10"
            >
              <Copy className="h-3.5 w-3.5" /> Copy text
            </button>
            <button
              onClick={downloadReport}
              className="inline-flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-medium text-[#3c4f3d]/70 transition-colors hover:bg-[#3c4f3d]/10"
            >
              <Download className="h-3.5 w-3.5" /> Download .txt
            </button>
            <button
              onClick={downloadPDF}
              disabled={isPdfLoading}
              className="inline-flex items-center gap-1.5 rounded-md bg-indigo-600 px-3 py-1.5 text-xs font-semibold text-white shadow-sm transition-colors hover:bg-indigo-700 disabled:cursor-not-allowed disabled:opacity-60"
            >
              <FileDown className="h-3.5 w-3.5" />
              {isPdfLoading ? "Generating…" : "Download PDF"}
            </button>
          </div>
          <button
            onClick={() => setShowDeleteConfirm(true)}
            className="inline-flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-medium text-red-500 transition-colors hover:bg-red-50"
          >
            <Trash2 className="h-3.5 w-3.5" /> Delete report
          </button>
        </div>
      </div>
      {/* end white card */}

      {/* Toast */}
      {successToast && (
        <div className="animate-in fade-in slide-in-from-bottom-2 fixed bottom-6 left-1/2 z-[100] flex -translate-x-1/2 items-center gap-2 rounded-full bg-[#3c4f3d] px-4 py-2 text-sm text-white shadow-lg">
          <span>{successToast.icon}</span>
          <span>{successToast.message}</span>
        </div>
      )}

      {/* Delete confirm */}
      {showDeleteConfirm && (
        <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/40">
          <div className="mx-4 w-full max-w-xs rounded-xl bg-white p-5 shadow-xl">
            <div className="mb-3 flex items-center gap-3">
              <div className="flex h-9 w-9 items-center justify-center rounded-full bg-red-100">
                <Trash2 className="h-4 w-4 text-red-600" />
              </div>
              <h3 className="text-base font-semibold text-[#3c4f3d]">
                Delete Report
              </h3>
            </div>
            <p className="mb-4 text-sm text-[#3c4f3d]/70">
              Are you sure you want to delete this report? This cannot be
              undone.
            </p>
            <div className="flex justify-end gap-2">
              <button
                onClick={() => setShowDeleteConfirm(false)}
                className="rounded-md bg-gray-100 px-3 py-1.5 text-sm font-medium text-[#3c4f3d] transition-colors hover:bg-gray-200"
              >
                Cancel
              </button>
              <button
                onClick={confirmDelete}
                disabled={isDeleting}
                className="rounded-md bg-red-500 px-3 py-1.5 text-sm font-medium text-white transition-colors hover:bg-red-600 disabled:opacity-60"
              >
                {isDeleting ? "Deleting…" : "Delete"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export type { SavedReport };
