"use client";

import {
  type AnalysisResult,
  analyzeVariantWithAPI,
  fetchSingleBase,
  type ClinvarVariant,
  type GeneBounds,
  type GeneFromSearch,
} from "~/utils/genome-api";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Input } from "./ui/input";
import React, {
  forwardRef,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
} from "react";
import {
  getClassificationColorClasses,
  getNucleotideColorClass,
} from "~/utils/coloring-utils";
import { Button } from "./ui/button";
import { Zap, ShoppingCart } from "lucide-react";
import { FormattedClinicalSummary } from "./formatted-clinical-summary";
import { ISMHeatmap } from "./ism-heatmap";
import { CounterfactualCard } from "./counterfactual-card";
import { ACMGCriteriaTable } from "./acmg-criteria-table";
import { XAIPanel } from "./xai-panel";
import { KnowledgeGraphCard } from "./knowledge-graph-card";
import { ToolConcordance } from "./tool-concordance";
import { ACMGRefinedCard } from "./acmg-refined-card";
import { useActiveVariant } from "~/hooks/use-active-variant";
export interface VariantAnalysisHandle {
  focusAlternativeInput: () => void;
}

interface VariantAnalysisProps {
  gene: GeneFromSearch;
  genomeId: string;
  chromosome: string;
  clinvarVariants: Array<ClinvarVariant>;
  referenceSequence: string | null;
  sequencePosition: number | null;
  geneBounds: GeneBounds | null;
  onAnalysisComplete?: () => void;
}

const VariantAnalysis = forwardRef<VariantAnalysisHandle, VariantAnalysisProps>(
  (
    {
      gene,
      genomeId,
      chromosome,
      clinvarVariants = [],
      referenceSequence,
      sequencePosition,
      geneBounds,
      onAnalysisComplete,
    }: VariantAnalysisProps,
    ref,
  ) => {
    const [variantPosition, setVariantPosition] = useState<string>(
      geneBounds?.min?.toString() || "",
    );
    const [variantReference, setVariantReference] = useState("");
    const [variantAlternative, setVariantAlternative] = useState("");
    const [variantResult, setVariantResult] = useState<AnalysisResult | null>(
      null,
    );
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [variantError, setVariantError] = useState<string | null>(null);
    const [needsCredits, setNeedsCredits] = useState(false);
    const [enableISM, setEnableISM] = useState(false);
    const alternativeInputRef = useRef<HTMLInputElement>(null);
    const { setActiveVariant, clearActiveVariant } = useActiveVariant();

    // Clear active variant context when the component unmounts or gene changes
    useEffect(() => {
      return () => {
        clearActiveVariant();
      };
    }, [gene?.symbol, clearActiveVariant]);

    useImperativeHandle(ref, () => ({
      focusAlternativeInput: () => {
        if (alternativeInputRef.current) {
          alternativeInputRef.current.focus();
        }
      },
    }));

    useEffect(() => {
      if (sequencePosition && referenceSequence) {
        setVariantPosition(String(sequencePosition));
        setVariantReference(referenceSequence);
      }
    }, [sequencePosition, referenceSequence]);

    const handlePositionChange = (e: React.ChangeEvent<HTMLInputElement>) => {
      setVariantPosition(e.target.value);
      setVariantReference("");
    };

    const handleVariantSubmit = async (pos: string, alt: string) => {
      const position = parseInt(pos);
      if (isNaN(position)) {
        setVariantError("Please enter a valid position number");
        return;
      }

      // Allow multi-nucleotide sequences for indels (e.g., GTC, ATG)
      const validNucleotides = /^[ATGC]+$/i;
      if (!validNucleotides.test(alt)) {
        setVariantError(
          "Nucleotides must be A, C, G or T (multi-nucleotide sequences allowed)",
        );
        return;
      }

      setIsAnalyzing(true);
      setVariantError(null);
      setNeedsCredits(false);

      try {
        const data = await analyzeVariantWithAPI({
          position,
          alternative: alt,
          genomeId,
          chromosome,
          geneSymbol: gene?.symbol,
          runISMScan: enableISM,
        });
        setVariantResult(data);

        // Activate this variant as the chatbot context
        setActiveVariant({
          reportId: `live-${gene?.symbol}-${data.position}-${Date.now()}`,
          geneSymbol: gene?.symbol || "Unknown",
          chromosome,
          position: data.position,
          reference: data.reference || "",
          alternative: data.alternative,
          genomeId,
          prediction: data.prediction,
          deltaScore: data.delta_score,
          classificationConfidence: data.classification_confidence,
          clinvarClassification: data.clinvar_evidence?.status ?? null,
          variationType: null,
          clinvarId: data.clinvar_evidence?.variation_id ?? null,
          reportData: data as unknown as Record<string, unknown>,
        });

        // Trigger history refresh
        onAnalysisComplete?.();
        // Notify other components (e.g. AnalysisHistory) that a new report exists
        window.dispatchEvent(new CustomEvent("analysis-saved"));
        // Trigger credits refresh
        window.dispatchEvent(new CustomEvent("credits-updated"));
      } catch (err: unknown) {
        console.error(err);
        // Check if it's a credit-related error
        const error = err as Error & { needsCredits?: boolean };
        if (error.needsCredits) {
          setNeedsCredits(true);
          setVariantError(error.message || "Not enough credits");
        } else {
          setVariantError(error.message || "Failed to analyze variant");
        }
      } finally {
        setIsAnalyzing(false);
      }
    };

    return (
      <Card className="gap-0 border-none bg-white py-0 shadow-sm">
        <CardHeader className="pt-4 pb-2">
          <CardTitle className="text-sm font-normal text-[#3c4f3d]/70">
            Variant Analysis
          </CardTitle>
        </CardHeader>
        <CardContent className="pb-4">
          <p className="mb-4 text-xs text-[#3c4f3d]/80">
            Generate evidence-based pathogenicity predictions for any genomic
            coordinate.
          </p>
          <div className="flex flex-wrap items-end gap-4">
            <div>
              <label className="mb-1 block text-xs text-[#3c4f3d]/70">
                Position
              </label>
              <Input
                value={variantPosition}
                onChange={handlePositionChange}
                className="h-8 w-32 border-[#3c4f3d]/10 text-xs"
              />
            </div>
            <div>
              <label className="mb-1 block text-xs text-[#3c4f3d]/70">
                Alternative (variant)
              </label>
              <Input
                ref={alternativeInputRef}
                value={variantAlternative}
                onChange={(e) =>
                  setVariantAlternative(e.target.value.toUpperCase())
                }
                className="h-8 w-32 border-[#3c4f3d]/10 text-xs"
                placeholder="e.g., T or ATG"
              />
            </div>
            {variantReference && (
              <div className="mb-2 flex items-center gap-2 text-xs text-[#3c4f3d]">
                <span>Substitution</span>
                <span
                  className={`font-medium ${getNucleotideColorClass(variantReference)}`}
                >
                  {variantReference}
                </span>
                <span>→</span>
                <span
                  className={`font-medium ${getNucleotideColorClass(variantAlternative)}`}
                >
                  {variantAlternative ? variantAlternative : "?"}
                </span>
              </div>
            )}
            <label className="flex cursor-pointer items-center gap-2 select-none">
              <input
                type="checkbox"
                checked={enableISM}
                onChange={(e) => setEnableISM(e.target.checked)}
                disabled={isAnalyzing}
                className="h-3.5 w-3.5 cursor-pointer rounded border-slate-300 text-indigo-600 focus:ring-indigo-500"
              />
              <span className="text-[11px] text-slate-500">
                ISM Scan{" "}
                <span className="text-slate-400">(±20bp constraint map)</span>
              </span>
            </label>
            <Button
              disabled={isAnalyzing || !variantPosition || !variantAlternative}
              className="h-8 cursor-pointer bg-[#3c4f3d] text-xs text-white hover:bg-[#3c4f3d]/90"
              onClick={() =>
                handleVariantSubmit(
                  variantPosition.replaceAll(",", ""),
                  variantAlternative,
                )
              }
            >
              {isAnalyzing ? (
                <>
                  <span className="mr-2 inline-block h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent align-middle"></span>
                  Analyzing...
                </>
              ) : (
                "Analyze variant"
              )}
            </Button>
          </div>

          {variantPosition &&
            clinvarVariants
              .filter(
                (variant) =>
                  variant?.variation_type
                    ?.toLowerCase()
                    .includes("single nucleotide") &&
                  parseInt(variant?.location?.replaceAll(",", "")) ===
                    parseInt(variantPosition.replaceAll(",", "")),
              )
              .map((matchedVariant) => {
                const refAltMatch = matchedVariant.title.match(/(\w)>(\w)/);

                let ref = null;
                let alt = null;
                if (refAltMatch && refAltMatch.length === 3) {
                  ref = refAltMatch[1];
                  alt = refAltMatch[2];
                }

                if (!ref || !alt) return null;

                // Log when ClinVar card renders (confirms new code is loaded)
                console.log(
                  `🧬 [ClinVar-Card-Render] Position ${matchedVariant.location}: extracted ref=${ref}, alt=${alt} from title: "${matchedVariant.title}"`,
                );

                return (
                  <div
                    key={matchedVariant.clinvar_id}
                    className="mt-4 rounded-md border border-[#3c4f3d]/10 bg-[#e9eeea]/30 p-4"
                  >
                    <div className="mb-3 flex items-center justify-between">
                      <h4 className="text-sm font-medium text-[#3c4f3d]">
                        Known Variant Detected
                      </h4>
                      <span className="text-xs text-[#3c4f3d]/70">
                        Position: {matchedVariant.location}
                      </span>
                    </div>

                    <div className="grid gap-4 md:grid-cols-2">
                      <div>
                        <div className="mb-1 text-xs font-medium text-[#3c4f3d]/70">
                          Variant Details
                        </div>
                        <div className="text-sm">{matchedVariant.title}</div>
                        <div className="mt-2 text-sm">
                          {gene?.symbol} {variantPosition}{" "}
                          <span className="font-mono">
                            <span className={getNucleotideColorClass(ref)}>
                              {ref}
                            </span>
                            <span>{">"}</span>
                            <span className={getNucleotideColorClass(alt)}>
                              {alt}
                            </span>
                          </span>
                        </div>
                        <div className="mt-2 text-xs text-[#3c4f3d]/70">
                          ClinVar classification
                          <span
                            className={`ml-1 rounded-sm px-2 py-0.5 ${getClassificationColorClasses(matchedVariant.classification)}`}
                          >
                            {matchedVariant.classification || "Unknown"}
                          </span>
                        </div>
                      </div>
                      <div className="flex items-center justify-end">
                        <Button
                          disabled={isAnalyzing}
                          variant="outline"
                          size="sm"
                          className="h-7 cursor-pointer border-[#3c4f3d]/20 bg-[#e9eeea] text-xs text-[#3c4f3d] hover:bg-[#3c4f3d]/10"
                          onClick={async () => {
                            console.log(
                              `[ClinVar-Debug] Starting analysis for position ${variantPosition}`,
                            );
                            console.log(
                              `[ClinVar-Debug] From title: ref=${ref}, alt=${alt}`,
                            );

                            // Fetch genomic reference to detect strand orientation
                            const genomicRef = await fetchSingleBase(
                              chromosome,
                              parseInt(variantPosition.replaceAll(",", "")),
                              genomeId,
                            );
                            console.log(
                              `[ClinVar-Debug] Genomic reference fetched: ${genomicRef}`,
                            );

                            // Check if gene is on minus strand (transcript ref != genomic ref)
                            const isMinusStrand =
                              genomicRef &&
                              ref &&
                              genomicRef.toUpperCase() !== ref.toUpperCase();
                            console.log(
                              `[ClinVar-Debug] Strand detection: genomic=${genomicRef}, transcript=${ref}, isMinusStrand=${isMinusStrand}`,
                            );

                            let finalAlt = alt;

                            // For minus-strand genes, reverse complement the alternative
                            if (isMinusStrand && genomicRef) {
                              const complement: Record<string, string> = {
                                A: "T",
                                T: "A",
                                C: "G",
                                G: "C",
                              };
                              const originalAlt = alt;
                              finalAlt = complement[alt.toUpperCase()] || alt;
                              console.log(
                                `[ClinVar] Minus-strand gene detected. Transcript: ${ref}>${originalAlt}, Genomic: ${genomicRef}>${finalAlt}`,
                              );
                            } else {
                              console.log(
                                `[ClinVar-Debug] Plus-strand or detection failed, using original alt=${alt}`,
                              );
                            }

                            console.log(
                              `[ClinVar-Debug] Final alternative being sent: ${finalAlt}`,
                            );
                            setVariantAlternative(finalAlt);
                            handleVariantSubmit(
                              variantPosition.replaceAll(",", ""),
                              finalAlt,
                            );
                          }}
                        >
                          {isAnalyzing ? (
                            <>
                              <span className="mr-1 inline-block h-3 w-3 animate-spin rounded-full border-2 border-white border-t-transparent align-middle"></span>
                              Analyzing...
                            </>
                          ) : (
                            <>
                              <Zap className="mr-1 inline-block h-3 w-3" />
                              Analyze this Variant
                            </>
                          )}
                        </Button>
                      </div>
                    </div>
                  </div>
                );
              })[0]}
          {variantError && (
            <div className="mt-4 rounded-md bg-red-50 p-3 text-xs text-red-600">
              <div className="flex items-center justify-between">
                <span>{variantError}</span>
                {needsCredits && (
                  <button
                    onClick={async () => {
                      try {
                        const response = await fetch("/api/stripe/checkout", {
                          method: "POST",
                          headers: { "Content-Type": "application/json" },
                          body: JSON.stringify({ packageId: "credits_5" }),
                        });
                        const data = await response.json();
                        if (data.url) {
                          window.location.href = data.url;
                        }
                      } catch (e) {
                        console.error("Checkout error:", e);
                      }
                    }}
                    className="flex items-center gap-1.5 rounded-md bg-[#de8246] px-3 py-1.5 text-xs font-medium text-white transition-colors hover:bg-[#c97340]"
                  >
                    <ShoppingCart className="h-3.5 w-3.5" />
                    Buy Credits
                  </button>
                )}
              </div>
            </div>
          )}
          {variantResult && (
            <div className="mt-6 rounded-md border border-[#3c4f3d]/10 bg-[#e9eeea]/30 p-4 shadow-sm">
              <h4 className="mb-3 text-sm font-medium text-[#3c4f3d]">
                Analysis Report
              </h4>

              {/* Analysis Report Box - styled like modal */}
              <div className="rounded-md border border-[#3c4f3d]/10 bg-white p-4">
                <div className="grid gap-4 md:grid-cols-2">
                  {/* Left side - Variant Info */}
                  <div className="rounded-md bg-[#f5f5f4] p-4">
                    <div className="mb-4">
                      <div className="mb-1 text-xs font-medium text-[#3c4f3d]/70">
                        Variant
                      </div>
                      <div className="font-mono text-sm font-semibold tracking-tight text-[#3c4f3d]">
                        {gene?.symbol}{" "}
                        {variantResult.position?.toLocaleString()}{" "}
                        {variantResult.reference}
                        <span className="text-orange-500">{">"}</span>
                        {variantResult.alternative}
                      </div>
                    </div>
                    <div>
                      <div className="mb-1 text-xs font-medium text-[#3c4f3d]/70">
                        Delta likelihood score
                      </div>
                      <div
                        className={`font-mono text-lg font-semibold ${variantResult.prediction.toLowerCase().includes("pathogenic") ? "text-red-800" : variantResult.prediction.toLowerCase().includes("benign") ? "text-green-800" : "text-[#3c4f3d]"}`}
                      >
                        {variantResult.delta_score.toFixed(6)}
                      </div>

                      {/* Visual scale bar */}
                      <div className="mt-2">
                        {/* Scale labels */}
                        <div className="mb-1 flex justify-between text-[10px] text-[#3c4f3d]/60">
                          <span>Pathogenic</span>
                          <span>Neutral</span>
                          <span>Benign</span>
                        </div>

                        {/* Gradient bar with marker */}
                        <div className="relative h-3 rounded-full bg-gradient-to-r from-red-500 via-yellow-400 to-green-500">
                          {/* Center indicator for neutral (0) */}
                          <div className="absolute top-0 left-1/2 h-3 w-0.5 bg-white/70"></div>

                          {/* Score marker */}
                          <div
                            className="absolute top-1/2 h-4 w-4 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-white bg-[#3c4f3d] shadow-md"
                            style={{
                              left: `${Math.max(
                                5,
                                Math.min(
                                  95,
                                  // Use much tighter range for better sensitivity to small scores
                                  // Most delta scores are in range -0.01 to +0.01
                                  ((Math.max(
                                    -0.01,
                                    Math.min(0.01, variantResult.delta_score),
                                  ) +
                                    0.01) /
                                    0.02) *
                                    100,
                                ),
                              )}%`,
                            }}
                            title={`Score: ${variantResult.delta_score.toFixed(6)}`}
                          ></div>
                        </div>

                        {/* Scale numbers */}
                        <div className="mt-0.5 flex justify-between text-[9px] text-[#3c4f3d]/40">
                          <span>-1</span>
                          <span>0</span>
                          <span>+1</span>
                        </div>
                      </div>

                      <div className="mt-2 text-xs text-[#3c4f3d]/60">
                        {variantResult.delta_score < -0.001
                          ? "⚠️ Negative score indicates loss of function"
                          : variantResult.delta_score > 0.001
                            ? "✓ Positive score indicates neutral/benign function"
                            : "○ Score near zero suggests minimal impact"}
                      </div>
                    </div>
                  </div>

                  {/* Right side - Evo2 Prediction */}
                  <div className="rounded-md bg-[#f5f5f4] p-4">
                    <div className="mb-2 flex items-center gap-2 text-xs font-medium text-[#3c4f3d]">
                      <span className="flex h-5 w-5 items-center justify-center rounded-full bg-[#3c4f3d]/10">
                        <span className="h-3 w-3 rounded-full bg-[#de8246]"></span>
                      </span>
                      Evo2 Prediction
                    </div>
                    <div
                      className={`inline-flex items-center gap-1 rounded-lg px-3 py-1 text-xs ${getClassificationColorClasses(variantResult.prediction)}`}
                    >
                      {variantResult.prediction
                        .toLowerCase()
                        .includes("pathogenic") ? (
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
                      {variantResult.prediction}
                    </div>
                    <div className="mt-4">
                      <div className="mb-1 text-xs font-medium text-[#3c4f3d]/70">
                        Confidence:
                      </div>
                      <div className="h-2 w-full rounded-full bg-white">
                        <div
                          className={`h-2 rounded-full ${variantResult.prediction.toLowerCase().includes("pathogenic") ? "bg-red-500" : "bg-green-500"}`}
                          style={{
                            width: `${Math.min(100, variantResult.classification_confidence * 100)}%`,
                          }}
                        ></div>
                      </div>
                      <div className="mt-1 text-right text-xs text-[#3c4f3d]/60">
                        {Math.round(
                          variantResult.classification_confidence * 100,
                        )}
                        %
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Clinical Enrichment Section */}
              <div className="mt-4 border-t border-[#3c4f3d]/10 pt-4">
                <h5 className="mb-3 text-xs font-medium text-[#3c4f3d]/70">
                  Clinical Annotations
                </h5>

                <div className="grid gap-4 md:grid-cols-3">
                  {/* gnomAD Population Frequency */}
                  <div className="rounded-md bg-white p-3">
                    <div className="mb-1 text-xs font-medium text-[#3c4f3d]/70">
                      Population Frequency
                    </div>
                    {variantResult.population_frequency?.gnomad_af ? (
                      <div className="text-sm">
                        <span className="font-medium">
                          {(
                            variantResult.population_frequency.gnomad_af * 100
                          ).toFixed(4)}
                          %
                        </span>
                        <div className="mt-1 text-xs text-[#3c4f3d]/60">
                          {variantResult.population_frequency.is_common_variant
                            ? "Common variant"
                            : "Rare variant"}
                        </div>
                      </div>
                    ) : (
                      <div className="text-sm text-[#3c4f3d]/50">
                        Not found in gnomAD
                      </div>
                    )}
                    <div className="mt-1 text-xs text-[#3c4f3d]/40">
                      {variantResult.population_frequency?.source ||
                        "gnomAD v4.1"}
                    </div>
                  </div>

                  {/* ACMG Evidence */}
                  <div className="rounded-md bg-white p-3">
                    <div className="mb-1 text-xs font-medium text-[#3c4f3d]/70">
                      ACMG Evidence
                    </div>
                    {variantResult.acmg_evidence?.code &&
                    variantResult.acmg_evidence.code !== "None" ? (
                      <div>
                        <span className="inline-block rounded bg-blue-100 px-2 py-0.5 text-xs font-medium text-blue-800">
                          {variantResult.acmg_evidence.code}
                          {variantResult.acmg_evidence.strength &&
                            ` (${variantResult.acmg_evidence.strength})`}
                        </span>
                        <div className="mt-1 text-xs text-[#3c4f3d]/60">
                          {variantResult.acmg_evidence.description}
                        </div>
                      </div>
                    ) : (
                      <div className="text-sm text-[#3c4f3d]/50">
                        {variantResult.acmg_evidence?.description ||
                          "No evidence code assigned"}
                      </div>
                    )}
                  </div>

                  {/* Literature Context */}
                  <div className="rounded-md bg-white p-3">
                    <div className="mb-1 text-xs font-medium text-[#3c4f3d]/70">
                      Literature
                    </div>
                    {variantResult.literature_context?.articles_found ? (
                      <div>
                        <div className="text-xs text-[#3c4f3d]/60">
                          {Math.min(
                            5,
                            variantResult.literature_context.pubmed_ids
                              ?.length || 0,
                          )}{" "}
                          articles
                        </div>
                        {variantResult.literature_context.pubmed_ids?.length >
                          0 && (
                          <div className="mt-1 flex flex-wrap gap-1">
                            {variantResult.literature_context.pubmed_ids
                              .slice(0, 5)
                              .map((id) => (
                                <a
                                  key={id}
                                  href={`https://pubmed.ncbi.nlm.nih.gov/${id}/`}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="inline-flex items-center gap-1 rounded-full border border-slate-200 bg-slate-100 px-2 py-1 text-xs font-medium text-slate-600 transition-colors hover:bg-slate-200 hover:text-slate-800"
                                >
                                  <svg
                                    className="h-3 w-3"
                                    fill="currentColor"
                                    viewBox="0 0 24 24"
                                  >
                                    <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 17.93c-3.95-.49-7-3.85-7-7.93 0-.62.08-1.21.21-1.79L9 15v1c0 1.1.9 2 2 2v1.93zm6.9-2.54c-.26-.81-1-1.39-1.9-1.39h-1v-3c0-.55-.45-1-1-1H8v-2h2c.55 0 1-.45 1-1V7h2c1.1 0 2-.9 2-2v-.41c2.93 1.19 5 4.06 5 7.41 0 2.08-.8 3.97-2.1 5.39z" />
                                  </svg>
                                  PMID:{id}
                                </a>
                              ))}
                          </div>
                        )}
                      </div>
                    ) : (
                      <div className="text-sm text-[#3c4f3d]/50">
                        No literature found
                      </div>
                    )}
                  </div>
                </div>

                {/* Multi-Modal RAG Clinical Summary */}
                {variantResult.clinical_summary && (
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
                    <FormattedClinicalSummary
                      summary={variantResult.clinical_summary}
                    />

                    {/* Evidence Confidence Matrix */}
                    {variantResult.evidence_confidence && (
                      <div className="mt-4 border-t border-[#3c4f3d]/10 pt-3">
                        <div className="mb-2 text-[10px] font-medium tracking-wider text-[#3c4f3d]/50 uppercase">
                          Evidence Confidence
                        </div>
                        <div className="grid grid-cols-3 gap-2">
                          {Object.entries(variantResult.evidence_confidence)
                            .filter(([key]) => key !== "overall")
                            .map(([key, val]) => {
                              const color =
                                val.confidence === "High"
                                  ? "bg-green-50 text-green-700 border-green-200"
                                  : val.confidence === "Medium"
                                    ? "bg-yellow-50 text-yellow-700 border-yellow-200"
                                    : val.confidence === "Low"
                                      ? "bg-orange-50 text-orange-700 border-orange-200"
                                      : "bg-gray-50 text-gray-500 border-gray-200";
                              return (
                                <div
                                  key={key}
                                  className={`rounded border px-2 py-1.5 ${color}`}
                                  title={val.note}
                                >
                                  <div className="flex items-center justify-between">
                                    <span className="text-[10px] font-semibold tracking-wide uppercase">
                                      {key}
                                    </span>
                                    <span className="text-[10px] font-medium">
                                      {val.confidence}
                                    </span>
                                  </div>
                                  <div className="mt-0.5 text-[9px] leading-tight opacity-80">
                                    {val.available ? "Available" : "Missing"}
                                  </div>
                                </div>
                              );
                            })}
                        </div>
                        {/* Overall confidence badge */}
                        {variantResult.evidence_confidence.overall && (
                          <div className="mt-2 flex items-center gap-2">
                            <span className="text-[10px] text-[#3c4f3d]/60">
                              Overall:
                            </span>
                            <span
                              className={`inline-flex items-center rounded-full px-2 py-0.5 text-[10px] font-semibold ${
                                variantResult.evidence_confidence.overall
                                  .level === "High"
                                  ? "bg-green-100 text-green-800"
                                  : variantResult.evidence_confidence.overall
                                        .level === "Medium"
                                    ? "bg-yellow-100 text-yellow-800"
                                    : "bg-red-100 text-red-800"
                              }`}
                            >
                              {variantResult.evidence_confidence.overall.level}
                            </span>
                            <span className="text-[10px] text-[#3c4f3d]/50">
                              {
                                variantResult.evidence_confidence.overall
                                  .sources_available
                              }{" "}
                              sources,{" "}
                              {
                                variantResult.evidence_confidence.overall
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
                          summary is generated by an AI model (Llama 3.3 70B)
                          using publicly available databases. It is intended for
                          research and educational purposes only and must not be
                          used as a substitute for professional clinical
                          judgment, genetic counseling, or laboratory
                          validation. Always verify critical findings with a
                          board-certified clinical geneticist or molecular
                          pathologist before making patient care decisions.
                        </p>
                      </div>
                    </div>
                  </div>
                )}

                {/* Legacy AI Summary (fallback) */}
                {!variantResult.clinical_summary &&
                  variantResult.literature_context?.summary && (
                    <div className="mt-4 rounded-md bg-white p-3">
                      <div className="mb-2 text-xs font-medium text-[#3c4f3d]/70">
                        AI Clinical Summary
                      </div>
                      <FormattedClinicalSummary
                        summary={variantResult.literature_context.summary}
                      />
                    </div>
                  )}

                {/* In-Silico Mutagenesis Scan */}
                {variantResult.ism_scan && (
                  <div className="mt-4">
                    <ISMHeatmap
                      data={variantResult.ism_scan}
                      geneSymbol={gene?.symbol}
                      variantPosition={variantResult.position}
                      chromosome={chromosome}
                    />
                  </div>
                )}

                {/* Counterfactual Analysis (from ISM position 0) */}
                {variantResult.counterfactuals && (
                  <div className="mt-4">
                    <CounterfactualCard
                      data={variantResult.counterfactuals}
                      observedAlternative={variantResult.alternative}
                    />
                  </div>
                )}

                {/* ACMG/AMP Criteria Mapping */}
                {variantResult.acmg_criteria && (
                  <div className="mt-4">
                    <ACMGCriteriaTable data={variantResult.acmg_criteria} />
                  </div>
                )}

                {/* LLM-Refined ACMG Criteria */}
                {variantResult.acmg_criteria_refined && (
                  <div className="mt-4">
                    <ACMGRefinedCard
                      refined={variantResult.acmg_criteria_refined}
                      ruleBased={variantResult.acmg_criteria ?? null}
                    />
                  </div>
                )}

                {/* Knowledge Graph (Gene → Disease → Drug) */}
                {variantResult.knowledge_graph && (
                  <div className="mt-4">
                    <KnowledgeGraphCard data={variantResult.knowledge_graph} />
                  </div>
                )}

                {/* Multi-Tool Concordance */}
                {variantResult.external_scores && (
                  <div className="mt-4">
                    <ToolConcordance
                      data={variantResult.external_scores}
                      evo2Prediction={variantResult.prediction}
                      evo2Delta={variantResult.delta_score}
                      clinvarClassification={
                        variantResult.clinvar_evidence?.status ?? null
                      }
                    />
                  </div>
                )}

                {/* XAI Panel */}
                <div className="mt-4">
                  <XAIPanel
                    geneSymbol={gene?.symbol ?? "Unknown"}
                    chromosome={chromosome}
                    position={variantResult.position}
                    reference={variantResult.reference}
                    alternative={variantResult.alternative}
                    prediction={variantResult.prediction}
                    deltaScore={variantResult.delta_score}
                    classificationConfidence={
                      variantResult.classification_confidence
                    }
                    variationType={null}
                    clinvarClassification={
                      variantResult.clinvar_evidence?.status ?? null
                    }
                    populationFrequency={
                      variantResult.population_frequency ?? null
                    }
                    acmgEvidence={variantResult.acmg_evidence ?? null}
                    xaiFactors={variantResult.xai_factors?.factors ?? null}
                    vepAnnotation={null}
                  />
                </div>

                {/* Evidence Summary */}
                <div className="mt-4 rounded-md border border-[#3c4f3d]/10 bg-gradient-to-br from-[#e9eeea]/50 to-white p-4">
                  <h5 className="mb-3 flex items-center gap-2 text-sm font-medium text-[#3c4f3d]">
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      className="h-4 w-4"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M20.618 5.984A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"
                      />
                    </svg>
                    Evidence Summary
                  </h5>
                  <div className="space-y-3 text-sm leading-relaxed text-[#3c4f3d]">
                    {/* Evo2 Prediction */}
                    <div className="flex items-start gap-2">
                      <span
                        className={`mt-1 h-2 w-2 flex-shrink-0 rounded-full ${variantResult.prediction.toLowerCase().includes("pathogenic") ? "bg-red-500" : "bg-green-500"}`}
                      ></span>
                      <p className="text-[#3c4f3d]">
                        <strong>
                          Evo2 predicts this variant as{" "}
                          {variantResult.prediction}
                        </strong>{" "}
                        with{" "}
                        {Math.round(
                          variantResult.classification_confidence * 100,
                        )}
                        % confidence.
                        {variantResult.delta_score < 0
                          ? ` The negative delta score (${variantResult.delta_score.toFixed(4)}) indicates the variant reduces protein fitness, suggesting loss of normal function.`
                          : ` The positive delta score (${variantResult.delta_score.toFixed(4)}) indicates preserved or improved protein fitness, suggesting the variant is tolerated.`}
                      </p>
                    </div>

                    {/* gnomAD */}
                    <div className="flex items-start gap-2">
                      <span
                        className={`mt-1 h-2 w-2 flex-shrink-0 rounded-full ${variantResult.population_frequency?.gnomad_af ? "bg-yellow-500" : "bg-red-500"}`}
                      ></span>
                      <p className="text-[#3c4f3d]">
                        {variantResult.population_frequency?.gnomad_af ? (
                          <>
                            <strong>
                              Population frequency:{" "}
                              {(
                                variantResult.population_frequency.gnomad_af *
                                100
                              ).toFixed(4)}
                              %
                            </strong>{" "}
                            in gnomAD v4.1.
                            {variantResult.population_frequency.gnomad_af > 0.01
                              ? " This is a common variant, less likely to be pathogenic."
                              : " This is a rare variant, which may support pathogenicity if functional evidence exists."}
                          </>
                        ) : (
                          <>
                            <strong>Not observed in gnomAD</strong> (v4.1 with
                            800,000+ individuals).
                            {variantResult.delta_score < 0
                              ? " Combined with Evo2's negative delta score, this supports pathogenicity—deleterious variants are rare because natural selection removes them."
                              : " While rarity can suggest pathogenicity, Evo2's positive delta score indicates preserved protein function. Many ultra-rare variants are benign but simply haven't been observed yet due to population sampling."}
                          </>
                        )}
                      </p>
                    </div>

                    {/* ACMG */}
                    <div className="flex items-start gap-2">
                      <span
                        className={`mt-1 h-2 w-2 flex-shrink-0 rounded-full ${
                          variantResult.acmg_evidence?.code?.includes("PP3")
                            ? "bg-red-400"
                            : variantResult.acmg_evidence?.code?.includes("BP4")
                              ? "bg-green-400"
                              : "bg-blue-500"
                        }`}
                      ></span>
                      <p className="text-[#3c4f3d]">
                        {variantResult.acmg_evidence?.code &&
                        variantResult.acmg_evidence.code !== "None" ? (
                          <>
                            <strong>
                              ACMG {variantResult.acmg_evidence.code}
                            </strong>
                            {variantResult.acmg_evidence.strength &&
                              ` (${variantResult.acmg_evidence.strength})`}
                            :{" "}
                            {variantResult.acmg_evidence.code.includes("PP3")
                              ? "Computational evidence supports a deleterious effect on the gene or gene product."
                              : variantResult.acmg_evidence.code.includes("BP4")
                                ? "Computational evidence suggests no impact on the gene or gene product."
                                : variantResult.acmg_evidence.description}
                          </>
                        ) : (
                          <>
                            <strong>Computational evidence inconclusive</strong>
                            : The prediction confidence is in the uncertain
                            range. Additional clinical or functional evidence is
                            recommended.
                          </>
                        )}
                      </p>
                    </div>

                    {/* Classification Note */}
                    <div className="mt-3 rounded-md border border-yellow-200 bg-yellow-50 p-3">
                      <div className="flex items-start gap-2">
                        <span className="text-lg leading-none text-yellow-600">
                          ⚠
                        </span>
                        <p className="text-sm text-yellow-800">
                          <strong>Classification Note:</strong> This is a
                          standalone analysis without ClinVar reference data.
                          {variantResult.prediction
                            .toLowerCase()
                            .includes("pathogenic")
                            ? " Evo2 predicts this variant as potentially pathogenic. Consider clinical validation and review of functional studies before clinical decision-making."
                            : variantResult.prediction
                                  .toLowerCase()
                                  .includes("benign")
                              ? " While Evo2 predicts this variant as likely benign, clinical interpretation should consider patient phenotype and family history."
                              : " Evo2 cannot confidently classify this variant. The delta score falls in the uncertain range — additional clinical or functional evidence is recommended."}
                        </p>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Export Report Buttons */}
                <div className="mt-4 flex justify-start gap-2">
                  <Button
                    variant="outline"
                    className="flex items-center gap-2 border-[#3c4f3d]/20 text-[#3c4f3d] hover:bg-[#e9eeea]"
                    onClick={() => {
                      const date = new Date().toLocaleDateString("en-US", {
                        month: "2-digit",
                        day: "2-digit",
                        year: "numeric",
                      });
                      const position =
                        variantResult.position?.toLocaleString() || "";
                      const gnomadValue = variantResult.population_frequency
                        ?.gnomad_af
                        ? `${(variantResult.population_frequency.gnomad_af * 100).toFixed(4)}%`
                        : "Not observed";
                      const pubmedIds =
                        variantResult.literature_context?.pubmed_ids?.slice(
                          0,
                          5,
                        ) || [];

                      const report = `VARIANT ANALYSIS REPORT
Generated: ${date}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

VARIANT INFORMATION
• Gene: ${gene?.symbol || "Unknown"}
• Position: ${chromosome}:${position}
• Change: ${variantResult.reference || "N"}>${variantResult.alternative}
• Genome: ${genomeId}

EVO2 PREDICTION
• Prediction: ${variantResult.prediction}
• Delta Score: ${variantResult.delta_score.toFixed(6)}
• Confidence: ${Math.round(variantResult.classification_confidence * 100)}%

POPULATION FREQUENCY
• gnomAD v4.1: ${gnomadValue}

ACMG EVIDENCE
• Code: ${variantResult.acmg_evidence?.code || "None"}
• ${variantResult.acmg_evidence?.description || "Computational evidence is inconclusive"}

${
  variantResult.clinical_summary
    ? `MULTI-MODAL RAG CLINICAL SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
${variantResult.clinical_summary}

EVIDENCE CONFIDENCE
• Overall: ${variantResult.evidence_confidence?.overall?.level || "N/A"} (${variantResult.evidence_confidence?.overall?.sources_available || "N/A"} sources)
• VEP: ${variantResult.evidence_confidence?.vep?.confidence || "N/A"}
• Evo2: ${variantResult.evidence_confidence?.evo2?.confidence || "N/A"}
• gnomAD: ${variantResult.evidence_confidence?.gnomad?.confidence || "N/A"}
• ClinVar: ${variantResult.evidence_confidence?.clinvar?.confidence || "N/A"}
• UniProt: ${variantResult.evidence_confidence?.uniprot?.confidence || "N/A"}
• PubMed: ${variantResult.evidence_confidence?.pubmed?.confidence || "N/A"}
`
    : `AI CLINICAL SUMMARY
${variantResult.literature_context?.summary || "No clinical summary available."}
`
}

REFERENCES
${pubmedIds.length > 0 ? pubmedIds.map((id: string) => `• PMID:${id}`).join("\n") : "• No references available"}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DISCLAIMER: This report is generated by computational models using publicly available databases. It is intended for research and educational purposes only and must not be used as a substitute for professional clinical judgment, genetic counseling, or laboratory validation. Always verify critical findings with a board-certified clinical geneticist or molecular pathologist before making patient care decisions.

Cross-reference: https://varsome.com/position/${genomeId}/${chromosome}-${variantResult.position}
`;

                      const blob = new Blob([report], { type: "text/plain" });
                      const url = URL.createObjectURL(blob);
                      const a = document.createElement("a");
                      a.href = url;
                      a.download = `variant-report-${gene?.symbol || "unknown"}-${variantResult.position}.txt`;
                      a.click();
                      URL.revokeObjectURL(url);
                    }}
                  >
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      className="h-4 w-4"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                      />
                    </svg>
                    Download Report
                  </Button>
                  <Button
                    variant="outline"
                    className="flex items-center gap-2 border-[#3c4f3d]/20 text-[#3c4f3d] hover:bg-[#e9eeea]"
                    onClick={async () => {
                      const date = new Date().toLocaleDateString("en-US", {
                        month: "2-digit",
                        day: "2-digit",
                        year: "numeric",
                      });
                      const position =
                        variantResult.position?.toLocaleString() || "";
                      const gnomadValue = variantResult.population_frequency
                        ?.gnomad_af
                        ? `${(variantResult.population_frequency.gnomad_af * 100).toFixed(4)}%`
                        : "Not observed";
                      const pubmedIds =
                        variantResult.literature_context?.pubmed_ids?.slice(
                          0,
                          5,
                        ) || [];

                      const report = `VARIANT ANALYSIS REPORT
Generated: ${date}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

VARIANT INFORMATION
• Gene: ${gene?.symbol || "Unknown"}
• Position: ${chromosome}:${position}
• Change: ${variantResult.reference || "N"}>${variantResult.alternative}
• Genome: ${genomeId}

EVO2 PREDICTION
• Prediction: ${variantResult.prediction}
• Delta Score: ${variantResult.delta_score.toFixed(6)}
• Confidence: ${Math.round(variantResult.classification_confidence * 100)}%

POPULATION FREQUENCY
• gnomAD v4.1: ${gnomadValue}

ACMG EVIDENCE
• Code: ${variantResult.acmg_evidence?.code || "None"}
• ${variantResult.acmg_evidence?.description || "Computational evidence is inconclusive"}

${
  variantResult.clinical_summary
    ? `MULTI-MODAL RAG CLINICAL SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
${variantResult.clinical_summary}

EVIDENCE CONFIDENCE
• Overall: ${variantResult.evidence_confidence?.overall?.level || "N/A"} (${variantResult.evidence_confidence?.overall?.sources_available || "N/A"} sources)
• VEP: ${variantResult.evidence_confidence?.vep?.confidence || "N/A"}
• Evo2: ${variantResult.evidence_confidence?.evo2?.confidence || "N/A"}
• gnomAD: ${variantResult.evidence_confidence?.gnomad?.confidence || "N/A"}
• ClinVar: ${variantResult.evidence_confidence?.clinvar?.confidence || "N/A"}
• UniProt: ${variantResult.evidence_confidence?.uniprot?.confidence || "N/A"}
• PubMed: ${variantResult.evidence_confidence?.pubmed?.confidence || "N/A"}
`
    : `AI CLINICAL SUMMARY
${variantResult.literature_context?.summary || "No clinical summary available."}
`
}

REFERENCES
${pubmedIds.length > 0 ? pubmedIds.map((id: string) => `• PMID:${id}`).join("\n") : "• No references available"}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DISCLAIMER: This report is generated by computational models using publicly available databases. It is intended for research and educational purposes only and must not be used as a substitute for professional clinical judgment, genetic counseling, or laboratory validation. Always verify critical findings with a board-certified clinical geneticist or molecular pathologist before making patient care decisions.

Cross-reference: https://varsome.com/position/${genomeId}/${chromosome}-${variantResult.position}
`;

                      try {
                        await navigator.clipboard.writeText(report);
                        alert("Report copied to clipboard!");
                      } catch {
                        alert("Failed to copy to clipboard");
                      }
                    }}
                  >
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      className="h-4 w-4"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"
                      />
                    </svg>
                    Copy Report
                  </Button>
                </div>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    );
  },
);

export default VariantAnalysis;
