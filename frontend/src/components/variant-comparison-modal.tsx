import { useEffect } from "react";
import type { ClinvarVariant } from "~/utils/genome-api";
import { Button } from "./ui/button";
import { Check, ExternalLink, Shield, X } from "lucide-react";
import {
  getClassificationColorClasses,
  getNucleotideColorClass,
} from "~/utils/coloring-utils";
import { FormattedClinicalSummary } from "./formatted-clinical-summary";

export function VariantComparisonModal({
  comparisonVariant,
  onClose,
}: {
  comparisonVariant: ClinvarVariant | null;
  onClose: () => void;
}) {
  // Lock body scroll when modal is open
  useEffect(() => {
    if (comparisonVariant && comparisonVariant.evo2Result) {
      document.body.style.overflow = 'hidden';
      return () => {
        document.body.style.overflow = '';
      };
    }
  }, [comparisonVariant]);

  if (!comparisonVariant || !comparisonVariant.evo2Result) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4 overflow-y-auto h-screen min-h-screen">
      <div className="max-h-[90vh] w-full max-w-3xl overflow-hidden rounded-2xl bg-white shadow-xl flex flex-col">
        {/* Modal header */}
        <div className="border-b border-[#3c4f3d]/10 p-5 flex-shrink-0">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-medium text-[#3c4f3d]">
              Variant Analysis Comparison
            </h3>
            <Button
              variant="ghost"
              size="sm"
              onClick={onClose}
              className="h-7 w-7 cursor-pointer p-0 text-[#3c4f3d]/70 hover:bg-[#9eeea]/70 hover:text-[#3c4f3d]"
            >
              <X className="h-5 w-5" />
            </Button>
          </div>
        </div>

        {/* Modal content */}
        <div className="flex-1 overflow-y-auto p-5">
          {comparisonVariant && comparisonVariant.evo2Result && (
            <div className="space-y-6">
              <div className="rounded-md border border-[#3c4f3d]/10 bg-[#e9eeea]/30 p-4">
                <h4 className="mb-3 text-sm font-medium text-[#3c4f3d]">
                  Variant Information
                </h4>
                <div className="grid gap-4 md:grid-cols-2">
                  <div>
                    <div className="space-y-2">
                      <div className="flex">
                        <span className="w-28 text-xs text-[#3c4f3d]/70">
                          Position:
                        </span>
                        <span className="text-xs">
                          {comparisonVariant.location}
                        </span>
                      </div>
                      <div className="flex">
                        <span className="w-28 text-xs text-[#3c4f3d]/70">
                          Type:
                        </span>
                        <span className="text-xs">
                          {comparisonVariant.variation_type}
                        </span>
                      </div>
                    </div>
                  </div>

                  <div>
                    <div className="space-y-2">
                      <div className="flex">
                        <span className="w-28 text-xs text-[#3c4f3d]/70">
                          Variant:
                        </span>
                        <span className="font-mono text-xs">
                          {(() => {
                            const match =
                              comparisonVariant.title.match(/(\w)>(\w)/);
                            if (match && match.length === 3) {
                              const [_, ref, alt] = match;
                              return (
                                <>
                                  <span
                                    className={getNucleotideColorClass(ref!)}
                                  >
                                    {ref}
                                  </span>
                                  <span className="text-orange-500">{">"}</span>
                                  <span
                                    className={getNucleotideColorClass(alt!)}
                                  >
                                    {alt}
                                  </span>
                                </>
                              );
                            }
                            return comparisonVariant.title;
                          })()}
                        </span>
                      </div>
                      <div className="flex items-center">
                        <span className="w-28 text-xs text-[#3c4f3d]/70">
                          ClinVar ID:
                        </span>
                        <a
                          href={`https://www.ncbi.nlm.nih.gov/clinvar/variation/${comparisonVariant.clinvar_id}`}
                          className="text-xs text-[#de8246] hover:underline"
                          target="_blank"
                        >
                          {comparisonVariant.clinvar_id}
                        </a>
                        <ExternalLink className="ml-1 inline-block h-3 w-3 text-[#de8246]" />
                      </div>
                    </div>
                  </div>
                </div>

                {/* Cross-reference Links */}
                <div className="mt-4 flex flex-wrap gap-3">
                  <a
                    href={`https://www.ncbi.nlm.nih.gov/clinvar/variation/${comparisonVariant.clinvar_id}/`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-1.5 rounded-md bg-[#de8246]/10 px-3 py-1.5 text-xs font-medium text-[#de8246] hover:bg-[#de8246]/20 transition-colors"
                  >
                    <ExternalLink className="h-3 w-3" />
                    View in ClinVar
                  </a>
                  {(() => {
                    // Build VarSome URL from variant details
                    // VarSome expects reference strand notation, so we may need to complement
                    const match = comparisonVariant.title.match(/(\w)>(\w)/);
                    const ref = match?.[1] || '';
                    const alt = match?.[2] || '';
                    const position = comparisonVariant.location?.replaceAll(',', '') || '';
                    const chrom = comparisonVariant.chromosome?.replace('chr', '') || '';

                    // Complement mapping for strand conversion
                    const complement: Record<string, string> = { 'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G' };

                    // Try both orientations - VarSome will redirect to the correct one
                    // Use position-based URL which shows all variants at this location
                    const varsomePositionUrl = `https://varsome.com/position/hg38/chr${chrom}-${position}`;

                    return position && chrom ? (
                      <a
                        href={varsomePositionUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-flex items-center gap-1.5 rounded-md bg-teal-500/10 px-3 py-1.5 text-xs font-medium text-teal-700 hover:bg-teal-500/20 transition-colors"
                      >
                        <ExternalLink className="h-3 w-3" />
                        Cross-check on VarSome
                      </a>
                    ) : null;
                  })()}
                </div>
              </div>

              {/* Experimental warning for indels */}
              {!comparisonVariant.variation_type.toLowerCase().includes("single nucleotide") && (
                <div className="rounded-md border border-blue-200 bg-blue-50 p-3 text-sm text-blue-800">
                  <strong>ℹ️ Experimental Analysis:</strong> This is a structural variant ({comparisonVariant.variation_type}).
                  Evo2 scores may be influenced by sequence length changes.
                  Interpret the "Delta Score" with caution compared to Single Nucleotide Variants.
                </div>
              )}

              {/* Variant results */}
              <div>
                <h4 className="mb-3 text-sm font-medium text-[#3c4f3d]">
                  Analysis Comparison
                </h4>
                <div className="rounded-md border border-[#3c4f3d]/10 bg-white p-4">
                  <div className="grid gap-4 md:grid-cols-2">
                    {/* ClinVar Assesment */}
                    <div className="rounded-md bg-[#e9eeea]/50 p-4">
                      <h5 className="mb-2 flex items-center gap-2 text-xs font-medium text-[#3c4f3d]">
                        <span className="flex h-5 w-5 items-center justify-center rounded-full bg-[#3c4f3d]/10">
                          <span className="h-3 w-3 rounded-full bg-[#3c4f3d]"></span>
                        </span>
                        ClinVar Assessment
                      </h5>
                      <div className="mt-2">
                        <div
                          className={`w-fit rounded-md px-2 py-1 text-xs font-normal ${getClassificationColorClasses(comparisonVariant.classification)}`}
                        >
                          {comparisonVariant.classification ||
                            "Unknown significance"}
                        </div>
                      </div>
                    </div>

                    {/* Evo2 Prediction */}
                    <div className="rounded-md bg-[#e9eeea]/50 p-4">
                      <h5 className="mb-2 flex items-center gap-2 text-xs font-medium text-[#3c4f3d]">
                        <span className="flex h-5 w-5 items-center justify-center rounded-full bg-[#3c4f3d]/10">
                          <span className="h-3 w-3 rounded-full bg-[#de8246]"></span>
                        </span>
                        Evo2 Prediction
                      </h5>
                      <div className="mt-2">
                        <div
                          className={`flex w-fit items-center gap-1 rounded-md px-2 py-1 text-xs font-normal ${getClassificationColorClasses(comparisonVariant.evo2Result.prediction)}`}
                        >
                          {comparisonVariant.evo2Result.prediction.toLowerCase().includes("pathogenic") ? (
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                            </svg>
                          ) : (
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                            </svg>
                          )}
                          {comparisonVariant.evo2Result.prediction}
                        </div>
                      </div>
                      {/* Delta score with visual gradient bar */}
                      <div className="mt-3">
                        <div className="mb-1 text-xs text-[#3c4f3d]/70">
                          Delta Likelihood Score:
                        </div>
                        <div className={`text-sm font-semibold font-mono ${comparisonVariant.evo2Result.prediction.toLowerCase().includes('pathogenic') ? 'text-red-800' : comparisonVariant.evo2Result.prediction.toLowerCase().includes('benign') ? 'text-green-800' : 'text-[#3c4f3d]'}`}>
                          {comparisonVariant.evo2Result.delta_score.toFixed(6)}
                        </div>

                        {/* Visual scale bar */}
                        <div className="mt-2">
                          {/* Scale labels */}
                          <div className="flex justify-between text-[10px] text-[#3c4f3d]/60 mb-1">
                            <span>Pathogenic</span>
                            <span>Neutral</span>
                            <span>Benign</span>
                          </div>

                          {/* Gradient bar with marker */}
                          <div className="relative h-3 rounded-full bg-gradient-to-r from-red-500 via-yellow-400 to-green-500">
                            {/* Center indicator for neutral (0) */}
                            <div className="absolute top-0 left-1/2 h-3 w-0.5 bg-white/70"></div>

                            {/* Score marker - maps score (-1 to +1) to position (0% to 100%) */}
                            <div
                              className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 h-4 w-4 rounded-full border-2 border-white bg-[#3c4f3d] shadow-md"
                              style={{
                                left: `${Math.max(5, Math.min(95,
                                  // Clamp score to -0.01 to +0.01 range and map to 0-100%
                                  // Use much tighter range for better sensitivity (scores are typically very small)
                                  ((Math.max(-0.01, Math.min(0.01, comparisonVariant.evo2Result.delta_score)) + 0.01) / 0.02) * 100
                                ))}%`
                              }}
                              title={`Score: ${comparisonVariant.evo2Result.delta_score.toFixed(6)}`}
                            ></div>
                          </div>

                          {/* Scale numbers */}
                          <div className="flex justify-between text-[9px] text-[#3c4f3d]/40 mt-0.5">
                            <span>-1</span>
                            <span>0</span>
                            <span>+1</span>
                          </div>
                        </div>

                        <div className="text-xs text-[#3c4f3d]/60 mt-2">
                          {comparisonVariant.evo2Result.delta_score < -0.001
                            ? "⚠️ Negative score indicates loss of function"
                            : comparisonVariant.evo2Result.delta_score > 0.001
                              ? "✓ Positive score indicates neutral/benign function"
                              : "○ Score near zero suggests minimal impact"}
                        </div>
                      </div>
                      {/* Confidence bar */}
                      <div className="mt-3">
                        <div className="mb-1 text-xs text-[#3c4f3d]/70">
                          Confidence:
                        </div>
                        <div className="mt-1 h-2 w-full rounded-full bg-[#e9eeea]/80">
                          <div
                            className={`h-2 rounded-full ${comparisonVariant.evo2Result.prediction.includes("pathogenic") ? "bg-red-600" : "bg-green-600"}`}
                            style={{
                              width: `${Math.min(100, comparisonVariant.evo2Result.classification_confidence * 100)}%`,
                            }}
                          ></div>
                        </div>
                        <div className="mt-1 text-right text-xs text-[#3c4f3d]/60">
                          {Math.round(
                            comparisonVariant.evo2Result
                              .classification_confidence * 100,
                          )}
                          %
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Assesment Agreement */}
                  <div className="mt-4 rounded-md bg-[#e9eeea]/20 p-3 text-xs leading-relaxed">
                    <div className="flex items-center gap-2">
                      <span
                        className={`flex h-5 w-5 items-center justify-center rounded-full ${comparisonVariant.classification.toLowerCase() === comparisonVariant.evo2Result.prediction.toLowerCase() ? "bg-green-100" : "bg-yellow-100"}`}
                      >
                        {comparisonVariant.classification.toLowerCase() ===
                          comparisonVariant.evo2Result.prediction.toLowerCase() ? (
                          <Check className="h-3 w-3 text-green-600" />
                        ) : (
                          <span className="flex h-3 w-3 items-center justify-center text-yellow-600">
                            <p>!</p>
                          </span>
                        )}
                      </span>
                      <span className="font-medium text-[#3c4f3d]">
                        {comparisonVariant.classification.toLowerCase() ===
                          comparisonVariant.evo2Result.prediction.toLowerCase()
                          ? "Evo2 prediction agrees with ClinVar classification"
                          : "Evo2 prediction differs from ClinVar classification"}
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Clinical Enrichment Section */}
              {(comparisonVariant.evo2Result.population_frequency ||
                comparisonVariant.evo2Result.acmg_evidence ||
                comparisonVariant.evo2Result.literature_context) && (
                  <div className="rounded-md border border-[#3c4f3d]/10 bg-[#e9eeea]/30 p-4">
                    <h4 className="mb-3 text-sm font-medium text-[#3c4f3d]">
                      Clinical Annotations
                    </h4>

                    <div className="grid gap-4 md:grid-cols-3">
                      {/* gnomAD Population Frequency */}
                      <div className="rounded-md bg-white p-3">
                        <div className="text-xs font-medium text-[#3c4f3d]/70 mb-1">
                          Population Frequency
                        </div>
                        {comparisonVariant.evo2Result.population_frequency?.gnomad_af ? (
                          <div className="text-sm">
                            <span className="font-medium">
                              {(comparisonVariant.evo2Result.population_frequency.gnomad_af * 100).toFixed(4)}%
                            </span>
                            <div className="text-xs text-[#3c4f3d]/60 mt-1">
                              {comparisonVariant.evo2Result.population_frequency.is_common_variant
                                ? "Common variant"
                                : "Rare variant"}
                            </div>
                          </div>
                        ) : (
                          <div className="text-sm text-[#3c4f3d]/50">
                            Not found in gnomAD
                          </div>
                        )}
                      </div>

                      {/* ACMG Evidence */}
                      <div className="rounded-md bg-white p-3">
                        <div className="text-xs font-medium text-[#3c4f3d]/70 mb-1">
                          ACMG Evidence
                        </div>
                        {comparisonVariant.evo2Result.acmg_evidence?.code &&
                          comparisonVariant.evo2Result.acmg_evidence.code !== "None" ? (
                          <div>
                            <span className="inline-block rounded bg-blue-100 px-2 py-0.5 text-xs font-medium text-blue-800">
                              {comparisonVariant.evo2Result.acmg_evidence.code}
                              {comparisonVariant.evo2Result.acmg_evidence.strength &&
                                ` (${comparisonVariant.evo2Result.acmg_evidence.strength})`}
                            </span>
                            <div className="text-xs text-[#3c4f3d]/60 mt-1">
                              {comparisonVariant.evo2Result.acmg_evidence.description}
                            </div>
                          </div>
                        ) : (
                          <div className="text-sm text-[#3c4f3d]/50">
                            {comparisonVariant.evo2Result.acmg_evidence?.description || "No evidence code"}
                          </div>
                        )}
                      </div>

                      {/* Literature Context */}
                      <div className="rounded-md bg-white p-3">
                        <div className="text-xs font-medium text-[#3c4f3d]/70 mb-1">
                          Literature
                        </div>
                        {comparisonVariant.evo2Result.literature_context?.articles_found ? (
                          <div>
                            <div className="text-xs text-[#3c4f3d]/60">
                              {Math.min(5, comparisonVariant.evo2Result.literature_context.pubmed_ids?.length || 0)} articles
                            </div>
                            {comparisonVariant.evo2Result.literature_context.pubmed_ids?.length > 0 && (
                              <div className="mt-1 flex flex-wrap gap-1">
                                {comparisonVariant.evo2Result.literature_context.pubmed_ids.slice(0, 5).map((id: string) => (
                                  <a
                                    key={id}
                                    href={`https://pubmed.ncbi.nlm.nih.gov/${id}/`}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="inline-flex items-center gap-1 px-2 py-1 text-xs font-medium text-slate-600 bg-slate-100 rounded-full hover:bg-slate-200 hover:text-slate-800 transition-colors border border-slate-200"
                                  >
                                    <svg className="h-3 w-3" fill="currentColor" viewBox="0 0 24 24">
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

                    {/* AI Summary */}
                    {comparisonVariant.evo2Result.literature_context?.summary && (
                      <div className="mt-4 rounded-md bg-white p-3">
                        <div className="text-xs font-medium text-[#3c4f3d]/70 mb-2">
                          AI Clinical Summary
                        </div>
                        <FormattedClinicalSummary summary={comparisonVariant.evo2Result.literature_context.summary} />
                      </div>
                    )}
                  </div>
                )}

              {/* Evidence Summary - Clinical Interpretation */}
              <div className="rounded-md border border-[#3c4f3d]/10 bg-gradient-to-br from-[#e9eeea]/50 to-white p-4">
                <h4 className="mb-3 text-sm font-medium text-[#3c4f3d] flex items-center gap-2">
                  <Shield className="h-4 w-4" />
                  Evidence Summary
                </h4>
                <div className="space-y-3 text-sm text-[#3c4f3d] leading-relaxed">
                  {/* Prediction Explanation */}
                  <div className="flex items-start gap-2">
                    <span className={`mt-1 h-2 w-2 rounded-full flex-shrink-0 ${comparisonVariant.evo2Result.prediction.toLowerCase().includes("pathogenic")
                      ? "bg-red-500"
                      : "bg-green-500"
                      }`}></span>
                    <p>
                      <strong>Evo2 predicts this variant as {comparisonVariant.evo2Result.prediction}</strong>
                      {" "}with {Math.round(comparisonVariant.evo2Result.classification_confidence * 100)}% confidence
                      {comparisonVariant.evo2Result.delta_score < 0
                        ? `. The negative delta score (${comparisonVariant.evo2Result.delta_score.toFixed(4)}) indicates the variant reduces protein fitness, suggesting loss of normal function.`
                        : `. The positive delta score (${comparisonVariant.evo2Result.delta_score.toFixed(4)}) indicates preserved or improved protein fitness, suggesting the variant is tolerated.`
                      }
                    </p>
                  </div>

                  {/* Population Frequency Evidence */}
                  <div className="flex items-start gap-2">
                    <span className={`mt-1 h-2 w-2 rounded-full flex-shrink-0 ${comparisonVariant.evo2Result.population_frequency?.gnomad_af
                      ? (comparisonVariant.evo2Result.population_frequency.is_common_variant ? "bg-green-500" : "bg-yellow-500")
                      : comparisonVariant.evo2Result.prediction.toLowerCase().includes("benign") ? "bg-yellow-500" : "bg-red-500"
                      }`}></span>
                    <p>
                      {!comparisonVariant.evo2Result.population_frequency?.gnomad_af ? (
                        comparisonVariant.evo2Result.prediction.toLowerCase().includes("benign") ? (
                          <>
                            <strong>Not observed in gnomAD</strong> (v4.1 with 800,000+ individuals).
                            While rarity can suggest pathogenicity, <strong>Evo2's positive delta score indicates preserved protein function</strong>.
                            Many ultra-rare variants are benign but simply haven't been observed yet due to population sampling.
                          </>
                        ) : (
                          <>
                            <strong>Not observed in gnomAD</strong> (v4.1 with 800,000+ individuals).
                            Combined with Evo2's negative delta score, this supports pathogenicity—deleterious variants are rare because natural selection removes them.
                          </>
                        )
                      ) : comparisonVariant.evo2Result.population_frequency.is_common_variant ? (
                        <>
                          <strong>Common variant</strong> (AF: {(comparisonVariant.evo2Result.population_frequency.gnomad_af * 100).toFixed(4)}%).
                          Variants present at high frequency in healthy populations are typically benign (ACMG BA1 criterion).
                        </>
                      ) : (
                        <>
                          <strong>Rare variant</strong> (AF: {(comparisonVariant.evo2Result.population_frequency.gnomad_af * 100).toFixed(4)}%).
                          Low population frequency alone is not definitive—functional evidence from Evo2 provides additional context.
                        </>
                      )}
                    </p>
                  </div>

                  {/* ACMG Evidence */}
                  {comparisonVariant.evo2Result.acmg_evidence && (
                    <div className="flex items-start gap-2">
                      <span className={`mt-1 h-2 w-2 rounded-full flex-shrink-0 ${comparisonVariant.evo2Result.acmg_evidence.code?.includes("PP3") ? "bg-red-400" :
                        comparisonVariant.evo2Result.acmg_evidence.code?.includes("BP4") ? "bg-green-400" :
                          "bg-gray-400"
                        }`}></span>
                      <p>
                        {comparisonVariant.evo2Result.acmg_evidence.code && comparisonVariant.evo2Result.acmg_evidence.code !== "None" ? (
                          <>
                            <strong>ACMG {comparisonVariant.evo2Result.acmg_evidence.code}</strong>
                            {comparisonVariant.evo2Result.acmg_evidence.strength && ` (${comparisonVariant.evo2Result.acmg_evidence.strength})`}: {" "}
                            {comparisonVariant.evo2Result.acmg_evidence.code.includes("PP3")
                              ? "Computational evidence supports a deleterious effect on the gene or gene product."
                              : comparisonVariant.evo2Result.acmg_evidence.code.includes("BP4")
                                ? "Computational evidence suggests no impact on the gene or gene product."
                                : comparisonVariant.evo2Result.acmg_evidence.description
                            }
                          </>
                        ) : (
                          <>
                            <strong>Computational evidence inconclusive</strong>: The prediction confidence is in the uncertain range.
                            Additional clinical or functional evidence is recommended.
                          </>
                        )}
                      </p>
                    </div>
                  )}

                  {/* Clinical Discordance Warning */}
                  {comparisonVariant.classification.toLowerCase() !== comparisonVariant.evo2Result.prediction.toLowerCase() && (
                    <div className="mt-3 p-3 bg-yellow-50 border border-yellow-200 rounded-md">
                      <div className="flex items-start gap-2">
                        <span className="mt-0.5 text-yellow-600 font-bold">⚠</span>
                        <p className="text-yellow-800">
                          <strong>Classification Discordance:</strong> ClinVar reports "{comparisonVariant.classification}"
                          while Evo2 predicts "{comparisonVariant.evo2Result.prediction}".
                          {comparisonVariant.classification.toLowerCase().includes("uncertain")
                            ? " This VUS may warrant reclassification based on computational evidence. Consider functional studies or family segregation analysis."
                            : " This disagreement suggests the variant may need clinical review with additional evidence."
                          }
                        </p>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Modal footer */}
        <div className="flex justify-between border-t border-[#3c4f3d]/10 bg-[#e9eeea]/30 p-4 flex-shrink-0">
          <div className="flex gap-2">
            <Button
              variant="outline"
              onClick={() => {
                // Generate report for download
                const match = comparisonVariant.title.match(/(\w)>(\w)/);
                let changeNotation = '';
                if (match) {
                  changeNotation = `${match[1]}>${match[2]}`;
                } else if (comparisonVariant.variation_type?.toLowerCase().includes('dup')) {
                  changeNotation = `Duplication at ${comparisonVariant.location}`;
                } else if (comparisonVariant.variation_type?.toLowerCase().includes('del')) {
                  changeNotation = `Deletion at ${comparisonVariant.location}`;
                } else if (comparisonVariant.variation_type?.toLowerCase().includes('ins')) {
                  changeNotation = `Insertion at ${comparisonVariant.location}`;
                } else {
                  changeNotation = comparisonVariant.variation_type || 'Unknown';
                }

                const report = `VARIANT ANALYSIS REPORT
Generated: ${new Date().toLocaleDateString('en-US', { month: '2-digit', day: '2-digit', year: 'numeric' })}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

VARIANT INFORMATION
• Position: ${comparisonVariant.chromosome}:${comparisonVariant.location}
• Change: ${changeNotation}
• Type: ${comparisonVariant.variation_type}
• ClinVar ID: ${comparisonVariant.clinvar_id}

CLASSIFICATION COMPARISON
• ClinVar: ${comparisonVariant.classification || 'Unknown'}
• Evo2 Prediction: ${comparisonVariant.evo2Result?.prediction || 'N/A'}
• Delta Score: ${comparisonVariant.evo2Result?.delta_score?.toFixed(6) || 'N/A'}
• Confidence: ${comparisonVariant.evo2Result?.classification_confidence ? Math.round(comparisonVariant.evo2Result.classification_confidence * 100) : 'N/A'}%

POPULATION FREQUENCY
• gnomAD: ${comparisonVariant.evo2Result?.population_frequency?.gnomad_af
                    ? `${(comparisonVariant.evo2Result.population_frequency.gnomad_af * 100).toFixed(4)}%`
                    : 'Not observed'}

ACMG EVIDENCE
• Code: ${comparisonVariant.evo2Result?.acmg_evidence?.code || 'None'}
• ${comparisonVariant.evo2Result?.acmg_evidence?.description || 'Computational evidence is inconclusive'}

AI CLINICAL SUMMARY
${comparisonVariant.evo2Result?.literature_context?.summary || 'No summary available'}

REFERENCES
${comparisonVariant.evo2Result?.literature_context?.pubmed_ids?.slice(0, 5).map((id: string) => `• PMID:${id}`).join('\n') || '• No references available'}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Cross-reference: https://varsome.com/position/hg38/${comparisonVariant.chromosome}-${comparisonVariant.location?.replaceAll(',', '')}
`;

                const blob = new Blob([report], { type: 'text/plain' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `variant-report-${comparisonVariant.clinvar_id}.txt`;
                a.click();
                URL.revokeObjectURL(url);
              }}
              className="cursor-pointer border-[#3c4f3d]/20 bg-white text-[#3c4f3d] hover:bg-[#e9eeea]/70 flex items-center gap-2"
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              Download Report
            </Button>
            <Button
              variant="outline"
              onClick={async () => {
                // Generate report for clipboard
                const match = comparisonVariant.title.match(/(\w)>(\w)/);
                let changeNotation = '';
                if (match) {
                  changeNotation = `${match[1]}>${match[2]}`;
                } else if (comparisonVariant.variation_type?.toLowerCase().includes('dup')) {
                  changeNotation = `Duplication at ${comparisonVariant.location}`;
                } else if (comparisonVariant.variation_type?.toLowerCase().includes('del')) {
                  changeNotation = `Deletion at ${comparisonVariant.location}`;
                } else if (comparisonVariant.variation_type?.toLowerCase().includes('ins')) {
                  changeNotation = `Insertion at ${comparisonVariant.location}`;
                } else {
                  changeNotation = comparisonVariant.variation_type || 'Unknown';
                }

                const report = `VARIANT ANALYSIS REPORT
Generated: ${new Date().toLocaleDateString('en-US', { month: '2-digit', day: '2-digit', year: 'numeric' })}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

VARIANT INFORMATION
• Position: ${comparisonVariant.chromosome}:${comparisonVariant.location}
• Change: ${changeNotation}
• Type: ${comparisonVariant.variation_type}
• ClinVar ID: ${comparisonVariant.clinvar_id}

CLASSIFICATION COMPARISON
• ClinVar: ${comparisonVariant.classification || 'Unknown'}
• Evo2 Prediction: ${comparisonVariant.evo2Result?.prediction || 'N/A'}
• Delta Score: ${comparisonVariant.evo2Result?.delta_score?.toFixed(6) || 'N/A'}
• Confidence: ${comparisonVariant.evo2Result?.classification_confidence ? Math.round(comparisonVariant.evo2Result.classification_confidence * 100) : 'N/A'}%

POPULATION FREQUENCY
• gnomAD: ${comparisonVariant.evo2Result?.population_frequency?.gnomad_af
                    ? `${(comparisonVariant.evo2Result.population_frequency.gnomad_af * 100).toFixed(4)}%`
                    : 'Not observed'}

ACMG EVIDENCE
• Code: ${comparisonVariant.evo2Result?.acmg_evidence?.code || 'None'}
• ${comparisonVariant.evo2Result?.acmg_evidence?.description || 'Computational evidence is inconclusive'}

AI CLINICAL SUMMARY
${comparisonVariant.evo2Result?.literature_context?.summary || 'No summary available'}

REFERENCES
${comparisonVariant.evo2Result?.literature_context?.pubmed_ids?.slice(0, 5).map((id: string) => `• PMID:${id}`).join('\n') || '• No references available'}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Cross-reference: https://varsome.com/position/hg38/${comparisonVariant.chromosome}-${comparisonVariant.location?.replaceAll(',', '')}
`;

                try {
                  await navigator.clipboard.writeText(report);
                  alert('Report copied to clipboard!');
                } catch {
                  alert('Failed to copy to clipboard');
                }
              }}
              className="cursor-pointer border-[#3c4f3d]/20 bg-white text-[#3c4f3d] hover:bg-[#e9eeea]/70 flex items-center gap-2"
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
              </svg>
              Copy Report
            </Button>
          </div>
          <Button
            variant="outline"
            onClick={onClose}
            className="cursor-pointer border-[#3c4f3d]/10 bg-white text-[#3c4f3d] hover:bg-[#e9eeea]/70"
          >
            Close
          </Button>
        </div>
      </div>
    </div>
  );
}
