"use client";

import { useEffect, useState } from "react";
import {
  analyzeVariantWithAPI,
  fetchSingleBase,
  type ClinvarVariant,
  type GeneFromSearch,
} from "~/utils/genome-api";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Button } from "./ui/button";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "./ui/table";
import { Viaoda_Libre } from "next/font/google";
import {
  BarChart2,
  ExternalLink,
  RefreshCw,
  Search,
  Shield,
  Zap,
  CheckCircle,
} from "lucide-react";
import { getClassificationColorClasses } from "~/utils/coloring-utils";
import { SavedReportModal, type SavedReport } from "./saved-report-modal";

export default function KnownVariants({
  refreshVariants,
  showComparison,
  updateClinvarVariant,
  clinvarVariants,
  isLoadingClinvar,
  clinvarError,
  genomeId,
  gene,
  onAnalysisComplete,
}: {
  refreshVariants: () => void;
  showComparison: (variant: ClinvarVariant) => void;
  updateClinvarVariant: (id: string, newVariant: ClinvarVariant) => void;
  clinvarVariants: ClinvarVariant[];
  isLoadingClinvar: boolean;
  clinvarError: string | null;
  genomeId: string;
  gene: GeneFromSearch;
  onAnalysisComplete?: () => void;
}) {
  // Track which positions have been analyzed by this user
  const [analyzedReports, setAnalyzedReports] = useState<Map<number, SavedReport>>(new Map());
  const [selectedReport, setSelectedReport] = useState<SavedReport | null>(null);
  const [enableISM, setEnableISM] = useState(false);

  // Fetch user's analyzed variants for this gene
  useEffect(() => {
    const fetchAnalyzedVariants = async () => {
      try {
        const response = await fetch(`/api/history?gene=${encodeURIComponent(gene.symbol)}`);
        if (response.ok) {
          const data = await response.json();
          const reportsMap = new Map<number, SavedReport>();
          (data.reports || []).forEach((r: SavedReport) => {
            reportsMap.set(r.position, r);
          });
          setAnalyzedReports(reportsMap);
        }
      } catch (error) {
        console.error("Failed to fetch analyzed variants:", error);
      }
    };
    fetchAnalyzedVariants();
  }, [gene.symbol]);

  const getAnalyzedReport = (variant: ClinvarVariant): SavedReport | null => {
    const position = variant.location ? parseInt(variant.location.replaceAll(",", "")) : null;
    if (position !== null && analyzedReports.has(position)) {
      return analyzedReports.get(position) || null;
    }
    return null;
  };

  const isAlreadyAnalyzed = (variant: ClinvarVariant): boolean => {
    return getAnalyzedReport(variant) !== null;
  };

  const analyzeVariant = async (variant: ClinvarVariant) => {
    let variantDetails: { position: number | null; reference: string; alternative: string } | null = null;
    const position = variant.location
      ? parseInt(variant.location.replaceAll(",", ""))
      : null;

    const variationType = variant.variation_type.toLowerCase();

    // Parse based on variant type
    if (variationType.includes("single nucleotide")) {
      // SNV: Match pattern like "G>T" or "A>C"
      const refAltMatch = variant.title.match(/([ATCG])>([ATCG])/i);
      if (refAltMatch && refAltMatch.length === 3 && position) {
        const transcriptRef = refAltMatch[1]!;
        const transcriptAlt = refAltMatch[2]!;
        
        console.log(`🧬 [ClinVar-Table] Analyzing ${gene.symbol} position ${position}: transcript ${transcriptRef}>${transcriptAlt}`);
        
        // Fetch genomic reference to detect strand orientation
        const genomicRef = await fetchSingleBase(gene.chrom, position, genomeId);
        console.log(`🧬 [ClinVar-Table] Genomic reference at position ${position}: ${genomicRef}`);
        
        // Check if gene is on minus strand
        const isMinusStrand = genomicRef && transcriptRef && 
                              genomicRef.toUpperCase() !== transcriptRef.toUpperCase();
        
        let finalAlt = transcriptAlt;
        
        // For minus-strand genes, reverse complement the alternative
        if (isMinusStrand) {
          const complement: Record<string, string> = { 
            'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C' 
          };
          finalAlt = complement[transcriptAlt.toUpperCase()] || transcriptAlt;
          console.log(`🧬 [ClinVar-Table] Minus-strand detected! Transcript: ${transcriptRef}>${transcriptAlt}, Genomic: ${genomicRef}>${finalAlt}`);
        } else {
          console.log(`🧬 [ClinVar-Table] Plus-strand (or detection failed), using transcript alt: ${transcriptAlt}`);
        }
        
        variantDetails = {
          position,
          reference: transcriptRef,
          alternative: finalAlt,
        };
      }
    } else if (variationType.includes("deletion")) {
      // Deletion: The backend will handle empty alternative as deletion
      // For deletions, we send position and let backend fetch reference
      variantDetails = {
        position,
        reference: "",  // Backend will fetch from genome
        alternative: "", // Empty = deletion
      };
    } else if (variationType.includes("insertion") || variationType.includes("duplication")) {
      // Insertion/Duplication: Try to parse inserted sequence from title
      // Pattern examples: "insATG", "dupA"
      const insMatch = variant.title.match(/ins([ATCG]+)/i);
      const dupMatch = variant.title.match(/dup([ATCG]*)/i);

      if (insMatch) {
        variantDetails = {
          position,
          reference: "",
          alternative: insMatch[1]!,
        };
      } else if (dupMatch) {
        variantDetails = {
          position,
          reference: "",
          alternative: dupMatch[1] || "N", // Default if no sequence specified
        };
      } else {
        // Generic indel - try anyway
        variantDetails = {
          position,
          reference: "",
          alternative: "N", // Placeholder
        };
      }
    } else if (variationType.includes("indel") || variationType.includes("delins")) {
      // Complex indels like "AT>GTC" (delins = deletion-insertion)
      const refAltMatch = variant.title.match(/([ATCG]+)>([ATCG]+)/i);
      if (refAltMatch && refAltMatch.length === 3) {
        variantDetails = {
          position,
          reference: refAltMatch[1]!,
          alternative: refAltMatch[2]!,
        };
      }
    } else {
      // Other variant types - try generic multi-nucleotide parsing
      const refAltMatch = variant.title.match(/([ATCG]+)>([ATCG]+)/i);
      if (refAltMatch && refAltMatch.length === 3) {
        variantDetails = {
          position,
          reference: refAltMatch[1]!,
          alternative: refAltMatch[2]!,
        };
      }
    }

    if (!variantDetails || !variantDetails.position) {
      console.warn("Could not parse variant details from:", variant.title);
      return;
    }

    updateClinvarVariant(variant.clinvar_id, {
      ...variant,
      isAnalyzing: true,
    });

    try {
      const data = await analyzeVariantWithAPI({
        position: variantDetails.position,
        alternative: variantDetails.alternative,
        genomeId: genomeId,
        chromosome: gene.chrom,
        geneSymbol: gene.symbol,
        runISMScan: enableISM,
      });

      const updatedVariant: ClinvarVariant = {
        ...variant,
        isAnalyzing: false,
        evo2Result: data,
      };

      updateClinvarVariant(variant.clinvar_id, updatedVariant);

      // Trigger history refresh
      onAnalysisComplete?.();
      // Trigger credits refresh
      window.dispatchEvent(new CustomEvent('credits-updated'));

      showComparison(updatedVariant);
    } catch (error) {
      updateClinvarVariant(variant.clinvar_id, {
        ...variant,
        isAnalyzing: false,
        evo2Error: error instanceof Error ? error.message : "Analysis failed",
      });
    }
  };

  // Badge styling for variant types - semantic colors by impact
  const getTypeBadge = (type: string) => {
    const t = type.toLowerCase();
    let styles = "bg-slate-100 text-slate-600 border border-slate-200"; // Default Gray

    if (t.includes("single nucleotide") || t.includes("snv")) {
      styles = "bg-blue-50 text-blue-700 border border-blue-100";
    } else if (t.includes("deletion") || t.includes("indel") || t.includes("frameshift")) {
      styles = "bg-rose-50 text-rose-700 border border-rose-100";
    } else if (t.includes("duplication") || t.includes("insertion")) {
      styles = "bg-amber-50 text-amber-700 border border-amber-100";
    }

    return (
      <span className={`px-2.5 py-0.5 rounded-full text-xs font-medium whitespace-nowrap ${styles}`}>
        {type}
      </span>
    );
  };

  return (
    <>
      <Card className="gap-0 border-none bg-white py-0 shadow-sm">
        <CardHeader className="flex flex-row items-center justify-between pt-4 pb-2">
          <CardTitle className="text-sm font-normal text-[#3c4f3d]/70">
            Known Variants in Gene (<span className="font-semibold font-mono text-[#3c4f3d]">{gene.symbol}</span>) from ClinVar
          </CardTitle>
          <div className="flex items-center gap-3">
            <label className="flex items-center gap-1.5 cursor-pointer select-none">
              <input
                type="checkbox"
                checked={enableISM}
                onChange={(e) => setEnableISM(e.target.checked)}
                className="h-3 w-3 rounded border-slate-300 text-indigo-600 focus:ring-indigo-500 cursor-pointer"
              />
              <span className="text-[10px] text-slate-500">ISM Scan</span>
            </label>
            <Button
              variant="ghost"
              size="sm"
              onClick={refreshVariants}
              disabled={isLoadingClinvar}
              className="h-7 cursor-pointer text-xs text-[#3c4f3d] hover:bg-[#e9eeea]/70"
            >
              <RefreshCw className="mr-1 h-3 w-3" />
              Refresh
            </Button>
          </div>
        </CardHeader>
        <CardContent className="pb-4">
          {clinvarError && (
            <div className="mb-4 rounded-md bg-red-50 p-3 text-xs text-red-600">
              {clinvarError}
            </div>
          )}

          {isLoadingClinvar ? (
            <div className="flex justify-center py-6">
              <div className="h-5 w-5 animate-spin rounded-full border-2 border-[#3c4f3d]/30 border-t-[#3c4f3d]"></div>
            </div>
          ) : clinvarVariants.length > 0 ? (
            <div className="relative">
              {/* Scroll hint gradient - shows there's more content */}
              <div className="pointer-events-none absolute right-0 top-0 bottom-0 w-8 bg-gradient-to-l from-white to-transparent z-10 md:hidden"></div>
              <div className="h-96 max-h-96 overflow-x-auto overflow-y-scroll rounded-md border border-[#3c4f3d]/5">
                <Table className="min-w-[600px] w-full table-fixed">
                  <TableHeader className="sticky top-0 z-10">
                    <TableRow className="bg-[#e9eeea]/80 hover:bg-[#e9eeea]/30">
                      <TableHead className="py-2 pl-6 text-left text-xs font-medium text-[#3c4f3d] w-[40%]">
                        Variant
                      </TableHead>
                      <TableHead className="py-2 text-left text-xs font-medium text-[#3c4f3d] w-[20%]">
                        Type
                      </TableHead>
                      <TableHead className="py-2 text-left text-xs font-medium text-[#3c4f3d] w-[25%]">
                        Clinical Significance
                      </TableHead>
                      <TableHead className="py-2 pr-2 text-xs font-medium text-[#3c4f3d] text-center w-[15%]">
                        Actions
                      </TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {clinvarVariants.map((variant) => (
                      <TableRow
                        key={variant.clinvar_id}
                        className="border-b border-[#3c4f3d]/5 hover:bg-slate-50/50 transition-colors"
                      >
                        <TableCell className="py-2 pl-6 align-middle w-[40%]">
                          <div className="flex items-center gap-1.5 text-xs font-medium text-[#3c4f3d]">
                            {isAlreadyAnalyzed(variant) && (
                              <button
                                onClick={() => setSelectedReport(getAnalyzedReport(variant))}
                                className="flex items-center gap-0.5 rounded-full bg-emerald-100 px-1.5 py-0.5 text-[10px] font-medium text-emerald-700 hover:bg-emerald-200 transition-colors cursor-pointer"
                                title="Click to view saved report"
                              >
                                <CheckCircle className="h-3 w-3" />
                                Analyzed
                              </button>
                            )}
                            {variant.title}
                          </div>
                          <div className="mt-1 flex items-center gap-1 text-xs text-[#3c4f3d]/70">
                            <p>Location: {variant.location}</p>
                            <Button
                              variant="link"
                              size="sm"
                              className="h-6 cursor-pointer px-0 text-xs text-[#de8246] hover:text-[#de8246]/80"
                              onClick={() =>
                                window.open(
                                  `https://www.ncbi.nlm.nih.gov/clinvar/variation/${variant.clinvar_id}`,
                                  "_blank",
                                )
                              }
                            >
                              View in ClinVar
                              <ExternalLink className="ml-1 inline-block h-2 w-2" />
                            </Button>
                          </div>
                        </TableCell>
                        <TableCell className="py-2 text-xs align-middle w-[20%]">
                          {getTypeBadge(variant.variation_type)}
                        </TableCell>
                        <TableCell className="py-2 text-xs align-middle w-[25%]">
                          <div
                            className={`w-fit rounded-md px-2 py-1 text-center font-normal ${getClassificationColorClasses(variant.classification)}`}
                          >
                            {variant.classification || "Unknown"}
                          </div>
                          {variant.evo2Result && (
                            <div className="mt-2">
                              <div
                                className={`flex w-fit items-center gap-1 rounded-md px-2 py-1 text-center ${getClassificationColorClasses(variant.evo2Result.prediction)}`}
                              >
                                <Shield className="h-3 w-3" />
                                <span>Evo2: {variant.evo2Result.prediction}</span>
                              </div>
                            </div>
                          )}
                        </TableCell>
                        <TableCell className="py-2 text-xs align-middle text-center w-[15%] pr-2">
                          <div className="flex justify-center">
                            {!variant.evo2Result ? (
                              <button
                                className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-[#de8246] border border-[#de8246]/30 rounded-lg hover:bg-[#de8246]/10 transition-colors cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
                                disabled={variant.isAnalyzing}
                                onClick={() => analyzeVariant(variant)}
                                title="Run Evo2 inference on Modal H100"
                              >
                                {variant.isAnalyzing ? (
                                  <>
                                    <span className="inline-block h-3.5 w-3.5 animate-spin rounded-full border-2 border-[#de8246]/30 border-t-[#de8246]"></span>
                                    Analyzing...
                                  </>
                                ) : (
                                  <>
                                    <Zap className="h-3.5 w-3.5" />
                                    Analyze
                                  </>
                                )}
                              </button>
                            ) : (
                              <button
                                className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-green-700 bg-green-50 rounded-lg hover:bg-green-100 transition-colors cursor-pointer"
                                onClick={() => showComparison(variant)}
                                title="View Evo2 prediction vs ClinVar classification"
                              >
                                <BarChart2 className="h-3.5 w-3.5" />
                                Results
                              </button>
                            )}
                          </div>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            </div>
          ) : (
            <div className="flex h-48 flex-col items-center justify-center text-center text-gray-400">
              <Search className="mb-4 h-10 w-10 text-gray-300" />
              <p className="text-sm leading-relaxed">
                No ClinVar variants found for this gene.
              </p>
            </div>
          )}
        </CardContent>
      </Card>
      {/* Saved Report Modal */}
      <SavedReportModal
        report={selectedReport}
        onClose={() => setSelectedReport(null)}
        onDelete={(reportId) => {
          // Remove from analyzedReports map
          setAnalyzedReports((prev) => {
            const newMap = new Map(prev);
            // Find and delete by report ID (need to find position first)
            for (const [position, report] of newMap.entries()) {
              if (report.id === reportId) {
                newMap.delete(position);
                break;
              }
            }
            return newMap;
          });
        }}
      />
    </>
  );
}
