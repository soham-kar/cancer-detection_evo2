"use client";

import React, { useEffect, useState } from "react";
import {
  History,
  Trash2,
  ChevronDown,
  ChevronUp,
  Dna,
  Eye,
} from "lucide-react";
import { SavedReportModal, type SavedReport } from "./saved-report-modal";

interface AnalysisReport {
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
  analysisSource?: "clinvar" | "custom" | null;
  createdAt: string;
}

interface AnalysisHistoryProps {
  currentGeneSymbol?: string;
}

export interface AnalysisHistoryHandle {
  refresh: () => void;
}

export const AnalysisHistory = React.forwardRef<
  AnalysisHistoryHandle,
  AnalysisHistoryProps
>(function AnalysisHistory({ currentGeneSymbol }, ref) {
  const [reports, setReports] = useState<AnalysisReport[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isExpanded, setIsExpanded] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedReportId, setSelectedReportId] = useState<string | null>(null);
  const [selectedReport, setSelectedReport] = useState<SavedReport | null>(
    null,
  );
  const [isLoadingReport, setIsLoadingReport] = useState(false);
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);

  const fetchHistory = React.useCallback(async () => {
    try {
      setIsLoading(true);
      setError(null);
      // Fetch ALL reports for the user, filter client-side for reliability
      const url = "/api/history?limit=50";
      console.log("[AnalysisHistory] Fetching all reports for user");
      const response = await fetch(url, { cache: "no-store" });
      const data = await response.json();
      console.log("[AnalysisHistory] Response:", {
        ok: response.ok,
        total: data.total,
        error: data.error,
      });

      if (response.ok) {
        const allReports = data.reports || [];
        // Filter client-side by gene symbol (case-insensitive)
        const filtered = currentGeneSymbol
          ? allReports.filter(
              (r: AnalysisReport) =>
                r.geneSymbol.toUpperCase() === currentGeneSymbol.toUpperCase(),
            )
          : allReports;
        console.log("[AnalysisHistory] Filtered:", {
          total: allReports.length,
          filtered: filtered.length,
          gene: currentGeneSymbol,
        });
        setReports(filtered);
      } else {
        setError(data.error || "Failed to fetch history");
        setReports([]);
      }
    } catch (err) {
      console.error("[AnalysisHistory] Fetch error:", err);
      setError("Failed to load analysis history");
      setReports([]);
    } finally {
      setIsLoading(false);
    }
  }, [currentGeneSymbol]);

  // Expose refresh function to parent
  React.useImperativeHandle(
    ref,
    () => ({
      refresh: fetchHistory,
    }),
    [fetchHistory],
  );

  useEffect(() => {
    fetchHistory();
  }, [fetchHistory]);

  // Refresh when a new analysis completes anywhere in the app
  useEffect(() => {
    const handleAnalysisSaved = () => {
      console.log(
        "[AnalysisHistory] analysis-saved event received, refreshing...",
      );
      fetchHistory();
    };
    window.addEventListener("analysis-saved", handleAnalysisSaved);
    return () =>
      window.removeEventListener("analysis-saved", handleAnalysisSaved);
  }, [fetchHistory]);

  const handleDelete = (reportId: string) => {
    // Show confirmation modal
    setDeleteConfirmId(reportId);
  };

  const confirmDelete = async () => {
    if (!deleteConfirmId) return;
    try {
      const response = await fetch(`/api/history?id=${deleteConfirmId}`, {
        method: "DELETE",
      });

      if (response.ok) {
        setReports(reports.filter((r) => r.id !== deleteConfirmId));
      }
    } catch (err) {
      console.error("Failed to delete report:", err);
    } finally {
      setDeleteConfirmId(null);
    }
  };

  const handleViewReport = async (reportId: string) => {
    setIsLoadingReport(true);
    setSelectedReportId(reportId);
    try {
      const response = await fetch(`/api/history?id=${reportId}`);
      if (response.ok) {
        const data = await response.json();
        setSelectedReport(data.report);
      }
    } catch (err) {
      console.error("Failed to load report:", err);
    } finally {
      setIsLoadingReport(false);
    }
  };

  const getPredictionColor = (prediction: string) => {
    const lower = prediction.toLowerCase();
    if (lower.includes("pathogenic")) return "text-red-600 bg-red-50";
    if (lower.includes("benign")) return "text-green-600 bg-green-50";
    return "text-amber-600 bg-amber-50";
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center gap-2 py-2 text-sm text-[#3c4f3d]/50">
        <svg
          className="h-4 w-4 animate-spin"
          xmlns="http://www.w3.org/2000/svg"
          fill="none"
          viewBox="0 0 24 24"
        >
          <circle
            className="opacity-25"
            cx="12"
            cy="12"
            r="10"
            stroke="currentColor"
            strokeWidth="4"
          ></circle>
          <path
            className="opacity-75"
            fill="currentColor"
            d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
          ></path>
        </svg>
        <span>
          Looking for previous analyses
          {currentGeneSymbol ? ` for ${currentGeneSymbol}` : ""}...
        </span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="overflow-hidden rounded-lg border border-red-200 bg-red-50">
        <div className="flex items-center gap-2 p-4 text-sm font-medium text-red-700">
          <History className="h-4 w-4 text-red-600" />
          Previous Analysis Reports
        </div>
        <div className="border-t border-red-100 p-4 text-xs text-red-600/80">
          Unable to load analysis history. {error}
        </div>
      </div>
    );
  }

  // Show empty state instead of hiding the section
  if (reports.length === 0) {
    return (
      <div className="overflow-hidden rounded-lg border border-[#3c4f3d]/10 bg-white">
        <div className="flex items-center gap-2 p-4 text-sm font-medium text-[#3c4f3d]">
          <History className="h-4 w-4 text-[#de8246]" />
          Previous Analysis Reports (0)
        </div>
        <div className="border-t border-[#3c4f3d]/10 p-4 text-xs text-[#3c4f3d]/60">
          No previous analyses found for {currentGeneSymbol || "this gene"}.
          Analyze a variant above to generate a report.
        </div>
      </div>
    );
  }

  return (
    <>
      <div className="overflow-hidden rounded-lg border border-[#3c4f3d]/10 bg-white">
        <button
          onClick={() => setIsExpanded(!isExpanded)}
          className="flex w-full items-center justify-between p-4 transition-colors hover:bg-[#f4f7f5]"
        >
          <div className="flex items-center gap-2 text-sm font-medium text-[#3c4f3d]">
            <History className="h-4 w-4 text-[#de8246]" />
            Previous Analysis Reports ({reports.length})
          </div>
          {isExpanded ? (
            <ChevronUp className="h-4 w-4 text-[#3c4f3d]/50" />
          ) : (
            <ChevronDown className="h-4 w-4 text-[#3c4f3d]/50" />
          )}
        </button>

        {isExpanded && (
          <div className="border-t border-[#3c4f3d]/10">
            <div className="max-h-64 overflow-y-auto">
              {reports.map((report) => (
                <div
                  key={report.id}
                  className="flex items-center justify-between border-b border-[#3c4f3d]/5 p-3 last:border-b-0 hover:bg-[#f4f7f5]/50"
                >
                  <div className="flex items-center gap-3">
                    {/* Source Badge */}
                    {report.analysisSource && (
                      <span
                        className={`rounded px-1.5 py-0.5 text-[10px] font-semibold tracking-wide uppercase ${
                          report.analysisSource === "clinvar"
                            ? "border border-teal-200 bg-teal-100 text-teal-700"
                            : "border border-orange-200 bg-orange-100 text-orange-700"
                        }`}
                      >
                        {report.analysisSource}
                      </span>
                    )}
                    <div className="flex items-center gap-1.5">
                      <Dna className="h-3.5 w-3.5 text-[#de8246]" />
                      <span className="font-mono text-xs font-semibold text-[#3c4f3d]">
                        {report.geneSymbol}
                      </span>
                    </div>
                    <span className="text-xs text-[#3c4f3d]/60">
                      chr{report.chromosome}:{report.position.toLocaleString()}
                    </span>
                    <span className="font-mono text-xs text-[#3c4f3d]/70">
                      {report.reference} → {report.alternative}
                    </span>
                    <span
                      className={`rounded px-1.5 py-0.5 text-xs font-medium ${getPredictionColor(report.prediction)}`}
                    >
                      {report.prediction}
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-[#3c4f3d]/50">
                      {formatDate(report.createdAt)}
                    </span>
                    <button
                      onClick={() => handleViewReport(report.id)}
                      className="rounded bg-[#de8246]/10 px-2 py-0.5 text-[10px] font-medium text-[#de8246] transition-colors hover:bg-[#de8246]/20"
                    >
                      View Report
                    </button>
                    <button
                      onClick={() => handleDelete(report.id)}
                      className="p-1 text-[#3c4f3d]/30 transition-colors hover:text-red-500"
                      title="Delete report"
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Saved Report Modal */}
      {selectedReport && (
        <SavedReportModal
          report={selectedReport}
          onClose={() => {
            setSelectedReport(null);
            setSelectedReportId(null);
          }}
        />
      )}

      {/* Delete Confirmation Modal */}
      {deleteConfirmId && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
          <div className="mx-4 w-full max-w-xs rounded-lg bg-white p-5 shadow-xl">
            <div className="mb-3 flex items-center gap-3">
              <div className="flex h-9 w-9 items-center justify-center rounded-full bg-red-100">
                <Trash2 className="h-4 w-4 text-red-600" />
              </div>
              <h3 className="text-base font-semibold text-[#3c4f3d]">
                Delete Report
              </h3>
            </div>
            <p className="mb-4 text-sm text-[#3c4f3d]/70">
              Are you sure you want to delete this saved report? This action
              cannot be undone.
            </p>
            <div className="flex justify-end gap-2">
              <button
                onClick={() => setDeleteConfirmId(null)}
                className="rounded-md bg-gray-100 px-3 py-1.5 text-sm font-medium text-[#3c4f3d] transition-colors hover:bg-gray-200"
              >
                Cancel
              </button>
              <button
                onClick={confirmDelete}
                className="rounded-md bg-red-500 px-3 py-1.5 text-sm font-medium text-white transition-colors hover:bg-red-600"
              >
                Delete
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
});
