"use client";

import React, { useEffect, useState } from "react";
import { History, Trash2, ChevronDown, ChevronUp, Dna, Eye } from "lucide-react";
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
    analysisSource?: 'clinvar' | 'custom' | null;
    createdAt: string;
}

interface AnalysisHistoryProps {
    currentGeneSymbol?: string;
}

export interface AnalysisHistoryHandle {
    refresh: () => void;
}

export const AnalysisHistory = React.forwardRef<AnalysisHistoryHandle, AnalysisHistoryProps>(
    function AnalysisHistory({ currentGeneSymbol }, ref) {
        const [reports, setReports] = useState<AnalysisReport[]>([]);
        const [isLoading, setIsLoading] = useState(true);
        const [isExpanded, setIsExpanded] = useState(false);
        const [error, setError] = useState<string | null>(null);
        const [selectedReportId, setSelectedReportId] = useState<string | null>(null);
        const [selectedReport, setSelectedReport] = useState<SavedReport | null>(null);
        const [isLoadingReport, setIsLoadingReport] = useState(false);
        const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);

        const fetchHistory = async () => {
            try {
                setIsLoading(true);
                const url = currentGeneSymbol
                    ? `/api/history?gene=${encodeURIComponent(currentGeneSymbol)}&limit=20`
                    : "/api/history?limit=20";
                const response = await fetch(url);
                const data = await response.json();

                if (response.ok) {
                    setReports(data.reports || []);
                } else {
                    setError(data.error || "Failed to fetch history");
                }
            } catch (err) {
                setError("Failed to load analysis history");
            } finally {
                setIsLoading(false);
            }
        };

        // Expose refresh function to parent
        React.useImperativeHandle(ref, () => ({
            refresh: fetchHistory,
        }));

        useEffect(() => {
            fetchHistory();
        }, [currentGeneSymbol]);

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
                    setReports(reports.filter(r => r.id !== deleteConfirmId));
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
                    <svg className="animate-spin h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    <span>Looking for previous analyses{currentGeneSymbol ? ` for ${currentGeneSymbol}` : ""}...</span>
                </div>
            );
        }

        if (error) {
            return null; // Don't show section if there's an error
        }

        // Don't show anything if there are no previous analyses
        if (reports.length === 0) {
            return null;
        }

        return (
            <>
                <div className="rounded-lg border border-[#3c4f3d]/10 bg-white overflow-hidden">
                    <button
                        onClick={() => setIsExpanded(!isExpanded)}
                        className="w-full flex items-center justify-between p-4 hover:bg-[#f4f7f5] transition-colors"
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
                                        className="flex items-center justify-between p-3 border-b border-[#3c4f3d]/5 last:border-b-0 hover:bg-[#f4f7f5]/50"
                                    >
                                        <div className="flex items-center gap-3">
                                            {/* Source Badge */}
                                            {report.analysisSource && (
                                                <span className={`px-1.5 py-0.5 rounded text-[10px] font-semibold uppercase tracking-wide ${report.analysisSource === 'clinvar'
                                                    ? 'bg-teal-100 text-teal-700 border border-teal-200'
                                                    : 'bg-orange-100 text-orange-700 border border-orange-200'
                                                    }`}>
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
                                            <span className={`px-1.5 py-0.5 rounded text-xs font-medium ${getPredictionColor(report.prediction)}`}>
                                                {report.prediction}
                                            </span>
                                        </div>
                                        <div className="flex items-center gap-2">
                                            <span className="text-xs text-[#3c4f3d]/50">
                                                {formatDate(report.createdAt)}
                                            </span>
                                            <button
                                                onClick={() => handleViewReport(report.id)}
                                                className="px-2 py-0.5 text-[10px] font-medium text-[#de8246] bg-[#de8246]/10 rounded hover:bg-[#de8246]/20 transition-colors"
                                            >
                                                View Report
                                            </button>
                                            <button
                                                onClick={() => handleDelete(report.id)}
                                                className="p-1 text-[#3c4f3d]/30 hover:text-red-500 transition-colors"
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
                        <div className="bg-white rounded-lg p-5 shadow-xl max-w-xs w-full mx-4">
                            <div className="flex items-center gap-3 mb-3">
                                <div className="flex items-center justify-center w-9 h-9 rounded-full bg-red-100">
                                    <Trash2 className="h-4 w-4 text-red-600" />
                                </div>
                                <h3 className="text-base font-semibold text-[#3c4f3d]">Delete Report</h3>
                            </div>
                            <p className="text-sm text-[#3c4f3d]/70 mb-4">
                                Are you sure you want to delete this saved report? This action cannot be undone.
                            </p>
                            <div className="flex gap-2 justify-end">
                                <button
                                    onClick={() => setDeleteConfirmId(null)}
                                    className="px-3 py-1.5 text-sm font-medium text-[#3c4f3d] bg-gray-100 rounded-md hover:bg-gray-200 transition-colors"
                                >
                                    Cancel
                                </button>
                                <button
                                    onClick={confirmDelete}
                                    className="px-3 py-1.5 text-sm font-medium text-white bg-red-500 rounded-md hover:bg-red-600 transition-colors"
                                >
                                    Delete
                                </button>
                            </div>
                        </div>
                    </div>
                )}
            </>
        );
    }
);
