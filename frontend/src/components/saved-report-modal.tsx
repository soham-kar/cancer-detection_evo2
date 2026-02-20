"use client";

import { useEffect, useState } from "react";
import { Button } from "./ui/button";
import { Check, X, Shield, ExternalLink, Download, Copy, Trash2 } from "lucide-react";
import { getClassificationColorClasses } from "~/utils/coloring-utils";

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
    createdAt: string;
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
    const [successToast, setSuccessToast] = useState<{ message: string; icon: string } | null>(null);

    // Lock body scroll when modal is open
    useEffect(() => {
        if (report) {
            document.body.style.overflow = 'hidden';
            return () => {
                document.body.style.overflow = '';
            };
        }
    }, [report]);

    if (!report) return null;

    const generateReportText = () => {
        const date = new Date().toLocaleDateString('en-US', { month: '2-digit', day: '2-digit', year: 'numeric' });
        const changeNotation = `${report.reference || 'N'}>${report.alternative}`;

        return `VARIANT ANALYSIS REPORT
Generated: ${date}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

VARIANT INFORMATION
• Position: ${report.chromosome}:${report.position.toLocaleString()}
• Change: ${changeNotation}
• Type: ${report.variationType || 'Single Nucleotide Variant'}
• Gene: ${report.geneSymbol}

CLASSIFICATION COMPARISON
• ClinVar: ${report.clinvarClassification || 'Unknown'}
• Evo2 Prediction: ${report.prediction}
• Delta Score: ${report.deltaScore?.toFixed(6) || '0.000000'}
• Confidence: ${Math.round((report.classificationConfidence || 0) * 100)}%

POPULATION FREQUENCY
• gnomAD: ${report.populationFrequency?.gnomad_af
                ? `${(report.populationFrequency.gnomad_af * 100).toFixed(4)}%`
                : 'Not observed'}

ACMG EVIDENCE
• Code: ${report.acmgEvidence?.code || 'None'}
• ${report.acmgEvidence?.description || 'Computational evidence is inconclusive'}

AI CLINICAL SUMMARY
${report.literatureContext?.summary || 'No summary available'}

REFERENCES
${report.literatureContext?.pubmed_ids?.slice(0, 5).map(id => `• PMID:${id}`).join('\n') || '• No references available'}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Cross-reference: https://varsome.com/position/${report.genomeId}/chr${report.chromosome}-${report.position}
`;
    };

    const showSuccessToast = (message: string, icon: string) => {
        setSuccessToast({ message, icon });
        setTimeout(() => setSuccessToast(null), 3000);
    };

    const copyReport = async () => {
        try {
            await navigator.clipboard.writeText(generateReportText());
            showSuccessToast('Report copied to clipboard!', '📋');
        } catch {
            showSuccessToast('Failed to copy to clipboard', '❌');
        }
    };

    const downloadReport = () => {
        const blob = new Blob([generateReportText()], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `variant-report-${report.geneSymbol}-${report.position}.txt`;
        a.click();
        URL.revokeObjectURL(url);
        showSuccessToast('Report downloaded successfully!', '📥');
    };

    const handleDeleteClick = () => {
        setShowDeleteConfirm(true);
    };

    const confirmDelete = async () => {
        setIsDeleting(true);
        try {
            const response = await fetch('/api/history/delete', {
                method: 'DELETE',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ reportId: report.id }),
            });

            if (response.ok) {
                onDelete?.(report.id);
                onClose();
            } else {
                const data = await response.json();
                alert(data.error || 'Failed to delete report');
            }
        } catch {
            alert('Failed to delete report');
        } finally {
            setIsDeleting(false);
            setShowDeleteConfirm(false);
        }
    };

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4 overflow-y-auto h-screen min-h-screen">
            <div className="max-h-[90vh] w-full max-w-3xl overflow-hidden rounded-2xl bg-white shadow-xl">
                {/* Modal header */}
                <div className="border-b border-[#3c4f3d]/10 p-5">
                    <div className="flex items-center justify-between">
                        <h3 className="text-lg font-medium text-[#3c4f3d]">
                            Variant Analysis Comparison
                        </h3>
                        <Button
                            variant="ghost"
                            size="sm"
                            onClick={onClose}
                            className="h-7 w-7 cursor-pointer p-0 text-[#3c4f3d]/70 hover:bg-[#e9eeea]/70 hover:text-[#3c4f3d]"
                        >
                            <X className="h-5 w-5" />
                        </Button>
                    </div>
                </div>

                {/* Modal content */}
                <div className="max-h-[calc(90vh-130px)] overflow-y-auto p-5">
                    <div className="space-y-6">
                        {/* Variant Information */}
                        <div className="rounded-md border border-[#3c4f3d]/10 bg-[#e9eeea]/30 p-4">
                            <h4 className="mb-3 text-sm font-medium text-[#3c4f3d]">
                                Variant Information
                            </h4>
                            <div className="grid gap-4 md:grid-cols-2">
                                <div>
                                    <div className="space-y-2">
                                        <div className="flex">
                                            <span className="w-28 text-xs text-[#3c4f3d]/70">Position:</span>
                                            <span className="text-xs">{report.position.toLocaleString()}</span>
                                        </div>
                                        <div className="flex">
                                            <span className="w-28 text-xs text-[#3c4f3d]/70">Type:</span>
                                            <span className="text-xs">Single Nucleotide Variant</span>
                                        </div>
                                    </div>
                                </div>
                                <div>
                                    <div className="space-y-2">
                                        <div className="flex">
                                            <span className="w-28 text-xs text-[#3c4f3d]/70">Variant:</span>
                                            <span className="text-xs font-mono">
                                                <span className="text-green-600">{report.reference || "N"}</span>
                                                <span className="text-orange-500">&gt;</span>
                                                <span className="text-blue-600">{report.alternative}</span>
                                            </span>
                                        </div>
                                        <div className="flex">
                                            <span className="w-28 text-xs text-[#3c4f3d]/70">Gene:</span>
                                            <span className="text-xs font-medium">{report.geneSymbol}</span>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            {/* Cross-reference Links */}
                            <div className="mt-4 flex flex-wrap gap-3">
                                <a
                                    href={report.clinvarId
                                        ? `https://www.ncbi.nlm.nih.gov/clinvar/variation/${report.clinvarId}/`
                                        : `https://www.ncbi.nlm.nih.gov/clinvar/?term=${report.chromosome}%5Bchr%5D+AND+${report.position}%5Bpos%5D`
                                    }
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="inline-flex items-center gap-1.5 rounded-md bg-[#de8246]/10 px-3 py-1.5 text-xs font-medium text-[#de8246] hover:bg-[#de8246]/20 transition-colors"
                                >
                                    <ExternalLink className="h-3 w-3" />
                                    View in ClinVar
                                </a>
                                <a
                                    href={`https://varsome.com/position/${report.genomeId}/${report.chromosome}-${report.position}`}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="inline-flex items-center gap-1.5 rounded-md bg-teal-500/10 px-3 py-1.5 text-xs font-medium text-teal-700 hover:bg-teal-500/20 transition-colors"
                                >
                                    <ExternalLink className="h-3 w-3" />
                                    Cross-check on VarSome
                                </a>
                            </div>
                        </div>

                        {/* Analysis Comparison - Two columns */}
                        <div>
                            <h4 className="mb-3 text-sm font-medium text-[#3c4f3d]">
                                Analysis Comparison
                            </h4>
                            <div className="rounded-md border border-[#3c4f3d]/10 bg-white p-4">
                                <div className="grid gap-4 md:grid-cols-2">
                                    {/* ClinVar Assessment */}
                                    <div className="rounded-md bg-[#e9eeea]/50 p-4">
                                        <h5 className="mb-2 flex items-center gap-2 text-xs font-medium text-[#3c4f3d]">
                                            <span className="flex h-5 w-5 items-center justify-center rounded-full bg-[#3c4f3d]/10">
                                                <span className="h-3 w-3 rounded-full bg-[#3c4f3d]"></span>
                                            </span>
                                            ClinVar Assessment
                                        </h5>
                                        <div className="mt-2">
                                            <div className={`w-fit rounded-md px-2 py-1 text-xs font-normal ${getClassificationColorClasses(report.clinvarClassification || 'Unknown')}`}>
                                                {report.clinvarClassification || "Unknown significance"}
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
                                            <div className={`flex w-fit items-center gap-1 rounded-md px-2 py-1 text-xs font-normal ${getClassificationColorClasses(report.prediction)}`}>
                                                {report.prediction.toLowerCase().includes("pathogenic") ? (
                                                    <svg xmlns="http://www.w3.org/2000/svg" className="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                                                    </svg>
                                                ) : (
                                                    <svg xmlns="http://www.w3.org/2000/svg" className="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                                                    </svg>
                                                )}
                                                {report.prediction}
                                            </div>
                                        </div>

                                        {/* Delta score with gradient bar */}
                                        <div className="mt-3">
                                            <div className="mb-1 text-xs text-[#3c4f3d]/70">
                                                Delta Likelihood Score:
                                            </div>
                                            <div className={`text-sm font-semibold font-mono ${report.prediction.toLowerCase().includes('pathogenic') ? 'text-red-800' : report.prediction.toLowerCase().includes('benign') ? 'text-green-800' : 'text-[#3c4f3d]'}`}>
                                                {report.deltaScore?.toFixed(6) || "0.000000"}
                                            </div>

                                            {/* Gradient bar with marker */}
                                            <div className="mt-2">
                                                <div className="flex justify-between text-[10px] text-[#3c4f3d]/60 mb-1">
                                                    <span>Pathogenic</span>
                                                    <span>Neutral</span>
                                                    <span>Benign</span>
                                                </div>
                                                <div className="relative h-3 rounded-full bg-gradient-to-r from-red-500 via-yellow-400 to-green-500">
                                                    <div className="absolute top-0 left-1/2 h-3 w-0.5 bg-white/70"></div>
                                                    <div
                                                        className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 h-4 w-4 rounded-full border-2 border-white bg-[#3c4f3d] shadow-md"
                                                        style={{
                                                            left: `${Math.max(5, Math.min(95,
                                                                // Use much tighter range for better sensitivity to small scores
                                                                ((Math.max(-0.01, Math.min(0.01, report.deltaScore || 0)) + 0.01) / 0.02) * 100
                                                            ))}%`
                                                        }}
                                                    ></div>
                                                </div>
                                            </div>

                                            <div className="mt-2 text-xs text-[#3c4f3d]/60">
                                                ≈ Score near zero suggests minimal impact
                                            </div>
                                        </div>

                                        {/* Confidence */}
                                        <div className="mt-3">
                                            <div className="mb-1 text-xs text-[#3c4f3d]/70">
                                                Confidence:
                                            </div>
                                            <div className="h-2 w-full rounded-full bg-gray-200 overflow-hidden">
                                                <div
                                                    className="h-full rounded-full bg-gradient-to-r from-green-400 to-green-600"
                                                    style={{ width: `${(report.classificationConfidence || 0) * 100}%` }}
                                                />
                                            </div>
                                            <div className="mt-1 text-right text-xs font-medium text-[#3c4f3d]">
                                                {Math.round((report.classificationConfidence || 0) * 100)}%
                                            </div>
                                        </div>
                                    </div>
                                </div>

                                {/* Discordance warning - only show if classifications differ */}
                                {report.clinvarClassification && report.clinvarClassification.toLowerCase() !== report.prediction.toLowerCase() && (
                                    <div className="mt-4 flex items-center gap-2 text-xs text-amber-700">
                                        <span className="flex h-4 w-4 items-center justify-center text-amber-600">!</span>
                                        <span>Evo2 prediction differs from ClinVar classification</span>
                                    </div>
                                )}
                            </div>
                        </div>

                        {/* Clinical Annotations - 3 columns */}
                        {(report.populationFrequency || report.acmgEvidence || report.literatureContext) && (
                            <div className="rounded-md border border-[#3c4f3d]/10 bg-[#e9eeea]/30 p-4">
                                <h4 className="mb-3 text-sm font-medium text-[#3c4f3d]">
                                    Clinical Annotations
                                </h4>

                                <div className="grid gap-4 md:grid-cols-3">
                                    {/* Population Frequency */}
                                    <div className="rounded-md bg-white p-3">
                                        <div className="text-xs font-medium text-[#3c4f3d]/70 mb-1">
                                            Population Frequency
                                        </div>
                                        {report.populationFrequency?.gnomad_af ? (
                                            <div className="text-sm text-[#3c4f3d]">
                                                {(report.populationFrequency.gnomad_af * 100).toFixed(4)}%
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
                                        <div className="text-sm text-[#3c4f3d]/50">
                                            {report.acmgEvidence?.description || "Computational evidence is inconclusive"}
                                        </div>
                                    </div>

                                    {/* Literature */}
                                    <div className="rounded-md bg-white p-3">
                                        <div className="text-xs font-medium text-[#3c4f3d]/70 mb-1">
                                            Literature
                                        </div>
                                        <div className="text-sm text-[#3c4f3d]/60 mb-2">
                                            {report.literatureContext?.articles_found || 0} articles
                                        </div>
                                        {report.literatureContext?.pubmed_ids && report.literatureContext.pubmed_ids.length > 0 && (
                                            <div className="flex flex-wrap gap-1">
                                                {report.literatureContext.pubmed_ids.slice(0, 5).map((id: string) => (
                                                    <a
                                                        key={id}
                                                        href={`https://pubmed.ncbi.nlm.nih.gov/${id}/`}
                                                        target="_blank"
                                                        rel="noopener noreferrer"
                                                        className="inline-flex items-center gap-1 px-2 py-1 text-xs font-medium rounded-full border bg-slate-100 text-slate-600 border-slate-200 hover:bg-slate-200 hover:text-slate-800 transition-colors"
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
                                </div>

                                {/* AI Clinical Summary */}
                                {report.literatureContext?.summary && (
                                    <div className="mt-4 rounded-md bg-white p-3">
                                        <div className="text-xs font-medium text-[#3c4f3d]/70 mb-2">
                                            AI Clinical Summary
                                        </div>
                                        <div className="text-sm text-[#3c4f3d] leading-relaxed">
                                            {report.literatureContext.summary}
                                        </div>
                                    </div>
                                )}
                            </div>
                        )}

                        {/* Evidence Summary */}
                        <div className="rounded-md border border-[#3c4f3d]/10 bg-gradient-to-br from-[#e9eeea]/50 to-white p-4">
                            <h4 className="mb-3 text-sm font-medium text-[#3c4f3d] flex items-center gap-2">
                                <Shield className="h-4 w-4" />
                                Evidence Summary
                            </h4>
                            <div className="space-y-3 text-sm text-[#3c4f3d] leading-relaxed">
                                {/* Prediction */}
                                <div className="flex items-start gap-2">
                                    <span className={`mt-1 h-2 w-2 rounded-full flex-shrink-0 ${report.prediction.toLowerCase().includes("pathogenic") ? "bg-red-500" : "bg-green-500"
                                        }`}></span>
                                    <p>
                                        <strong>Evo2 predicts this variant as {report.prediction}</strong>
                                        {" "}with {Math.round((report.classificationConfidence || 0) * 100)}% confidence.
                                        {report.deltaScore < 0
                                            ? ` The negative delta score (${report.deltaScore?.toFixed(4)}) indicates the variant reduces protein fitness, suggesting loss of normal function.`
                                            : ` The positive delta score (${report.deltaScore?.toFixed(4)}) indicates preserved or improved protein fitness, suggesting the variant is tolerated.`}
                                    </p>
                                </div>

                                {/* Population */}
                                <div className="flex items-start gap-2">
                                    <span className="mt-1 h-2 w-2 rounded-full flex-shrink-0 bg-amber-500"></span>
                                    <p>
                                        <strong>Not observed in gnomAD</strong> (v4.1 with 800,000+ individuals). While rarity can suggest pathogenicity, Evo2's positive delta score indicates preserved protein function. Many ultra-rare variants are benign but simply haven't been observed yet due to population sampling.
                                    </p>
                                </div>

                                {/* ACMG */}
                                <div className="flex items-start gap-2">
                                    <span className="mt-1 h-2 w-2 rounded-full flex-shrink-0 bg-purple-500"></span>
                                    <p>
                                        <strong>Computational evidence inconclusive</strong>: The prediction confidence is in the uncertain range. Additional clinical or functional evidence is recommended.
                                    </p>
                                </div>
                            </div>

                            {/* Classification Discordance Warning */}
                            <div className="mt-4 rounded-md border border-amber-300 bg-amber-50 p-3">
                                <div className="flex items-start gap-2 text-sm text-amber-800">
                                    <span className="text-amber-600 font-bold">⚠</span>
                                    <p>
                                        <strong>Classification Discordance:</strong> ClinVar reports "Uncertain significance" while Evo2 predicts "{report.prediction}". This VUS may warrant reclassification based on computational evidence. Consider functional studies or family segregation analysis.
                                    </p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Modal footer */}
                <div className="border-t border-[#3c4f3d]/10 p-4 flex justify-between">
                    <div className="flex gap-2">
                        <Button
                            variant="outline"
                            size="sm"
                            className="cursor-pointer text-xs"
                            onClick={downloadReport}
                        >
                            <Download className="h-3 w-3 mr-1" />
                            Download Report
                        </Button>
                        <Button
                            variant="outline"
                            size="sm"
                            className="cursor-pointer text-xs"
                            onClick={copyReport}
                        >
                            <Copy className="h-3 w-3 mr-1" />
                            Copy Report
                        </Button>
                        <Button
                            variant="outline"
                            size="sm"
                            className="cursor-pointer text-xs text-red-600 border-red-200 hover:bg-red-50 hover:text-red-700"
                            onClick={handleDeleteClick}
                            disabled={isDeleting}
                        >
                            <Trash2 className="h-3 w-3 mr-1" />
                            {isDeleting ? 'Deleting...' : 'Delete'}
                        </Button>
                    </div>
                    <Button
                        variant="default"
                        size="sm"
                        onClick={onClose}
                        className="cursor-pointer bg-[#3c4f3d] hover:bg-[#3c4f3d]/90"
                    >
                        Close
                    </Button>
                </div>
            </div>

            {/* Custom Delete Confirmation Modal */}
            {showDeleteConfirm && (
                <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/60 p-4">
                    <div className="w-full max-w-sm rounded-xl bg-white p-6 shadow-2xl">
                        <div className="flex items-center gap-3 mb-4">
                            <div className="flex h-10 w-10 items-center justify-center rounded-full bg-red-100">
                                <Trash2 className="h-5 w-5 text-red-600" />
                            </div>
                            <h3 className="text-lg font-semibold text-[#3c4f3d]">Delete Report</h3>
                        </div>
                        <p className="text-sm text-[#3c4f3d]/70 mb-6">
                            Are you sure you want to delete this saved report? This action cannot be undone.
                        </p>
                        <div className="flex gap-3 justify-end">
                            <Button
                                variant="outline"
                                size="sm"
                                onClick={() => setShowDeleteConfirm(false)}
                                className="cursor-pointer"
                                disabled={isDeleting}
                            >
                                Cancel
                            </Button>
                            <Button
                                variant="destructive"
                                size="sm"
                                onClick={confirmDelete}
                                disabled={isDeleting}
                                className="cursor-pointer bg-red-600 hover:bg-red-700 text-white"
                            >
                                {isDeleting ? 'Deleting...' : 'Delete'}
                            </Button>
                        </div>
                    </div>
                </div>
            )}

            {/* Success Toast with Confetti */}
            {successToast && (
                <div className="fixed inset-0 z-[70] pointer-events-none flex items-start justify-center pt-8">
                    {/* Confetti pieces */}
                    <div className="absolute inset-0 overflow-hidden">
                        {[...Array(30)].map((_, i) => (
                            <div
                                key={i}
                                className="absolute animate-confetti"
                                style={{
                                    left: `${Math.random() * 100}%`,
                                    top: '-10px',
                                    animationDelay: `${Math.random() * 0.5}s`,
                                    animationDuration: `${1.5 + Math.random()}s`,
                                }}
                            >
                                <div
                                    className="w-3 h-3 rotate-45"
                                    style={{
                                        backgroundColor: ['#ff6b6b', '#4ecdc4', '#ffe66d', '#95e1d3', '#f38181', '#aa96da', '#fcbad3', '#a8d8ea'][i % 8],
                                    }}
                                />
                            </div>
                        ))}
                    </div>

                    {/* Toast message */}
                    <div className="bg-white rounded-xl shadow-2xl px-6 py-4 flex items-center gap-3 border border-green-200 animate-bounce-in">
                        <span className="text-3xl">{successToast.icon}</span>
                        <div>
                            <p className="font-semibold text-[#3c4f3d]">{successToast.message}</p>
                            <p className="text-xs text-[#3c4f3d]/60">🎉 Success!</p>
                        </div>
                    </div>

                    {/* Inline CSS for animations */}
                    <style>{`
                        @keyframes confetti-fall {
                            0% {
                                transform: translateY(0) rotate(0deg);
                                opacity: 1;
                            }
                            100% {
                                transform: translateY(100vh) rotate(720deg);
                                opacity: 0;
                            }
                        }
                        @keyframes bounce-in {
                            0% {
                                transform: scale(0.3) translateY(-100px);
                                opacity: 0;
                            }
                            50% {
                                transform: scale(1.05) translateY(0);
                            }
                            70% {
                                transform: scale(0.95);
                            }
                            100% {
                                transform: scale(1);
                                opacity: 1;
                            }
                        }
                        .animate-confetti {
                            animation: confetti-fall linear forwards;
                        }
                        .animate-bounce-in {
                            animation: bounce-in 0.5s ease-out forwards;
                        }
                    `}</style>
                </div>
            )}
        </div>
    );
}

export type { SavedReport };
