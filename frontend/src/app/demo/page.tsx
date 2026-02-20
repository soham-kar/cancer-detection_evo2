"use client";

import { useSearchParams, useRouter } from "next/navigation";
import { useEffect, useState, Suspense } from "react";
import { Dna, FileText, ArrowLeft, ExternalLink, CheckCircle, AlertTriangle, Beaker, Sparkles, ArrowRight } from "lucide-react";
import { Button } from "~/components/ui/button";
import ScoreGauge from "~/components/ScoreGauge";
import TriModalGrid from "~/components/TriModalGrid";
import MetadataSidebar from "~/components/MetadataSidebar";

interface DemoResult {
    success: boolean;
    isDemo: boolean;
    variant: {
        gene: string;
        variant: string;
        chromosome: string;
        position: number;
    };
    prediction: string;
    delta_score: number;
    classification_confidence: number;
    rag: {
        summary: string;
        clinvar_status: string;
        pmids: string[];
        protein_function: string;
        sources: {
            clinvar: { status: string; id: string };
            uniprot: { id: string; has_function: boolean };
            pubmed: { count: number; level: string };
        };
    } | null;
}

// Demo variant cards with clinical summaries
const DEMO_VARIANTS = [
    {
        id: "brca1",
        gene: "BRCA1",
        variant: "c.68_69delAG",
        description: "Hereditary Breast & Ovarian Cancer",
        color: "from-pink-500 to-rose-600",
        bgColor: "bg-pink-50",
        textColor: "text-pink-700",
        clinvarStatus: "Pathogenic",
        clinvarStatusColor: "bg-red-100 text-red-800",
        proteinFunction: "DNA repair via BRCA1/BARD1 complex",
        clinicalSummary: "Disrupts DNA repair → Increased breast/ovarian cancer risk. Ashkenazi Jewish founder mutation.",
        keyFindings: ["DNA repair loss", "Cancer risk", "Founder mutation"],
    },
    {
        id: "tp53",
        gene: "TP53",
        variant: "R248W",
        description: "Li-Fraumeni Syndrome",
        color: "from-purple-500 to-indigo-600",
        bgColor: "bg-purple-50",
        textColor: "text-purple-700",
        clinvarStatus: "Likely Pathogenic",
        clinvarStatusColor: "bg-orange-100 text-orange-800",
        proteinFunction: "Tumor suppressor (cell cycle arrest)",
        clinicalSummary: "DNA-binding domain disruption → Loss of tumor suppressor function.",
        keyFindings: ["Tumor suppressor loss", "Multi-cancer"],
    },
];

function DemoContent() {
    const searchParams = useSearchParams();
    const router = useRouter();
    const [phase, setPhase] = useState<"select" | "loading" | "result">("select");
    const [selectedVariant, setSelectedVariant] = useState<string | null>(null);
    const [result, setResult] = useState<DemoResult | null>(null);
    const [error, setError] = useState<string | null>(null);

    // Check if variant was passed in URL
    const urlVariant = searchParams.get("variant");

    useEffect(() => {
        if (urlVariant && (urlVariant === "brca1" || urlVariant === "tp53")) {
            runAnalysis(urlVariant);
        }
    }, [urlVariant]);

    const runAnalysis = async (variantId: string) => {
        setSelectedVariant(variantId);
        setPhase("loading");
        setError(null);

        try {
            const response = await fetch("/api/demo-analyze", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ variantId }),
            });

            if (!response.ok) {
                throw new Error("Demo analysis failed");
            }

            const data = await response.json();
            setResult(data);
            setPhase("result");
        } catch (err) {
            setError(err instanceof Error ? err.message : "Analysis failed");
            setPhase("select");
        }
    };

    // =========================================================================
    // PHASE 1: VARIANT SELECTION
    // =========================================================================
    if (phase === "select") {
        return (
            <div className="flex min-h-screen items-center justify-center overflow-hidden relative animate-fadeIn">
                {/* Background */}
                <div
                    className="absolute inset-0 bg-cover bg-center bg-no-repeat"
                    style={{
                        backgroundImage: 'url("/Gemini_Generated_Image_953ram953ram953r.png")',
                        opacity: 0.85,
                    }}
                />
                <div className="absolute inset-0 bg-gradient-to-br from-[#e9eeea]/35 via-[#d4ddd6]/30 to-[#c5d1c7]/35" />

                <div className="relative w-full max-w-2xl px-4">
                    {/* Header */}
                    <div className="text-center mb-8">
                        <div className="flex items-center justify-center gap-3 mb-4">
                            <Dna className="h-12 w-12 text-[#de8246]" />
                            <h1 className="text-4xl font-bold text-[#3c4f3d]">HelixMind</h1>
                        </div>
                        <p className="text-lg text-[#3c4f3d]/70 mb-2">Guest Demo Mode</p>
                        <p className="text-sm text-[#3c4f3d]/50">
                            Select a variant to analyze with our Tri-Modal RAG
                        </p>
                    </div>

                    {/* Variant Cards with Clinical Summaries */}
                    <div className="grid grid-cols-1 gap-10 mb-10">
                        {DEMO_VARIANTS.map((variant) => (
                            <button
                                key={variant.id}
                                onClick={() => runAnalysis(variant.id)}
                                className={`group relative overflow-hidden rounded-2xl p-6 text-left transition-all duration-300 hover:scale-[1.01] hover:shadow-2xl bg-white border border-slate-200 shadow-lg`}
                            >
                                {/* Gradient overlay on hover */}
                                <div className={`absolute inset-0 bg-gradient-to-br ${variant.color} opacity-0 group-hover:opacity-5 transition-opacity`} />

                                <div className="relative">
                                    {/* Header row */}
                                    <div className="flex items-start justify-between mb-4">
                                        <div>
                                            <div className="flex items-center gap-3 mb-2">
                                                <h3 className="text-2xl font-bold text-slate-900">
                                                    {variant.gene}
                                                </h3>
                                                <span className="font-mono text-sm text-slate-500 bg-slate-100 px-2 py-0.5 rounded">
                                                    {variant.variant}
                                                </span>
                                            </div>
                                            <p className="text-sm text-slate-500">
                                                {variant.description}
                                            </p>
                                        </div>
                                        <span className={`px-3 py-1 rounded-full text-sm font-semibold ${variant.clinvarStatusColor}`}>
                                            {variant.clinvarStatus}
                                        </span>
                                    </div>

                                    {/* Clinical Summary Section */}
                                    <div className="bg-slate-50 rounded-lg p-4 mb-4">
                                        <div className="flex items-start gap-3">
                                            <FileText className="h-5 w-5 text-slate-400 flex-shrink-0 mt-0.5" />
                                            <div>
                                                <p className="text-xs font-medium text-slate-500 uppercase tracking-wider mb-1">
                                                    Clinical Summary (Tri-Modal RAG)
                                                </p>
                                                <p className="text-sm text-slate-700 leading-relaxed">
                                                    {variant.clinicalSummary}
                                                </p>
                                            </div>
                                        </div>
                                    </div>

                                    {/* Protein Function */}
                                    <div className="mb-4">
                                        <p className="text-xs font-medium text-slate-500 uppercase tracking-wider mb-1">
                                            Protein Function (UniProt)
                                        </p>
                                        <p className="text-sm text-slate-600">
                                            {variant.proteinFunction}
                                        </p>
                                    </div>

                                    {/* Key Findings Tags */}
                                    <div className="flex flex-wrap gap-2 mb-4">
                                        {variant.keyFindings.map((finding, idx) => (
                                            <span
                                                key={idx}
                                                className={`px-2 py-1 rounded text-xs font-medium ${variant.bgColor} ${variant.textColor}`}
                                            >
                                                {finding}
                                            </span>
                                        ))}
                                    </div>

                                    {/* CTA Button */}
                                    <div className={`inline-flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-semibold text-white bg-gradient-to-r ${variant.color} shadow-md group-hover:shadow-lg transition-shadow`}>
                                        <Sparkles className="h-4 w-4" />
                                        Run Full Analysis
                                        <ArrowRight className="h-4 w-4" />
                                    </div>
                                </div>
                            </button>
                        ))}
                    </div>

                    {/* Sign up CTA */}
                    <div className="text-center">
                        <p className="text-sm text-[#3c4f3d]/50 mb-3">
                            Want to analyze any variant?
                        </p>
                        <Button
                            onClick={() => router.push("/sign-up")}
                            variant="outline"
                            className="border-[#3c4f3d]/30 hover:bg-[#3c4f3d]/5"
                        >
                            Create Free Account
                        </Button>
                    </div>

                    {/* Error message */}
                    {error && (
                        <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg text-center">
                            <p className="text-sm text-red-700">{error}</p>
                        </div>
                    )}
                </div>
            </div>
        );
    }

    // =========================================================================
    // PHASE 2: LOADING (Branded Splash)
    // =========================================================================
    if (phase === "loading") {
        const variant = DEMO_VARIANTS.find(v => v.id === selectedVariant);
        return (
            <div className="flex min-h-screen items-center justify-center overflow-hidden relative animate-fadeIn">
                {/* Background */}
                <div
                    className="absolute inset-0 bg-cover bg-center bg-no-repeat"
                    style={{
                        backgroundImage: 'url("/Gemini_Generated_Image_953ram953ram953r.png")',
                        opacity: 0.85,
                    }}
                />
                <div className="absolute inset-0 bg-gradient-to-br from-[#e9eeea]/35 via-[#d4ddd6]/30 to-[#c5d1c7]/35" />

                <div className="relative text-center">
                    {/* Big Logo with animation */}
                    <div className="flex items-center justify-center gap-4 mb-6">
                        <div className="relative">
                            <Dna
                                className="h-20 w-20 text-[#de8246]"
                                style={{
                                    animation: 'breathe 2s ease-in-out infinite',
                                }}
                            />
                            <style jsx global>{`
                                @keyframes breathe {
                                    0%, 100% {
                                        transform: scale(1);
                                        filter: drop-shadow(0 0 8px rgba(222, 130, 70, 0.4));
                                    }
                                    50% {
                                        transform: scale(1.1);
                                        filter: drop-shadow(0 0 20px rgba(222, 130, 70, 0.6));
                                    }
                                }
                            `}</style>
                            <div className="absolute inset-0 h-20 w-20 rounded-full bg-[#de8246]/20 blur-2xl animate-pulse" />
                        </div>
                        <h1 className="text-5xl font-bold text-[#3c4f3d]">
                            HelixMind
                        </h1>
                    </div>

                    {/* Status text */}
                    <p className="text-xl text-[#3c4f3d]/80 mb-2">
                        Analyzing <span className="font-mono font-bold text-[#de8246]">{variant?.gene} {variant?.variant}</span>
                    </p>
                    <p className="text-sm text-[#3c4f3d]/60 mb-8">
                        Fetching evidence from ClinVar, UniProt, and PubMed
                    </p>

                    {/* Bouncing dots */}
                    <div className="flex justify-center gap-2">
                        <div className="h-3 w-3 rounded-full bg-[#de8246] animate-bounce" style={{ animationDelay: '0ms' }} />
                        <div className="h-3 w-3 rounded-full bg-[#de8246] animate-bounce" style={{ animationDelay: '150ms' }} />
                        <div className="h-3 w-3 rounded-full bg-[#de8246] animate-bounce" style={{ animationDelay: '300ms' }} />
                    </div>

                    {/* Progress hint */}
                    <div className="mt-8 flex justify-center gap-6 text-xs text-[#3c4f3d]/50">
                        <span className="flex items-center gap-1">
                            <span className="w-2 h-2 bg-blue-400 rounded-full animate-pulse" />
                            ClinVar
                        </span>
                        <span className="flex items-center gap-1">
                            <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse" style={{ animationDelay: '100ms' }} />
                            UniProt
                        </span>
                        <span className="flex items-center gap-1">
                            <span className="w-2 h-2 bg-purple-400 rounded-full animate-pulse" style={{ animationDelay: '200ms' }} />
                            PubMed
                        </span>
                    </div>
                </div>
            </div>
        );
    }

    // =========================================================================
    // PHASE 3: RESULTS
    // =========================================================================
    if (!result) {
        return (
            <div className="min-h-screen bg-slate-50 flex flex-col items-center justify-center p-4">
                <div className="bg-white rounded-xl shadow-lg p-8 max-w-md text-center">
                    <AlertTriangle className="h-12 w-12 text-amber-500 mx-auto mb-4" />
                    <h2 className="text-xl font-semibold text-slate-800 mb-2">Analysis Failed</h2>
                    <p className="text-slate-600 mb-6">{error || "Unable to complete demo analysis"}</p>
                    <Button onClick={() => setPhase("select")} variant="outline">
                        <ArrowLeft className="mr-2 h-4 w-4" />
                        Try Again
                    </Button>
                </div>
            </div>
        );
    }

    const isPredictionPathogenic = result.prediction.toLowerCase().includes("pathogenic");

    return (
        <div className="min-h-screen bg-gradient-to-b from-slate-50 to-white animate-fadeIn">
            {/* Header */}
            <header className="border-b border-slate-200 bg-white sticky top-0 z-10">
                <div className="container mx-auto flex items-center justify-between px-6 py-4">
                    <div className="flex items-center gap-3">
                        <Dna className="h-7 w-7 text-[#de8246]" />
                        <h1 className="text-xl font-semibold text-[#3c4f3d]">
                            Helix<span className="text-[#de8246]">Mind</span>
                            <span className="ml-2 text-xs bg-amber-100 text-amber-800 px-2 py-0.5 rounded-full font-medium">
                                DEMO
                            </span>
                        </h1>
                    </div>
                    <div className="flex items-center gap-3">
                        <Button variant="outline" onClick={() => setPhase("select")}>
                            <ArrowLeft className="mr-2 h-4 w-4" />
                            Try Another
                        </Button>
                        <Button onClick={() => router.push("/sign-up")} className="bg-[#de8246] hover:bg-[#c97339]">
                            Sign Up for Full Access
                        </Button>
                    </div>
                </div>
            </header>

            <main className="container mx-auto px-6 py-8">
                {/* Two-column layout: Main + Sidebar */}
                <div className="flex flex-col lg:flex-row gap-6">
                    {/* Main Content */}
                    <div className="flex-1 space-y-6">
                        {/* Variant Header - Compact */}
                        <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
                            <div className="flex items-center gap-3 mb-2">
                                <Beaker className="h-5 w-5 text-amber-500" />
                                <span className="text-xs font-bold text-amber-600 uppercase tracking-wide">Demo Analysis</span>
                            </div>
                            <h2 className="text-3xl font-bold text-slate-900">
                                {result.variant.gene}{" "}
                                <span className="font-mono text-[#de8246]">{result.variant.variant}</span>
                            </h2>
                        </div>

                        {/* 1. Score Gauge */}
                        <ScoreGauge
                            score={result.delta_score}
                            confidence={result.classification_confidence}
                            prediction={result.prediction}
                        />

                        {/* 2. Tri-Modal Evidence Grid */}
                        {result.rag && (
                            <TriModalGrid
                                clinvarText={`${result.rag.clinvar_status}. Clinical variant.`}
                                clinvarStatus={result.rag.clinvar_status}
                                uniprotText={result.rag.protein_function || "Protein function data from UniProtKB."}
                                uniprotId={result.rag.sources?.uniprot?.id}
                                pubmedText={`${result.rag.pmids?.length || 0} publications`}
                                pubmedCount={result.rag.sources?.pubmed?.count || result.rag.pmids?.length || 0}
                                pmids={result.rag.pmids}
                            />
                        )}

                        {/* 3. Full Synthesized Report */}
                        {result.rag && (
                            <div className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
                                <div className="border-b border-slate-200 bg-gradient-to-r from-slate-800 to-slate-900 px-6 py-4">
                                    <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                                        <FileText className="h-5 w-5 text-[#de8246]" />
                                        Llama-3.3-70B Synthesized Report
                                    </h3>
                                    <p className="text-sm text-slate-400 mt-1">
                                        AI-generated clinical narrative from tri-modal evidence
                                    </p>
                                </div>

                                <div className="p-8 space-y-4">
                                    {/* Parse markdown sections */}
                                    {result.rag.summary.split('\n\n').filter(Boolean).map((section, i) => {
                                        // Split title and body
                                        const parts = section.split(':');
                                        const title = (parts[0] || '').replace(/\*\*/g, '').trim();
                                        const body = parts.slice(1).join(':').replace(/\*\*/g, '').trim();

                                        if (!body) return null;

                                        // Generate section ID based on title
                                        let sectionId = "";
                                        if (title.toLowerCase().includes("clinical")) sectionId = "clinical-section";
                                        if (title.toLowerCase().includes("mechanism") || title.toLowerCase().includes("biological")) sectionId = "mechanism-section";
                                        if (title.toLowerCase().includes("evidence") || title.toLowerCase().includes("implication")) sectionId = "evidence-section";

                                        // Assign colors based on section type
                                        const dotColors = ['bg-blue-500', 'bg-purple-500', 'bg-amber-500', 'bg-emerald-500', 'bg-rose-500'];
                                        const dotColor = dotColors[i % dotColors.length];

                                        return (
                                            <div
                                                key={i}
                                                id={sectionId}
                                                className="bg-slate-50 p-4 rounded-lg border border-slate-100 scroll-mt-24 transition-all duration-500"
                                            >
                                                <h3 className="font-bold text-slate-900 mb-2 flex items-center gap-2">
                                                    <span className={`w-2 h-2 rounded-full ${dotColor}`}></span>
                                                    {title}
                                                </h3>
                                                <p className="text-slate-700 leading-relaxed text-sm">
                                                    {body}
                                                </p>
                                            </div>
                                        );
                                    })}
                                </div>

                                {/* References / PMID Links - Always show this section */}
                                <div className="pt-6 mt-4 mx-8 mb-8 border-t border-slate-100">
                                    <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3">
                                        References
                                    </h4>
                                    <div className="flex flex-wrap gap-2 items-center">
                                        {result.rag.pmids && result.rag.pmids.length > 0 ? (
                                            <>
                                                {result.rag.pmids.map((pmid: string) => (
                                                    <a
                                                        key={pmid}
                                                        href={`https://pubmed.ncbi.nlm.nih.gov/${pmid}/`}
                                                        target="_blank"
                                                        rel="noopener noreferrer"
                                                        className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-white border border-slate-200 rounded-md text-xs font-medium text-slate-600 hover:text-indigo-600 hover:border-indigo-200 transition-colors shadow-sm"
                                                    >
                                                        <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                                                        </svg>
                                                        PMID:{pmid}
                                                    </a>
                                                ))}
                                                {/* Safety Net Link */}
                                                <a
                                                    href={`https://pubmed.ncbi.nlm.nih.gov/?term=${result.variant.gene}+${encodeURIComponent(result.variant.variant)}`}
                                                    target="_blank"
                                                    rel="noopener noreferrer"
                                                    className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-indigo-50 border border-indigo-100 rounded-md text-xs font-medium text-indigo-700 hover:bg-indigo-100 transition-colors"
                                                >
                                                    See all on PubMed ↗
                                                </a>
                                            </>
                                        ) : (
                                            <>
                                                <span className="text-xs text-slate-400 italic">
                                                    No direct PMIDs returned
                                                </span>
                                                <a
                                                    href={`https://pubmed.ncbi.nlm.nih.gov/?term=${result.variant.gene}+${encodeURIComponent(result.variant.variant)}`}
                                                    target="_blank"
                                                    rel="noopener noreferrer"
                                                    className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-indigo-50 border border-indigo-100 rounded-md text-xs font-medium text-indigo-700 hover:bg-indigo-100 transition-colors"
                                                >
                                                    Search PubMed ↗
                                                </a>
                                            </>
                                        )}
                                    </div>
                                </div>
                            </div>
                        )}

                        {/* Fallback when RAG is not available */}
                        {!result.rag && (
                            <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-8 text-center">
                                <AlertTriangle className="h-12 w-12 text-amber-500 mx-auto mb-4" />
                                <h4 className="text-lg font-medium text-slate-800 mb-2">RAG Pipeline Loading...</h4>
                                <p className="text-sm text-slate-600 mb-4">
                                    The Tri-Modal RAG service is warming up. This happens on first request.
                                </p>
                                <Button
                                    onClick={() => runAnalysis(selectedVariant || "brca1")}
                                    className="bg-[#de8246] hover:bg-[#c97339] text-white"
                                >
                                    <Sparkles className="mr-2 h-4 w-4" />
                                    Retry Analysis
                                </Button>
                            </div>
                        )}
                    </div>

                    {/* 3. Metadata Sidebar */}
                    <div className="lg:w-72 shrink-0">
                        <MetadataSidebar
                            gene={result.variant.gene}
                            variant={result.variant.variant}
                            chromosome={result.variant.chromosome}
                            position={result.variant.position}
                            genomeBuild="GRCh38 / hg38"
                            transcript={result.variant.gene === "BRCA1" ? "NM_007294.4" : "NM_000546.6"}
                            dbSnpId={result.variant.gene === "BRCA1" ? "rs80357713" : "rs28934578"}
                        />
                    </div>
                </div>

                {/* CTA */}
                <div className="mt-8 bg-gradient-to-r from-[#de8246] to-[#e69a5c] rounded-xl p-6 text-center text-white">
                    <h3 className="text-xl font-semibold mb-2">Ready for Full Access?</h3>
                    <p className="text-white/80 mb-4">
                        Analyze any variant with real-time Evo-2 scoring, gnomAD frequencies, and ACMG evidence.
                    </p>
                    <Button
                        onClick={() => router.push("/sign-up")}
                        className="bg-white text-[#de8246] hover:bg-orange-50"
                    >
                        Create Free Account
                    </Button>
                </div>
            </main>

            {/* Footer */}
            <footer className="border-t border-slate-200 bg-white mt-12 py-4">
                <div className="container mx-auto px-6 text-center text-xs text-slate-400">
                    Generated by HelixMind | Evo2-7B + Llama-3.3-70B | FOR RESEARCH USE ONLY
                </div>
            </footer>
        </div>
    );
}

export default function DemoPage() {
    return (
        <Suspense fallback={
            <div className="min-h-screen bg-slate-50 flex items-center justify-center">
                <div className="h-12 w-12 animate-spin rounded-full border-4 border-slate-200 border-t-[#de8246]"></div>
            </div>
        }>
            <DemoContent />
        </Suspense>
    );
}
