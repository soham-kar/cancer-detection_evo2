"use client";

import { Activity, Database, BookOpen, ExternalLink } from "lucide-react";

// Helper: Removes "(PubMed:12345...)" clutter from UniProt text
const cleanUniProtText = (text: string) => {
    if (!text) return "Function data unavailable.";
    // Remove stuff in parentheses like (PubMed:1234)
    const noParens = text.replace(/\(PubMed:[^)]+\)/g, "");
    // Remove trailing PubMed refs
    const clean = noParens.replace(/PubMed:\d+/g, "").trim();
    // Truncate to first sentence or 80 chars
    const firstSentence = clean.split('.')[0] || '';
    return firstSentence.slice(0, 80).trim() + (firstSentence.length > 80 ? "..." : ".");
};

// Smooth scroll helper with highlight flash
const scrollToSection = (id: string) => {
    const element = document.getElementById(id);
    if (element) {
        element.scrollIntoView({ behavior: 'smooth', block: 'start' });
        // Flash the section to highlight it
        element.classList.add('bg-indigo-50', 'ring-2', 'ring-indigo-200');
        setTimeout(() => {
            element.classList.remove('bg-indigo-50', 'ring-2', 'ring-indigo-200');
        }, 1500);
    }
};

interface TriModalGridProps {
    clinvarText: string;
    clinvarStatus: string;
    uniprotText: string;
    uniprotId?: string;
    pubmedText: string;
    pubmedCount: number;
    pmids?: string[];
}

export default function TriModalGrid({
    clinvarStatus,
    uniprotText,
    uniprotId,
    pubmedCount,
    pmids = [],
}: TriModalGridProps) {
    const isPathogenic = clinvarStatus?.toLowerCase().includes("pathogenic");

    return (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 items-stretch py-8">
            {/* Card 1: ClinVar -> Scrolls to Clinical Section */}
            <div
                onClick={() => scrollToSection('clinical-section')}
                className="bg-white p-5 rounded-xl border-l-4 border-blue-500 shadow-sm flex flex-col h-full hover:shadow-md hover:bg-slate-50 transition-all cursor-pointer group"
            >
                <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2 text-blue-700 font-bold text-xs uppercase tracking-wider group-hover:text-blue-800">
                        <Activity className="w-4 h-4" /> ClinVar
                    </div>
                    <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${isPathogenic
                            ? "bg-rose-100 text-rose-600"
                            : "bg-emerald-100 text-emerald-600"
                        }`}>
                        Official
                    </span>
                </div>
                {/* Main Text: Big classification */}
                <div className={`text-2xl font-bold mt-auto ${isPathogenic ? "text-rose-700" : "text-emerald-700"}`}>
                    {clinvarStatus || "Not Listed"}
                </div>
                <p className="text-xs text-slate-400 mt-1 group-hover:text-slate-500">Click to see details ↓</p>
            </div>

            {/* Card 2: UniProt -> Scrolls to Mechanism Section */}
            <div
                onClick={() => scrollToSection('mechanism-section')}
                className="bg-white p-5 rounded-xl border-l-4 border-purple-500 shadow-sm flex flex-col h-full hover:shadow-md hover:bg-slate-50 transition-all cursor-pointer group"
            >
                <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2 text-purple-700 font-bold text-xs uppercase tracking-wider group-hover:text-purple-800">
                        <Database className="w-4 h-4" /> Mechanism
                    </div>
                    {uniprotId && (
                        <a
                            href={`https://www.uniprot.org/uniprotkb/${uniprotId}`}
                            target="_blank"
                            rel="noopener noreferrer"
                            onClick={(e) => e.stopPropagation()}
                            className="text-xs text-purple-600 hover:underline flex items-center gap-1"
                        >
                            {uniprotId} <ExternalLink className="w-3 h-3" />
                        </a>
                    )}
                </div>
                {/* Main Text: Clamped to 3 lines max */}
                <div className="text-sm text-slate-700 leading-snug line-clamp-3 mt-auto">
                    {cleanUniProtText(uniprotText)}
                </div>
                <p className="text-xs text-slate-400 mt-2 group-hover:text-slate-500">Click to see details ↓</p>
            </div>

            {/* Card 3: PubMed -> Scrolls to Evidence Section */}
            <div
                onClick={() => scrollToSection('evidence-section')}
                className="bg-white p-5 rounded-xl border-l-4 border-amber-500 shadow-sm flex flex-col h-full hover:shadow-md hover:bg-slate-50 transition-all cursor-pointer group"
            >
                <div className="flex items-center gap-2 mb-3 text-amber-700 font-bold text-xs uppercase tracking-wider group-hover:text-amber-800">
                    <BookOpen className="w-4 h-4" /> Evidence
                </div>
                {/* Main Text: Big Number focus */}
                <div className="text-2xl font-bold text-slate-800 mt-auto flex items-baseline gap-2">
                    {pmids?.length || pubmedCount || 0}
                    <span className="text-sm font-normal text-slate-500">Key Papers</span>
                </div>
                {/* Tiny pill citations */}
                {pmids && pmids.length > 0 && (
                    <div className="flex gap-1 mt-2 flex-wrap">
                        {pmids.slice(0, 3).map((pmid: string) => (
                            <a
                                key={pmid}
                                href={`https://pubmed.ncbi.nlm.nih.gov/${pmid}/`}
                                target="_blank"
                                rel="noopener noreferrer"
                                onClick={(e) => e.stopPropagation()}
                                className="text-[10px] bg-amber-50 text-amber-700 px-1.5 py-0.5 rounded border border-amber-100 hover:bg-amber-100 transition-colors"
                            >
                                PMID:{pmid}
                            </a>
                        ))}
                    </div>
                )}
                <p className="text-xs text-slate-400 mt-2 group-hover:text-slate-500">Click to see details ↓</p>
            </div>
        </div>
    );
}
