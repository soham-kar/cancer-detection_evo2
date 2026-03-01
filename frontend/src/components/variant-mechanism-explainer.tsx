"use client";

import { useMemo } from "react";
import { inferMechanism } from "~/lib/variant-mechanism/engine";
import type { VariantContext, MechanismExplanation } from "~/lib/variant-mechanism/types";
import { getGeneData, genomicToAA, getDomainAtAA, getDomainsAfterAA } from "~/utils/domain-lookup";
import type { VEPAnnotation } from "~/app/api/vep/route";

interface Props {
    geneSymbol: string;
    chromosome: string;
    genomicPosition: number;
    reference: string;
    alternative: string;
    variantType: string;
    deltaScore: number;
    prediction: string;
    vepAnnotation?: VEPAnnotation | null;
}

export function VariantMechanismExplainer({
    geneSymbol,
    chromosome,
    genomicPosition,
    reference,
    alternative,
    variantType,
    deltaScore,
    prediction,
    vepAnnotation,
}: Props) {
    const explanation: MechanismExplanation = useMemo(() => {
        const geneData = getGeneData(geneSymbol);
        const proteinPosition = geneData ? genomicToAA(genomicPosition, geneData) : null;
        const hitDomain = (geneData && proteinPosition) ? getDomainAtAA(proteinPosition, geneData) : null;
        const domainsAfterPosition = (geneData && proteinPosition) ? getDomainsAfterAA(proteinPosition, geneData) : [];

        const ctx: VariantContext = {
            variantType,
            deltaScore,
            reference,
            alternative,
            geneSymbol,
            genomicPosition,
            chromosome,
            prediction,
            proteinPosition,
            hitDomain,
            domainsAfterPosition,
            vep: vepAnnotation ?? null,
        };
        return inferMechanism(ctx);
    }, [geneSymbol, genomicPosition, chromosome, reference, alternative, variantType, deltaScore, prediction, vepAnnotation]);

    const { title, confidence, primaryMechanism, molecularDetail, biologicalImpact, domainContext, comparisonNote } = explanation;

    const defaultConfColor = { bg: "bg-slate-800/60", border: "border-slate-600/50", title: "text-slate-300", badge: "bg-slate-700 text-slate-300", icon: "❓" };
    const confColor = {
        high: { bg: "bg-emerald-950/40", border: "border-emerald-700/50", title: "text-emerald-300", badge: "bg-emerald-800/60 text-emerald-200", icon: "💡" },
        medium: { bg: "bg-amber-950/40", border: "border-amber-700/50", title: "text-amber-300", badge: "bg-amber-800/60 text-amber-200", icon: "⚠️" },
        low: { bg: "bg-slate-800/60", border: "border-slate-600/50", title: "text-slate-300", badge: "bg-slate-700 text-slate-300", icon: "❓" },
    }[confidence] ?? defaultConfColor;

    return (
        <div className={`rounded-lg border ${confColor.bg} ${confColor.border} p-3.5 mt-3 space-y-2.5`}>
            {/* ── Header ── */}
            <div className="flex items-start justify-between gap-2">
                <div className={`text-xs font-semibold ${confColor.title} flex items-center gap-1.5`}>
                    <span>{confColor.icon}</span>
                    <span>{title}</span>
                </div>
                <span className={`text-[9px] px-1.5 py-0.5 rounded-full flex-shrink-0 font-medium ${confColor.badge}`}>
                    {confidence} confidence
                </span>
            </div>

            {/* ── Primary mechanism ── */}
            <p className="text-[11px] text-slate-300 leading-relaxed">
                {primaryMechanism}
            </p>

            {/* ── Molecular detail (codon / AA change) ── */}
            {molecularDetail && (
                <div className="rounded-md bg-slate-900/70 border border-slate-700/60 p-2.5 space-y-1.5">
                    <p className="text-[9px] font-semibold text-slate-400 uppercase tracking-wider">Molecular Change</p>
                    {molecularDetail.codonChange && (
                        <div className="flex items-center gap-2">
                            <span className="text-[10px] text-slate-500 w-14 flex-shrink-0">Codon</span>
                            <span className="font-mono text-[11px] text-slate-200 bg-slate-800/80 px-2 py-0.5 rounded">
                                {molecularDetail.codonChange}
                            </span>
                            <span className="text-[9px] text-slate-500 italic">lowercase = unchanged base</span>
                        </div>
                    )}
                    {molecularDetail.aminoAcidChange && (
                        <div className="flex items-center gap-2">
                            <span className="text-[10px] text-slate-500 w-14 flex-shrink-0">Protein</span>
                            <span className={`font-mono text-[11px] px-2 py-0.5 rounded ${molecularDetail.aaChange?.includes("=") || molecularDetail.aaChange?.includes("synonymous")
                                    ? "text-emerald-300 bg-emerald-900/40"
                                    : molecularDetail.aaChange?.includes("fs") || molecularDetail.aaChange?.includes("Ter")
                                        ? "text-red-300 bg-red-900/40"
                                        : "text-amber-300 bg-amber-900/40"
                                }`}>
                                {molecularDetail.aminoAcidChange}
                            </span>
                        </div>
                    )}
                    {molecularDetail.aaChange && (
                        <div className="flex items-center gap-2">
                            <span className="text-[10px] text-slate-500 w-14 flex-shrink-0">HGVS</span>
                            <span className="font-mono text-[10px] text-slate-400">{molecularDetail.aaChange}</span>
                        </div>
                    )}
                </div>
            )}

            {/* ── Biological impact ── */}
            <p className="text-[11px] text-slate-100 leading-relaxed font-medium">
                {biologicalImpact}
            </p>

            {/* ── Domain context ── */}
            {domainContext && (
                <div className="rounded-md bg-indigo-950/40 border border-indigo-700/40 px-2.5 py-2 space-y-1.5">
                    <p className="text-[10px] text-white leading-relaxed font-medium">
                        📍 {domainContext.sentence}
                    </p>
                    {domainContext.truncatedDomains.length > 0 && (
                        <div className="space-y-1">
                            <p className="text-[9px] font-bold text-red-200 uppercase tracking-wider">Domains Lost to Truncation</p>
                            {domainContext.truncatedDomains.map((d) => (
                                <div key={d.name} className="flex items-start gap-1.5">
                                    <span
                                        className="w-2 h-2 rounded-full flex-shrink-0 mt-0.5"
                                        style={{ backgroundColor: d.color }}
                                    />
                                    <div>
                                        <span className="text-[10px] font-semibold text-white">{d.name}</span>
                                        <span className="text-[9px] text-slate-200 ml-1">(aa {d.start}–{d.end})</span>
                                        <p className="text-[9px] text-slate-200 leading-relaxed">{d.description}</p>
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            )}

            {/* ── Comparison note ── */}
            {comparisonNote && (
                <div className="pt-0.5 border-t border-slate-700/50">
                    <p className="text-[10px] text-slate-500 italic">↔ {comparisonNote}</p>
                </div>
            )}
        </div>
    );
}
