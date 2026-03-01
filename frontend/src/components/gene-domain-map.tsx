"use client";

import { useEffect, useState } from "react";

// ── Static domain data for top 5 cancer genes ─────────────────────────────
// Positions are in amino acid coordinates (UniProt canonical isoform)
// Genomic position → AA uses approximate linear mapping over the CDS

interface Domain {
    name: string;
    start: number; // aa
    end: number;   // aa
    color: string;
    description: string;
}

interface GeneStaticData {
    totalAA: number;        // protein length in aa
    genomicCDSStart: number; // hg38 start of CDS (approx)
    genomicCDSEnd: number;   // hg38 end of CDS (approx)
    strand: '+' | '-';       // DNA strand: plus or minus
    domains: Domain[];
    uniprotId: string;
}

const GENE_DB: Record<string, GeneStaticData> = {
    BRCA1: {
        totalAA: 1863,
        genomicCDSStart: 43044295,
        genomicCDSEnd: 43125364,
        strand: '-',
        uniprotId: "P38398",
        domains: [
            { name: "RING finger", start: 24, end: 65, color: "#ef4444", description: "E3 ubiquitin ligase activity; BARD1 interaction; most pathogenic missense mutations cluster here" },
            { name: "Coiled-coil", start: 466, end: 932, color: "#f59e0b", description: "Mediates protein-protein interactions; binds PALB2 and other HR repair partners; central to BRCA1 complex assembly" },
            { name: "BRCT (I)", start: 1646, end: 1736, color: "#3b82f6", description: "Binds phosphoproteins; phospho-BACH1/ABRAXAS recognition; DNA damage response" },
            { name: "BRCT (II)", start: 1756, end: 1855, color: "#6366f1", description: "Tandem BRCT repeat; key for tumor suppression; p53 and CtIP interaction" },
        ],
    },
    BRCA2: {
        totalAA: 3418,
        genomicCDSStart: 32315474,
        genomicCDSEnd: 32400268,
        strand: '-',
        uniprotId: "P51587",
        domains: [
            { name: "PALB2 binding", start: 10, end: 40, color: "#ef4444", description: "Interacts with PALB2 for nuclear localisation" },
            { name: "OB fold (F1)", start: 2402, end: 2667, color: "#f59e0b", description: "ssDNA binding" },
            { name: "OB fold (F2)", start: 2670, end: 2803, color: "#f59e0b", description: "ssDNA binding" },
            { name: "OB fold (F3)", start: 2809, end: 3102, color: "#f59e0b", description: "ssDNA binding" },
            { name: "Tower", start: 2810, end: 2872, color: "#22c55e", description: "Interacts with RAD51" },
        ],
    },
    TP53: {
        totalAA: 393,
        genomicCDSStart: 7668421,
        genomicCDSEnd: 7687550,
        strand: '-',
        uniprotId: "P04637",
        domains: [
            { name: "Transactivation I", start: 1, end: 40, color: "#f59e0b", description: "MDM2 binding; p300 interaction" },
            { name: "Transactivation II", start: 40, end: 67, color: "#f59e0b", description: "Secondary transactivation domain" },
            { name: "Proline-rich", start: 67, end: 98, color: "#22c55e", description: "Apoptosis regulation" },
            { name: "DNA binding", start: 102, end: 292, color: "#ef4444", description: "Core domain; 90% of cancer mutations" },
            { name: "Tetramerization", start: 323, end: 356, color: "#3b82f6", description: "Forms functional p53 tetramer" },
            { name: "Regulatory", start: 364, end: 393, color: "#6366f1", description: "Post-translational modification hub" },
        ],
    },
    PTEN: {
        totalAA: 403,
        genomicCDSStart: 89622870,
        genomicCDSEnd: 89731687,
        strand: '+',
        uniprotId: "P60484",
        domains: [
            { name: "PIP2 binding", start: 1, end: 15, color: "#f59e0b", description: "Membrane targeting" },
            { name: "Phosphatase", start: 14, end: 185, color: "#ef4444", description: "Catalytic domain; dephosphorylates PIP3" },
            { name: "C2", start: 186, end: 351, color: "#3b82f6", description: "Membrane binding, calcium-independent" },
            { name: "PDZ-binding", start: 401, end: 403, color: "#22c55e", description: "Interactions with PDZ domain proteins" },
        ],
    },
    ATM: {
        totalAA: 3056,
        genomicCDSStart: 108222832,
        genomicCDSEnd: 108369102,
        strand: '+',
        uniprotId: "Q13315",
        domains: [
            { name: "FAT", start: 1960, end: 2566, color: "#f59e0b", description: "FRAP-ATM-TRRAP domain; regulatory" },
            { name: "Kinase", start: 2712, end: 3011, color: "#ef4444", description: "PI3K-like kinase; phosphorylates H2AX, BRCA1" },
            { name: "FATC", start: 3024, end: 3056, color: "#6366f1", description: "C-terminal; required for kinase activity" },
        ],
    },
};

// ── UniProt fallback ───────────────────────────────────────────────────────
interface UniProtFeature {
    type: string;
    location: { start: { value: number }; end: { value: number } };
    description?: string;
}

async function fetchUniprotDomains(geneSymbol: string): Promise<{
    domains: Domain[];
    totalAA: number;
    uniprotId: string;
} | null> {
    try {
        // Search UniProt for the human gene
        const searchRes = await fetch(
            `https://rest.uniprot.org/uniprotkb/search?query=gene:${encodeURIComponent(geneSymbol)}+AND+organism_id:9606+AND+reviewed:true&fields=accession,sequence,ft_domain,ft_region,length&format=json&size=1`
        );
        if (!searchRes.ok) return null;
        const searchData = await searchRes.json();
        const entry = searchData?.results?.[0];
        if (!entry) return null;

        const uniprotId: string = entry.primaryAccession ?? "?";
        const totalAA: number = entry.sequence?.length ?? 500;
        const features: UniProtFeature[] = entry.features ?? [];

        const palette = ["#ef4444", "#f59e0b", "#22c55e", "#3b82f6", "#6366f1", "#ec4899", "#14b8a6"];
        const domains: Domain[] = features
            .filter((f) => f.type === "Domain" || f.type === "Region")
            .slice(0, 8)
            .map((f, i) => ({
                name: f.description ?? f.type,
                start: f.location.start.value,
                end: f.location.end.value,
                color: palette[i % palette.length]!,
                description: `UniProt annotation: ${f.type}`,
            }));

        return { domains, totalAA, uniprotId };
    } catch {
        return null;
    }
}

// ── Helper: map genomic position → approximate AA position ────────────────
// Accounts for plus and minus strand orientations
function genomicToAA(pos: number, geneData: GeneStaticData): number | null {
    const cdsLen = geneData.genomicCDSEnd - geneData.genomicCDSStart;
    if (cdsLen <= 0) return null;
    
    let fraction: number;
    if (geneData.strand === '+') {
        // Plus strand: position increases left to right
        fraction = (pos - geneData.genomicCDSStart) / cdsLen;
    } else {
        // Minus strand: position mapping is reversed
        fraction = 1 - (pos - geneData.genomicCDSStart) / cdsLen;
    }
    
    // Clamp fraction to [0, 1]
    fraction = Math.max(0, Math.min(1, fraction));
    
    const aa = Math.round(fraction * geneData.totalAA);
    return Math.max(1, Math.min(geneData.totalAA, aa));
}

// ── Component ─────────────────────────────────────────────────────────────
interface GeneDomainMapProps {
    geneSymbol: string;
    genomicPosition: number;
    chromosome?: string;
    prediction?: string;  // e.g. "Likely pathogenic" — used to contextualise the domain note
}

export function GeneDomainMap({ geneSymbol, genomicPosition, chromosome, prediction }: GeneDomainMapProps) {
    const gene = geneSymbol.toUpperCase();
    const isPredPathogenic = (prediction ?? "").toLowerCase().includes("pathogenic");
    const isPredBenign = (prediction ?? "").toLowerCase().includes("benign");
    const staticData = GENE_DB[gene];

    const [uniprotData, setUniprotData] = useState<{
        domains: Domain[];
        totalAA: number;
        uniprotId: string;
    } | null>(null);
    const [loading, setLoading] = useState(!staticData);
    const [failed, setFailed] = useState(false);

    useEffect(() => {
        if (staticData) return; // use the static DB
        setLoading(true);
        fetchUniprotDomains(geneSymbol).then((data) => {
            setLoading(false);
            if (data) setUniprotData(data);
            else setFailed(true);
        });
    }, [geneSymbol, staticData]);

    if (loading) {
        return (
            <div className="rounded-md bg-white border border-indigo-100 p-3">
                <div className="text-xs text-indigo-600 animate-pulse">🔬 Fetching {geneSymbol} protein domains from UniProt…</div>
            </div>
        );
    }

    const domains = staticData?.domains ?? uniprotData?.domains ?? [];
    const totalAA = staticData?.totalAA ?? uniprotData?.totalAA ?? 500;
    const uniprotId = staticData?.uniprotId ?? uniprotData?.uniprotId ?? null;

    if (failed || domains.length === 0) {
        return (
            <div className="rounded-md bg-white border border-indigo-100 p-3">
                <div className="text-xs font-medium text-indigo-800 mb-1">🔬 Protein Domain Map</div>
                <p className="text-[10px] text-indigo-600/70">
                    Domain architecture not available for <span className="font-semibold">{geneSymbol}</span>.{" "}
                    <a
                        href={`https://www.uniprot.org/uniprotkb?query=${geneSymbol}+AND+human&fields=accession`}
                        target="_blank" rel="noopener noreferrer"
                        className="underline text-indigo-500 hover:text-indigo-700"
                    >
                        View on UniProt
                    </a>
                </p>
            </div>
        );
    }

    // Calculate variant position in AA space
    const variantAA: number | null = staticData
        ? genomicToAA(genomicPosition, staticData)
        : null; // for UniProt data we can't map genomic→AA without CDS coordinates

    // Find which domain (if any) the variant falls in
    const hitDomain = variantAA
        ? domains.find((d) => variantAA >= d.start && variantAA <= d.end)
        : null;

    // SVG dimensions
    const svgW = 460;
    const svgH = 54;
    const barY = 20;
    const barH = 22;
    const barX = 10;
    const barWidth = svgW - 20;

    function domainX(aa: number) {
        return barX + (aa / totalAA) * barWidth;
    }

    return (
        <div className="rounded-md bg-white border border-indigo-100 p-3">
            <div className="flex items-center justify-between mb-2">
                <div className="text-xs font-medium text-indigo-800 flex items-center gap-1">
                    🔬 {geneSymbol} Protein Domain Map
                    {uniprotId && (
                        <a
                            href={`https://www.uniprot.org/uniprotkb/${uniprotId}`}
                            target="_blank" rel="noopener noreferrer"
                            className="text-[10px] text-indigo-400 hover:text-indigo-600 underline ml-1"
                        >
                            UniProt:{uniprotId}
                        </a>
                    )}
                </div>
                <span className="text-[9px] text-indigo-400">{totalAA} aa</span>
            </div>

            {/* SVG protein bar */}
            <div className="overflow-x-auto">
                <svg viewBox={`0 0 ${svgW} ${svgH}`} className="w-full max-w-full" style={{ minWidth: 280 }}>
                    {/* Protein backbone */}
                    <rect x={barX} y={barY + barH / 2 - 3} width={barWidth} height={6} rx={3} fill="#e2e8f0" />

                    {/* Domains */}
                    {domains.map((d) => {
                        const x = domainX(d.start);
                        const w = Math.max(8, domainX(d.end) - x);
                        return (
                            <g key={d.name}>
                                <rect x={x} y={barY} width={w} height={barH} rx={4}
                                    fill={d.color} fillOpacity={0.85}>
                                    <title>{d.name} (aa {d.start}–{d.end}): {d.description}</title>
                                </rect>
                                {w > 30 && (
                                    <text x={x + w / 2} y={barY + barH / 2 + 4}
                                        textAnchor="middle" fontSize={8} fill="white" fontWeight="600">
                                        {d.name.length > 12 ? d.name.slice(0, 10) + "…" : d.name}
                                    </text>
                                )}
                            </g>
                        );
                    })}

                    {/* Variant marker */}
                    {variantAA !== null && (
                        <g>
                            <line
                                x1={domainX(variantAA)} y1={barY - 6}
                                x2={domainX(variantAA)} y2={barY + barH + 4}
                                stroke="#1e1b4b" strokeWidth={2} strokeDasharray="2,1"
                            />
                            <circle cx={domainX(variantAA)} cy={barY - 8} r={4}
                                fill={hitDomain ? hitDomain.color : "#64748b"} stroke="white" strokeWidth={1.5}>
                                <title>Variant at ~aa {variantAA}{hitDomain ? ` (in ${hitDomain.name})` : " (outside known domains)"}</title>
                            </circle>
                        </g>
                    )}

                    {/* AA scale ticks */}
                    {[0, 0.25, 0.5, 0.75, 1].map((frac) => {
                        const x = barX + frac * barWidth;
                        const aa = Math.round(frac * totalAA);
                        return (
                            <g key={frac}>
                                <line x1={x} y1={barY + barH} x2={x} y2={barY + barH + 4} stroke="#94a3b8" strokeWidth={1} />
                                <text x={x} y={svgH - 1} textAnchor="middle" fontSize={7} fill="#94a3b8">{aa}</text>
                            </g>
                        );
                    })}
                </svg>
            </div>

            {/* Domain legend */}
            <div className="mt-2 flex flex-wrap gap-1.5">
                {domains.map((d) => (
                    <div key={d.name} className="flex items-center gap-1 text-[10px] text-[#3c4f3d]/70">
                        <span className="h-2 w-2 rounded-sm flex-shrink-0" style={{ backgroundColor: d.color }} />
                        {d.name}
                    </div>
                ))}
            </div>

            {/* Variant domain note */}
            {variantAA !== null && (
                <div className={`mt-2 text-[10px] leading-relaxed rounded px-2 py-1 ${hitDomain
                    ? "bg-red-50 text-red-700 border border-red-100"
                    : isPredPathogenic
                        ? "bg-amber-50 text-amber-800 border border-amber-100"
                        : "bg-green-50 text-green-700 border border-green-100"
                    }`}>
                    {hitDomain
                        ? <>Variant at ~aa {variantAA} falls in <strong>{hitDomain.name}</strong> — {hitDomain.description}.</>
                        : isPredPathogenic
                            ? <>Variant at ~aa {variantAA} is <strong>outside annotated functional domains</strong>. Evo2&apos;s pathogenic call is driven by the delta score (sequence-level signal), not domain disruption — this may reflect subtle splice or regulatory effects not captured by domain annotations.</>
                            : isPredBenign
                                ? <>Variant at ~aa {variantAA} is <strong>outside known functional domains</strong> — no critical catalytic or structural region disrupted. Consistent with benign classification.</>
                                : <>Variant at ~aa {variantAA} is <strong>outside known functional domains</strong>. Domain location alone is not conclusive — interpret alongside delta score and population data.</>
                    }
                </div>
            )}

            <p className="mt-1.5 text-[9px] text-indigo-400">
                {staticData ? "Domain boundaries from UniProt canonical sequence." : "Domain data fetched live from UniProt REST API."}
                {" "}Amino acid position estimated from genomic coordinates.
            </p>
        </div>
    );
}
