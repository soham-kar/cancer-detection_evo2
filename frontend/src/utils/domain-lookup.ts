/**
 * domain-lookup.ts
 * Shared utility: static gene domain database + genomic→AA position helper.
 * Extracted from gene-domain-map.tsx so the mechanism engine can use it too.
 */

export interface Domain {
    name: string;
    start: number; // amino acid
    end: number;   // amino acid
    color: string;
    description: string;
}

export interface GeneStaticData {
    totalAA: number;
    genomicCDSStart: number; // hg38
    genomicCDSEnd: number;   // hg38
    strand: '+' | '-';       // DNA strand: plus or minus
    domains: Domain[];
    uniprotId: string;
}

// ── Static domain database ──────────────────────────────────────────────────
// Positions: UniProt canonical isoform amino acid coordinates
export const GENE_DB: Record<string, GeneStaticData> = {
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
            { name: "OB fold (F1)", start: 2402, end: 2667, color: "#f59e0b", description: "ssDNA binding; RAD51 loading" },
            { name: "OB fold (F2)", start: 2670, end: 2803, color: "#f59e0b", description: "ssDNA binding" },
            { name: "OB fold (F3)", start: 2809, end: 3102, color: "#f59e0b", description: "ssDNA binding" },
            { name: "Tower", start: 2810, end: 2872, color: "#22c55e", description: "Interacts with RAD51; strand invasion" },
        ],
    },
    TP53: {
        totalAA: 393,
        genomicCDSStart: 7668421,
        genomicCDSEnd: 7687550,
        strand: '-',
        uniprotId: "P04637",
        domains: [
            { name: "Transactivation I", start: 1, end: 40, color: "#f59e0b", description: "MDM2 binding; p300 interaction; transcriptional activation" },
            { name: "Transactivation II", start: 40, end: 67, color: "#f59e0b", description: "Secondary transactivation domain" },
            { name: "Proline-rich", start: 67, end: 98, color: "#22c55e", description: "Apoptosis regulation; APAF-1 interaction" },
            { name: "DNA binding", start: 102, end: 292, color: "#ef4444", description: "Core domain; sequence-specific DNA binding; 90% of cancer mutations cluster here" },
            { name: "Tetramerization", start: 323, end: 356, color: "#3b82f6", description: "Forms functional p53 tetramer; required for transcriptional activity" },
            { name: "Regulatory", start: 364, end: 393, color: "#6366f1", description: "Post-translational modification hub; ubiquitination, acetylation" },
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
            { name: "Phosphatase", start: 14, end: 185, color: "#ef4444", description: "Catalytic domain; dephosphorylates PIP3; primary tumor suppressor activity" },
            { name: "C2", start: 186, end: 351, color: "#3b82f6", description: "Membrane binding, calcium-independent; regulatory" },
            { name: "PDZ-binding", start: 401, end: 403, color: "#22c55e", description: "Interactions with PDZ domain proteins; nuclear localization" },
        ],
    },
    ATM: {
        totalAA: 3056,
        genomicCDSStart: 108222832,
        genomicCDSEnd: 108369102,
        strand: '+',
        uniprotId: "Q13315",
        domains: [
            { name: "FAT", start: 1960, end: 2566, color: "#f59e0b", description: "FRAP-ATM-TRRAP domain; regulatory scaffold" },
            { name: "Kinase", start: 2712, end: 3011, color: "#ef4444", description: "PI3K-like kinase; phosphorylates H2AX, BRCA1, p53 on DNA damage" },
            { name: "FATC", start: 3024, end: 3056, color: "#6366f1", description: "C-terminal; required for kinase activity" },
        ],
    },
};

// ── Helper: genomic position → approximate amino acid position ───────────────
// NOTE: This function uses a linear approximation and does not account for introns.
// For precise mapping, exon boundaries should be added to GeneStaticData and processed here.
export function genomicToAA(pos: number, geneData: GeneStaticData): number | null {
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

// ── Helper: find domain at a given AA position ───────────────────────────────
// Returns the most specific (smallest-range) domain when multiple domains contain the queried residue.
export function getDomainAtAA(aaPos: number, geneData: GeneStaticData): Domain | null {
    const matchingDomains = geneData.domains.filter((d) => aaPos >= d.start && aaPos <= d.end);
    if (matchingDomains.length === 0) return null;
    
    // Return the domain with the smallest span (most specific)
    return matchingDomains.reduce((smallest, current) => {
        const currentSpan = current.end - current.start;
        const smallestSpan = smallest.end - smallest.start;
        return currentSpan < smallestSpan ? current : smallest;
    });
}

// ── Helper: find all domains after a given AA position (truncation impact) ──-
export function getDomainsAfterAA(aaPos: number, geneData: GeneStaticData): Domain[] {
    return geneData.domains.filter((d) => d.start > aaPos);
}

// ── Helper: get gene data by symbol ─────────────────────────────────────────
export function getGeneData(geneSymbol: string): GeneStaticData | null {
    return GENE_DB[geneSymbol.toUpperCase()] ?? null;
}
