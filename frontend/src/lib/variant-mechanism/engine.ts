/**
 * engine.ts — Variant Mechanism Inference Engine
 *
 * Generates biological explanations for variants based on:
 *   - VEP consequence (when available)
 *   - Frameshift math (ref/alt length)
 *   - Domain position (GENE_DB)
 *   - Evo2 delta score direction
 *
 * No variant-specific or gene-specific hardcoding — all rules are biological.
 */
import type { VariantContext, MechanismExplanation } from "./types";
import type { Domain } from "~/utils/domain-lookup";

// ── Amino acid single-letter → full name map ─────────────────────────────── 
const AA_NAMES: Record<string, string> = {
    A: "Alanine", R: "Arginine", N: "Asparagine", D: "Aspartate",
    C: "Cysteine", Q: "Glutamine", E: "Glutamate", G: "Glycine",
    H: "Histidine", I: "Isoleucine", L: "Leucine", K: "Lysine",
    M: "Methionine", F: "Phenylalanine", P: "Proline", S: "Serine",
    T: "Threonine", W: "Tryptophan", Y: "Tyrosine", V: "Valine",
    "*": "Stop codon",
};

// ── Amino acid biochemical groups (for conservative vs non-conservative) ─────
const AA_GROUP: Record<string, string> = {
    // Nonpolar aliphatic
    G: "nonpolar", A: "nonpolar", V: "nonpolar", L: "nonpolar", I: "nonpolar", P: "nonpolar", M: "nonpolar",
    // Aromatic
    F: "aromatic", W: "aromatic", Y: "aromatic",
    // Polar uncharged
    S: "polar", T: "polar", C: "polar", N: "polar", Q: "polar",
    // Positively charged
    K: "positive", R: "positive", H: "positive",
    // Negatively charged
    D: "negative", E: "negative",
    // Stop
    "*": "stop",
};

function isConservativeChange(ref: string, alt: string): boolean {
    if (!ref || !alt || ref === alt) return true;
    const refGroup = AA_GROUP[ref.toUpperCase()];
    const altGroup = AA_GROUP[alt.toUpperCase()];
    return refGroup !== undefined && refGroup === altGroup;
}

/** Parse "p.Thr1074AsnfsTer12" → { refAA, pos, altAA, fsOffset } */
function parseHGVSp(hgvsp: string | null): {
    refAA1: string; pos: number; altAA1: string; isStop: boolean; fsOffset?: number;
} | null {
    if (!hgvsp) return null;
    // p.Thr1074AsnfsTer12  or  p.Thr1074Ala  or  p.Thr1074=
    const match = hgvsp.match(/p\.([A-Z][a-z]{2})(\d+)([A-Z?*][a-z]*)(fs)?(?:Ter(\d+))?/);
    if (!match) return null;

    // Three-letter → single-letter
    const THREE_TO_ONE: Record<string, string> = {
        Ala: "A", Arg: "R", Asn: "N", Asp: "D", Cys: "C", Gln: "Q",
        Glu: "E", Gly: "G", His: "H", Ile: "I", Leu: "L", Lys: "K",
        Met: "M", Phe: "F", Pro: "P", Ser: "S", Thr: "T", Trp: "W",
        Tyr: "Y", Val: "V", Ter: "*",
    };

    const refAA3 = match[1]!;
    const pos = parseInt(match[2]!, 10);
    const altAA3 = match[3]!.slice(0, 3);
    const isFrameshift = !!match[4];
    const fsOffset = match[5] ? parseInt(match[5], 10) : undefined;

    const refAA1 = THREE_TO_ONE[refAA3] ?? refAA3[0]!;
    const altAA1 = THREE_TO_ONE[altAA3] ?? (match[3]![0] ?? "?");
    const isStop = altAA1 === "*" || match[3] === "Ter";

    return { refAA1, pos, altAA1: isFrameshift ? altAA1 + "fs" : altAA1, isStop, fsOffset };
}

/** Build human-readable codons line: "acA → acG" */
function formatCodons(codons: string | null): string | null {
    if (!codons) return null;
    const [ref, alt] = codons.split("/");
    if (!ref || !alt) return codons;
    return `${ref} → ${alt}`;
}

/** Format amino acid change for display */
function formatAAChange(parsed: ReturnType<typeof parseHGVSp>, raw: string | null): string | null {
    if (!parsed) return raw;
    const refName = AA_NAMES[parsed.refAA1] ?? parsed.refAA1;
    const altName = parsed.altAA1.endsWith("fs")
        ? (AA_NAMES[parsed.altAA1[0]!] ?? parsed.altAA1[0])
        : (AA_NAMES[parsed.altAA1] ?? parsed.altAA1);

    if (parsed.altAA1 === parsed.refAA1) return `${refName}-${parsed.pos} (unchanged)`;
    if (parsed.altAA1.endsWith("fs")) {
        return `${refName}-${parsed.pos} → ${altName} (frameshift${parsed.fsOffset ? `, stop at +${parsed.fsOffset}` : ""})`;
    }
    if (parsed.isStop) return `${refName}-${parsed.pos} → premature stop codon`;
    return `${refName}-${parsed.pos} → ${altName}`;
}

/** Describe domains truncated by a stop codon at the given AA position */
function describeTruncatedDomains(stopAA: number, lostDomains: Domain[], geneName: string): string {
    if (lostDomains.length === 0) return "";
    const names = lostDomains.map((d) => `${d.name} (aa ${d.start}–${d.end})`).join(", ");
    const descriptions = lostDomains.map((d) => d.description.trim().replace(/[.!?]+$/, "")).join("; ");
    return `Premature stop at aa ${stopAA} means the following ${geneName} domains are never synthesized: ${names}. ${descriptions}.`;
}

// ─────────────────────────────────────────────────────────────────────────────
export function inferMechanism(ctx: VariantContext): MechanismExplanation {
    const { vep, variantType, reference, alternative, deltaScore, hitDomain, domainsAfterPosition, geneSymbol, proteinPosition } = ctx;

    const isDup = /duplication|dup/i.test(variantType);
    const isDel = /deletion|del/i.test(variantType);
    const isIns = /insertion|ins/i.test(variantType) && !isDup;
    const isIndel = isDup || isDel || isIns;

    // ── Length-based frameshift check (works even without VEP) ─────────────
    const lenDiff = alternative.length - reference.length;
    const isLengthFrameshift = isIndel && lenDiff !== 0 && Math.abs(lenDiff) % 3 !== 0;

    const isFrameshift = vep?.isFrameshift ?? isLengthFrameshift;
    const isSynonymous = vep?.isSynonymous ?? false;
    const isNonsense = vep?.isNonsense ?? false;

    const parsedAA = parseHGVSp(vep?.aaChange ?? null);
    const codonDisplay = formatCodons(vep?.codons ?? null);
    const aaDisplay = formatAAChange(parsedAA, vep?.aaChange ?? null);

    const molDetail = (codonDisplay || aaDisplay || vep?.aaChange) ? {
        codonChange: codonDisplay,
        aaChange: vep?.aaChange ?? null,
        aminoAcidChange: aaDisplay,
    } : null;

    // Stop codon position for truncation analysis
    const stopAA = parsedAA?.fsOffset && proteinPosition
        ? proteinPosition + parsedAA.fsOffset
        : null;
    
    // Include both fully lost domains (start > stopAA) and partially truncated (start <= stopAA < end)
    const truncatedByStop = stopAA
        ? domainsAfterPosition.filter((d) => d.start > stopAA || (d.start <= stopAA && stopAA < d.end))
        : domainsAfterPosition;

    // ── FRAMESHIFT (Dup / Del / Ins) ────────────────────────────────────────
    if (isFrameshift) {
        const insSize = Math.abs(lenDiff);
        const domainCtxSentence = hitDomain
            ? `Variant falls within ${hitDomain.name} (aa ${hitDomain.start}–${hitDomain.end}) — frameshift also destroys this domain directly.`
            : `Variant is outside annotated domains, but the reading frame shift affects all downstream protein.`;
        const truncDesc = describeTruncatedDomains(stopAA ?? (proteinPosition ?? 0), truncatedByStop, geneSymbol);

        return {
            title: isDel ? "Frameshift-Causing Deletion" : "Frameshift-Causing Duplication/Insertion",
            confidence: "high",
            primaryMechanism: `${isDel ? "Deletion" : "Insertion"} of ${insSize} base${insSize > 1 ? "s" : ""} (${Math.abs(lenDiff)} mod 3 ≠ 0) disrupts the reading frame. Every amino acid from position ${proteinPosition ?? "?"} onward is altered, and a premature stop codon typically appears within tens of residues.`,
            molecularDetail: molDetail,
            biologicalImpact: [
                "Frameshift variants are highly deleterious — the resulting truncated protein is usually degraded by nonsense-mediated mRNA decay (NMD).",
                truncDesc,
                `Evo2 Δ = ${deltaScore.toFixed(6)} ${deltaScore < 0 ? "confirms strong purifying selection against this change" : "— note: weak Evo2 signal does not negate the frameshift mechanism"}.`,
            ].filter(Boolean).join(" "),
            domainContext: {
                sentence: domainCtxSentence,
                truncatedDomains: truncatedByStop,
            },
            comparisonNote: `An in-frame ${isDel ? "deletion" : "duplication"} (multiple of 3 bases) would insert/remove whole amino acids without disrupting the reading frame — a potentially less severe outcome depending on location.`,
        };
    }

    // ── IN-FRAME INDEL ──────────────────────────────────────────────────────
    if (isIndel && !isFrameshift) {
        const aaCount = Math.abs(lenDiff) / 3;
        return {
            title: `In-Frame ${isDel ? "Deletion" : "Insertion/Duplication"} (${aaCount} amino acid${aaCount !== 1 ? "s" : ""})`,
            confidence: "medium",
            primaryMechanism: `${isDel ? "Deletion" : "Insertion"} of ${Math.abs(lenDiff)} bases (multiple of 3) preserves the reading frame but ${isDel ? "removes" : "adds"} ${aaCount} amino acid${aaCount !== 1 ? "s" : ""} from the protein.`,
            molecularDetail: molDetail,
            biologicalImpact: `In-frame variants are context-dependent. ${hitDomain ? `Located in ${hitDomain.name} — disruption of this domain's structure could impair function.` : "Outside annotated domains — may be more tolerated."} Evo2 Δ = ${deltaScore.toFixed(6)} ${Math.abs(deltaScore) < 0.001 ? "suggests low evolutionary constraint at this position" : deltaScore < 0 ? "indicates pathogenic pressure" : "suggests tolerance"}.`,
            domainContext: hitDomain ? {
                sentence: `Within ${hitDomain.name} (aa ${hitDomain.start}–${hitDomain.end}): ${hitDomain.description}.`,
                truncatedDomains: [],
            } : null,
            comparisonNote: `A frameshift variant at this position (inserting/deleting non-multiple of 3 bases) would be far more severe — destroying all downstream protein sequence.`,
        };
    }

    // ── NONSENSE (stop gained) ───────────────────────────────────────────────
    if (isNonsense) {
        const truncDesc = describeTruncatedDomains(proteinPosition ?? 0, truncatedByStop, geneSymbol);
        return {
            title: "Nonsense Variant — Premature Stop Codon",
            confidence: "high",
            primaryMechanism: `The ${vep?.aaChange ?? "substitution"} introduces a premature stop codon${proteinPosition ? ` at position ${proteinPosition}` : ""}. The protein is truncated and typically degraded by nonsense-mediated decay.`,
            molecularDetail: molDetail,
            biologicalImpact: [
                "Nonsense variants are equivalent to LOF (loss of function).",
                truncDesc,
            ].filter(Boolean).join(" "),
            domainContext: truncatedByStop.length > 0 ? {
                sentence: `Domains not synthesized: ${truncatedByStop.map((d) => d.name).join(", ")}.`,
                truncatedDomains: truncatedByStop,
            } : null,
            comparisonNote: null,
        };
    }

    // ── SYNONYMOUS ───────────────────────────────────────────────────────────
    if (isSynonymous) {
        return {
            title: "Synonymous Variant — No Protein Change",
            confidence: "high",
            primaryMechanism: `The ${vep?.aaChange ?? "nucleotide substitution"} does not alter the amino acid sequence due to the degeneracy of the genetic code. The protein product is identical to reference.`,
            molecularDetail: molDetail,
            biologicalImpact: `Synonymous variants are typically benign. Evo2 Δ = ${deltaScore.toFixed(6)} ${Math.abs(deltaScore) < 0.001 ? "is consistent — near-zero evolutionary pressure confirms tolerance" : deltaScore > 0 ? "positive signal reflects evolutionary preference for this sequence context" : "slightly negative despite synonymous change — may affect splicing or codon usage"}.`,
            domainContext: hitDomain ? {
                sentence: `Located in ${hitDomain.name} (aa ${hitDomain.start}–${hitDomain.end}), but synonymous change preserves protein integrity.`,
                truncatedDomains: [],
            } : null,
            comparisonNote: `A missense change at this codon altering the amino acid would require further functional assessment.`,
        };
    }

    // ── MISSENSE ─────────────────────────────────────────────────────────────
    if (vep?.consequence === "missense_variant" && parsedAA) {
        const conservative = isConservativeChange(parsedAA.refAA1, parsedAA.altAA1[0]!);
        const refName = AA_NAMES[parsedAA.refAA1] ?? parsedAA.refAA1;
        const altName = AA_NAMES[parsedAA.altAA1] ?? parsedAA.altAA1;

        return {
            title: conservative ? "Conservative Missense Substitution" : "Non-Conservative Missense Substitution",
            confidence: "medium",
            primaryMechanism: `${refName}-${parsedAA.pos} is replaced by ${altName}. ${conservative ? "Both amino acids share similar biochemical properties (same group), so structural impact may be limited." : `These amino acids have different biochemical properties — ${refName} is ${AA_GROUP[parsedAA.refAA1] ?? "unknown"}, ${altName} is ${AA_GROUP[parsedAA.altAA1[0]!] ?? "unknown"} — increasing the likelihood of structural disruption.`}`,
            molecularDetail: molDetail,
            biologicalImpact: `${hitDomain ? `Located in ${hitDomain.name} — ${hitDomain.description}.` : "Outside annotated functional domains."} Evo2 Δ = ${deltaScore.toFixed(6)} ${deltaScore < -0.001 ? "— negative signal suggests evolutionary constraint at this position; the reference residue is conserved across species" : deltaScore > 0.001 ? "— positive signal suggests this alternate is tolerated evolutionarily" : "— near-zero signal indicates Evo2 is uncertain; additional functional evidence recommended"}.`,
            domainContext: hitDomain ? {
                sentence: `${hitDomain.name} (aa ${hitDomain.start}–${hitDomain.end}): ${hitDomain.description}.`,
                truncatedDomains: [],
            } : null,
            comparisonNote: conservative
                ? `A non-conservative substitution at ${refName}-${parsedAA.pos} (e.g., replacing it with a charged residue) would likely be more damaging.`
                : null,
        };
    }

    // ── FALLBACK: VEP unavailable — reason from delta + domain only ─────────
    const isPathogenic = deltaScore < -0.001;
    const isBenign = deltaScore > 0.001;
    return {
        title: isPathogenic ? "Pathogenic Signal Detected" : isBenign ? "Benign Signal Detected" : "Variant of Uncertain Significance",
        confidence: "low",
        primaryMechanism: isPathogenic
            ? `Evo2 Δ = ${deltaScore.toFixed(6)} — strong negative signal indicating this variant is evolutionarily disfavored. Molecular annotation (VEP) was unavailable; mechanism inferred from sequence-level scoring.`
            : isBenign
                ? `Evo2 Δ = ${deltaScore.toFixed(6)} — positive signal indicating evolutionary tolerance. VEP annotation unavailable.`
                : `Evo2 Δ = ${deltaScore.toFixed(6)} — near-zero signal, insufficient to classify. Molecular annotation (VEP) was unavailable.`,
        molecularDetail: null,
        biologicalImpact: hitDomain
            ? `Variant falls in ${hitDomain.name} (aa ${hitDomain.start}–${hitDomain.end}): ${hitDomain.description}.`
            : "Variant is outside annotated functional domains.",
        domainContext: hitDomain ? {
            sentence: `${hitDomain.name}: ${hitDomain.description}.`,
            truncatedDomains: [],
        } : null,
        comparisonNote: "Re-run analysis when VEP annotation is available for a more precise mechanism explanation.",
    };
}
