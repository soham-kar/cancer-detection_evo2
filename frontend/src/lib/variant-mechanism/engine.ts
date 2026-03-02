// =============================================================================
// Variant Mechanism Inference Engine - Explainable AI (XAI)
// =============================================================================
// Generates human-readable biological explanations for genomic variants based on:
//   - VEP molecular consequences (when available)
//   - Frameshift mathematics (reference/alternative length analysis)
//   - Protein domain positioning (GENE_DB functional annotations)
//   - Evo2 evolutionary constraint signals (delta score direction/magnitude)
//
// Design Philosophy:
//   - No variant-specific hardcoding: All rules are biological principles
//   - No gene-specific hardcoding: Domain data drives gene context
//   - Graceful degradation: Works without VEP annotation (sequence-only mode)
//   - Evidence transparency: Shows reasoning chain for clinical review
//
// Mechanism Categories:
//   1. Frameshift (HIGH impact): Reading frame disruption → truncation
//   2. Nonsense (HIGH impact): Premature stop codon → protein truncation
//   3. In-frame indel (MODERATE impact): AA insertion/deletion
//   4. Missense (VARIABLE impact): Conservative vs non-conservative substitution
//   5. Synonymous (LOW impact): Silent mutation, no protein change
//   6. Fallback (uncertain): VEP unavailable, delta score only
// =============================================================================

import type { VariantContext, MechanismExplanation } from "./types";
import type { Domain } from "~/utils/domain-lookup";

// =============================================================================
// Amino Acid Reference Data
// =============================================================================
/**
 * Single-letter amino acid code to full name mapping.
 * Used for generating human-readable protein change descriptions.
 */
const AA_NAMES: Record<string, string> = {
    A: "Alanine", R: "Arginine", N: "Asparagine", D: "Aspartate",
    C: "Cysteine", Q: "Glutamine", E: "Glutamate", G: "Glycine",
    H: "Histidine", I: "Isoleucine", L: "Leucine", K: "Lysine",
    M: "Methionine", F: "Phenylalanine", P: "Proline", S: "Serine",
    T: "Threonine", W: "Tryptophan", Y: "Tyrosine", V: "Valine",
    "*": "Stop codon",
};

/**
 * Amino acid biochemical property groups for conservative change analysis.
 * 
 * Groups:
 *   - nonpolar: Hydrophobic residues, typically buried in protein core
 *   - aromatic: Large ring structures, important for pi-stacking interactions
 *   - polar: Uncharged hydrophilic residues, often at surface or H-bonding
 *   - positive: Lysine, Arginine, Histidine (basic residues)
 *   - negative: Aspartate, Glutamate (acidic residues)
 *   - stop: Termination codon
 * 
 * Conservative changes: Within-group substitutions (e.g., Leu → Ile)
 * Non-conservative changes: Between-group substitutions (e.g., Lys → Asp)
 */
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
    // Stop codon
    "*": "stop",
};

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * Determine if amino acid substitution is biochemically conservative.
 * 
 * Conservative changes maintain similar physicochemical properties and are
 * often better tolerated structurally. Examples:
 *   - Leu → Ile (both nonpolar aliphatic)
 *   - Asp → Glu (both negatively charged)
 *   - Ser → Thr (both polar uncharged)
 * 
 * Non-conservative changes alter charge, polarity, or size and are more
 * likely to disrupt protein structure/function.
 * 
 * @param ref - Reference amino acid single-letter code
 * @param alt - Alternative amino acid single-letter code
 * @returns true if substitution is conservative (same biochemical group)
 */
function isConservativeChange(ref: string, alt: string): boolean {
    if (!ref || !alt || ref === alt) return true;
    const refGroup = AA_GROUP[ref.toUpperCase()];
    const altGroup = AA_GROUP[alt.toUpperCase()];
    return refGroup !== undefined && refGroup === altGroup;
}

/**
 * Parse HGVS protein notation into structured components.
 * 
 * Supported formats:
 *   - Missense: p.Thr1074Ala (three-letter AA codes)
 *   - Synonymous: p.Thr1074= (unchanged)
 *   - Nonsense: p.Thr1074Ter or p.Thr1074*
 *   - Frameshift: p.Thr1074AsnfsTer12 (frameshift to Ter at +12)
 * 
 * Returns:
 *   - refAA1: Reference amino acid single-letter code
 *   - pos: Amino acid position in protein
 *   - altAA1: Alternative amino acid (or "fs" suffix for frameshift)
 *   - isStop: true if nonsense mutation
 *   - fsOffset: Frameshift offset to termination codon
 * 
 * @param hgvsp - HGVS protein notation string (e.g., "p.Thr1074Ala")
 * @returns Parsed components or null if format unrecognized
 */
function parseHGVSp(hgvsp: string | null): {
    refAA1: string; pos: number; altAA1: string; isStop: boolean; fsOffset?: number;
} | null {
    if (!hgvsp) return null;
    
    // Match HGVS protein notation pattern
    // Captures: (refAA3)(position)(altAA3)(frameshift?)(termination offset?)
    const match = hgvsp.match(/p\.([A-Z][a-z]{2})(\d+)([A-Z?*][a-z]*)(fs)?(?:Ter(\d+))?/);
    if (!match) return null;

    // Three-letter to single-letter amino acid code mapping
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

/**
 * Format codon change for human-readable display.
 * 
 * Input format: "acA/acG" (lowercase = unchanged, uppercase = mutated)
 * Output format: "acA → acG" (directional arrow)
 * 
 * @param codons - Codon change string from VEP (ref/alt)
 * @returns Formatted codon change or original string if format unrecognized
 */
function formatCodons(codons: string | null): string | null {
    if (!codons) return null;
    const [ref, alt] = codons.split("/");
    if (!ref || !alt) return codons;
    return `${ref} → ${alt}`;
}

/**
 * Format amino acid change for display with full residue names.
 * 
 * Examples:
 *   - Missense: "Threonine-1074 → Alanine"
 *   - Synonymous: "Threonine-1074 (unchanged)"
 *   - Nonsense: "Threonine-1074 → premature stop codon"
 *   - Frameshift: "Threonine-1074 → Asparagine (frameshift, stop at +12)"
 * 
 * @param parsed - Parsed HGVS components from parseHGVSp
 * @param raw - Raw HGVS string as fallback
 * @returns Human-readable amino acid change description
 */
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

/**
 * Generate description of protein domains truncated by premature stop codon.
 * 
 * Used for nonsense and frameshift variants to explain functional consequences
 * of protein truncation. Catalogs all domains lost beyond the stop position.
 * 
 * Output format:
 *   "Premature stop at aa 1074 means the following BRCA1 domains are never
 *    synthesized: BRCT (I) (aa 1646–1736), BRCT (II) (aa 1756–1855).
 *    Binds phosphoproteins; key for DNA repair."
 * 
 * @param stopAA - Amino acid position of premature stop codon
 * @param lostDomains - Array of domains beyond stop position
 * @param geneName - Gene symbol for context
 * @returns Human-readable truncation description
 */
function describeTruncatedDomains(stopAA: number, lostDomains: Domain[], geneName: string): string {
    if (lostDomains.length === 0) return "";
    const names = lostDomains.map((d) => `${d.name} (aa ${d.start}–${d.end})`).join(", ");
    const descriptions = lostDomains.map((d) => d.description.trim().replace(/[.!?]+$/, "")).join("; ");
    return `Premature stop at aa ${stopAA} means the following ${geneName} domains are never synthesized: ${names}. ${descriptions}.`;
}

// =============================================================================
// Main Mechanism Inference Engine
// =============================================================================

/**
 * Infer biological mechanism and generate XAI explanation for variant.
 * 
 * This is the main entry point for mechanism inference. It analyzes variant
 * attributes in priority order:
 *   1. VEP consequence (if available): High-confidence molecular annotation
 *   2. Length-based frameshift: Mathematical certainty from indel length
 *   3. Domain context: Functional impact based on protein region
 *   4. Evo2 delta score: Evolutionary constraint signal
 * 
 * The function generates:
 *   - Title: Concise mechanism category
 *   - Primary mechanism: Technical description of molecular change
 *   - Biological impact: Functional consequences and evolutionary evidence
 *   - Molecular detail: Codon and amino acid changes
 *   - Domain context: Affected protein regions
 *   - Comparison note: Contrast with alternative scenarios
 * 
 * Confidence levels:
 *   - high: VEP-confirmed PTVs (frameshift, nonsense, synonymous)
 *   - medium: Missense or in-frame indels with domain context
 *   - low: Fallback mode without VEP annotation
 * 
 * @param ctx - Variant context with VEP, delta score, and domain data
 * @returns MechanismExplanation with structured XAI content
 */
export function inferMechanism(ctx: VariantContext): MechanismExplanation {
    const { vep, variantType, reference, alternative, deltaScore, hitDomain, domainsAfterPosition, geneSymbol, proteinPosition } = ctx;

    // Classify variant type from ClinVar or VEP data
    const isDup = /duplication|dup/i.test(variantType);
    const isDel = /deletion|del/i.test(variantType);
    const isIns = /insertion|ins/i.test(variantType) && !isDup;
    const isIndel = isDup || isDel || isIns;

    // ===== Frameshift Detection =====
    // Length-based detection works even without VEP annotation
    // Frameshift condition: indel length not divisible by 3
    const lenDiff = alternative.length - reference.length;
    const isLengthFrameshift = isIndel && lenDiff !== 0 && Math.abs(lenDiff) % 3 !== 0;

    // Prefer VEP classification, fall back to length-based detection
    const isFrameshift = vep?.isFrameshift ?? isLengthFrameshift;
    const isSynonymous = vep?.isSynonymous ?? false;
    const isNonsense = vep?.isNonsense ?? false;

    // Parse protein-level changes from VEP annotation
    const parsedAA = parseHGVSp(vep?.aaChange ?? null);
    const codonDisplay = formatCodons(vep?.codons ?? null);
    const aaDisplay = formatAAChange(parsedAA, vep?.aaChange ?? null);

    // Structure molecular details for display
    const molDetail = (codonDisplay || aaDisplay || vep?.aaChange) ? {
        codonChange: codonDisplay,
        aaChange: vep?.aaChange ?? null,
        aminoAcidChange: aaDisplay,
    } : null;

    // Calculate truncation consequences for PTVs
    // Stop codon position: current position + frameshift offset
    const stopAA = parsedAA?.fsOffset && proteinPosition
        ? proteinPosition + parsedAA.fsOffset
        : null;
    
    // Identify truncated domains: both fully lost and partially truncated
    const truncatedByStop = stopAA
        ? domainsAfterPosition.filter((d) => d.start > stopAA || (d.start <= stopAA && stopAA < d.end))
        : domainsAfterPosition;

    // ===== MECHANISM CLASSIFICATION HIERARCHY =====
    // Priority order: Frameshift → Nonsense → In-frame indel → Missense → Synonymous → Fallback

    // ===== FRAMESHIFT VARIANTS (Duplication / Deletion / Insertion) =====
    if (isFrameshift) {
        const insSize = Math.abs(lenDiff);
        const domainCtxSentence = hitDomain
            ? `Variant falls within ${hitDomain.name} (aa ${hitDomain.start}–${hitDomain.end}) — frameshift destroys this domain and all downstream protein.`
            : `Variant is outside annotated domains, but reading frame shift affects all downstream protein sequence.`;
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

    // ===== IN-FRAME INDEL VARIANTS =====
    // Preserves reading frame but removes/adds complete amino acids
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

    // ===== NONSENSE VARIANTS (Stop-Gained) =====
    // Premature termination codon → protein truncation
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

    // ===== SYNONYMOUS VARIANTS (Silent Mutations) =====
    // Codon change with no amino acid change due to genetic code degeneracy
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

    // ===== MISSENSE VARIANTS (Amino Acid Substitutions) =====
    // Single amino acid change with variable functional impact
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

    // ===== FALLBACK MODE =====
    // VEP annotation unavailable — infer from Evo2 delta score and backend classification
    // Uses gene-specific thresholds and 3-tier classification from backend
    const isPathogenic = ctx.prediction === "Likely pathogenic";
    const isBenign = ctx.prediction === "Likely benign";
    return {
        title: isPathogenic ? "Pathogenic Signal Detected" : isBenign ? "Benign Signal Detected" : "Variant of Uncertain Significance",
        confidence: "low",
        primaryMechanism: isPathogenic
            ? `Evo2 Δ = ${deltaScore.toFixed(6)} — strong negative signal indicating evolutionary constraint. This variant is disfavored by natural selection. Molecular annotation (VEP) was unavailable; mechanism inferred from sequence-level conservation analysis.`
            : isBenign
                ? `Evo2 Δ = ${deltaScore.toFixed(6)} — positive signal indicating evolutionary tolerance. Variant is permissive across species. VEP molecular annotation unavailable.`
                : `Evo2 Δ = ${deltaScore.toFixed(6)} — near-zero signal suggests weak or conflicting evolutionary pressure. Classification uncertain without molecular annotation.`,
        molecularDetail: null,
        biologicalImpact: hitDomain
            ? `Variant at ~aa ${proteinPosition} falls within ${hitDomain.name} (aa ${hitDomain.start}–${hitDomain.end}): ${hitDomain.description}. Functional assessment recommended.`
            : proteinPosition
                ? `Variant at ~aa ${proteinPosition} is outside annotated functional domains. May have regulatory or unknown effects.`
                : "Protein position could not be determined. Variant impact on protein domains unknown.",
        domainContext: hitDomain ? {
            sentence: `${hitDomain.name}: ${hitDomain.description}.`,
            truncatedDomains: [],
        } : null,
        comparisonNote: "Re-run analysis when VEP annotation becomes available for precise molecular mechanism explanation.",
    };
}
