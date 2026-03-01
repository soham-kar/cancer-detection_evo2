/**
 * types.ts — Mechanism Engine type definitions
 */

import type { VEPAnnotation } from "~/app/api/vep/route";
import type { Domain } from "~/utils/domain-lookup";

/** All data available about a variant when the mechanism is inferred */
export interface VariantContext {
    variantType: string;           // "SNV", "Duplication", "Deletion", "Insertion", etc.
    deltaScore: number;            // Evo2 delta score
    reference: string;             // ref allele, e.g. "T" or "TC"
    alternative: string;           // alt allele, e.g. "A" or "TCC"
    geneSymbol: string;
    genomicPosition: number;
    chromosome: string;
    prediction: string;            // "Pathogenic" | "Benign" | "Uncertain"

    // Domain context — computed from GENE_DB + genomicToAA
    proteinPosition: number | null;
    hitDomain: Domain | null;        // domain the variant falls in (or null)
    domainsAfterPosition: Domain[]; // domains downstream (truncated if frameshift)

    // VEP annotation — null if VEP call failed or not yet run
    vep: VEPAnnotation | null;
}

/** What the mechanism engine returns for the UI card */
export interface MechanismExplanation {
    title: string;
    confidence: "high" | "medium" | "low";
    primaryMechanism: string;

    // Codon/AA level detail (shown when VEP has data)
    molecularDetail: {
        codonChange: string | null;      // "acA → acG"
        aaChange: string | null;         // "p.Thr1074AsnfsTer12"
        aminoAcidChange: string | null;  // "Thr → Asn (then frameshift, stop at +12)"
    } | null;

    biologicalImpact: string;

    // Domain context
    domainContext: {
        sentence: string;
        truncatedDomains: Domain[];  // domains lost if frameshift/truncation
    } | null;

    comparisonNote: string | null;  // "Compare to: ..."
}
