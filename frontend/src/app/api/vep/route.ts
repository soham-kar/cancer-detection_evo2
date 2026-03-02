// =============================================================================
// VEP Annotation API Route - Ensembl VEP Proxy
// =============================================================================
// Server-side proxy for Ensembl Variant Effect Predictor (VEP) REST API.
// 
// Purpose:
//   - Fetch molecular consequence predictions for variants
//   - Provide override logic for high-confidence classifications
//   - Enrich XAI explanations with structural information
// 
// Features:
//   - Canonical transcript selection for consistency
//   - HGVS notation builder for SNVs, insertions, deletions
//   - Ambiguous base rejection (N, R, Y, W, S, K, M, B, D, H, V)
//   - Timeout protection (8 seconds)
//   - Graceful failure handling (never blocks analysis)
// 
// Ensembl VEP API:
//   - Free public API, no authentication required
//   - Rate limits: Generous for academic use
//   - Documentation: https://rest.ensembl.org/documentation/info/vep_hgvs_post
// =============================================================================
import type { NextRequest } from "next/server";
import { NextResponse } from "next/server";

// =============================================================================
// Type Definitions
// =============================================================================

/**
 * VEP annotation structured data returned from Ensembl and parsed by this endpoint.
 * 
 * This interface provides the molecular consequence information used by:
 *   - Backend VEP override logic (nonsense/frameshift/synonymous)
 *   - Frontend XAI mechanism engine (structural explanations)
 *   - RAG context enrichment (molecular mechanism descriptions)
 * 
 * Impact Levels:
 *   - HIGH: Protein-truncating variants (nonsense, frameshift, splice)
 *   - MODERATE: Missense variants, in-frame indels
 *   - LOW: Synonymous variants, intronic variants
 *   - MODIFIER: Intergenic, regulatory variants
 */
export interface VEPAnnotation {
    consequence: string;        // Variant consequence term (e.g., "missense_variant", "stop_gained")
    impact: "HIGH" | "MODERATE" | "LOW" | "MODIFIER";  // VEP impact classification
    aaChange: string | null;    // HGVS protein notation (e.g., "p.Thr1074Ala")
    codons: string | null;      // Ref/alt codons with lowercase unchanged bases (e.g., "acA/acG")
    aminoAcids: string | null;  // Amino acid change (e.g., "T" or "T/A")
    isSynonymous: boolean;      // Silent mutation flag
    isFrameshift: boolean;      // Frameshift mutation flag
    isNonsense: boolean;        // Nonsense/stop-gained mutation flag
    transcriptId: string | null;  // Ensembl transcript ID
    exonNumber: string | null;  // Exon location (e.g., "10/23")
    geneId: string | null;      // Ensembl gene ID
}

// =============================================================================
// HGVS Notation Builder
// =============================================================================

/**
 * Build HGVS genomic notation for Ensembl VEP query.
 * 
 * HGVS (Human Genome Variation Society) notation is the standard format for
 * describing genetic variants. This function converts input coordinates to
 * HGVS format compatible with Ensembl VEP API.
 * 
 * Supported variant types:
 *   - SNV: 17:g.43093278T>A
 *   - Insertion: 17:g.43093278_43093279insAG
 *   - Deletion: 17:g.43093279_43093280del
 *   - Indel: Fallback to SNV notation
 * 
 * @param chrom - Chromosome (with or without "chr" prefix)
 * @param pos - 1-based genomic position
 * @param ref - Reference allele sequence
 * @param alt - Alternative allele sequence
 * @returns HGVS genomic notation string
 */
function buildHGVS(chrom: string, pos: number, ref: string, alt: string): string {
    // Ensembl uses chromosome identifiers without "chr" prefix
    const c = chrom.replace(/^chr/i, "");

    console.log("[VEP-HGVS] Input: chrom=" + chrom + " pos=" + pos + " ref=" + ref + " alt=" + alt);
    console.log("[VEP-HGVS] ref.length=" + ref.length + " alt.length=" + alt.length);

    if (ref.length === 1 && alt.length === 1) {
        // Single nucleotide variant (SNV)
        const result = `${c}:g.${pos}${ref}>${alt}`;
        console.log("[VEP-HGVS] Detected SNV:", result);
        return result;
    }
    if (alt.length > ref.length && alt.startsWith(ref)) {
        // Insertion: alternative contains reference + inserted sequence
        const inserted = alt.slice(ref.length);
        const result = `${c}:g.${pos}_${pos + ref.length}ins${inserted}`;
        console.log("[VEP-HGVS] Detected insertion:", result);
        return result;
    }
    if (ref.length > alt.length && ref.startsWith(alt)) {
        // Deletion: reference contains alternative + deleted sequence
        const delStart = pos + alt.length;
        const delEnd = pos + ref.length - 1;
        const result = `${c}:g.${delStart}_${delEnd}del`;
        console.log("[VEP-HGVS] Detected deletion:", result);
        return result;
    }
    // Complex indel or unmatched pattern - use generic substitution notation
    const result = `${c}:g.${pos}${ref}>${alt}`;
    console.log("[VEP-HGVS] Using fallback notation:", result);
    return result;
}

// =============================================================================
// VEP Response Parser
// =============================================================================

/**
 * Parse Ensembl VEP API response into structured VEPAnnotation object.
 * 
 * Ensembl returns complex nested JSON with multiple transcripts. This function:
 *   1. Selects canonical transcript for consistent gene representation
 *   2. Extracts consequence terms and impact classification
 *   3. Parses protein-level changes (amino acid substitutions)
 *   4. Identifies variant categories (synonymous, frameshift, nonsense)
 * 
 * Canonical Transcript Selection:
 *   - Ensembl marks one transcript per gene as "canonical"
 *   - Represents the most biologically significant isoform
 *   - Ensures consistent reporting across tools
 * 
 * @param data - Raw VEP API response array
 * @returns Parsed VEPAnnotation or null if parsing fails
 */
function parseConsequence(data: Record<string, unknown>[]): VEPAnnotation | null {
    if (!data || data.length === 0) return null;

    const hit = data[0] as {
        transcript_consequences?: {
            canonical?: number;
            consequence_terms?: string[];
            impact?: string;
            hgvsp?: string;
            codons?: string;
            amino_acids?: string;
            transcript_id?: string;
            exon?: string;
            gene_id?: string;
        }[];
        most_severe_consequence?: string;
    };

    // Extract transcript consequences and prefer canonical transcript
    // Canonical transcript represents the primary/reference isoform
    const transcripts = hit.transcript_consequences ?? [];
    const canonical = transcripts.find((t) => t.canonical === 1) ?? transcripts[0];

    if (!canonical) return null;

    const consequences: string[] = canonical.consequence_terms ?? [];
    const topConsequence = consequences[0] ?? hit.most_severe_consequence ?? "unknown";
    const impact = (canonical.impact ?? "MODIFIER") as VEPAnnotation["impact"];

    // Classify variant into major functional categories for backend override logic
    const isFrameshift = consequences.some((c) => c.includes("frameshift"));
    const isSynonymous = consequences.some((c) => c.includes("synonymous"));
    const isNonsense = consequences.some((c) => c === "stop_gained");

    // Extract protein-level change in HGVS notation
    // Format: "ENST00000357654.8:p.Thr1074Ala" → "p.Thr1074Ala"
    const rawHgvsp = canonical.hgvsp ?? null;
    const aaChange = rawHgvsp
        ? (rawHgvsp.includes(":") ? rawHgvsp.split(":")[1]! : rawHgvsp)
        : null;

    return {
        consequence: topConsequence,
        impact,
        aaChange,
        codons: canonical.codons ?? null,
        aminoAcids: canonical.amino_acids ?? null,
        isSynonymous,
        isFrameshift,
        isNonsense,
        transcriptId: canonical.transcript_id ?? null,
        exonNumber: canonical.exon ?? null,
        geneId: canonical.gene_id ?? null,
    };
}

export async function POST(request: NextRequest) {
    try {
        const body = await request.json() as {
            chromosome: string;
            position: number;
            reference: string;
            alternative: string;
        };

        const { chromosome, position, reference, alternative } = body;
        
        // Log input for debugging and monitoring
        console.log("[VEP] Input variant:", JSON.stringify({
            chromosome,
            position,
            reference,
            alternative,
        }));

        if (!chromosome || !position || !reference || !alternative) {
            console.warn("[VEP] Missing required fields");
            return NextResponse.json({ vep: null }, { status: 200 });
        }

        // Reject IUPAC ambiguous nucleotide codes
        // Ensembl VEP requires unambiguous bases (A, C, G, T only)
        const ambiguousBases = /[NRYWSKMBDHV]/i;
        if (ambiguousBases.test(reference) || ambiguousBases.test(alternative)) {
            console.warn("[VEP] Ambiguous bases detected in variant:", { reference, alternative });
            return NextResponse.json({ vep: null }, { status: 200 });
        }

        const hgvs = buildHGVS(chromosome, position, reference, alternative);
        console.log("[VEP] Built HGVS:", hgvs);

        const url = `https://rest.ensembl.org/vep/human/hgvs/${encodeURIComponent(hgvs)}?canonical=1&content-type=application/json`;
        console.log("[VEP] Ensembl API URL:", url);

        // Timeout protection to prevent hanging requests
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 8000);

        let vep: VEPAnnotation | null = null;
        try {
            console.log("[VEP] Querying Ensembl VEP API...");
            const res = await fetch(url, { signal: controller.signal });
            clearTimeout(timeout);

            console.log("[VEP] Response status:", res.status, res.statusText);

            if (res.ok) {
                const data = await res.json() as Record<string, unknown>[];
                console.log("[VEP] Response data length:", data.length);
                console.log("[VEP] Raw response:", JSON.stringify(data).substring(0, 500)); // First 500 chars
                
                vep = parseConsequence(data);
                console.log("[VEP] Parsed consequence:", vep ? JSON.stringify(vep) : "null");
            } else {
                console.warn("[VEP] Ensembl returned non-OK status:", res.status);
                const errorText = await res.text();
                console.warn("[VEP] Error response:", errorText.substring(0, 300));
            }
        } catch (error) {
            clearTimeout(timeout);
            const errorMsg = error instanceof Error ? error.message : String(error);
            console.error("[VEP] Fetch error:", errorMsg);
            if (errorMsg.includes("abort")) {
                console.error("[VEP] Request timed out (8s limit)");
            }
        }

        console.log("[VEP] Final result - vep:", vep);
        return NextResponse.json({ vep }, { status: 200 });
    } catch (error) {
        const errorMsg = error instanceof Error ? error.message : String(error);
        console.error("[VEP] POST handler error:", errorMsg);
        return NextResponse.json({ vep: null }, { status: 200 });
    }
}
