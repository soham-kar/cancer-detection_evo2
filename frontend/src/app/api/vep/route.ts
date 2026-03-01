/**
 * /api/vep — Server-side Ensembl VEP proxy
 *
 * Accepts: POST { chromosome, position, reference, alternative }
 * Returns: VEPAnnotation or null (failure never blocks analysis)
 *
 * Uses canonical transcript only for consistency.
 * Ensembl REST is free, no API key needed.
 */
import type { NextRequest } from "next/server";
import { NextResponse } from "next/server";

export interface VEPAnnotation {
    consequence: string;        // "synonymous_variant" | "missense_variant" | "frameshift_variant" | ...
    impact: "HIGH" | "MODERATE" | "LOW" | "MODIFIER";
    aaChange: string | null;    // "p.Thr1074=" | "p.Thr1074Ala" | "p.Thr1074AsnfsTer12"
    codons: string | null;      // "acA/acG" — ref/alt codon (lower case = unchanged base)
    aminoAcids: string | null;  // "T" (synonymous) or "T/A" (ref/alt AA)
    isSynonymous: boolean;
    isFrameshift: boolean;
    isNonsense: boolean;        // stop_gained
    transcriptId: string | null;
    exonNumber: string | null;  // "10/23"
    geneId: string | null;
}

// Build HGVS notation for VEP query
function buildHGVS(chrom: string, pos: number, ref: string, alt: string): string {
    // Strip "chr" prefix for Ensembl
    const c = chrom.replace(/^chr/i, "");

    console.log("[VEP-HGVS] Input: chrom=" + chrom + " pos=" + pos + " ref=" + ref + " alt=" + alt);
    console.log("[VEP-HGVS] ref.length=" + ref.length + " alt.length=" + alt.length);

    if (ref.length === 1 && alt.length === 1) {
        // SNV: 17:g.43093278T>A
        const result = `${c}:g.${pos}${ref}>${alt}`;
        console.log("[VEP-HGVS] Branch: SNV →", result);
        return result;
    }
    if (alt.length > ref.length && alt.startsWith(ref)) {
        // Insertion / duplication: 17:g.43093278_43093279insNN
        const inserted = alt.slice(ref.length);
        const result = `${c}:g.${pos}_${pos + ref.length}ins${inserted}`;
        console.log("[VEP-HGVS] Branch: Insertion →", result);
        return result;
    }
    if (ref.length > alt.length && ref.startsWith(alt)) {
        // Deletion: 17:g.43093279_43093280del
        const delStart = pos + alt.length;
        const delEnd = pos + ref.length - 1;
        const result = `${c}:g.${delStart}_${delEnd}del`;
        console.log("[VEP-HGVS] Branch: Deletion →", result);
        return result;
    }
    // Indel fallback
    const result = `${c}:g.${pos}${ref}>${alt}`;
    console.log("[VEP-HGVS] Branch: Fallback →", result);
    return result;
}

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

    // Prefer canonical transcript
    const transcripts = hit.transcript_consequences ?? [];
    const canonical = transcripts.find((t) => t.canonical === 1) ?? transcripts[0];

    if (!canonical) return null;

    const consequences: string[] = canonical.consequence_terms ?? [];
    const topConsequence = consequences[0] ?? hit.most_severe_consequence ?? "unknown";
    const impact = (canonical.impact ?? "MODIFIER") as VEPAnnotation["impact"];

    const isFrameshift = consequences.some((c) => c.includes("frameshift"));
    const isSynonymous = consequences.some((c) => c.includes("synonymous"));
    const isNonsense = consequences.some((c) => c === "stop_gained");

    // aaChange: use hgvsp, strip transcript prefix (e.g. "ENST000…p.Thr123Ala" → "p.Thr123Ala")
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
        
        // Log input
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

        // Reject ambiguous bases (N, R, Y, W, S, K, M, B, D, H, V) — Ensembl can't handle them
        const ambiguousBases = /[NRYWSKMBDHV]/i;
        if (ambiguousBases.test(reference) || ambiguousBases.test(alternative)) {
            console.warn("[VEP] Ambiguous bases detected in variant:", { reference, alternative });
            return NextResponse.json({ vep: null }, { status: 200 });
        }

        const hgvs = buildHGVS(chromosome, position, reference, alternative);
        console.log("[VEP] Built HGVS:", hgvs);

        const url = `https://rest.ensembl.org/vep/human/hgvs/${encodeURIComponent(hgvs)}?canonical=1&content-type=application/json`;
        console.log("[VEP] Ensembl API URL:", url);

        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 8000); // 8s timeout

        let vep: VEPAnnotation | null = null;
        try {
            console.log("[VEP] Fetching from Ensembl...");
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
