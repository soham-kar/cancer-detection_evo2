import { NextRequest, NextResponse } from "next/server";

// NEW: Use the real Evo2 analysis endpoint with multi-modal RAG
const MODAL_API_URL = process.env.NEXT_PUBLIC_ANALYZE_SINGLE_VARIANT_BASE_URL;

// Demo variants with pre-calculated scores (mock GPU results)
const DEMO_VARIANTS: Record<string, {
    gene: string;
    variant: string;
    chromosome: string;
    position: number;
    reference: string;
    alternative: string;
    mockScore: number;
    prediction: string;
    confidence: number;
}> = {
    "brca1": {
        gene: "BRCA1",
        variant: "c.68_69delAG",
        chromosome: "chr17",
        position: 43047665,
        reference: "CAG",
        alternative: "C",
        mockScore: -0.0523,
        prediction: "Pathogenic",
        confidence: 0.97,
    },
    "tp53": {
        gene: "TP53",
        variant: "R248W",
        chromosome: "chr17",
        position: 7674220,
        reference: "C",
        alternative: "T",
        mockScore: -0.0341,
        prediction: "Likely Pathogenic",
        confidence: 0.94,
    },
};

export async function POST(request: NextRequest) {
    try {
        const body = await request.json();
        const variantKey = (body.variantId || "brca1").toLowerCase();

        // Get demo variant data
        const demoVariant = DEMO_VARIANTS[variantKey];
        if (!demoVariant) {
            return NextResponse.json(
                { error: "Invalid demo variant" },
                { status: 400 }
            );
        }

        console.log("🚀 Guest Demo: Analyzing", demoVariant.gene, demoVariant.variant);

        // NEW: Call the real Evo2 + Multi-Modal RAG endpoint
        let analysisResult = null;
        try {
            console.log("📡 Calling Evo2 Analysis API...");
            const apiResponse = await fetch(MODAL_API_URL!, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    variant_position: demoVariant.position,
                    alternative: demoVariant.alternative,
                    genome: "hg38",
                    chromosome: demoVariant.chromosome,
                    gene_symbol: demoVariant.gene,
                }),
            });

            if (apiResponse.ok) {
                analysisResult = await apiResponse.json();
                console.log("✅ Analysis Success:", analysisResult.prediction);
            } else {
                console.error("❌ Analysis Error:", await apiResponse.text());
            }
        } catch (apiError) {
            console.error("⚠️ Analysis call failed:", apiError);
        }

        // Return combined result (mock GPU + real analysis)
        return NextResponse.json({
            success: true,
            isDemo: true,
            variant: {
                gene: demoVariant.gene,
                variant: demoVariant.variant,
                chromosome: demoVariant.chromosome,
                position: demoVariant.position,
                reference: demoVariant.reference,
                alternative: demoVariant.alternative,
            },
            // Mock GPU results (saves H100 credits)
            prediction: demoVariant.prediction,
            delta_score: demoVariant.mockScore,
            classification_confidence: demoVariant.confidence,
            classification_source: "demo_mock",
            // NEW: Real Multi-Modal RAG results
            clinical_summary: analysisResult?.clinical_summary ?? null,
            evidence_confidence: analysisResult?.evidence_confidence ?? null,
            // Legacy RAG structure for backward compatibility with demo UI
            rag: analysisResult ? {
                summary: analysisResult.clinical_summary || analysisResult.literature_context?.summary || "",
                clinvar_status: analysisResult.clinvar_evidence?.status || "Unknown",
                pmids: analysisResult.literature_context?.pubmed_ids || [],
                protein_function: analysisResult.protein_context?.function || "",
                sources: {
                    clinvar: { status: analysisResult.clinvar_evidence?.status || "Unknown", id: analysisResult.clinvar_evidence?.variation_id || "" },
                    uniprot: { id: analysisResult.protein_context?.accession || "", has_function: !!analysisResult.protein_context?.function },
                    pubmed: { count: analysisResult.literature_context?.articles_found || 0, level: analysisResult.evidence_confidence?.pubmed?.confidence || "N/A" },
                },
            } : null,
        });

    } catch (error) {
        console.error("Demo API error:", error);
        return NextResponse.json(
            { error: "Demo analysis failed" },
            { status: 500 }
        );
    }
}

// GET endpoint to list available demo variants
export async function GET() {
    return NextResponse.json({
        variants: Object.entries(DEMO_VARIANTS).map(([key, v]) => ({
            id: key,
            gene: v.gene,
            variant: v.variant,
            prediction: v.prediction,
        })),
    });
}
