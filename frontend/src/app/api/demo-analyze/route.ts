import { NextRequest, NextResponse } from "next/server";

// Modal RAG URL (web endpoint)
const MODAL_RAG_URL = "https://karsoham529--multimodal-rag-search-and-synthesize-web.modal.run";

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

        // Call the real MultiModalRAG (cheap CPU, real value)
        let ragResult = null;
        try {
            console.log("📡 Calling MultiModalRAG...");
            const ragResponse = await fetch(MODAL_RAG_URL, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    gene: demoVariant.gene,
                    variant: demoVariant.variant,
                }),
            });

            if (ragResponse.ok) {
                ragResult = await ragResponse.json();
                console.log("✅ RAG Success:", ragResult.clinvar_status);
            } else {
                console.error("❌ RAG Error:", await ragResponse.text());
            }
        } catch (ragError) {
            console.error("⚠️ RAG call failed:", ragError);
        }

        // Return combined result (mock GPU + real RAG)
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
            // Real RAG results (the valuable part)
            rag: ragResult ? {
                summary: ragResult.summary,
                clinvar_status: ragResult.clinvar_status,
                pmids: ragResult.pmids || [],
                protein_function: ragResult.protein_function,
                sources: ragResult.sources,
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
