import { NextRequest, NextResponse } from "next/server";
import { getOrCreateUser, deductCredit, incrementFreeRuns, setCooldown } from "~/lib/user-utils";
import { db } from "~/lib/db";

const FREE_LIMIT_PER_DAY = 10;
const COOLDOWN_DAYS = 2;

// Modal backend URL
const MODAL_API_URL = process.env.NEXT_PUBLIC_ANALYZE_SINGLE_VARIANT_BASE_URL;

interface AnalysisRequestBody {
    variant_position: number;
    alternative: string;
    genome: string;
    chromosome: string;
    reference?: string;
    gene_symbol?: string;
    clinvar_classification?: string;
    variation_type?: string;
    clinvar_id?: string;
    analysis_source?: 'clinvar' | 'custom';
}

async function callModalAnalysis(body: AnalysisRequestBody) {
    if (!MODAL_API_URL) {
        throw new Error("MODAL_API_URL not configured");
    }

    console.log("🔬 Calling Modal API:", { url: MODAL_API_URL, body });

    const response = await fetch(MODAL_API_URL, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify(body),
    });

    if (!response.ok) {
        const errorText = await response.text();
        console.error("❌ Modal API Error:", errorText);
        throw new Error(`Modal API failed: ${errorText}`);
    }

    const result = await response.json();
    console.log("✅ Modal API Response:", result);
    return result;
}

export async function POST(request: NextRequest) {
    try {
        // Get or create user
        const user = await getOrCreateUser();

        // COOLDOWN CHECK: If cooldown is active, reject
        if (user.cooldownUntil && new Date(user.cooldownUntil) > new Date()) {
            const remainingTime = Math.ceil(
                (new Date(user.cooldownUntil).getTime() - Date.now()) / (1000 * 60 * 60)
            );
            return NextResponse.json(
                {
                    error: "cooldown_active",
                    message: `You've used all free credits. Please wait ${remainingTime} hours or purchase credits.`,
                    cooldownUntil: user.cooldownUntil,
                    needsCredits: true,
                },
                { status: 429 }
            );
        }

        // FREE LIMIT CHECK: If no paid credits and free limit reached
        if (user.freeRunsToday >= FREE_LIMIT_PER_DAY && user.credits <= 0) {
            // Set cooldown for 2 days
            const cooldownUntil = new Date();
            cooldownUntil.setDate(cooldownUntil.getDate() + COOLDOWN_DAYS);

            await setCooldown(user.clerkId, cooldownUntil);

            return NextResponse.json(
                {
                    error: "free_limit_reached",
                    message: `You've used your ${FREE_LIMIT_PER_DAY} free analyses today. Purchase credits or wait 2 days.`,
                    cooldownUntil,
                    needsCredits: true,
                },
                { status: 429 }
            );
        }

        // CREDIT DEDUCTION LOGIC
        let creditUsed: "paid" | "free";

        if (user.credits > 0) {
            // Use paid credits first
            await deductCredit(user.clerkId);
            creditUsed = "paid";
        } else {
            // Use free credit
            await incrementFreeRuns(user.clerkId);
            creditUsed = "free";
        }

        // Get the request body for the actual analysis
        const body = await request.json() as AnalysisRequestBody;

        // Call the Modal/GPU analysis endpoint
        const analysisResult = await callModalAnalysis(body);

        // Save the analysis result to database
        try {
            await db.analysisReport.create({
                data: {
                    clerkUserId: user.clerkId,
                    geneSymbol: body.gene_symbol || "Unknown",
                    chromosome: body.chromosome,
                    position: body.variant_position,
                    reference: body.reference || "",
                    alternative: body.alternative,
                    genomeId: body.genome,
                    prediction: analysisResult.prediction || "",
                    deltaScore: analysisResult.delta_score || 0,
                    classificationConfidence: analysisResult.classification_confidence || 0,
                    classificationSource: analysisResult.classification_source || null,
                    clinvarClassification: body.clinvar_classification || null,
                    variationType: body.variation_type || null,
                    clinvarId: body.clinvar_id || null,
                    populationFrequency: analysisResult.population_frequency || null,
                    acmgEvidence: analysisResult.acmg_evidence || null,
                    literatureContext: analysisResult.literature_context || null,
                    analysisSource: body.analysis_source || null,
                },
            });
            console.log("✅ Analysis saved to database");
        } catch (saveError) {
            console.error("Failed to save analysis to database:", saveError);
            // Don't fail the request if save fails - the analysis was successful
        }

        // Calculate updated credits
        const updatedCredits = creditUsed === "paid" ? user.credits - 1 : user.credits;
        const updatedFreeRuns = creditUsed === "free" ? user.freeRunsToday + 1 : user.freeRunsToday;

        // Return success with analysis result and credit info
        return NextResponse.json({
            success: true,
            creditUsed,
            remainingCredits: updatedCredits,
            freeRunsToday: updatedFreeRuns,
            freeRunsRemaining: FREE_LIMIT_PER_DAY - updatedFreeRuns,
            ...analysisResult, // Spread the Modal API response
        });

    } catch (error) {
        console.error("Analysis API error:", error);

        if (error instanceof Error && error.message === "Unauthorized: No user found") {
            return NextResponse.json(
                { error: "unauthorized", message: "Please sign in to continue" },
                { status: 401 }
            );
        }

        if (error instanceof Error && error.message.includes("Modal API failed")) {
            return NextResponse.json(
                { error: "analysis_failed", message: error.message },
                { status: 502 }
            );
        }

        return NextResponse.json(
            { error: "internal_error", message: "Internal server error" },
            { status: 500 }
        );
    }
}

// GET endpoint to check user's credit status
export async function GET() {
    try {
        const user = await getOrCreateUser();

        return NextResponse.json({
            credits: user.credits,
            freeRunsToday: user.freeRunsToday,
            freeRunsRemaining: Math.max(0, FREE_LIMIT_PER_DAY - user.freeRunsToday),
            cooldownUntil: user.cooldownUntil,
            isOnCooldown: user.cooldownUntil ? new Date(user.cooldownUntil) > new Date() : false,
        });

    } catch (error) {
        if (error instanceof Error && error.message === "Unauthorized: No user found") {
            return NextResponse.json(
                { error: "Unauthorized" },
                { status: 401 }
            );
        }
        return NextResponse.json(
            { error: "Internal server error" },
            { status: 500 }
        );
    }
}
