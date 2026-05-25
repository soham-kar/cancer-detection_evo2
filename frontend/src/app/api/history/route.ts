import { NextRequest, NextResponse } from "next/server";
import { currentUser } from "@clerk/nextjs/server";
import { db } from "~/lib/db";

export async function GET(request: NextRequest) {
    try {
        const user = await currentUser();

        if (!user) {
            return NextResponse.json(
                { error: "Unauthorized" },
                { status: 401 }
            );
        }

        // Get query params for optional filtering
        const { searchParams } = new URL(request.url);
        const reportId = searchParams.get("id");
        const geneSymbol = searchParams.get("gene");
        const limit = parseInt(searchParams.get("limit") || "50", 10);

        // If ID is provided, fetch a single report
        if (reportId) {
            try {
                const report = await db.analysisReport.findFirst({
                    where: {
                        id: reportId,
                        clerkUserId: user.id,
                    },
                });

                if (!report) {
                    return NextResponse.json(
                        { error: "Report not found" },
                        { status: 404 }
                    );
                }

                // Parse JSON string fields for frontend consumption
                const parsedReport = {
                    ...report,
                    populationFrequency: report.populationFrequency ? JSON.parse(report.populationFrequency) : null,
                    acmgEvidence: report.acmgEvidence ? JSON.parse(report.acmgEvidence) : null,
                    literatureContext: report.literatureContext ? JSON.parse(report.literatureContext) : null,
                    evidenceConfidence: report.evidenceConfidence ? JSON.parse(report.evidenceConfidence) : null,
                    vepAnnotation: report.vepAnnotation ? JSON.parse(report.vepAnnotation) : null,
                    ismScanData: report.ismScanData ? JSON.parse(report.ismScanData) : null,
                    xaiFactors: report.xaiFactors ? JSON.parse(report.xaiFactors) : null,
                    counterfactuals: report.counterfactuals ? JSON.parse(report.counterfactuals) : null,
                    acmgCriteria: report.acmgCriteria ? JSON.parse(report.acmgCriteria) : null,
                    knowledgeGraph: report.knowledgeGraph ? JSON.parse(report.knowledgeGraph) : null,
                    externalScores: report.externalScores ? JSON.parse(report.externalScores) : null,
                    acmgCriteriaRefined: report.acmgCriteriaRefined ? JSON.parse(report.acmgCriteriaRefined) : null,
                };

                return NextResponse.json({ report: parsedReport });
            } catch (dbError) {
                console.warn("[DB] Database unavailable for history:", (dbError as Error).message);
                return NextResponse.json({ reports: [], total: 0, dbAvailable: false });
            }
        }

        // Fetch user's analysis history
        try {
            const reports = await db.analysisReport.findMany({
                where: {
                    clerkUserId: user.id,
                    ...(geneSymbol && { geneSymbol }),
                },
                orderBy: {
                    createdAt: "desc",
                },
                take: limit,
            });

            return NextResponse.json({
                reports: reports.map(r => ({
                    ...r,
                    populationFrequency: r.populationFrequency ? JSON.parse(r.populationFrequency) : null,
                    acmgEvidence: r.acmgEvidence ? JSON.parse(r.acmgEvidence) : null,
                    literatureContext: r.literatureContext ? JSON.parse(r.literatureContext) : null,
                    evidenceConfidence: r.evidenceConfidence ? JSON.parse(r.evidenceConfidence) : null,
                    vepAnnotation: r.vepAnnotation ? JSON.parse(r.vepAnnotation) : null,
                    ismScanData: r.ismScanData ? JSON.parse(r.ismScanData) : null,
                    xaiFactors: r.xaiFactors ? JSON.parse(r.xaiFactors) : null,
                    counterfactuals: r.counterfactuals ? JSON.parse(r.counterfactuals) : null,
                    acmgCriteria: r.acmgCriteria ? JSON.parse(r.acmgCriteria) : null,
                    knowledgeGraph: r.knowledgeGraph ? JSON.parse(r.knowledgeGraph) : null,
                    externalScores: r.externalScores ? JSON.parse(r.externalScores) : null,
                    acmgCriteriaRefined: r.acmgCriteriaRefined ? JSON.parse(r.acmgCriteriaRefined) : null,
                })),
                total: reports.length,
            });
        } catch (dbError) {
            console.warn("[DB] Database unavailable for history:", (dbError as Error).message);
            return NextResponse.json({ reports: [], total: 0, dbAvailable: false });
        }

    } catch (error) {
        console.error("History API error:", error);
        return NextResponse.json(
            { error: "Internal server error" },
            { status: 500 }
        );
    }
}

// Delete a specific report
export async function DELETE(request: NextRequest) {
    try {
        const user = await currentUser();

        if (!user) {
            return NextResponse.json(
                { error: "Unauthorized" },
                { status: 401 }
            );
        }

        const { searchParams } = new URL(request.url);
        const reportId = searchParams.get("id");

        if (!reportId) {
            return NextResponse.json(
                { error: "Report ID is required" },
                { status: 400 }
            );
        }

        // Verify ownership before deletion
        const report = await db.analysisReport.findFirst({
            where: {
                id: reportId,
                clerkUserId: user.id,
            },
        });

        if (!report) {
            return NextResponse.json(
                { error: "Report not found" },
                { status: 404 }
            );
        }

        await db.analysisReport.delete({
            where: { id: reportId },
        });

        return NextResponse.json({ success: true });

    } catch (error) {
        console.error("Delete report error:", error);
        return NextResponse.json(
            { error: "Internal server error" },
            { status: 500 }
        );
    }
}
