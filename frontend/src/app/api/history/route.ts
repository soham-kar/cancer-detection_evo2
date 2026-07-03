import { NextRequest, NextResponse } from "next/server";
import { currentUser } from "@clerk/nextjs/server";
import { db } from "~/lib/db";

export async function GET(request: NextRequest) {
  try {
    const user = await currentUser();

    if (!user) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
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
            { status: 404 },
          );
        }

        // Native JSON fields are already parsed by Prisma
        return NextResponse.json({ report });
      } catch (dbError) {
        console.warn(
          "[DB] Database unavailable for history:",
          (dbError as Error).message,
        );
        return NextResponse.json({ reports: [], total: 0, dbAvailable: false });
      }
    }

    // Fetch user's analysis history
    try {
      console.log(
        "[History API] Querying for user:",
        user.id,
        "gene:",
        geneSymbol,
      );
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
      console.log("[History API] Found", reports.length, "reports");

      return NextResponse.json({
        reports,
        total: reports.length,
      });
    } catch (dbError) {
      console.warn(
        "[DB] Database unavailable for history:",
        (dbError as Error).message,
      );
      return NextResponse.json({ reports: [], total: 0, dbAvailable: false });
    }
  } catch (error) {
    console.error("History API error:", error);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 },
    );
  }
}

// Delete a specific report
export async function DELETE(request: NextRequest) {
  try {
    const user = await currentUser();

    if (!user) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const { searchParams } = new URL(request.url);
    const reportId = searchParams.get("id");

    if (!reportId) {
      return NextResponse.json(
        { error: "Report ID is required" },
        { status: 400 },
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
      return NextResponse.json({ error: "Report not found" }, { status: 404 });
    }

    await db.analysisReport.delete({
      where: { id: reportId },
    });

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error("Delete report error:", error);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 },
    );
  }
}
