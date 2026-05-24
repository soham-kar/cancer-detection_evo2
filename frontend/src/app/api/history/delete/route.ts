import { NextRequest, NextResponse } from "next/server";
import { getOrCreateUser } from "~/lib/user-utils";
import { db } from "~/lib/db";

export async function DELETE(request: NextRequest) {
    try {
        let clerkId: string;
        try {
            const user = await getOrCreateUser();
            clerkId = user.clerkId;
        } catch (dbError) {
            console.warn("[DB] Database unavailable for delete:", (dbError as Error).message);
            return NextResponse.json({ success: true, dbAvailable: false });
        }

        const { reportId } = await request.json();

        if (!reportId) {
            return NextResponse.json(
                { error: "Report ID is required" },
                { status: 400 }
            );
        }

        // Verify the report belongs to this user
        const report = await db.analysisReport.findFirst({
            where: {
                id: reportId,
                clerkUserId: clerkId,
            },
        });

        if (!report) {
            return NextResponse.json(
                { error: "Report not found or access denied" },
                { status: 404 }
            );
        }

        // Delete the report
        await db.analysisReport.delete({
            where: { id: reportId },
        });

        return NextResponse.json({ success: true });
    } catch (error) {
        console.error("Delete report error:", error);

        if (error instanceof Error && error.message === "Unauthorized: No user found") {
            return NextResponse.json(
                { error: "Unauthorized" },
                { status: 401 }
            );
        }

        return NextResponse.json(
            { error: "Failed to delete report" },
            { status: 500 }
        );
    }
}
