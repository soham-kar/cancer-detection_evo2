import { NextResponse } from "next/server";
import { stripe } from "~/lib/stripe";
import { db } from "~/lib/db";
import { addCredits } from "~/lib/user-utils";

export async function POST(request: Request) {
    try {
        const body = await request.json();
        const { sessionId } = body;

        if (!sessionId) {
            return NextResponse.json(
                { error: "Session ID is required" },
                { status: 400 }
            );
        }

        // IDEMPOTENCY CHECK: Check if this session has already been processed
        const existingPayment = await db.payment.findUnique({
            where: { sessionId },
        });

        if (existingPayment) {
            // Already processed - return success without granting credits again
            return NextResponse.json({
                success: true,
                message: "Payment already processed",
                creditsGranted: existingPayment.creditsGranted,
                alreadyProcessed: true,
            });
        }

        // Retrieve the session from Stripe to verify payment
        const session = await stripe.checkout.sessions.retrieve(sessionId);

        if (session.payment_status !== "paid") {
            return NextResponse.json(
                { error: "Payment not completed" },
                { status: 400 }
            );
        }

        // Get user info from session metadata
        const clerkId = session.metadata?.clerkId;
        const creditsToGrant = parseInt(session.metadata?.creditsToGrant || "5", 10);

        if (!clerkId) {
            return NextResponse.json(
                { error: "Invalid session: No user ID found" },
                { status: 400 }
            );
        }

        // Verify user exists
        const user = await db.user.findUnique({
            where: { clerkId },
        });

        if (!user) {
            return NextResponse.json(
                { error: "User not found" },
                { status: 404 }
            );
        }

        // GRANT CREDITS
        await addCredits(clerkId, creditsToGrant);

        // LOG TRANSACTION for idempotency
        await db.payment.create({
            data: {
                sessionId,
                creditsGranted: creditsToGrant,
                clerkUserId: clerkId,
            },
        });

        return NextResponse.json({
            success: true,
            message: `Successfully granted ${creditsToGrant} credits`,
            creditsGranted: creditsToGrant,
            alreadyProcessed: false,
        });

    } catch (error) {
        console.error("Grant credits error:", error);

        // Handle Stripe errors
        if (error instanceof Error && error.message.includes("No such checkout.session")) {
            return NextResponse.json(
                { error: "Invalid session ID" },
                { status: 400 }
            );
        }

        return NextResponse.json(
            { error: "Failed to grant credits" },
            { status: 500 }
        );
    }
}
