import { headers } from "next/headers";
import { NextResponse } from "next/server";
import { stripe } from "~/lib/stripe";
import { db } from "~/lib/db";
import Stripe from "stripe";

// Disable body parsing to get raw body for signature verification
export const config = {
    api: {
        bodyParser: false,
    },
};

export async function POST(request: Request) {
    const body = await request.text();
    const headersList = await headers();
    const signature = headersList.get("stripe-signature");

    if (!signature) {
        return NextResponse.json(
            { error: "No signature provided" },
            { status: 400 }
        );
    }

    const webhookSecret = process.env.STRIPE_WEBHOOK_SECRET;

    if (!webhookSecret) {
        console.error("STRIPE_WEBHOOK_SECRET not configured");
        return NextResponse.json(
            { error: "Webhook secret not configured" },
            { status: 500 }
        );
    }

    let event: Stripe.Event;

    try {
        event = stripe.webhooks.constructEvent(body, signature, webhookSecret);
    } catch (err) {
        console.error("Webhook signature verification failed:", err);
        return NextResponse.json(
            { error: "Invalid signature" },
            { status: 400 }
        );
    }

    // Handle the event
    switch (event.type) {
        case "checkout.session.completed": {
            const session = event.data.object as Stripe.Checkout.Session;

            // Only process paid sessions
            if (session.payment_status !== "paid") {
                console.log("Session not paid yet, skipping");
                break;
            }

            const clerkId = session.metadata?.clerkId;
            const creditsToGrant = parseInt(session.metadata?.creditsToGrant || "5", 10);
            const sessionId = session.id;

            if (!clerkId) {
                console.error("No clerkId in session metadata");
                break;
            }

            // Check if already processed (idempotency)
            const existingPayment = await db.payment.findUnique({
                where: { sessionId },
            });

            if (existingPayment) {
                console.log(`Payment ${sessionId} already processed, skipping`);
                break;
            }

            // Grant credits
            try {
                await db.user.update({
                    where: { clerkId },
                    data: {
                        credits: { increment: creditsToGrant },
                        cooldownUntil: null,
                        freeRunsToday: 0,
                    },
                });

                // Log the transaction
                await db.payment.create({
                    data: {
                        sessionId,
                        creditsGranted: creditsToGrant,
                        clerkUserId: clerkId,
                    },
                });

                console.log(`Granted ${creditsToGrant} credits to user ${clerkId} via webhook`);
            } catch (error) {
                console.error("Error granting credits via webhook:", error);
            }

            break;
        }

        case "checkout.session.expired": {
            console.log("Checkout session expired:", event.data.object.id);
            break;
        }

        default:
            console.log(`Unhandled event type: ${event.type}`);
    }

    return NextResponse.json({ received: true });
}
