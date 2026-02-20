import { NextResponse } from "next/server";
import { getOrCreateUser } from "~/lib/user-utils";
import { stripe } from "~/lib/stripe";
import { db } from "~/lib/db";

// Credit packages
const CREDIT_PACKAGES = [
    { id: "credits_5", credits: 5, priceInCents: 500, name: "5 Credits" },
    { id: "credits_20", credits: 20, priceInCents: 1500, name: "20 Credits" },
    { id: "credits_50", credits: 50, priceInCents: 3000, name: "50 Credits" },
];

export async function POST(request: Request) {
    try {
        const user = await getOrCreateUser();
        const body = await request.json();
        const packageId = body.packageId || "credits_5"; // Default to 5 credits

        // Find the selected package
        const selectedPackage = CREDIT_PACKAGES.find((pkg) => pkg.id === packageId);
        if (!selectedPackage) {
            return NextResponse.json(
                { error: "Invalid package selected" },
                { status: 400 }
            );
        }

        // Create or retrieve Stripe customer
        let stripeCustomerId = user.stripeCustomerId;

        if (!stripeCustomerId) {
            const customer = await stripe.customers.create({
                email: user.email,
                metadata: {
                    clerkId: user.clerkId,
                },
            });
            stripeCustomerId = customer.id;

            // Update user with Stripe customer ID
            await db.user.update({
                where: { clerkId: user.clerkId },
                data: { stripeCustomerId },
            });
        }

        // Create Stripe Checkout Session
        const session = await stripe.checkout.sessions.create({
            customer: stripeCustomerId,
            payment_method_types: ["card"],
            line_items: [
                {
                    price_data: {
                        currency: "usd",
                        product_data: {
                            name: selectedPackage.name,
                            description: `${selectedPackage.credits} analysis credits for HelixMind`,
                        },
                        unit_amount: selectedPackage.priceInCents,
                    },
                    quantity: 1,
                },
            ],
            mode: "payment",
            success_url: `${process.env.NEXT_PUBLIC_APP_URL || "http://localhost:3000"}/billing/success?session_id={CHECKOUT_SESSION_ID}`,
            cancel_url: `${process.env.NEXT_PUBLIC_APP_URL || "http://localhost:3000"}/billing/cancel`,
            metadata: {
                clerkId: user.clerkId,
                packageId: selectedPackage.id,
                creditsToGrant: selectedPackage.credits.toString(),
            },
        });

        return NextResponse.json({ url: session.url });

    } catch (error) {
        console.error("Stripe checkout error:", error);

        if (error instanceof Error && error.message === "Unauthorized: No user found") {
            return NextResponse.json(
                { error: "Unauthorized", message: "Please sign in to continue" },
                { status: 401 }
            );
        }

        return NextResponse.json(
            { error: "Failed to create checkout session" },
            { status: 500 }
        );
    }
}

// GET endpoint to retrieve available packages
export async function GET() {
    return NextResponse.json({ packages: CREDIT_PACKAGES });
}
