// =============================================================================
// User Management Utilities - Credit System & Authentication
// =============================================================================
// Helper functions for user authentication, credit management, and access control.
// 
// Credit System:
//   - New users: 5 paid credits + 10 free analyses per day
//   - Paid credits: Purchased via Stripe, no expiration
//   - Free credits: Daily allowance, resets at midnight UTC
//   - Cooldown: 48-hour lockout after free limit exhausted
// 
// Database Operations:
//   - User creation and retrieval
//   - Credit balance management
//   - Free tier tracking and cooldown enforcement
//   - Stripe integration for payment processing
// =============================================================================

import { currentUser } from "@clerk/nextjs/server";
import { db } from "./db";

/**
 * Get authenticated user from Clerk and ensure database record exists.
 * 
 * This function:
 *   1. Retrieves current authenticated user from Clerk session
 *   2. Looks up user in local database by Clerk ID
 *   3. Creates new database record if first-time user
 * 
 * New User Initialization:
 *   - 5 paid credits (promotional welcome bonus)
 *   - 0 free runs used today
 *   - No cooldown restriction
 * 
 * @returns Promise resolving to User database record
 * @throws Error if no authenticated user found (should be protected by middleware)
 */
export async function getOrCreateUser() {
    const clerkUser = await currentUser();

    if (!clerkUser) {
        throw new Error("Unauthorized: No user found");
    }

    // Check if user record exists in local database
    let dbUser = await db.user.findUnique({
        where: { clerkId: clerkUser.id },
    });

    // Create new user record on first authentication
    if (!dbUser) {
        dbUser = await db.user.create({
            data: {
                clerkId: clerkUser.id,
                email: clerkUser.emailAddresses[0]?.emailAddress ?? "",
                credits: 5,
                freeRunsToday: 0,
            },
        });
    }

    return dbUser;
}

/**
 * Get user's credit balance and access restrictions.
 * 
 * Returns:
 *   - credits: Paid credit balance
 *   - freeRunsToday: Number of free analyses used today
 *   - cooldownUntil: Cooldown expiration timestamp (null if no cooldown)
 * 
 * @param clerkId - Clerk user identifier
 * @returns Promise resolving to credit info or null if user not found
 */
export async function getUserCredits(clerkId: string) {
    const user = await db.user.findUnique({
        where: { clerkId },
        select: { credits: true, freeRunsToday: true, cooldownUntil: true },
    });

    return user;
}

/**
 * Deduct one paid credit from user's balance.
 * 
 * Used when user has paid credits available (credits > 0).
 * Preferred over free credits to incentivize purchases.
 * 
 * @param clerkId - Clerk user identifier
 * @returns Promise resolving to updated user record
 */
export async function deductCredit(clerkId: string) {
    return db.user.update({
        where: { clerkId },
        data: {
            credits: { decrement: 1 },
        },
    });
}

/**
 * Increment free analysis counter for daily free tier tracking.
 * 
 * Used when user has no paid credits (credits = 0).
 * Counter resets to 0 at midnight UTC via scheduled job.
 * 
 * Limit: 10 free analyses per day before cooldown activation.
 * 
 * @param clerkId - Clerk user identifier
 * @returns Promise resolving to updated user record
 */
export async function incrementFreeRuns(clerkId: string) {
    return db.user.update({
        where: { clerkId },
        data: {
            freeRunsToday: { increment: 1 },
        },
    });
}

/**
 * Activate cooldown period for user who exhausted free tier.
 * 
 * Cooldown prevents analysis requests for 48 hours after free limit reached.
 * Encourages credit purchase or wait period to prevent abuse.
 * 
 * Cooldown is cleared when:
 *   - User purchases credits (automatic)
 *   - Cooldown period expires (48 hours)
 * 
 * @param clerkId - Clerk user identifier
 * @param until - Cooldown expiration timestamp
 * @returns Promise resolving to updated user record
 */
export async function setCooldown(clerkId: string, until: Date) {
    return db.user.update({
        where: { clerkId },
        data: {
            cooldownUntil: until,
            freeRunsToday: 0, // Reset counter when activating cooldown
        },
    });
}

/**
 * Add paid credits to user's balance after successful purchase.
 * 
 * Called by Stripe webhook after payment confirmation.
 * Automatically clears any active cooldown and resets free tier counter.
 * 
 * Credit packages:
 *   - 50 credits: $19.99
 *   - 100 credits: $34.99
 *   - 500 credits: $149.99
 * 
 * @param clerkId - Clerk user identifier
 * @param amount - Number of credits to add
 * @returns Promise resolving to updated user record
 */
export async function addCredits(clerkId: string, amount: number) {
    return db.user.update({
        where: { clerkId },
        data: {
            credits: { increment: amount },
            cooldownUntil: null, // Clear any active cooldown
            freeRunsToday: 0,     // Reset free tier counter
        },
    });
}
