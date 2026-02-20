import { currentUser } from "@clerk/nextjs/server";
import { db } from "./db";

export async function getOrCreateUser() {
    const clerkUser = await currentUser();

    if (!clerkUser) {
        throw new Error("Unauthorized: No user found");
    }

    // Check if user exists in our database
    let dbUser = await db.user.findUnique({
        where: { clerkId: clerkUser.id },
    });

    // If user doesn't exist, create them with default credits
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

export async function getUserCredits(clerkId: string) {
    const user = await db.user.findUnique({
        where: { clerkId },
        select: { credits: true, freeRunsToday: true, cooldownUntil: true },
    });

    return user;
}

export async function deductCredit(clerkId: string) {
    return db.user.update({
        where: { clerkId },
        data: {
            credits: { decrement: 1 },
        },
    });
}

export async function incrementFreeRuns(clerkId: string) {
    return db.user.update({
        where: { clerkId },
        data: {
            freeRunsToday: { increment: 1 },
        },
    });
}

export async function setCooldown(clerkId: string, until: Date) {
    return db.user.update({
        where: { clerkId },
        data: {
            cooldownUntil: until,
            freeRunsToday: 0, // Reset free runs when cooldown is set
        },
    });
}

export async function addCredits(clerkId: string, amount: number) {
    return db.user.update({
        where: { clerkId },
        data: {
            credits: { increment: amount },
            cooldownUntil: null, // Clear cooldown when credits are purchased
            freeRunsToday: 0, // Reset free runs
        },
    });
}
