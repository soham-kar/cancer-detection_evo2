// =============================================================================
// Prisma Database Client - PostgreSQL Connection
// =============================================================================
// Singleton Prisma client for database operations with connection pooling.
// 
// Features:
//   - Global singleton pattern prevents multiple client instances
//   - Development logging for debugging queries
//   - Production optimization with minimal logging
//   - Connection pool management for serverless environments
// 
// Database Schema:
//   - User: Authentication and credit management
//   - AnalysisReport: Variant analysis results and history
//   - Billing: Stripe payment tracking and subscriptions
// 
// Environment Variables Required:
//   - DATABASE_URL: PostgreSQL connection string
// =============================================================================

import { PrismaClient } from "@prisma/client";

/**
 * Global Prisma client singleton to prevent multiple instances in development.
 * 
 * Next.js hot-reloading in development creates new module instances,
 * which would create multiple Prisma clients and exhaust database connections.
 * This pattern stores the client in the global scope to persist across reloads.
 */
const globalForPrisma = globalThis as unknown as {
    prisma: PrismaClient | undefined;
};

/**
 * Prisma database client with environment-specific logging.
 * 
 * Development: Logs errors and warnings for debugging
 * Production: Logs only errors to minimize performance impact
 */
export const db =
    globalForPrisma.prisma ??
    new PrismaClient({
        log:
            process.env.NODE_ENV === "development"
                ? ["error", "warn"]
                : ["error"],
    });

// Store client in global scope in non-production environments
if (process.env.NODE_ENV !== "production") globalForPrisma.prisma = db;
