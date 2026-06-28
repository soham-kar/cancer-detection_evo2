// =============================================================================
// Next.js Middleware - Authentication & Route Protection
// =============================================================================
// Clerk-based authentication middleware for protecting routes and API endpoints.
//
// Authentication Strategy:
//   - Homepage (/): Protected - requires sign-in
//   - Demo routes: Public - guest access without authentication
//   - VEP API: Public - called from authenticated context
//   - Sign-in/Sign-up: Public - authentication flows
//
// Clerk Features:
//   - JWT-based session management
//   - Social OAuth providers (Google, GitHub, etc.)
//   - Email/password authentication
//   - Session protection and CSRF prevention
// =============================================================================

import { clerkMiddleware, createRouteMatcher } from "@clerk/nextjs/server";

/**
 * Define public routes accessible without authentication.
 *
 * Public Routes:
 *   - /demo: Guest mode for trying the platform
 *   - /api/demo-analyze: Demo analysis endpoint (limited functionality)
 *   - /api/vep: VEP annotation proxy (called from authenticated routes)
 *   - /sign-in, /sign-up: Authentication flows
 *
 * Protected Routes (default):
 *   - /: Homepage with full analysis features
 *   - /api/analyze: Production analysis endpoint
 *   - /history: Analysis history dashboard
 *   - /billing: Credit purchase and subscription management
 */
const isPublicRoute = createRouteMatcher([
  "/demo(.*)", // Demo page for guest mode
  "/api/demo-analyze", // Guest demo API endpoint
  "/api/vep", // VEP annotation (called from authenticated context)
  "/sign-in(.*)",
  "/sign-up(.*)",
]);

/**
 * Clerk middleware function protecting non-public routes.
 *
 * Behavior:
 *   - Public routes: Allow access without authentication
 *   - Protected routes: Redirect to /sign-in if not authenticated
 *   - Authenticated users: Allow access to all routes
 *
 * @param auth - Clerk authentication context
 * @param request - Next.js request object
 */
export default clerkMiddleware(async (auth, request) => {
  if (!isPublicRoute(request)) {
    await auth.protect();
  }
});

/**
 * Next.js middleware matcher configuration.
 *
 * Applies middleware to:
 *   - All pages except Next.js internals (_next/*, static assets)
 *   - All API routes (/api/*)
 *   - All tRPC routes (/trpc/*) if used
 *
 * Excluded:
 *   - Static files (images, fonts, CSS, JS)
 *   - Next.js build artifacts
 *   - Public assets in /public directory
 */
export const config = {
  matcher: [
    // Match all routes except Next.js internals and static files
    "/((?!_next|[^?]*\\.(?:html?|css|js(?!on)|jpe?g|webp|png|gif|svg|ttf|woff2?|ico|csv|docx?|xlsx?|zip|webmanifest)).*)",
    // Always apply to API and tRPC routes
    "/(api|trpc)(.*)",
  ],
};
