import { clerkMiddleware, createRouteMatcher } from "@clerk/nextjs/server";

// Define public routes that don't require authentication
// Homepage (/) is NOT public - users must sign in or use demo mode
const isPublicRoute = createRouteMatcher([
    "/demo(.*)",          // Demo page for guest mode
    "/api/demo-analyze",  // Guest demo API endpoint
    "/api/vep",           // VEP annotation (called from authenticated context)
    "/sign-in(.*)",
    "/sign-up(.*)",
]);

export default clerkMiddleware(async (auth, request) => {
    if (!isPublicRoute(request)) {
        await auth.protect();
    }
});

export const config = {
    matcher: [
        // Skip Next.js internals and all static files, unless found in search params
        "/((?!_next|[^?]*\\.(?:html?|css|js(?!on)|jpe?g|webp|png|gif|svg|ttf|woff2?|ico|csv|docx?|xlsx?|zip|webmanifest)).*)",
        // Always run for API routes
        "/(api|trpc)(.*)",
    ],
};
