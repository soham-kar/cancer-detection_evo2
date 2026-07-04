/**
 * Run `build` or `dev` with `SKIP_ENV_VALIDATION` to skip env validation. This is especially useful
 * for Docker builds.
 */
import "./src/env.js";

/** @type {import("next").NextConfig} */
const config = {
  reactStrictMode: false,
  // Bundle pg and the Prisma pg adapter into the server bundle so Next.js
  // does not emit dynamic import chunks that fail to resolve at runtime.
  serverExternalPackages: ["pg", "@prisma/adapter-pg"],
};

export default config;
