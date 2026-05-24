-- CreateTable
CREATE TABLE "users" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "clerkId" TEXT NOT NULL,
    "email" TEXT NOT NULL,
    "credits" INTEGER NOT NULL DEFAULT 5,
    "stripeCustomerId" TEXT,
    "freeRunsToday" INTEGER NOT NULL DEFAULT 0,
    "cooldownUntil" DATETIME,
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" DATETIME NOT NULL
);

-- CreateTable
CREATE TABLE "payments" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "sessionId" TEXT NOT NULL,
    "creditsGranted" INTEGER NOT NULL,
    "clerkUserId" TEXT NOT NULL,
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "payments_clerkUserId_fkey" FOREIGN KEY ("clerkUserId") REFERENCES "users" ("clerkId") ON DELETE RESTRICT ON UPDATE CASCADE
);

-- CreateTable
CREATE TABLE "analysis_reports" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "clerkUserId" TEXT NOT NULL,
    "geneSymbol" TEXT NOT NULL,
    "chromosome" TEXT NOT NULL,
    "position" INTEGER NOT NULL,
    "reference" TEXT NOT NULL,
    "alternative" TEXT NOT NULL,
    "genomeId" TEXT NOT NULL,
    "prediction" TEXT NOT NULL,
    "deltaScore" REAL NOT NULL,
    "classificationConfidence" REAL NOT NULL,
    "classificationSource" TEXT,
    "clinvarClassification" TEXT,
    "variationType" TEXT,
    "clinvarId" TEXT,
    "populationFrequency" TEXT,
    "acmgEvidence" TEXT,
    "literatureContext" TEXT,
    "analysisSource" TEXT,
    "vepAnnotation" TEXT,
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "analysis_reports_clerkUserId_fkey" FOREIGN KEY ("clerkUserId") REFERENCES "users" ("clerkId") ON DELETE RESTRICT ON UPDATE CASCADE
);

-- CreateIndex
CREATE UNIQUE INDEX "users_clerkId_key" ON "users"("clerkId");

-- CreateIndex
CREATE UNIQUE INDEX "users_email_key" ON "users"("email");

-- CreateIndex
CREATE UNIQUE INDEX "users_stripeCustomerId_key" ON "users"("stripeCustomerId");

-- CreateIndex
CREATE UNIQUE INDEX "payments_sessionId_key" ON "payments"("sessionId");

-- CreateIndex
CREATE INDEX "analysis_reports_clerkUserId_idx" ON "analysis_reports"("clerkUserId");

-- CreateIndex
CREATE INDEX "analysis_reports_geneSymbol_idx" ON "analysis_reports"("geneSymbol");
