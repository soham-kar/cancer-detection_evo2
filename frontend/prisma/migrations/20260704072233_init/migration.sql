-- CreateTable
CREATE TABLE "users" (
    "id" TEXT NOT NULL,
    "clerkId" TEXT NOT NULL,
    "email" TEXT NOT NULL,
    "credits" INTEGER NOT NULL DEFAULT 5,
    "stripeCustomerId" TEXT,
    "freeRunsToday" INTEGER NOT NULL DEFAULT 0,
    "cooldownUntil" TIMESTAMP(3),
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "users_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "payments" (
    "id" TEXT NOT NULL,
    "sessionId" TEXT NOT NULL,
    "creditsGranted" INTEGER NOT NULL,
    "clerkUserId" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "payments_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "analysis_reports" (
    "id" TEXT NOT NULL,
    "clerkUserId" TEXT NOT NULL,
    "geneSymbol" TEXT NOT NULL,
    "chromosome" TEXT NOT NULL,
    "position" INTEGER NOT NULL,
    "reference" TEXT NOT NULL,
    "alternative" TEXT NOT NULL,
    "genomeId" TEXT NOT NULL,
    "prediction" TEXT NOT NULL,
    "deltaScore" DOUBLE PRECISION NOT NULL,
    "classificationConfidence" DOUBLE PRECISION NOT NULL,
    "classificationSource" TEXT,
    "clinvarClassification" TEXT,
    "variationType" TEXT,
    "clinvarId" TEXT,
    "populationFrequency" JSONB,
    "acmgEvidence" JSONB,
    "literatureContext" JSONB,
    "clinicalSummary" TEXT,
    "evidenceConfidence" JSONB,
    "ismScanData" JSONB,
    "xaiFactors" JSONB,
    "counterfactuals" JSONB,
    "acmgCriteria" JSONB,
    "knowledgeGraph" JSONB,
    "externalScores" JSONB,
    "acmgCriteriaRefined" JSONB,
    "analysisSource" TEXT,
    "vepAnnotation" JSONB,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "analysis_reports_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "chat_sessions" (
    "id" TEXT NOT NULL,
    "clerkUserId" TEXT NOT NULL,
    "analysisReportId" TEXT,
    "title" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "chat_sessions_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "chat_messages" (
    "id" TEXT NOT NULL,
    "sessionId" TEXT NOT NULL,
    "role" TEXT NOT NULL,
    "content" TEXT NOT NULL,
    "toolCalls" JSONB,
    "toolResults" JSONB,
    "metadata" JSONB,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "chat_messages_pkey" PRIMARY KEY ("id")
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

-- CreateIndex
CREATE INDEX "chat_sessions_clerkUserId_idx" ON "chat_sessions"("clerkUserId");

-- CreateIndex
CREATE INDEX "chat_sessions_analysisReportId_idx" ON "chat_sessions"("analysisReportId");

-- CreateIndex
CREATE INDEX "chat_messages_sessionId_idx" ON "chat_messages"("sessionId");

-- CreateIndex
CREATE INDEX "chat_messages_createdAt_idx" ON "chat_messages"("createdAt");

-- AddForeignKey
ALTER TABLE "payments" ADD CONSTRAINT "payments_clerkUserId_fkey" FOREIGN KEY ("clerkUserId") REFERENCES "users"("clerkId") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "analysis_reports" ADD CONSTRAINT "analysis_reports_clerkUserId_fkey" FOREIGN KEY ("clerkUserId") REFERENCES "users"("clerkId") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "chat_sessions" ADD CONSTRAINT "chat_sessions_clerkUserId_fkey" FOREIGN KEY ("clerkUserId") REFERENCES "users"("clerkId") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "chat_sessions" ADD CONSTRAINT "chat_sessions_analysisReportId_fkey" FOREIGN KEY ("analysisReportId") REFERENCES "analysis_reports"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "chat_messages" ADD CONSTRAINT "chat_messages_sessionId_fkey" FOREIGN KEY ("sessionId") REFERENCES "chat_sessions"("id") ON DELETE CASCADE ON UPDATE CASCADE;
