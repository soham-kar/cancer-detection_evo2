import { prisma } from "../src/lib/prisma";

async function main() {
  const user = await prisma.user.create({
    data: {
      clerkId: "seed_user_001",
      email: "seed@example.com",
      credits: 5,
      freeRunsToday: 0,
    },
  });

  await prisma.analysisReport.create({
    data: {
      clerkUserId: user.clerkId,
      geneSymbol: "BRCA1",
      chromosome: "chr17",
      position: 43094609,
      reference: "C",
      alternative: "T",
      genomeId: "hg38",
      prediction: "Pathogenic",
      deltaScore: -0.1234,
      classificationConfidence: 0.87,
      externalScores: {
        cadd: { phred: 25.3, interpretation: "Likely deleterious" },
        alphamissense: { score: 0.82, classification: "Likely pathogenic" },
      },
      populationFrequency: {
        gnomad_af: 0.00002,
        is_common_variant: false,
      },
    },
  });

  console.log("✅ Seeded database");
}

main()
  .then(async () => {
    await prisma.$disconnect();
  })
  .catch(async (e) => {
    console.error(e);
    await prisma.$disconnect();
    process.exit(1);
  });
