import { db } from "../src/lib/db";

async function main() {
  const reports = await db.analysisReport.findMany({
    orderBy: { createdAt: "desc" },
    take: 5,
    select: {
      id: true,
      geneSymbol: true,
      position: true,
      prediction: true,
      clerkUserId: true,
      createdAt: true,
    },
  });
  console.log("Reports:", JSON.stringify(reports, null, 2));
  await db.$disconnect();
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
