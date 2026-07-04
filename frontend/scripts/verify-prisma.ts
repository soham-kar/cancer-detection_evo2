import { prisma } from "../src/lib/prisma";

async function main() {
  const user = await prisma.user.findFirst();
  const report = await prisma.analysisReport.findFirst();

  console.log("✅ Connected to Prisma Postgres");
  console.log("User:", user?.email);
  console.log("Report:", report?.geneSymbol, report?.prediction);
}

main()
  .then(async () => {
    await prisma.$disconnect();
  })
  .catch(async (e) => {
    console.error("❌ Connection failed:", e);
    await prisma.$disconnect();
    process.exit(1);
  });
