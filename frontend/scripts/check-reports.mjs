const { PrismaClient } = require("./generated/prisma");
const { PrismaPg } = require("@prisma/adapter-pg");
const { Pool } = require("pg");

const url = process.env.DATABASE_URL;
const p = new PrismaClient({
  adapter: new PrismaPg(new Pool({ connectionString: url, max: 1 })),
});

p.analysisReport
  .findMany({
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
  })
  .then((r) => {
    console.log("Reports:", JSON.stringify(r, null, 2));
    return p.$disconnect();
  })
  .catch((e) => {
    console.error(e.message);
    return p.$disconnect();
  });
