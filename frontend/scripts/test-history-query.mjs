import { db } from '../src/lib/db.ts';

async function test() {
  try {
    const user = await db.user.findUnique({ where: { clerkId: 'test_user_123' } });
    console.log('user found:', user ? user.id : null);
    const report = await db.analysisReport.findFirst({
      where: { clerkUserId: 'test_user_123', geneSymbol: 'BRCA1' },
      orderBy: { createdAt: 'desc' }
    });
    console.log('report found:', report ? report.id : null);
  } catch (e) {
    console.error('ERROR:', e);
  } finally {
    await db.$disconnect();
  }
}

test();
