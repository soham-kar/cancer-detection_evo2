import { db } from '../src/lib/db.ts';

async function test() {
  try {
    const user = await db.user.findUnique({ where: { clerkId: 'test_user_123' } });
    console.log('user found:', user ? user.id : null);
    const report = await db.analysisReport.create({
      data: {
        clerkUserId: 'test_user_123',
        geneSymbol: 'TEST',
        chromosome: 'chr1',
        position: 12345,
        reference: 'A',
        alternative: 'T',
        genomeId: 'hg38',
        prediction: 'Pathogenic',
        deltaScore: 0.5,
        classificationConfidence: 0.8,
      }
    });
    console.log('created report:', report.id);
  } catch (e) {
    console.error('ERROR:', e);
  } finally {
    await db.$disconnect();
  }
}

test();
