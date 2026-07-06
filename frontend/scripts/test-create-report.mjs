import { db } from '../src/lib/db.ts';

async function test() {
  try {
    const report = await db.analysisReport.create({
      data: {
        clerkUserId: 'test_user_123',
        geneSymbol: 'BRCA1',
        chromosome: 'chr17',
        position: 43044294,
        reference: 'A',
        alternative: 'T',
        genomeId: 'hg38',
        prediction: 'Pathogenic',
        deltaScore: 0.5,
        classificationConfidence: 0.8,
        analysisSource: 'custom',
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
