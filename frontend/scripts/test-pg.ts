import { Pool } from 'pg';

const pool = new Pool({ connectionString: process.env.DATABASE_URL });

pool.query('SELECT count(*) FROM "AnalysisReport"')
  .then((r) => {
    console.log('Direct pg query result:', r.rows);
    pool.end();
  })
  .catch((e) => {
    console.error('Direct pg query error:', e);
    pool.end();
  });
