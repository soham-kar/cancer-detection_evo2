const { Pool } = require('pg');

const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
  ssl: { rejectUnauthorized: false },
});

pool.query("SELECT table_name FROM information_schema.tables WHERE table_schema='public'")
  .then((r) => {
    console.log('Tables:', r.rows);
    pool.end();
  })
  .catch((e) => {
    console.error('Direct pg query error:', e);
    pool.end();
  });
