# Database Migration Guide: SQLite → Neon PostgreSQL

This guide documents the migration of the HelixMind frontend database from SQLite to Neon serverless PostgreSQL.

## What Changed

- `frontend/prisma/schema.prisma` provider switched from `sqlite` to `postgresql`.
- All JSON-string columns in `AnalysisReport` converted to native Prisma `Json?` fields.
- Added `ChatSession` and `ChatMessage` models for persistent design-assistant chat history.
- `frontend/src/env.js` now validates `DATABASE_URL`.
- `frontend/.env.example` and `frontend/.env.local` include a PostgreSQL `DATABASE_URL`.
- `frontend/src/app/api/analyze/route.ts` and `frontend/src/app/api/history/route.ts` no longer manually `JSON.stringify` / `JSON.parse`.

## Local Development

### Option A: Local PostgreSQL (Docker)

```bash
docker run -d \
  --name helixmind-postgres \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=helixmind_dev \
  -p 5432:5432 \
  postgres:16
```

Set in `frontend/.env.local`:

```env
DATABASE_URL="postgresql://postgres:postgres@localhost:5432/helixmind_dev?schema=public"
```

### Option B: Neon Serverless (recommended for prod parity)

1. Create a project at https://neon.tech.
2. Copy the connection string from **Dashboard → Connection Details**.
3. Paste it into `frontend/.env.local` and Vercel environment variables.

## Apply Migrations

```bash
cd frontend
npx prisma migrate dev --name init_postgres
npx prisma generate
```

> If you already have a SQLite `dev.db` with data you want to keep, export it with `prisma db pull` + a custom script, or use `pgloader` to migrate data into PostgreSQL before running `prisma migrate deploy`.

## Production Deployment

On Vercel (or any host):

1. Add `DATABASE_URL` environment variable with the Neon connection string.
2. Run migrations as part of the build or via a separate CI step:

```bash
npx prisma migrate deploy
npx prisma generate
```

3. Remove or unset any `dev.db` / SQLite references.

## Neon-Specific Notes

- Use `?sslmode=require` in the connection string.
- For serverless/edge deployments, consider Prisma Accelerate or the `@prisma/adapter-pg` driver adapter.
- Neon has an idle timeout; the first request after idle may be slower (cold start). Connection pooling is enabled automatically via `pgbouncer` when using `?sslmode=require`.

## Verification

```bash
cd frontend
npx prisma validate
npx prisma db pull --print   # confirm schema matches database
npx tsc --noEmit             # type-check frontend
```

## Rollback

If you need to revert to SQLite temporarily:

1. Restore `provider = "sqlite"` and `url = env("DATABASE_URL")` in `schema.prisma`.
2. Change `DATABASE_URL` to `file:./dev.db`.
3. Regenerate the client: `npx prisma generate`.

Note: native JSON fields will not work in SQLite without reverting them to `String?` and re-adding manual `JSON.parse`/`JSON.stringify` in the API routes.
