# boilerplate

FastAPI microservice boilerplate: async SQLAlchemy (asyncpg) DB access, Alembic
migrations, health routes, pytest.

Copy this folder to start a new service, rename it, and update `service_name` /
`database_url` in `.env`.

## Local dev (Docker)

```bash
cp .env.example .env
docker compose up --build
```

- App: http://localhost:8000
- Docs: http://localhost:8000/docs
- Health: http://localhost:8000/health, http://localhost:8000/health/db

## Migrations

Run inside the `app` container (or locally with `DATABASE_URL` set):

```bash
alembic upgrade head                    # apply migrations
alembic revision --autogenerate -m "x"  # generate a new migration from model changes
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

## Production image

```bash
docker build -t boilerplate .
docker run -p 8000:8000 -e DATABASE_URL=... boilerplate
```

## Layout

```
app/
  main.py         # FastAPI app + router registration
  config.py       # env-based settings
  db.py           # async SQLAlchemy engine/session
  models/
    items.py      # example model (Item) — replace with real domain models
  api/
    health.py     # /health, /health/db
    items.py      # example CRUD wired to the DB
  lib/            # shared, framework-agnostic helpers (API clients, utilities)
  crons/          # standalone scheduled entrypoints (python -m app.crons.<name>)
alembic/          # migrations (async env)
tests/            # pytest + example tests (httpx.AsyncClient)
Dockerfile        # multistage prod build
Dockerfile.dev    # dev image with --reload
docker-compose.yml # app + postgres for local dev
```
