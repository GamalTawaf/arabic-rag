# boilerplate (FastAPI service template)

Stack: FastAPI, SQLAlchemy async (asyncpg), Alembic, Postgres, pytest (async), ruff, pip-audit.

## Testing

```bash
pip install -r requirements-dev.txt
pytest              # bare pytest — pytest.ini sets pythonpath=. and asyncio_mode=auto
ruff check .
pip-audit -r requirements.txt
```

- Global testing rules (80% coverage, AAA, TDD, unit/integration/e2e) apply, with one
  override for this repo: there is no user-facing flow, so E2E tests don't apply —
  unit + integration (endpoint tests via the `client` fixture) is the full bar here.
- All endpoint tests go through the async `client` fixture in `tests/conftest.py`
  (`httpx.AsyncClient` over `ASGITransport`, in-memory `sqlite+aiosqlite` with
  `StaticPool`). Don't build a second engine/fixture per test file — StaticPool is
  required or each connection gets its own empty `:memory:` db. Test functions are
  plain `async def test_...` — no `@pytest.mark.asyncio` needed (`asyncio_mode = auto`).
- One test file per router (`tests/test_<name>.py`), mirroring `app/api/`.
- CI (`.github/workflows/ci.yml`) runs `pytest`, `ruff check .`, `pip-audit` on every
  push/PR — run all three locally before considering a change done.

## Conventions

- Async by default: async routes, `AsyncSession`, `await db.execute(...)`. No sync
  engine/session anywhere — don't mix `sqlalchemy.orm.Session` back in.
- New model → `app/models/<name>.py`, exported from `app/models/__init__.py` (that
  import is what makes Alembic autogenerate see it — `alembic/env.py` does
  `from app import models`).
- New endpoint → one router per resource in `app/api/<name>.py`, registered in
  `app/main.py`.
- Shared, framework-agnostic helpers (API clients, utility functions) → `app/lib/`.
  Not routes, not models.
- Scheduled/standalone jobs → `app/crons/<name>.py`, run via
  `python -m app.crons.<name>` (see `example_report_items.py`). Not wired into the
  FastAPI app.
- Config only through `app/config.py` (pydantic-settings) — never read `os.environ`
  directly in route/model code.
- Migrations: `alembic revision --autogenerate -m "..."` then read the generated file
  before committing — autogenerate misses renames and some constraint changes.
- This folder is the template. Changes here should stay generic; service-specific logic
  belongs in the copy, not upstream in this repo.

## Skills for this repo

- `ecc:python-reviewer` — after any Python change
- `superpowers:test-driven-development` / `ecc:tdd-guide` — new endpoints or models
- `ecc:database-migrations` — writing/reviewing Alembic migrations
- `ecc:database-reviewer` — schema or query changes
- `ecc:api-design` — new endpoints/routers
- `ecc:docker-patterns` — Dockerfile / docker-compose changes
- `ecc:security-review` — before adding auth, input handling, or secrets
- `ecc:backend-patterns` — general FastAPI/service structure questions
