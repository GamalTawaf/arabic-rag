import os
import sys

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app import db as app_db
from app.config import settings
from app.db import Base
from app.main import app

TEST_DATABASE_URL = os.getenv(
    "TEST_DATABASE_URL",
    "postgresql+asyncpg://rag_user:rag_pass@localhost:5433/rag_test",
)


@pytest.fixture(scope="session")
def pg_url():
    """Skip DB-backed tests when no pgvector Postgres is reachable."""
    import asyncio

    async def probe():
        engine = create_async_engine(TEST_DATABASE_URL)
        try:
            async with engine.connect():
                return True
        except Exception:  # noqa: BLE001 — a probe: any failure means "no usable DB"
            return False
        finally:
            await engine.dispose()

    if not asyncio.run(probe()):
        pytest.skip("no pgvector Postgres at TEST_DATABASE_URL")
    return TEST_DATABASE_URL


@pytest.fixture(scope="session")
def pg_schema(pg_url):
    """Build the test schema once, from the Alembic migrations rather than metadata.

    Using `alembic upgrade head` instead of Base.metadata.create_all means the suite
    runs against the DDL that production actually gets, so migration/model drift fails
    a test instead of hiding until deploy. The schema is dropped first so alembic always
    starts from base — otherwise a leftover table from a previous run makes `upgrade
    head` die with DuplicateTableError while alembic_version still reads empty.
    """
    import asyncio
    import subprocess

    from sqlalchemy import text

    async def reset_schema():
        engine = create_async_engine(pg_url, isolation_level="AUTOCOMMIT")
        async with engine.connect() as conn:
            await conn.execute(text("DROP SCHEMA public CASCADE"))
            await conn.execute(text("CREATE SCHEMA public"))
        await engine.dispose()

    asyncio.run(reset_schema())
    # `python -m alembic` with THIS interpreter, not a hardcoded `.venv/bin/alembic`:
    # CI installs into the runner's system Python with no virtualenv, so that path
    # does not exist there and subprocess.run raises FileNotFoundError (which
    # check=False does not suppress) — every DB-backed test errors out. Same
    # breakage for anyone on conda or a venv by another name.
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "-c", "config/alembic.ini", "upgrade", "head"],
        env={**os.environ, "DATABASE_URL": pg_url},
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        check=False,
    )
    if result.returncode != 0:
        # Fail, never skip. A skip here is the dangerous outcome: a broken
        # migration would take every DB-backed test out of the run, pytest would
        # exit 0, CI would go green, and the migration would ship — the exact
        # drift this fixture exists to catch.
        raise RuntimeError(
            "alembic upgrade head failed against the test database "
            f"(exit {result.returncode}).\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
        )
    return pg_url


@pytest.fixture()
async def engine(pg_url, pg_schema, monkeypatch):
    """Point ``app.db`` at the test database and hand back its engine.

    Nothing private is patched: ``app.db.engine()`` builds itself from
    ``settings.database_url`` on first use, so redirecting the whole app is one
    setting plus a cache reset. The reset is also what keeps the pool inside the
    test's event loop — a cached engine outliving its loop hands the next test
    connections bound to a closed one.
    """
    from sqlalchemy import text

    monkeypatch.setattr(settings, "database_url", pg_url)
    app_db.reset_sessions()
    eng = app_db.engine()
    # trade-off: TRUNCATE per test rather than recreating the schema — the schema comes
    # from alembic once per session (pg_schema). Ceiling: tests share one database, so
    # they cannot run in parallel against it. Upgrade path: a database per xdist worker.
    tables = ", ".join(table.name for table in Base.metadata.sorted_tables)
    async with eng.begin() as conn:
        await conn.execute(text(f"TRUNCATE TABLE {tables} RESTART IDENTITY CASCADE"))
    yield eng
    async with eng.begin() as conn:
        await conn.execute(text(f"TRUNCATE TABLE {tables} RESTART IDENTITY CASCADE"))
    await eng.dispose()
    # Dispose *then* clear: leaving a disposed engine in the cache would hand the
    # next caller connections bound to this test's closed event loop.
    app_db.reset_sessions()


@pytest.fixture()
async def db_session(engine):
    """A session for arranging rows and asserting on them.

    Deliberately *not* the session the code under test uses — services open their
    own now. It commits, they commit, and both read the same database.
    """
    async with async_sessionmaker(engine, expire_on_commit=False)() as session:
        yield session


@pytest.fixture()
async def client(db_session):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
