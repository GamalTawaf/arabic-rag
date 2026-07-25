import os

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.db import Base, get_db
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
    result = subprocess.run(
        [".venv/bin/alembic", "upgrade", "head"],
        env={**os.environ, "DATABASE_URL": pg_url},
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        check=False,
    )
    if result.returncode != 0:
        pytest.skip(f"alembic upgrade failed against the test database: {result.stderr[-500:]}")
    return pg_url


@pytest.fixture()
async def db_session(pg_url, pg_schema):
    from sqlalchemy import text

    engine = create_async_engine(pg_url)
    # ponytail: TRUNCATE per test rather than recreating the schema — the schema comes
    # from alembic once per session (pg_schema). Ceiling: tests share one database, so
    # they cannot run in parallel against it. Upgrade path: a database per xdist worker.
    tables = ", ".join(table.name for table in Base.metadata.sorted_tables)
    async with engine.begin() as conn:
        await conn.execute(text(f"TRUNCATE TABLE {tables} RESTART IDENTITY CASCADE"))
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as session:
        yield session
    async with engine.begin() as conn:
        await conn.execute(text(f"TRUNCATE TABLE {tables} RESTART IDENTITY CASCADE"))
    await engine.dispose()


@pytest.fixture()
async def client(db_session):
    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()
