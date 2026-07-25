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


@pytest.fixture()
async def db_session(pg_url):
    from sqlalchemy import text

    engine = create_async_engine(pg_url)
    # ponytail: TRUNCATE, not drop_all/create_all. Tests share the dev database, and
    # dropping the tables left it schema-less while alembic_version still read "0001",
    # which broke `python -m ingestion ingest` right after a test run. Truncating gives
    # the same clean slate per test without the collateral damage. Ceiling: no schema
    # isolation between a test run and local data. Upgrade path: point
    # TEST_DATABASE_URL at a dedicated database.
    tables = ", ".join(table.name for table in Base.metadata.sorted_tables)
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)
        await conn.execute(text(f"TRUNCATE TABLE {tables} RESTART IDENTITY CASCADE"))
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as session:
        yield session
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
