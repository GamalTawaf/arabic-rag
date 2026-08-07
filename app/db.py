"""The one engine, the one session factory, and the only way to get a session.

Nothing in ``app/api`` opens a session: routes take a request and return a body,
and the service or library function they call is what talks to the database. So
there is no FastAPI dependency here — :func:`session_scope` is the whole surface,
and it works identically inside a route, a cron and a benchmark script.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import cache

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.config import settings


class Base(DeclarativeBase):
    pass


@cache
def engine() -> AsyncEngine:
    """The connection pool, built on first use.

    Lazy rather than module-level: a pool used to be created at import time, so
    every process that imported anything under ``app`` — a CLI, a cron, the test
    collector — paid for one whether or not it ran a query, and the URL was
    pinned before a caller could point it anywhere else.
    """
    return create_async_engine(settings.database_url)


@cache
def sessions() -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine(), expire_on_commit=False)


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """One session, closed on exit. The single seam everything else goes through."""
    async with sessions()() as session:
        yield session


def reset_sessions() -> None:
    """Drop the cached engine and factory. For tests; the service never calls it."""
    sessions.cache_clear()
    engine.cache_clear()
