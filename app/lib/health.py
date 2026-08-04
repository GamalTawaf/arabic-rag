"""The liveness probe's one query."""

from __future__ import annotations

from sqlalchemy import text

from app.db import session_scope


async def database_reachable() -> None:
    """Round-trip the pool. Raises whatever the driver raises — a probe wants that."""
    async with session_scope() as session:
        await session.execute(text("SELECT 1"))
