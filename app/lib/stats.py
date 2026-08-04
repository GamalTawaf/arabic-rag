"""The read-only queries behind ``GET /stats``."""

from __future__ import annotations

from sqlalchemy import distinct, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.chunks import EMBEDDING_COLUMNS, Chunk
from app.models.query_cache import QueryCache


async def corpus_stats(session: AsyncSession) -> dict:
    """Chunk count, document count, and per-model embedding coverage in one row.

    ``count(column)`` counts non-NULLs, which is exactly "how many chunks are in
    this model's index" — a NULL vector is out of the index, not a zero vector
    (see ``dense_search``).
    """
    columns = [Chunk.__table__.c[name] for name in EMBEDDING_COLUMNS.values()]
    row = (
        await session.execute(
            select(
                func.count(Chunk.id),
                func.count(distinct(Chunk.doc_id)),
                *[func.count(column) for column in columns],
            )
        )
    ).one()
    total, docs, *embedded = row
    return {
        "chunks": total,
        "documents": docs,
        "embeddings": {
            model_key: {
                "chunks": count,
                "coverage": round(count / total, 4) if total else 0.0,
            }
            for model_key, count in zip(EMBEDDING_COLUMNS, embedded)
        },
    }


async def cache_stats(session: AsyncSession) -> dict:
    """Cache size and hit ratio.

    ``hits`` counts answers *served* from the cache; every stored row is one
    answer that had to be generated, i.e. one miss. So
    ``ratio = hits / (hits + rows)`` — an approximation that ignores refusals and
    lookups made while the cache was disabled, and that is stated here rather
    than dressed up as an exact figure.
    """
    entries, hits = (
        await session.execute(
            select(func.count(QueryCache.id), func.coalesce(func.sum(QueryCache.hits), 0))
        )
    ).one()
    lookups = entries + hits
    return {
        "entries": entries,
        "hits": int(hits),
        "hit_ratio": round(hits / lookups, 4) if lookups else 0.0,
        "enabled": settings.semantic_cache_enabled,
        "threshold": settings.semantic_cache_threshold,
    }
