"""``GET /stats`` — what this instance actually has, right now.

The dashboard question a reviewer asks in an interview: *is the corpus loaded,
which models is it embedded with, is the cache doing anything, how much has this
thing spent today, and would a generation call even work?* One indexed query
answers the first three; the rest is process state.

Deliberately cheap and side-effect free: no model is loaded, no provider is
constructed, nothing is written. ``/stats`` must stay callable on a cold
instance that has never served an ``/ask``.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy import distinct, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db import get_db
from app.deps import (
    SERVICE_MODEL_KEY,
    SERVICE_PLANNER,
    SERVICE_RERANKER,
    SpendTrackerDep,
)
from app.generation.providers import available_providers
from app.models.chunks import EMBEDDING_COLUMNS, Chunk
from app.models.query_cache import QueryCache

router = APIRouter(tags=["stats"])

DbSession = Annotated[AsyncSession, Depends(get_db)]


async def _corpus(session: AsyncSession) -> dict:
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


async def _cache(session: AsyncSession) -> dict:
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


@router.get("/stats")
async def stats(db: DbSession, spend: SpendTrackerDep):
    today = spend.today()
    configured = [name.strip() for name in settings.providers.split(",") if name.strip()]
    available = available_providers()
    return {
        "corpus": await _corpus(db),
        "cache": await _cache(db),
        "spend": {
            "date": today.date,
            "usd": round(today.usd, 6),
            "calls": today.calls,
            "cap_usd": spend.cap_usd,
            "remaining_usd": round(spend.remaining(), 6),
        },
        "providers": {
            "configured": configured,
            # Configured but not available means "no API key in this process" —
            # the difference between the two lists is the whole reason to print
            # both instead of one "providers" field.
            "available": available,
            "missing_keys": [name for name in configured if name not in available],
        },
        "retrieval": {
            "model_key": SERVICE_MODEL_KEY,
            "reranker": SERVICE_RERANKER,
            "planner": SERVICE_PLANNER,
            "rerank_enabled": settings.rerank_enabled,
            "rerank_min_score": settings.rerank_min_score,
            "top_k_retrieve": settings.top_k_retrieve,
            "top_k_context": settings.top_k_context,
        },
    }
