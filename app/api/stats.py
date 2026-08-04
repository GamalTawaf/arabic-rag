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
from app.lib.stats import cache_stats, corpus_stats

router = APIRouter(tags=["stats"])

DbSession = Annotated[AsyncSession, Depends(get_db)]


@router.get("/stats")
async def stats(db: DbSession, spend: SpendTrackerDep):
    today = spend.today()
    configured = [name.strip() for name in settings.providers.split(",") if name.strip()]
    available = available_providers()
    return {
        "corpus": await corpus_stats(db),
        "cache": await cache_stats(db),
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
