"""Lazily built singletons, and the FastAPI dependencies that hand them out.

The hard constraint: **importing ``app.main`` must load no model.** The
reranker alone is ~2 GB of weights and ``import torch`` costs seconds; a process
that only ever serves ``/health`` or ``/stats``, and every test run, must not pay
that. So nothing here is constructed at import time — each builder is
``functools.cache``d and runs on first request.

``functools.cache`` does not cache exceptions, so a keyless environment gets a
fresh, honest failure on every request instead of one poisoned singleton.
"""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING, Annotated

from fastapi import Depends, HTTPException, status

from app.config import settings
from app.observability.cost import SpendTracker, get_spend_tracker
from app.retrieval.embed import Embedder, get_embedder
from app.retrieval.rerank import Reranker, get_reranker
from app.service import RagService

if TYPE_CHECKING:
    # Import-time only: the runtime imports inside build_planner/build_provider
    # are what keep torch and the provider SDKs off the /health import path.
    from app.generation.base import Provider
    from app.planning.planner import Planner

#: The embedding model the *service* runs. Not ``settings.embedding_model``:
#: that field names an API model (`text-embedding-3-large`) this deployment has
#: no key for, and it is the benchmark's variable, not the service's.
#:
#: bge-m3 is the measured choice — recall@10 0.946 on the full eval set at
#: ~2.6 ms mean / 3.9 ms p95, and a Gulf-dialect penalty of -0.4 points against
#: e5's -9.9. It is also 1024-dim, which is what `query_cache` stores.
#:
#: # trade-off: a module constant, because config lives in app/config.py and that
#: # file is not mine to edit. Upgrade path: add `retrieval_model_key: str =
#: # "bge"` to Settings and read it here.
SERVICE_MODEL_KEY = "bge"

#: Cross-encoder reranker. "noop" keeps fusion order (the ablation baseline).
SERVICE_RERANKER = "bge"

#: Rule-based Gulf→MSA planning: measurable offline, needs no API key, and is
#: the arm the phase-3 numbers were produced with.
SERVICE_PLANNER = "rules"


@cache
def build_embedder() -> Embedder:
    """The query embedder. Weights load on its first ``embed_queries`` call."""
    return get_embedder(SERVICE_MODEL_KEY)


@cache
def build_reranker() -> Reranker:
    """The cross-encoder. ``get_reranker`` is itself cached — one copy of ~2 GB."""
    return get_reranker(SERVICE_RERANKER)


@cache
def build_planner() -> Planner:
    from app.planning.planner import get_planner

    return get_planner(SERVICE_PLANNER)


@cache
def build_provider() -> Provider:
    """The failover chain from ``settings.providers``. Raises without API keys."""
    from app.generation.failover import FailoverProvider

    return FailoverProvider.from_settings()


@cache
def build_service() -> RagService:
    return RagService(
        build_embedder(),
        build_reranker(),
        build_planner(),
        build_provider(),
        get_spend_tracker(),
        settings,
    )


def get_service() -> RagService:
    """FastAPI dependency for the pipeline.

    A missing API key is a *configuration* failure, not a bug in the request, so
    it surfaces as a 503 naming the variable to set rather than a 500 with a
    traceback from inside a provider constructor.
    """
    try:
        return build_service()
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"generation is not configured: {exc}",
        ) from exc


def get_ingest_embedder() -> Embedder:
    """The embedder new chunks are indexed with — the same one ``/ask`` queries.

    Not optional: a chunk written with a NULL vector is invisible to dense
    retrieval, so an ingest that skipped embedding would report success and then
    never be found. The alternative is remembering to run
    ``python -m ingestion backfill`` after every POST, which nobody will.
    """
    return build_embedder()


def reset_singletons() -> None:
    """Drop every cached singleton. For tests; the service never calls it."""
    for builder in (
        build_embedder,
        build_reranker,
        build_planner,
        build_provider,
        build_service,
    ):
        builder.cache_clear()


ServiceDep = Annotated[RagService, Depends(get_service)]
SpendTrackerDep = Annotated[SpendTracker, Depends(get_spend_tracker)]
EmbedderDep = Annotated[Embedder, Depends(get_ingest_embedder)]
