"""Semantic query cache: one row per answered question.

Separate table from ``chunks`` on purpose. Chunks carry one vector column per
*benchmarked* model because the benchmark varies exactly one factor; the cache
only ever holds vectors from the **one** model the running service embeds with,
so it needs a single column plus a ``model_key`` tag to prove which model
produced it. Mixing the two would put four mostly-NULL vector columns and a
cache lifecycle on the corpus table.

# trade-off: one 1024-dim column, sized for the two local models (e5, bge) that
# the service actually runs. OpenAI (3072) and Cohere (1536) cannot be cached
# here — :func:`app.retrieval.cache.lookup` rejects them loudly rather than
# silently truncating. Upgrade path if the service ever runs an API embedder:
# add a second nullable column per dim, exactly as `chunks` does.
"""

from __future__ import annotations

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Index, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base

CACHE_DIM = 1024


class QueryCache(Base):
    __tablename__ = "query_cache"

    id: Mapped[str] = mapped_column(String, primary_key=True)  # uuid4().hex

    # The normalized query is stored, not the raw one: it is what the cheap
    # digit/negation guard re-derives its key from on every candidate hit.
    query_normalized: Mapped[str] = mapped_column(Text, nullable=False)

    embedding: Mapped[list[float]] = mapped_column(Vector(CACHE_DIM), nullable=False)
    model_key: Mapped[str] = mapped_column(String, nullable=False)

    # Everything *other than the question* that determines the answer: retrieval
    # config, depths, and the generating model. Without it the cache is keyed on
    # the question alone, so asking the same thing under `config=lexical` returns
    # the answer `config=hybrid+rerank` produced — which silently turns any A/B
    # through the HTTP API into a measurement of the cache. See
    # :func:`app.service.pipeline_key`.
    pipeline_key: Mapped[str] = mapped_column(String, nullable=False)

    answer: Mapped[str] = mapped_column(Text, nullable=False)
    # ``[{"chunk_id": str, "score": float}]``. Rows written before scores were
    # stored hold bare id strings; JSONB takes both, and there is no migration
    # because there is nothing to rewrite — app.service._cited_scores reads either
    # shape and an unscored legacy row keeps reporting 0.0.
    citations: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)

    hits: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    last_hit_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    __table_args__ = (
        Index("ix_query_cache_scope", "model_key", "pipeline_key"),
        # HNSW post-filters the model_key predicate rather than pre-filtering it.
        # Irrelevant at cache scale (thousands of rows, one model in practice);
        # upgrade path is a partial index per model_key if that ever changes.
        Index(
            "ix_query_cache_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )
