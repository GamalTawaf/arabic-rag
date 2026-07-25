"""Semantic answer cache: serve a stored answer when a near-identical question returns.

A generated answer costs an LLM call plus retrieval; a cache hit costs one
pgvector nearest-neighbour lookup over a table of thousands of rows. That is the
whole argument. Two rules make it safe rather than merely cheap:

**1. Never match across embedding models.** Every lookup filters on
``model_key`` first. Two vectors from different models live in unrelated spaces,
so a cosine similarity between them is a meaningless number that happens to be
in [-1, 1] — a cross-model "match" would confidently return the answer to some
other question. The dimension check exists for the same reason: e5 and bge are
both 1024-dim and would otherwise compare without error.

**2. A similarity threshold alone is not enough.** Measured with bge-m3, the
service's embedder (see the table below): "هل يجوز لصاحب العمل فصل العامل أثناء
الإجازة؟" and the same sentence with "لا" inserted score **0.9870** cosine.
That is comfortably over the 0.95 threshold, and the two answers are opposites —
the cache would confidently serve "yes" to a question asking "no". So before a
semantic hit is accepted it must also pass an exact-match guard on the
normalized digit sequence and on the negation particles present
(:func:`guard_key`). A guard mismatch is a miss, not an error: the caller
regenerates, which is the cheap failure.

Measured on bge-m3, cosine of question pairs differing in exactly one respect::

    negation inserted (يجوز / لا يجوز)        0.9870   over threshold, guard REJECTs
    paraphrase, same meaning                  0.9913   over threshold, guard passes -> hit
    digit changed (5 سنوات / 3 سنوات)         0.90-0.94  under threshold anyway
    spelled-out number (سنة واحدة / سنتين)    0.9448   under threshold by 0.006

Two honest readings of that table. The **negation** guard is the one earning its
keep — it is the only thing standing between the threshold and an inverted
answer. The **digit** guard is defence in depth: on this model digit-only pairs
already fall below 0.95, so it decides nothing today, and it starts mattering
only if the threshold is lowered or the embedder swapped.

# ponytail: the guard is exact-match on digits and a fixed particle list, and it
# is deliberately over-eager — it counts interrogative "ما" as a negation, so
# "ما مدة الإشعار" never matches "كم مدة الإشعار" (measured 0.9652: a real,
# accepted false miss). Telling interrogative "ما" from Gulf negation "ما يجوز"
# needs morphology, and a false miss costs one generation while a false hit
# costs a wrong legal answer.
#
# Real ceiling: spelled-out numbers. "بعد سنة واحدة" vs "بعد سنتين" carries no
# digits, so guard_key sees them as identical — measured at 0.9448, which the
# 0.95 threshold rejects by 0.006 and nothing else would. Lower
# `semantic_cache_threshold` below 0.94 and that pair silently returns the wrong
# notice period. Upgrade path before touching that setting: fold Arabic number
# words and duals (واحد..عشرة, سنتين/شهرين/يومين) into the digit key.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.chunks import EMBEDDING_DIMS
from app.models.query_cache import CACHE_DIM, QueryCache
from ingestion.normalize import normalize_query

_TOKEN = re.compile(r"\w+")
_DIGITS = re.compile(r"\d+")

# Written in *normalized* form (see ingestion.normalize: ة->ه, ى->ي, hamza
# seats folded), because guard_key compares tokens of the normalized query.
# Includes Gulf negations (مو/مب/ماكو) — the dialect questions are the point.
NEGATION_PARTICLES = frozenset(
    normalize_query(word)
    for word in (
        "لا",
        "ما",
        "لم",
        "لن",
        "ليس",
        "ليست",
        "غير",
        "بدون",
        "دون",
        "مو",
        "مب",
        "ماكو",
    )
)


@dataclass(frozen=True)
class CachedAnswer:
    answer: str
    citations: list[str]
    similarity: float  # cosine, 1.0 == identical vector
    age_seconds: float  # since the entry was created, not since its last hit


def guard_key(text: str) -> tuple[tuple[str, ...], frozenset[str]]:
    """The cheap exact-match key a semantic hit must also agree on.

    ``(ordered digit sequences, set of negation particles)`` over the normalized
    query. Ordered digits so "المادة 5 بعد 3 سنوات" and "المادة 3 بعد 5 سنوات"
    do not share a key.
    """
    normalized = normalize_query(text)
    digits = tuple(_DIGITS.findall(normalized))
    particles = frozenset(_TOKEN.findall(normalized)) & NEGATION_PARTICLES
    return digits, particles


def _check(query_vec: Sequence[float], model_key: str) -> list[float]:
    """Validate at the trust boundary: this is the only place a bad key is cheap."""
    if not model_key:
        raise ValueError("model_key is required — the cache never matches across models")
    if EMBEDDING_DIMS.get(model_key) != CACHE_DIM:
        raise ValueError(
            f"query_cache holds {CACHE_DIM}-dim vectors; model_key {model_key!r} is "
            f"{EMBEDDING_DIMS.get(model_key, 'unknown')}-dim"
        )
    vector = list(query_vec)
    if len(vector) != CACHE_DIM:
        raise ValueError(f"query_vec has {len(vector)} dims, expected {CACHE_DIM}")
    return vector


async def lookup(
    session: AsyncSession,
    query: str,
    query_vec: Sequence[float],
    model_key: str,
    threshold: float = settings.semantic_cache_threshold,
) -> CachedAnswer | None:
    """The cached answer for a semantically equivalent question, or ``None``.

    A hit requires all three of: same ``model_key``, cosine similarity
    ``>= threshold``, and an identical :func:`guard_key`. On a hit ``hits`` is
    incremented and ``last_hit_at`` set, so hit rate is queryable from the table
    itself rather than only from spans.

    # ponytail: only the single nearest row is considered. If the nearest row
    # fails the guard the call is a miss even when the second-nearest would have
    # passed — one extra generation, versus scanning a candidate list on every
    # request. Upgrade path: take the top-k above threshold and return the first
    # that passes the guard.
    """
    vector = _check(query_vec, model_key)

    distance = QueryCache.embedding.cosine_distance(vector)
    statement = (
        select(
            QueryCache.id,
            QueryCache.query_normalized,
            QueryCache.answer,
            QueryCache.citations,
            QueryCache.created_at,
            distance.label("d"),
        )
        .where(QueryCache.model_key == model_key)
        .order_by(distance, QueryCache.id)  # id breaks ties: reproducible
        .limit(1)
    )
    row = (await session.execute(statement)).first()
    if row is None:
        return None

    similarity = 1.0 - float(row.d)
    if similarity < threshold:
        return None
    if guard_key(row.query_normalized) != guard_key(query):
        return None  # near-identical wording, materially different question

    await session.execute(
        update(QueryCache)
        .where(QueryCache.id == row.id)
        .values(hits=QueryCache.hits + 1, last_hit_at=func.now())
    )
    await session.commit()

    return CachedAnswer(
        answer=row.answer,
        citations=list(row.citations or []),
        similarity=similarity,
        age_seconds=(datetime.now(UTC) - row.created_at).total_seconds(),
    )


async def store(
    session: AsyncSession,
    query: str,
    query_vec: Sequence[float],
    model_key: str,
    answer: str,
    citations: list[str],
) -> None:
    """Cache an answer. Silently ignores an empty answer — never cache a failure.

    # ponytail: append-only, no TTL and no invalidation on re-ingestion, so a
    # corpus update leaves stale answers behind. Acceptable while the corpus is a
    # frozen snapshot. Upgrade path: ``DELETE FROM query_cache`` at the end of
    # the ingestion pipeline, or an age filter in :func:`lookup`.
    """
    vector = _check(query_vec, model_key)
    if not answer.strip():
        return

    session.add(
        QueryCache(
            id=uuid4().hex,
            query_normalized=normalize_query(query),
            embedding=vector,
            model_key=model_key,
            answer=answer,
            citations=list(citations),
        )
    )
    await session.commit()
