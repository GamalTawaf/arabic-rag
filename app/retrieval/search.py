"""Hybrid retrieval: dense (pgvector cosine) + lexical (Postgres FTS) + RRF.

Three small pieces that stay independently testable and independently
benchmarkable, because the ablation study needs to run each stage alone:

- :func:`dense_search`   — one vector column per model, cosine distance, HNSW.
- :func:`lexical_search` — tsvector over the *normalized* text, ``ts_rank_cd``.
- :func:`rrf_fuse`       — pure function, no DB, deterministic.

Every statement is built with ``select()`` and bound parameters. User text
reaches Postgres only as a parameter to ``plainto_tsquery``, never as SQL.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.models.chunks import EMBEDDING_COLUMNS, EMBEDDING_DIMS, Chunk
from ingestion.normalize import normalize_query

DEFAULT_LIMIT = 20
RRF_K = 60  # the constant from Cormack et al. 2009; damps the top of each list

# Query-side tsquery guards. Word characters only, so nothing a user types can
# reach to_tsquery's operator syntax (& | ! <-> : parentheses); the caps keep a
# pathological query from building a tsquery Postgres refuses to parse.
_TOKEN = re.compile(r"\w+")
MAX_TOKEN_CHARS = 64  # pg errors above 2047 bytes; no real Arabic word is close
MAX_QUERY_TOKENS = 32  # longer than any question in the eval set


@dataclass(frozen=True)
class Hit:
    """One retrieved chunk. ``score`` is only comparable within one ``source``."""

    chunk_id: str
    doc_id: str
    article: str | None
    text: str
    score: float
    source: str  # "dense" | "lexical" | "rrf"


def _check_limit(limit: int) -> int:
    if limit < 1:
        raise ValueError(f"limit must be >= 1, got {limit}")
    return limit


def _vector_column(model_key: str):
    """The pgvector column for a benchmarked model, or a clear error."""
    if model_key not in EMBEDDING_COLUMNS:
        raise ValueError(
            f"unknown model_key {model_key!r}; "
            f"expected one of {sorted(EMBEDDING_COLUMNS)}"
        )
    return Chunk.__table__.c[EMBEDDING_COLUMNS[model_key]]


async def dense_search(
    session: AsyncSession,
    query_vec: Sequence[float],
    model_key: str,
    limit: int = DEFAULT_LIMIT,
) -> list[Hit]:
    """Nearest chunks by cosine distance in ``model_key``'s vector column.

    Rows whose vector column is NULL are excluded — a chunk that was never
    embedded with this model is not a miss, it is out of the index, and letting
    NULLs sort in would corrupt the benchmark. ``score`` is ``1 - distance``, so
    higher is better and an exact match scores 1.0.
    """
    _check_limit(limit)
    column = _vector_column(model_key)
    vector = list(query_vec)
    expected = EMBEDDING_DIMS[model_key]
    if len(vector) != expected:
        raise ValueError(
            f"query_vec has {len(vector)} dims, {model_key} expects {expected}"
        )

    distance = column.cosine_distance(vector)
    query = (
        select(Chunk.id, Chunk.doc_id, Chunk.article, Chunk.text, distance.label("d"))
        .where(column.is_not(None))
        .order_by(distance, Chunk.id)  # id breaks ties so runs are reproducible
        .limit(limit)
    )
    rows = await session.execute(query)
    return [
        Hit(
            chunk_id=row.id,
            doc_id=row.doc_id,
            article=row.article,
            text=row.text,
            score=1.0 - float(row.d),
            source="dense",
        )
        for row in rows
    ]


def _tsquery(query: str):
    """An OR-of-terms tsquery over the normalized query, or None if there is none.

    Not ``plainto_tsquery``: that ANDs every term, and a natural-language Arabic
    question carries interrogatives ("كم", "ايش", "هل") that appear nowhere in
    legislative text, so *every* term must match and none ever do. Measured on
    the 268 answerable eval pairs against the real 233-chunk corpus:

        plainto_tsquery (AND)   recall@10 = 0.000, 268/268 queries returned []
        to_tsquery      (OR)    recall@10 = 0.356, 0/268 queries returned []

    OR-ing terms and letting ``ts_rank_cd`` sort by term coverage/density is
    what makes the lexical leg contribute anything to the fusion at all.
    """
    tokens = [
        token
        for token in _TOKEN.findall(normalize_query(query))
        if len(token) <= MAX_TOKEN_CHARS
    ][:MAX_QUERY_TOKENS]
    if not tokens:
        return None
    return func.to_tsquery("simple", " | ".join(tokens))


async def lexical_search(
    session: AsyncSession, query: str, limit: int = DEFAULT_LIMIT
) -> list[Hit]:
    """Full-text search over the generated tsvector, ranked by ``ts_rank_cd``.

    The query is put through the same normalizer as the indexed text first:
    ``tsv`` is built from ``text_normalized`` with the 'simple' config, so a
    diacritised or hamza-spelled query would otherwise silently match nothing.

    # trade-off: 'simple' has no Arabic stemmer and no stopword list, so this is
    # exact token matching after normalization — "الأجور" will not find "الأجر",
    # broken plurals and clitics are missed, and common particles still score.
    # Ceiling accepted because the dense leg covers morphology and RRF only
    # needs the ranks. Upgrade path: an Arabic dictionary/stemmer text search
    # config (hunspell ar) plus a stopword list, or camel-tools lemmas written
    # into a second indexed column.
    """
    _check_limit(limit)
    tsquery = _tsquery(query)
    if tsquery is None:
        return []  # empty/punctuation-only query: nothing to match, not an error

    rank = func.ts_rank_cd(Chunk.tsv, tsquery)
    statement = (
        select(Chunk.id, Chunk.doc_id, Chunk.article, Chunk.text, rank.label("rank"))
        .where(Chunk.tsv.op("@@")(tsquery))
        .order_by(rank.desc(), Chunk.id)
        .limit(limit)
    )
    rows = await session.execute(statement)
    return [
        Hit(
            chunk_id=row.id,
            doc_id=row.doc_id,
            article=row.article,
            text=row.text,
            score=float(row.rank),
            source="lexical",
        )
        for row in rows
    ]


def rrf_fuse(
    ranked_lists: Sequence[Sequence[Hit]], k: int = RRF_K, limit: int = DEFAULT_LIMIT
) -> list[Hit]:
    """Reciprocal Rank Fusion: ``score(d) = sum over lists of 1 / (k + rank(d))``.

    Rank is 1-based. Only positions matter, so dense cosine scores and lexical
    ts_rank_cd scores never have to be made commensurable — which is the whole
    reason to use RRF here. Ties break on ``chunk_id`` so a benchmark rerun
    produces byte-identical output. Each hit keeps the text/doc_id/article of
    the first list it appeared in.
    """
    _check_limit(limit)
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    scores: dict[str, float] = {}
    seen: dict[str, Hit] = {}
    for hits in ranked_lists:
        for rank, hit in enumerate(hits, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k + rank)
            seen.setdefault(hit.chunk_id, hit)

    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [
        replace(seen[chunk_id], score=score, source="rrf")
        for chunk_id, score in ordered[:limit]
    ]


async def hybrid_search(
    session: AsyncSession,
    query: str,
    query_vec: Sequence[float],
    model_key: str,
    limit: int = DEFAULT_LIMIT,
) -> list[Hit]:
    """Dense and lexical concurrently, fused with RRF.

    Each leg gets its own short-lived session on the caller's engine: one
    AsyncSession is one connection, and two coroutines sharing it raise
    ``InvalidRequestError: concurrent operations are not permitted``.

    # trade-off: consequence of the fan-out — the two legs read their own
    # snapshots and cannot see rows the caller has written but not committed.
    # Fine for search over an already-ingested corpus. Upgrade path if that ever
    # bites: drop the gather and await the two searches on `session` in sequence
    # (both are single-digit ms at corpus scale).
    """
    _check_limit(limit)
    leg_session = async_sessionmaker(session.bind, expire_on_commit=False)
    async with leg_session() as dense_leg, leg_session() as lexical_leg:
        # return_exceptions=True: a bare gather propagates the first failure
        # immediately, which unwinds the `async with` and closes the session the
        # still-running sibling is querying on. Let both legs settle, then raise.
        dense, lexical = await asyncio.gather(
            dense_search(dense_leg, query_vec, model_key, limit),
            lexical_search(lexical_leg, query, limit),
            return_exceptions=True,
        )
    for leg in (dense, lexical):
        if isinstance(leg, BaseException):
            raise leg
    return rrf_fuse((dense, lexical), limit=limit)
