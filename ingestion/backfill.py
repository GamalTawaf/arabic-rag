"""Backfill one model's vector column over chunks already in the table.

Ingestion writes text and leaves every vector column NULL; this fills one column
per run. That split is what makes the 4-model benchmark cheap — re-running a
model never re-chunks the corpus, and the eval ground truth (chunk ids) never
moves underneath it.

Resumable by construction: rows are walked by primary key, each batch is
committed on its own, and the default query only sees rows whose column is still
NULL. Ctrl-C in the middle of a 233-chunk run costs at most one batch.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.chunks import EMBEDDING_COLUMNS, EMBEDDING_DIMS, Chunk
from app.retrieval.embed import get_embedder

DEFAULT_BATCH_SIZE = 32


@dataclass(frozen=True)
class BackfillStats:
    model_key: str
    chunks_embedded: int
    chunks_skipped: int
    seconds: float


async def backfill_embeddings(
    session: AsyncSession,
    model_key: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    only_missing: bool = True,
) -> BackfillStats:
    """Embed chunk text into `model_key`'s vector column, committing per batch.

    `only_missing=True` (the default) fills NULLs only, so an interrupted run
    resumes where it stopped. `only_missing=False` re-embeds every row — use it
    after changing the model or the text that is embedded.
    """
    column = _column(model_key)
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    embedder = get_embedder(model_key)
    total, pending = await _counts(session, column, only_missing)
    started = time.perf_counter()
    embedded = 0
    last_id = ""

    while True:
        rows = (await session.execute(_next_batch(column, last_id, batch_size, only_missing))).all()
        if not rows:
            break
        # trade-off: embeds Chunk.text (the original, diacritics and all), not
        # text_normalized — same assumption as ingestion.pipeline._with_embeddings,
        # and the two must stay consistent or the benchmark compares two corpora.
        # embed_passages, never embed_queries: e5 prefixes "passage: " here and
        # "query: " at search time, and mixing them costs real recall.
        vectors = await embedder.embed_passages([text for _, text in rows])
        _check_batch(vectors, rows, model_key)

        for (chunk_id, _), vector in zip(rows, vectors, strict=True):
            await session.execute(
                update(Chunk).where(Chunk.id == chunk_id).values(**{column: vector})
            )
        await session.commit()

        embedded += len(rows)
        last_id = rows[-1][0]
        _progress(model_key, embedded, pending, time.perf_counter() - started)

    return BackfillStats(
        model_key=model_key,
        chunks_embedded=embedded,
        chunks_skipped=max(total - embedded, 0),
        seconds=time.perf_counter() - started,
    )


def _column(model_key: str) -> str:
    """Vector column for a model key. Checked before the model loads."""
    try:
        return EMBEDDING_COLUMNS[model_key]
    except KeyError:
        raise ValueError(
            f"unknown embedding model_key {model_key!r}; "
            f"expected one of {sorted(EMBEDDING_COLUMNS)}"
        ) from None


async def _counts(
    session: AsyncSession, column: str, only_missing: bool
) -> tuple[int, int]:
    """(rows in the table, rows this run will embed)."""
    total = await session.scalar(select(func.count()).select_from(Chunk)) or 0
    if not only_missing:
        return total, total
    missing = await session.scalar(
        select(func.count()).select_from(Chunk).where(getattr(Chunk, column).is_(None))
    )
    return total, missing or 0


def _next_batch(column: str, last_id: str, batch_size: int, only_missing: bool):
    """Keyset page over chunk ids — stable while the current page is being written."""
    stmt = select(Chunk.id, Chunk.text).where(Chunk.id > last_id)
    if only_missing:
        stmt = stmt.where(getattr(Chunk, column).is_(None))
    return stmt.order_by(Chunk.id).limit(batch_size)


def _check_batch(
    vectors: list[list[float]], rows: Sequence[tuple[str, str]], model_key: str
) -> None:
    expected_dim = EMBEDDING_DIMS[model_key]
    if len(vectors) != len(rows):
        raise ValueError(
            f"embedder {model_key!r} returned {len(vectors)} vectors for {len(rows)} texts"
        )
    for vector in vectors:
        if len(vector) != expected_dim:
            raise ValueError(
                f"embedder {model_key!r} returned dim {len(vector)}, expected {expected_dim}"
            )


def _progress(model_key: str, embedded: int, total: int, elapsed: float) -> None:
    """One line per batch on stderr — a CPU run is slow enough that silence reads as a hang."""
    print(
        f"backfill {model_key}: {embedded}/{total} chunks in {elapsed:.1f}s",
        file=sys.stderr,
        flush=True,
    )
