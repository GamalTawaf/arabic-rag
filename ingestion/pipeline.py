"""Ingestion pipeline: corpus documents -> chunks -> Postgres.

The one place that joins the three pure pieces (`fetch`, `chunk`, `normalize`)
to the `chunks` table. Re-running it is safe: chunk ids are a pure function of
the document text, so every write is an upsert on that id and re-ingesting an
edited document updates rows in place instead of duplicating them.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.constants import EMBEDDING_COLUMNS, EMBEDDING_DIMS
from app.models.chunks import Chunk
from ingestion.chunk import chunk_document
from ingestion.fetch import CorpusDoc
from ingestion.normalize import normalize_for_index

# API providers cap batch size and local models cap VRAM; 64 is comfortably under
# both (OpenAI allows 2048 inputs, e5/bge-m3 handle 64 x 512 tokens on CPU).
EMBED_BATCH = 64

# Rows per INSERT statement. Keeps the bound-parameter count far below asyncpg's
# 32767 limit even with a 3072-dim vector column in play.
INSERT_BATCH = 200

_TEXT_COLUMNS = ("doc_id", "article", "seq", "text", "text_normalized")


class Embedder(Protocol):
    """Anything that turns texts into vectors for one benchmarked model."""

    model_key: str  # a key of app.models.chunks.EMBEDDING_COLUMNS

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


@dataclass(frozen=True)
class IngestStats:
    documents: int
    chunks_written: int
    chunks_skipped: int


async def ingest_documents(
    docs: Iterable[CorpusDoc],
    session: AsyncSession,
    embedder: Embedder | None = None,
) -> IngestStats:
    """Chunk every document and upsert the chunks. Commits before returning.

    trade-off: embedding at ingest time is optional and single-model. The normal
    path leaves every vector column NULL and lets the phase-2 backfill script
    fill one model at a time — that is what makes the 4-model benchmark cheap to
    re-run. Pass an `embedder` only when you want one model written inline.
    """
    rows: list[dict[str, Any]] = []
    documents = 0
    skipped = 0

    for doc in docs:
        documents += 1
        seen: dict[str, int] = {}
        for chunk in chunk_document(doc.doc_id, doc.text):
            normalized = normalize_for_index(chunk.text)
            if not normalized:
                # Punctuation/whitespace-only chunk: nothing to index or match on.
                skipped += 1
                continue
            chunk_id, seq = _unique_id(chunk.id, seen)
            rows.append(
                {
                    "id": chunk_id,
                    "doc_id": chunk.doc_id,
                    "article": chunk.article,
                    "seq": seq,
                    "text": chunk.text,
                    "text_normalized": normalized,
                }
            )

    if not rows:
        return IngestStats(documents=documents, chunks_written=0, chunks_skipped=skipped)

    columns = list(_TEXT_COLUMNS)
    if embedder is not None:
        column = _embedding_column(embedder)
        rows = await _with_embeddings(rows, embedder, column)
        columns.append(column)

    for start in range(0, len(rows), INSERT_BATCH):
        await session.execute(_upsert(rows[start : start + INSERT_BATCH], columns))
    await session.commit()

    return IngestStats(
        documents=documents, chunks_written=len(rows), chunks_skipped=skipped
    )


def _unique_id(chunk_id: str, seen: dict[str, int]) -> tuple[str, int]:
    """Renumber `seq` so ids are unique within a document.

    trade-off: `chunk_document` restarts `seq` at 0 for every article heading, and
    real statutes repeat an article number — Law 14/2004 has both "مواد الإصدار"
    1-4 and its own articles 1-4, plus "المادة 52 مكرر". Numbering per
    (doc, article) across the whole document keeps every id unique and still a
    pure function of the text. Ceiling: article numbers stop being a stable
    handle if a document is re-ordered. Upgrade path: move this counter into
    `ingestion.chunk` so evals and the pipeline derive ids from one place.
    """
    prefix = chunk_id.rpartition(":")[0]
    seq = seen.get(prefix, 0)
    seen[prefix] = seq + 1
    return f"{prefix}:{seq}", seq


def _upsert(rows: Sequence[dict[str, Any]], columns: Sequence[str]):
    """INSERT ... ON CONFLICT (id) DO UPDATE on the mutable columns.

    `tsv` is a generated column and `created_at` keeps its original value, so
    neither is ever written here.
    """
    stmt = insert(Chunk).values(list(rows))
    return stmt.on_conflict_do_update(
        index_elements=[Chunk.id],
        set_={name: getattr(stmt.excluded, name) for name in columns},
    )


def _embedding_column(embedder: Embedder) -> str:
    try:
        return EMBEDDING_COLUMNS[embedder.model_key]
    except KeyError:
        raise ValueError(
            f"unknown embedder model_key {embedder.model_key!r}; "
            f"expected one of {sorted(EMBEDDING_COLUMNS)}"
        ) from None


async def _with_embeddings(
    rows: list[dict[str, Any]], embedder: Embedder, column: str
) -> list[dict[str, Any]]:
    """Return copies of `rows` with the model's vector column filled in.

    trade-off: embeds `text` (the original), not `text_normalized`. Normalization is
    a lexical device — it folds hamza seats and ta marbuta so the tsvector matches —
    and the multilingual encoders were trained on natural Arabic, diacritics and
    all. Ceiling: it is an assumption, not a measurement. Upgrade path: phase 2 runs
    raw-vs-normalized as one more benchmark ablation and this line follows the numbers.
    """
    expected_dim = EMBEDDING_DIMS[embedder.model_key]
    vectors: list[list[float]] = []

    for start in range(0, len(rows), EMBED_BATCH):
        batch = rows[start : start + EMBED_BATCH]
        result = await embedder.embed([row["text"] for row in batch])
        if len(result) != len(batch):
            raise ValueError(
                f"embedder {embedder.model_key!r} returned {len(result)} vectors "
                f"for {len(batch)} texts"
            )
        for vector in result:
            if len(vector) != expected_dim:
                raise ValueError(
                    f"embedder {embedder.model_key!r} returned dim {len(vector)}, "
                    f"expected {expected_dim}"
                )
        vectors.extend(result)

    return [row | {column: vector} for row, vector in zip(rows, vectors, strict=True)]
