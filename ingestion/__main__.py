"""CLI for the ingestion pipeline — the whole corpus in one command.

    python -m ingestion fetch                 # re-download the corpus from its sources
    python -m ingestion ingest                # committed corpus -> chunks table
    python -m ingestion backfill --model e5   # fill one model's vector column
    python -m ingestion stats                 # rows per document, embedding coverage

The database comes from settings.database_url (env: DATABASE_URL).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Awaitable
from pathlib import Path
from typing import TypeVar

from sqlalchemy import func, select
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError, SQLAlchemyError

from app.config import settings
from app.constants import EMBEDDING_COLUMNS
from app.db import engine, session_scope
from app.models.chunks import Chunk
from ingestion.backfill import DEFAULT_BATCH_SIZE, backfill_embeddings
from ingestion.fetch import (
    DEFAULT_CORPUS_DIR,
    CorpusDoc,
    fetch_corpus,
    load_corpus,
)
from ingestion.pipeline import ingest_documents

T = TypeVar("T")

_MISSING_TABLE_HINT = (
    'run "alembic -c config/alembic.ini upgrade head" against the same DATABASE_URL to create the chunks table'
)
_NOT_RUNNING_HINT = "is Postgres running? (docker compose up -d db)"


def _safe_url() -> str:
    try:
        return make_url(settings.database_url).render_as_string(hide_password=True)
    except ArgumentError:  # pragma: no cover - the driver reports the real problem
        return "<unparseable DATABASE_URL>"


def _run(coro: Awaitable[T]) -> int:
    """Run a DB coroutine, turning connection/schema failures into one clear line."""

    async def wrapped() -> None:
        try:
            await coro
        finally:
            await engine().dispose()

    try:
        asyncio.run(wrapped())
    except (SQLAlchemyError, OSError) as exc:
        message = str(exc)
        hint = _MISSING_TABLE_HINT if "does not exist" in message else _NOT_RUNNING_HINT
        print(f"database error against {_safe_url()}: {hint}", file=sys.stderr)
        print(f"  {type(exc).__name__}: {message.splitlines()[0]}", file=sys.stderr)
        return 1
    except ValueError as exc:
        # Unknown model key, missing API key, wrong embedding dimension: the
        # message already says what to do, so print it without a traceback.
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


async def _ingest(docs: list[CorpusDoc]) -> None:
    async with session_scope() as session:
        stats = await ingest_documents(docs, session)
    print(
        f"ingested {stats.documents} documents -> {stats.chunks_written} chunks "
        f"({stats.chunks_skipped} skipped) into {_safe_url()}"
    )


async def _backfill(model_key: str, batch_size: int, only_missing: bool) -> None:
    async with session_scope() as session:
        stats = await backfill_embeddings(
            session, model_key, batch_size=batch_size, only_missing=only_missing
        )
    print(
        f"backfilled {stats.model_key}: {stats.chunks_embedded} chunks embedded "
        f"({stats.chunks_skipped} already had a vector) in {stats.seconds:.1f}s"
    )


async def _stats() -> None:
    embedding_counts = [
        func.count(getattr(Chunk, column)).label(key)
        for key, column in EMBEDDING_COLUMNS.items()
    ]
    stmt = (
        select(Chunk.doc_id, func.count().label("chunks"), *embedding_counts)
        .group_by(Chunk.doc_id)
        .order_by(Chunk.doc_id)
    )
    async with session_scope() as session:
        rows = (await session.execute(stmt)).all()

    if not rows:
        print("chunks table is empty — run: python -m ingestion ingest")
        return

    headers = ["doc_id", "chunks", *EMBEDDING_COLUMNS]
    widths = [max(len(headers[0]), *(len(row[0]) for row in rows)), *(8,) * (len(headers) - 1)]
    print("  ".join(head.ljust(width) for head, width in zip(headers, widths, strict=True)))
    for row in rows:
        cells = [row[0], *(str(value) for value in row[1:])]
        print("  ".join(cell.ljust(width) for cell, width in zip(cells, widths, strict=True)))
    print(f"total: {sum(row[1] for row in rows)} chunks across {len(rows)} documents")


def _fetch(corpus_dir: Path) -> int:
    try:
        written = fetch_corpus(corpus_dir)
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"fetch failed: {exc}", file=sys.stderr)
        return 1
    for path in written:
        print(f"wrote {path} ({path.stat().st_size} bytes)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m ingestion",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("fetch", "re-download the corpus from its public sources"),
        ("ingest", "load the committed corpus into the chunks table"),
        ("backfill", "embed the chunks into one model's vector column"),
        ("stats", "row counts per document and embedding coverage"),
    ):
        subparser = subparsers.add_parser(name, help=help_text)
        if name in ("fetch", "ingest"):
            subparser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_DIR)
        if name == "backfill":
            subparser.add_argument(
                "--model", required=True, choices=sorted(EMBEDDING_COLUMNS), help="model key"
            )
            subparser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
            subparser.add_argument(
                "--all",
                action="store_true",
                help="re-embed every chunk (default: only rows whose column is NULL)",
            )

    args = parser.parse_args(argv)
    if args.command == "fetch":
        return _fetch(args.corpus)
    if args.command == "ingest":
        try:
            docs = load_corpus(args.corpus)
        except (OSError, ValueError) as exc:
            print(f"corpus error: {exc}", file=sys.stderr)
            return 1
        return _run(_ingest(docs))
    if args.command == "backfill":
        return _run(_backfill(args.model, args.batch_size, only_missing=not args.all))
    return _run(_stats())


if __name__ == "__main__":
    raise SystemExit(main())
