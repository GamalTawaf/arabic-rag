"""``POST /ingest`` and ``POST /ingest/pubsub`` — the two ways a document arrives.

Same pipeline, two callers, and the difference between them is entirely in what
a status code *means*:

=============================  =========  ===============================================
``POST /ingest``               200        upserted; body carries the counts
                               422        the document is invalid — the caller's problem
                               503        the database is unreachable — try again later
``POST /ingest/pubsub``        200        **handled or permanently rejected**; ack it
                               5xx        transient; Pub/Sub should redeliver
=============================  =========  ===============================================

**Why a malformed push gets a 2xx.** Pub/Sub retries anything that is not a 2xx,
for as long as the subscription's retention allows. A message whose base64 is
corrupt fails identically on every delivery, so answering 4xx buys nothing and
costs a redelivery loop that never terminates — the message becomes poison and
the subscription's backlog never drains. Acking it stops the loop; the ``ERROR``
log line and the ``rag.requests{status="rejected"}`` counter are what keep it
from being silent. Conversely a dead database *is* worth retrying, so it must
return 5xx: acking it would drop the document on the floor with no record that
it was ever meant to exist.

That is why this route reads the raw body itself instead of declaring a pydantic
body model. A body FastAPI cannot even parse would otherwise become an automatic
422 — the exact answer that turns a malformed message into an infinite retry.
Unhandled failures still fall through to a 500, which is the right default for
"we don't know what happened": unknown means retry.
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.deps import EmbedderDep
from app.ingest_worker import (
    IngestDocument,
    RejectedMessage,
    decode_push,
    ingest_document,
)
from app.lib.auth import require_ingest_key
from app.observability.tracing import record_request, span
from app.retrieval.embed import Embedder
from ingestion.pipeline import IngestStats

log = logging.getLogger(__name__)

router = APIRouter(tags=["ingest"])

DbSession = Annotated[AsyncSession, Depends(get_db)]


_DB_UNAVAILABLE = "ingestion storage is unavailable; retry"


def _counts(doc_id: str, stats: IngestStats, model_key: str) -> dict:
    return {
        "doc_id": doc_id,
        "documents": stats.documents,
        "chunks_written": stats.chunks_written,
        "chunks_skipped": stats.chunks_skipped,
        "model_key": model_key,
    }


async def _ingest(
    db: AsyncSession, document: IngestDocument, embedder: Embedder, route: str
) -> IngestStats:
    """Run the pipeline, turning a database failure into a retryable 503.

    # trade-off: embedding runs on the request path. Measured on this laptop
    # (Apple Silicon/MPS, local Postgres): a 2.6k-character document, 9 chunks,
    # 0.40 s warm and 7.9 s on the first request because that one loads bge-m3's
    # weights. Ceiling: the request timeout — Cloud Run defaults to 300 s and its
    # instances are CPU-only, several times slower than MPS, so a document of a
    # few hundred chunks will time out and the client will retry it forever.
    # Upgrade path: write the chunks with NULL vectors, return 202, and let
    # `ingestion.backfill` fill the column out of band — that script already
    # exists and already skips rows that have a vector.
    """
    attributes = {
        "app.ingest.doc_id": document.doc_id,
        "app.retrieval.model_key": embedder.model_key,
    }
    async with span("ingest", **attributes):
        try:
            return await ingest_document(db, document, embedder)
        except SQLAlchemyError as exc:
            log.exception("ingest of %s failed against the database", document.doc_id)
            record_request(route, "db_error")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"{_DB_UNAVAILABLE}: {type(exc).__name__}",
            ) from exc


@router.post("/ingest", dependencies=[Depends(require_ingest_key)])
async def ingest(document: IngestDocument, db: DbSession, embedder: EmbedderDep):
    """Ingest one document synchronously: chunk -> normalize -> embed -> upsert.

    200 rather than 201 because every write is an upsert on a chunk id derived
    from the text — re-posting an edited document updates rows in place, and
    claiming "Created" for that would be a lie two thirds of the time.
    """
    stats = await _ingest(db, document, embedder, "/ingest")
    record_request("/ingest", "ok")
    return _counts(document.doc_id, stats, embedder.model_key)


@router.post("/ingest/pubsub")
async def ingest_pubsub(request: Request, db: DbSession, embedder: EmbedderDep):
    """Pub/Sub push endpoint. Ack semantics are in this module's docstring.

    Redelivery is safe because ingestion is idempotent by chunk id; see
    :mod:`app.ingest_worker`. This handler therefore keeps no record of which
    ``messageId`` it has seen — it is genuinely fine for the same message to be
    processed twice.

    No ``x-api-key`` check here, unlike ``/ingest``: the push subscription is
    already OIDC-authenticated at the infrastructure layer against a dedicated
    service account (terraform/pubsub.tf), which is the stronger control. Adding
    the header would mean carrying the secret through Terraform for no gain.
    """
    try:
        message = decode_push(await request.body())
    except RejectedMessage as exc:
        # 200 on purpose: permanent failure. See the module docstring.
        log.error("dropping unprocessable Pub/Sub message: %s", exc)
        record_request("/ingest/pubsub", "rejected")
        return {"status": "rejected", "reason": str(exc)}

    stats = await _ingest(db, message.document, embedder, "/ingest/pubsub")
    record_request("/ingest/pubsub", "ok")
    log.info(
        "ingested %s from message %s: %d chunks",
        message.document.doc_id,
        message.message_id or "<no id>",
        stats.chunks_written,
    )
    return {
        "status": "ok",
        "message_id": message.message_id,
        **_counts(message.document.doc_id, stats, embedder.model_key),
    }
