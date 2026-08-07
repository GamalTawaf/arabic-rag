"""The ingestion path shared by ``POST /ingest`` and the Pub/Sub push endpoint.

Framework-free on purpose. :mod:`app.api.ingest` is HTTP framing and status
codes; everything that decides *whether a message is ingestible at all* lives
here, so the two entrypoints cannot drift apart — the local endpoint and the
broker must accept and reject exactly the same documents.

**Why redelivery is safe.** Pub/Sub guarantees *at-least-once* delivery: the same
message will arrive twice, and no amount of care on the publisher side prevents
it. Nothing here dedupes on ``messageId``, because it does not have to —
:func:`ingestion.pipeline.ingest_documents` upserts (``ON CONFLICT (id) DO
UPDATE``) on a chunk id that is a pure function of the document text
(``doc:article:seq``), so ingesting the same document twice rewrites the same
rows and leaves the table in the same state. Two deliveries of one message are
indistinguishable from one. ``tests/test_ingest_api.py`` posts an identical
envelope twice and asserts the row count is unchanged, because an idempotency
claim nobody executes is a hope rather than a property.

**The failure taxonomy, which is the whole point of this module.** Exactly two
kinds of failure, and getting them backwards is the classic push-subscription
bug — a poison message redelivered forever, or a transient blip that silently
drops data:

* :class:`RejectedMessage` — **permanent**. A malformed envelope, bad base64,
  bad JSON, a document that fails validation. Redelivery hands us the identical
  bytes and fails identically, so the broker has to be told to stop: ack it
  (2xx), log it, count it.
* everything else — **transient**. A dead database, a connection reset. The same
  message may well succeed in thirty seconds, so it must *not* be acked: 5xx,
  and Pub/Sub retries with backoff.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
from typing import Any

from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError

from app.constants import MAX_PUSH_BODY_BYTES, POSTED_LICENSE
from app.data import IngestDocument, PushMessage
from app.db import session_scope
from app.observability.tracing import span
from ingestion.fetch import CorpusDoc
from ingestion.pipeline import Embedder, ingest_documents

log = logging.getLogger(__name__)


class RejectedMessage(ValueError):
    """A message that can never succeed. Ack it, log it, do not retry it."""


class StorageUnavailable(RuntimeError):
    """The database is unreachable. Transient — retry, never ack. See ``app.main``."""


def decode_push(body: bytes) -> PushMessage:
    """Decode a Pub/Sub push body into a validated document.

    The wire shape is Pub/Sub's, not ours::

        {"message": {"data": "<base64 of the JSON document>",
                     "messageId": "...", "publishTime": "..."},
         "subscription": "projects/p/subscriptions/s"}

    Every failure in here is permanent by construction — the bytes are fixed, so
    a retry re-runs the same parse — hence one exception type.
    """
    if len(body) > MAX_PUSH_BODY_BYTES:
        raise RejectedMessage(
            f"push body is {len(body)} bytes, over the {MAX_PUSH_BODY_BYTES} byte limit"
        )

    envelope = _json_object(body, "push body")
    message = envelope.get("message")
    if not isinstance(message, dict):
        raise RejectedMessage("push body has no 'message' object")

    data = message.get("data")
    if not isinstance(data, str):
        raise RejectedMessage("message.data is missing or is not a base64 string")
    try:
        payload = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise RejectedMessage(f"message.data is not valid base64: {exc}") from exc

    return PushMessage(
        document=parse_document(payload),
        message_id=str(message.get("messageId", "")),
        subscription=str(envelope.get("subscription", "")),
    )


def parse_document(payload: bytes) -> IngestDocument:
    """Validate the decoded message payload. Raises :class:`RejectedMessage`."""
    raw = _json_object(payload, "message payload")
    try:
        return IngestDocument.model_validate(raw)
    except ValidationError as exc:
        raise RejectedMessage(f"message payload is not a valid document: {_why(exc)}") from exc


async def ingest_document(document: IngestDocument, embedder: Embedder) -> dict:
    """Chunk, normalize, embed and upsert one document; return its counts.

    Opens its own session (``app.db.session_scope``) and commits before
    returning, so a caller needs no database of its own — ``app.api.ingest``
    passes a validated document and nothing else.

    Raises :class:`StorageUnavailable`, never an HTTP exception: the status-code
    mapping lives in ``app.main`` so this module keeps working under ``python
    -m`` with no ASGI app around it.

    ``title``, ``source_url`` and the license are accepted for parity with the
    corpus manifest but the ``chunks`` table has a column for none of them, so
    they reach the database only as part of nothing at all.

    # trade-off: a `documents` table would keep them. Not built, because nothing
    # reads them yet — /ask cites chunk ids, and provenance lives in
    # data/corpus/manifest.json for the committed corpus.

    # trade-off: embedding runs on the caller's request path. Measured on this
    # laptop (Apple Silicon/MPS, local Postgres): a 2.6k-character document, 9
    # chunks, 0.40 s warm and 7.9 s on the first call because that one loads
    # bge-m3's weights. Ceiling: the request timeout — Cloud Run defaults to
    # 300 s and its instances are CPU-only, several times slower than MPS, so a
    # document of a few hundred chunks will time out and the client will retry
    # it forever. Upgrade path: write the chunks with NULL vectors, return 202,
    # and let `ingestion.backfill` fill the column out of band — that script
    # already exists and already skips rows that have a vector.
    """
    doc = CorpusDoc(
        doc_id=document.doc_id,
        title=document.title,
        source_url=document.source_url,
        license=POSTED_LICENSE,
        text=document.text,
    )
    attributes = {
        "app.ingest.doc_id": document.doc_id,
        "app.retrieval.model_key": embedder.model_key,
    }
    async with span("ingest", **attributes), session_scope() as session:
        try:
            stats = await ingest_documents([doc], session, embedder)
        except SQLAlchemyError as exc:
            log.exception("ingest of %s failed against the database", document.doc_id)
            raise StorageUnavailable(
                f"ingestion storage is unavailable; retry: {type(exc).__name__}"
            ) from exc
    return {
        "doc_id": document.doc_id,
        "documents": stats.documents,
        "chunks_written": stats.chunks_written,
        "chunks_skipped": stats.chunks_skipped,
        "model_key": embedder.model_key,
    }


def _json_object(raw: bytes, what: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RejectedMessage(f"{what} is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise RejectedMessage(f"{what} is not a JSON object")
    return value


def _why(exc: ValidationError) -> str:
    """The first pydantic error as one line — the log has to be greppable."""
    first = exc.errors()[0]
    field = ".".join(str(part) for part in first["loc"]) or "<body>"
    return f"{field}: {first['msg']}"
