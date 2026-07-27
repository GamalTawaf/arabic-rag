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
import re
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from ingestion.fetch import CorpusDoc
from ingestion.pipeline import Embedder, IngestStats, ingest_documents

log = logging.getLogger(__name__)

#: Chunk ids are ``doc_id:article:seq``. A doc_id containing ``:`` would split
#: into the wrong fields on every downstream parse, so the character is banned
#: rather than escaped — see :meth:`IngestDocument._usable_in_a_chunk_id`.
#: ``\A``/``\Z``, not ``^``/``$``: Python's ``$`` also matches immediately before
#: a trailing newline, so ``"my-doc\n"`` passed this check and the newline went
#: straight into the chunk primary key — where it is invisible in logs, breaks
#: exact-match lookups against the clean id, and makes re-ingesting the same
#: document write a second set of rows.
MAX_DOC_ID_CHARS = 128
DOC_ID_RE = re.compile(rf"\A[A-Za-z0-9][A-Za-z0-9._-]{{0,{MAX_DOC_ID_CHARS - 1}}}\Z")

MAX_TITLE_CHARS = 300
MAX_URL_CHARS = 2048

#: The largest document in the committed corpus (Law 14/2004) is ~67k
#: characters, so this is roughly 3x the real ceiling and still small enough
#: that a body of this size cannot exhaust a Cloud Run instance's memory.
MAX_TEXT_CHARS = 200_000

#: Pub/Sub's own documented per-message limit. A push body larger than this did
#: not come from Pub/Sub, so it is rejected before any parsing happens.
MAX_PUSH_BODY_BYTES = 10 * 1024 * 1024

#: ``CorpusDoc.license`` is required by the manifest loader but the ``chunks``
#: table has no column for it, so a posted document carries a placeholder.
POSTED_LICENSE = "unspecified"


class RejectedMessage(ValueError):
    """A message that can never succeed. Ack it, log it, do not retry it."""


class IngestDocument(BaseModel):
    """One document to ingest. The trust boundary for both entrypoints.

    ``extra`` is left at pydantic's default (ignore) rather than ``forbid``:
    a push subscription that rejects unknown fields turns an additive change on
    the publisher side into a total ingestion outage, every message poisoned at
    once. Unknown fields are dropped instead.
    """

    doc_id: str = Field(min_length=1, max_length=MAX_DOC_ID_CHARS)
    title: str = Field(min_length=1, max_length=MAX_TITLE_CHARS)
    text: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)
    source_url: str | None = Field(default=None, max_length=MAX_URL_CHARS)

    @field_validator("doc_id")
    @classmethod
    def _usable_in_a_chunk_id(cls, value: str) -> str:
        if ":" in value:
            raise ValueError(
                "doc_id must not contain ':' — chunk ids are 'doc_id:article:seq', "
                "so a colon here splits into the wrong fields and silently breaks "
                "id parsing and every eval pair that names a chunk"
            )
        if not DOC_ID_RE.match(value):
            raise ValueError(
                "doc_id must start with a letter or digit and contain only "
                f"letters, digits, '.', '_' and '-' (got {value!r})"
            )
        return value

    @field_validator("title", "text")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty or whitespace")
        return value

    @field_validator("source_url")
    @classmethod
    def _http_url(cls, value: str | None) -> str | None:
        if value is not None and not value.startswith(("http://", "https://")):
            raise ValueError("source_url must be an http(s) URL")
        return value


@dataclass(frozen=True)
class PushMessage:
    """A decoded Pub/Sub push delivery. ``message_id`` is for logs only."""

    document: IngestDocument
    message_id: str
    subscription: str


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


async def ingest_document(
    session: AsyncSession,
    document: IngestDocument,
    embedder: Embedder | None = None,
) -> IngestStats:
    """Chunk, normalize, embed and upsert one document. Commits before returning.

    ``title``, ``source_url`` and the license are accepted for parity with the
    corpus manifest but the ``chunks`` table has a column for none of them, so
    they reach the database only as part of nothing at all.

    # trade-off: a `documents` table would keep them. Not built, because nothing
    # reads them yet — /ask cites chunk ids, and provenance lives in
    # data/corpus/manifest.json for the committed corpus.
    """
    doc = CorpusDoc(
        doc_id=document.doc_id,
        title=document.title,
        source_url=document.source_url,
        license=POSTED_LICENSE,
        text=document.text,
    )
    return await ingest_documents([doc], session, embedder)


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
