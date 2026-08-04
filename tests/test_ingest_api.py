"""``POST /ingest`` and the Pub/Sub push endpoint, over real HTTP and a real database.

The ack-semantics tests are the load-bearing ones. A push subscription is a
retry loop with a broker on the other end, so "which failures return 2xx" is not
a style question — get it backwards and either a corrupt message is redelivered
until the subscription's retention expires, or a database blip silently swallows
a document. Both are asserted here on the status code, because the status code
is the entire protocol.

Nothing loads a model: the embedder is a fake with the right ``model_key`` and
dimension, injected through ``get_ingest_embedder``.
"""

from __future__ import annotations

import base64
import json
import logging
import subprocess
import sys
from pathlib import Path

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import OperationalError

from app.config import settings
from app.db import get_db
from app.deps import get_ingest_embedder
from app.ingest_worker import MAX_TEXT_CHARS, RejectedMessage, decode_push
from app.main import app
from app.models.chunks import Chunk

REPO_ROOT = Path(__file__).resolve().parent.parent

DOC_ID = "qatar-test-decision-9-2026"
SUBSCRIPTION = "projects/arabic-rag/subscriptions/ingest-push"

# Preamble plus two articles: three chunks, and the Arabic-Indic digit exercises
# the same normalization path the committed corpus goes through.
DOC_TEXT = """قرار وزاري تجريبي رقم ٩ لسنة ٢٠٢٦
المادة (1)
يُعمَل بأحكام هذا القرار اعتباراً من تاريخ نشره في الجريدة الرسمية.
المادة (٢)
على العامل إخطار صاحب العمل قبل إنهاء العقد بمدة لا تقل عن شهر.
"""

EXPECTED_CHUNK_IDS = [f"{DOC_ID}:1:0", f"{DOC_ID}:2:0", f"{DOC_ID}:p:0"]


class FakeEmbedder:
    """Deterministic 1024-dim vectors, zero ML dependencies. bge's dimension."""

    def __init__(self) -> None:
        self.model_key = "bge"
        self.dim = 1024
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(len(text) % 10)] * self.dim for text in texts]


class BrokenSession:
    """A session whose every statement fails the way a dead Postgres fails."""

    async def execute(self, *_args, **_kwargs):
        raise OperationalError(
            "INSERT INTO chunks ...",
            {},
            Exception("server closed the connection unexpectedly"),
        )

    async def commit(self) -> None:  # pragma: no cover - never reached
        raise AssertionError("commit must not run after a failed statement")


@pytest.fixture()
def embedder():
    """Install the fake embedder for the ingest routes, and clean up after."""
    fake = FakeEmbedder()
    app.dependency_overrides[get_ingest_embedder] = lambda: fake
    yield fake
    app.dependency_overrides.pop(get_ingest_embedder, None)


def document(**overrides) -> dict:
    return {
        "doc_id": DOC_ID,
        "title": "قرار وزاري تجريبي",
        "text": DOC_TEXT,
        "source_url": "https://almeezan.qa/test",
    } | overrides


def b64(raw: str | bytes) -> str:
    if isinstance(raw, str):
        raw = raw.encode("utf-8")
    return base64.b64encode(raw).decode("ascii")


def envelope(data: str, message_id: str = "13548923456789") -> dict:
    """A Pub/Sub push body; ``data`` is the already-encoded ``message.data``."""
    return {
        "message": {
            "data": data,
            "messageId": message_id,
            "publishTime": "2026-07-25T09:14:03.123Z",
        },
        "subscription": SUBSCRIPTION,
    }


def push_body(**overrides) -> dict:
    return envelope(b64(json.dumps(document(**overrides))))


async def count_chunks(session) -> int:
    return (await session.execute(select(func.count(Chunk.id)))).scalar_one()


async def chunk_ids(session) -> list[str]:
    return list((await session.execute(select(Chunk.id).order_by(Chunk.id))).scalars())


# --------------------------------------------------------------- POST /ingest


async def test_ingest_chunks_normalizes_embeds_and_upserts_one_document(
    client, db_session, embedder
):
    # Act
    response = await client.post("/ingest", json=document())

    # Assert — the counts the caller gets back
    assert response.status_code == 200
    assert response.json() == {
        "doc_id": DOC_ID,
        "documents": 1,
        "chunks_written": 3,
        "chunks_skipped": 0,
        "model_key": "bge",
    }

    # Assert — and the rows that actually landed
    assert await chunk_ids(db_session) == EXPECTED_CHUNK_IDS
    row = await db_session.get(Chunk, f"{DOC_ID}:2:0")
    assert "إخطار" in row.text
    assert row.text_normalized and row.text_normalized != row.text  # normalized form
    assert row.article == "2"


async def test_ingested_chunks_are_embedded_so_dense_retrieval_can_find_them(
    client, db_session, embedder
):
    # Act
    await client.post("/ingest", json=document())

    # Assert — a NULL vector would be a chunk that /ask can never retrieve
    row = await db_session.get(Chunk, f"{DOC_ID}:1:0")
    assert row.emb_bge is not None
    assert len(row.emb_bge) == 1024
    assert [text for call in embedder.calls for text in call]  # the fake was used


async def test_ingest_returns_503_when_the_database_is_down(client, db_session, embedder):
    # Arrange
    app.dependency_overrides[get_db] = lambda: BrokenSession()

    # Act
    response = await client.post("/ingest", json=document())

    # Assert — transient: the caller may retry
    assert response.status_code == 503
    assert "unavailable" in response.json()["detail"]


# -------------------------------------------------------- POST /ingest input


async def test_doc_id_containing_a_colon_is_rejected(client, db_session, embedder):
    # Act — a colon would split "doc:article:seq" into the wrong fields
    response = await client.post("/ingest", json=document(doc_id="qatar:law:14"))

    # Assert
    assert response.status_code == 422
    assert "':'" in json.dumps(response.json())
    assert await count_chunks(db_session) == 0


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"doc_id": ""}, id="empty doc_id"),
        pytest.param({"doc_id": "-leading-dash"}, id="doc_id not starting alnum"),
        pytest.param({"doc_id": "has spaces"}, id="doc_id with whitespace"),
        pytest.param({"doc_id": "a" * 129}, id="doc_id too long"),
        pytest.param({"text": "   \n  "}, id="whitespace-only text"),
        pytest.param({"text": ""}, id="empty text"),
        pytest.param({"title": ""}, id="empty title"),
        pytest.param({"source_url": "ftp://example.com/doc"}, id="non-http source_url"),
    ],
)
async def test_invalid_documents_are_rejected_before_anything_is_written(
    client, db_session, embedder, overrides
):
    # Act
    response = await client.post("/ingest", json=document(**overrides))

    # Assert
    assert response.status_code == 422
    assert await count_chunks(db_session) == 0
    assert embedder.calls == []


async def test_an_oversized_document_is_rejected(client, db_session, embedder):
    # Arrange — one character past the cap
    oversized = document(text="ا" * (MAX_TEXT_CHARS + 1))

    # Act
    response = await client.post("/ingest", json=oversized)

    # Assert
    assert response.status_code == 422
    assert await count_chunks(db_session) == 0


async def test_a_document_at_the_size_limit_is_accepted(client, db_session, embedder):
    # Act — the cap itself must be inclusive; ~3x the largest corpus document
    response = await client.post("/ingest", json=document(text="ا " * (MAX_TEXT_CHARS // 2)))

    # Assert
    assert response.status_code == 200
    assert response.json()["chunks_written"] > 0


# -------------------------------------------------------- POST /ingest/pubsub


async def test_a_pubsub_envelope_is_decoded_and_ingested(client, db_session, embedder):
    # Act
    response = await client.post("/ingest/pubsub", json=push_body())

    # Assert
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["message_id"] == "13548923456789"
    assert body["doc_id"] == DOC_ID
    assert body["chunks_written"] == 3
    assert await chunk_ids(db_session) == EXPECTED_CHUNK_IDS


async def test_the_same_message_delivered_twice_leaves_the_table_unchanged(
    client, db_session, embedder
):
    """At-least-once delivery is a guarantee, not an edge case.

    Chunk ids are a pure function of the document text and every write is an
    upsert on that id, which is exactly what makes a redelivery a no-op — so no
    dedupe table keyed on messageId is needed anywhere in the push path.
    """
    # Arrange — one delivery
    body = push_body()
    first = await client.post("/ingest/pubsub", json=body)
    assert first.status_code == 200
    before = await count_chunks(db_session)

    # Act — the identical message again, same messageId
    second = await client.post("/ingest/pubsub", json=body)

    # Assert
    assert second.status_code == 200
    assert second.json()["chunks_written"] == first.json()["chunks_written"]
    assert await count_chunks(db_session) == before == 3
    assert await chunk_ids(db_session) == EXPECTED_CHUNK_IDS


async def test_a_redelivery_of_an_edited_document_updates_rows_in_place(
    client, db_session, embedder
):
    # Arrange
    await client.post("/ingest/pubsub", json=push_body())

    # Act — same doc_id and article, different wording
    edited = DOC_TEXT.replace("شهر", "شهرين")
    await client.post("/ingest/pubsub", json=push_body(text=edited))

    # Assert — updated, not duplicated
    assert await count_chunks(db_session) == 3
    assert "شهرين" in (await db_session.get(Chunk, f"{DOC_ID}:2:0")).text


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        pytest.param(envelope("not base64 at all !!"), "base64", id="malformed base64"),
        pytest.param(envelope(b64("{not json")), "JSON", id="malformed JSON payload"),
        pytest.param(envelope(b64("[1, 2, 3]")), "JSON object", id="payload is not an object"),
        pytest.param(envelope(b64(json.dumps(document(doc_id="a:b")))), "':'", id="bad doc_id"),
        pytest.param(
            envelope(b64(json.dumps({"doc_id": DOC_ID}))), "Field required", id="missing fields"
        ),
        pytest.param({"subscription": SUBSCRIPTION}, "'message'", id="no message key"),
        pytest.param({"message": {"messageId": "1"}}, "message.data", id="no data key"),
        pytest.param({"message": {"data": 42}}, "message.data", id="data is not a string"),
    ],
)
async def test_a_permanently_bad_message_is_acked_and_logged_not_retried(
    client, db_session, embedder, caplog, body, reason
):
    """2xx on purpose: a redelivery re-runs the same parse and fails the same way.

    Answering 4xx here would make Pub/Sub redeliver the message until the
    subscription's retention runs out — the classic poison-message loop. The
    ERROR log and the rejected counter are what stop the ack from being silent.
    """
    # Act
    with caplog.at_level(logging.ERROR, logger="app.api.ingest"):
        response = await client.post("/ingest/pubsub", json=body)

    # Assert — acked, explained, and nothing written
    assert response.status_code == 200
    assert response.json()["status"] == "rejected"
    assert reason in response.json()["reason"]
    assert "dropping unprocessable Pub/Sub message" in caplog.text
    assert await count_chunks(db_session) == 0
    assert embedder.calls == []


async def test_a_body_that_is_not_json_at_all_is_acked(client, db_session, embedder):
    # Act — FastAPI would answer 422 for this if the route declared a body model,
    # which is exactly the retry loop the raw-body read avoids.
    response = await client.post(
        "/ingest/pubsub", content=b"\xff\xfe not json", headers={"content-type": "application/json"}
    )

    # Assert
    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


async def test_an_empty_body_is_acked(client, db_session, embedder):
    # Act
    response = await client.post("/ingest/pubsub", content=b"")

    # Assert
    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


async def test_a_database_failure_returns_5xx_so_pubsub_retries(
    client, db_session, embedder, caplog
):
    """The other half of the taxonomy: transient failures must NOT be acked.

    Acking this would drop the document with no record that it ever existed.
    """
    # Arrange
    app.dependency_overrides[get_db] = lambda: BrokenSession()

    # Act
    with caplog.at_level(logging.ERROR, logger="app.api.ingest"):
        response = await client.post("/ingest/pubsub", json=push_body())

    # Assert
    assert response.status_code >= 500
    assert response.status_code == 503
    assert "failed against the database" in caplog.text


# ------------------------------------------------------------- decode_push


def test_decode_push_returns_the_envelope_metadata():
    # Act
    message = decode_push(json.dumps(push_body()).encode("utf-8"))

    # Assert
    assert message.document.doc_id == DOC_ID
    assert message.document.text == DOC_TEXT
    assert message.message_id == "13548923456789"
    assert message.subscription == SUBSCRIPTION


def test_decode_push_rejects_a_body_over_the_pubsub_message_limit():
    # Arrange — Pub/Sub caps a message at 10 MB, so this did not come from Pub/Sub
    body = b'{"message": {"data": "' + b"A" * (11 * 1024 * 1024) + b'"}}'

    # Act / Assert
    with pytest.raises(RejectedMessage, match="over the"):
        decode_push(body)


def test_decode_push_rejects_a_payload_that_is_not_utf8():
    # Act / Assert — base64 is valid, the bytes inside are not text
    with pytest.raises(RejectedMessage, match="not valid JSON"):
        decode_push(json.dumps(envelope(b64(b"\xff\xfe\xfd"))).encode("utf-8"))


# ------------------------------------------------------------------ CI guard


def test_the_ingest_endpoint_imports_neither_torch_nor_sentence_transformers():
    """Ingestion embeds, so the temptation to import the model stack is real.

    A subprocess, because this session has already imported torch for the rerank
    tests — asserting on *this* process would prove nothing.
    """
    # Arrange
    probe = (
        "import sys, app.api.ingest, app.ingest_worker;"
        "print(int('torch' in sys.modules), int('sentence_transformers' in sys.modules))"
    )

    # Act
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    # Assert
    assert result.stdout.strip() == "0 0", result.stdout


# ------------------------------------------------------------ x-api-key gate


async def test_ingest_is_open_when_no_key_is_configured(client, db_session, embedder):
    # Arrange — the shipped default: empty key, local demo runs with no config
    assert settings.ingest_api_key == ""

    # Act
    response = await client.post("/ingest", json=document())

    # Assert
    assert response.status_code == 200


async def test_ingest_rejects_a_missing_or_wrong_key_once_one_is_configured(
    client, db_session, embedder, monkeypatch
):
    # Arrange
    monkeypatch.setattr(settings, "ingest_api_key", "s3cret")

    # Act
    missing = await client.post("/ingest", json=document())
    wrong = await client.post(
        "/ingest", json=document(), headers={"x-api-key": "s3cre7"}
    )

    # Assert — and nothing was written
    assert missing.status_code == 401
    assert wrong.status_code == 401
    assert await count_chunks(db_session) == 0


async def test_ingest_accepts_the_configured_key(
    client, db_session, embedder, monkeypatch
):
    # Arrange
    monkeypatch.setattr(settings, "ingest_api_key", "s3cret")

    # Act
    response = await client.post(
        "/ingest", json=document(), headers={"x-api-key": "s3cret"}
    )

    # Assert
    assert response.status_code == 200
    assert await count_chunks(db_session) == 3


async def test_the_pubsub_push_route_is_not_behind_the_api_key(
    client, db_session, embedder, monkeypatch
):
    """It is OIDC-authenticated at the infra layer instead; see the route docstring."""
    # Arrange
    monkeypatch.setattr(settings, "ingest_api_key", "s3cret")

    # Act
    response = await client.post("/ingest/pubsub", json=push_body())

    # Assert
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
