"""POST /ask — the HTTP surface, over real SSE frames.

The streaming assertions parse the bytes the client would actually receive
rather than the ``(event, payload)`` tuples the service yields. Frame order is
the contract a browser depends on, and only the wire format proves it.

Fakes come from ``tests/test_service.py``; ``conftest.py`` is not ours to extend.
"""

from __future__ import annotations

import json

import pytest

from app.deps import get_service, reset_singletons
from app.generation.base import ErrorKind, ProviderError
from app.generation.failover import AllProvidersFailed
from app.main import app
from app.observability.cost import SpendTracker
from app.retrieval.cache import store as cache_store
from app.service import (
    EVENT_CITATIONS,
    EVENT_DONE,
    EVENT_ERROR,
    EVENT_FINAL,
    EVENT_TOKEN,
)
from tests.test_service import (
    ANSWER_TEXT,
    GULF_QUESTION,
    MSA_QUESTION,
    NOTICE_TEXT,
    FakeProvider,
    FakeReranker,
    make_service,
    seed_corpus,
    unit_vector,
)


@pytest.fixture()
def use_service():
    """Install a RagService for the route to use, and clean up after."""

    def install(**kwargs):
        service = make_service(**kwargs)
        app.dependency_overrides[get_service] = lambda: service
        return service

    yield install
    app.dependency_overrides.pop(get_service, None)


def parse_sse(body: str) -> list[tuple[str, dict]]:
    """Parse real ``event:``/``data:`` frames into ``(event, payload)`` pairs."""
    frames = []
    for block in body.strip().split("\n\n"):
        event, data = None, []
        for line in block.splitlines():
            if line.startswith("event: "):
                event = line.removeprefix("event: ")
            elif line.startswith("data: "):
                data.append(line.removeprefix("data: "))
        frames.append((event, json.loads("\n".join(data))))
    return frames


async def post(client, **body):
    return await client.post("/ask", json=body)


# ------------------------------------------------------------- non-streaming


async def test_non_streaming_returns_the_generated_text_and_its_citations(
    client, db_session, use_service
):
    # Arrange
    await seed_corpus(db_session)
    provider = FakeProvider()
    use_service(provider=provider)

    # Act
    response = await post(client, question=MSA_QUESTION, stream=False)

    # Assert
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == ANSWER_TEXT
    assert body["cached"] is False and body["refused"] is False
    assert body["register"] == "msa"
    assert [citation["chunk_id"] for citation in body["citations"]] == [
        "law:49:0",
        "law:79:0",
    ]
    assert body["citations"][0]["excerpt"] == NOTICE_TEXT
    assert body["usage"]["output_tokens"] == 40
    assert body["cost_usd"] == pytest.approx(0.0004)
    assert body["stages_ms"]["generate"] >= 0.0
    assert provider.complete_calls == 1


async def test_a_gulf_question_is_answered_in_the_gulf_register_field(
    client, db_session, use_service
):
    # Arrange
    await seed_corpus(db_session)
    use_service()

    # Act
    response = await post(client, question=GULF_QUESTION, stream=False)

    # Assert
    assert response.json()["register"] == "gulf"


# ----------------------------------------------------------------- streaming


async def test_streaming_emits_citations_before_every_token_then_a_final_usage_event(
    client, db_session, use_service
):
    # Arrange
    await seed_corpus(db_session)
    use_service()

    # Act
    response = await post(client, question=MSA_QUESTION, stream=True)

    # Assert
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    frames = parse_sse(response.text)
    names = [name for name, _ in frames]

    assert names[0] == EVENT_CITATIONS
    assert EVENT_TOKEN in names
    assert all(
        names.index(EVENT_CITATIONS) < index
        for index, name in enumerate(names)
        if name == EVENT_TOKEN
    )
    assert names[-2:] == [EVENT_FINAL, EVENT_DONE]

    citations = frames[0][1]["citations"]
    assert [citation["chunk_id"] for citation in citations] == ["law:49:0", "law:79:0"]

    text = "".join(payload["text"] for name, payload in frames if name == EVENT_TOKEN)
    assert text.strip() == ANSWER_TEXT

    final = frames[-2][1]
    assert final["usage"] == {
        "input_tokens": 120,
        "output_tokens": 40,
        "cost_usd": pytest.approx(0.0004),
    }
    assert final["cost_usd"] == pytest.approx(0.0004)
    assert set(final["stages_ms"]) >= {"plan", "retrieve", "rerank", "generate", "total"}


async def test_streaming_reports_a_generation_failure_as_an_error_frame(
    client, db_session, use_service
):
    # Arrange — headers are already sent, so 502 is no longer available
    await seed_corpus(db_session)
    failure = AllProvidersFailed([("anthropic", "timeout: no response within 20.0s")])
    use_service(provider=FakeProvider(error=failure))

    # Act
    response = await post(client, question=MSA_QUESTION, stream=True)

    # Assert
    assert response.status_code == 200
    frames = parse_sse(response.text)
    assert [name for name, _ in frames] == [EVENT_CITATIONS, EVENT_ERROR, EVENT_DONE]
    assert frames[1][1]["attempts"] == [
        {"provider": "anthropic", "reason": "timeout: no response within 20.0s"}
    ]


async def test_streaming_reports_a_fatal_provider_error_as_an_error_frame(
    client, db_session, use_service
):
    # Arrange — a 401 never fails over, so it arrives as a bare ProviderError
    # mid-stream. Dropping the socket would tell the client nothing.
    await seed_corpus(db_session)
    use_service(provider=FakeProvider(error=ProviderError("fake", ErrorKind.FATAL, "HTTP 401")))

    # Act
    response = await post(client, question=MSA_QUESTION, stream=True)

    # Assert
    frames = parse_sse(response.text)
    assert [name for name, _ in frames] == [EVENT_CITATIONS, EVENT_ERROR, EVENT_DONE]
    assert frames[1][1]["attempts"] == [
        {"provider": "fake", "reason": "fatal: HTTP 401"}
    ]


# --------------------------------------------------------------------- cache


async def test_a_cache_hit_never_reaches_the_provider(client, db_session, use_service):
    # Arrange
    await seed_corpus(db_session)
    provider = FakeProvider()
    use_service(provider=provider)
    await cache_store(
        db_session, MSA_QUESTION, unit_vector(0), "bge", "جواب محفوظ", ["law:49:0"]
    )

    # Act
    response = await post(client, question=MSA_QUESTION, stream=False)

    # Assert
    body = response.json()
    assert body["cached"] is True
    assert body["answer"] == "جواب محفوظ"
    assert body["usage"] is None
    assert provider.calls == 0


# ------------------------------------------------------------- refusal gate


async def test_a_low_confidence_question_is_refused_without_an_llm_call(
    client, db_session, use_service
):
    # Arrange
    await seed_corpus(db_session)
    provider = FakeProvider()
    use_service(
        reranker=FakeReranker(score=0.01), provider=provider, rerank_min_score=0.15
    )

    # Act
    response = await post(client, question=MSA_QUESTION, stream=False)

    # Assert
    body = response.json()
    assert body["refused"] is True
    assert body["answer"] == "لا تتضمن المواد المتاحة إجابة عن هذا السؤال."
    assert body["citations"] == []
    assert body["usage"] is None
    assert provider.calls == 0


# --------------------------------------------------------------- cost/errors


async def test_the_daily_spend_cap_returns_503_with_the_numbers(
    client, db_session, use_service
):
    # Arrange
    await seed_corpus(db_session)
    provider = FakeProvider()
    use_service(provider=provider, spend=SpendTracker(0.0))

    # Act
    response = await post(client, question=MSA_QUESTION, stream=False)

    # Assert
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["error"] == "daily_spend_cap_exceeded"
    assert detail["cap_usd"] == 0.0
    assert detail["remaining_usd"] == 0.0
    assert "spend cap" in detail["message"]
    assert provider.calls == 0


async def test_the_spend_cap_returns_503_on_the_streaming_path_too(
    client, db_session, use_service
):
    # Arrange — the route pulls the first event before committing a response,
    # which is what keeps a status code available here.
    await seed_corpus(db_session)
    use_service(spend=SpendTracker(0.0))

    # Act
    response = await post(client, question=MSA_QUESTION, stream=True)

    # Assert
    assert response.status_code == 503
    assert response.json()["detail"]["error"] == "daily_spend_cap_exceeded"


async def test_all_providers_failing_returns_502_naming_every_attempt(
    client, db_session, use_service
):
    # Arrange
    await seed_corpus(db_session)
    failure = AllProvidersFailed(
        [("anthropic", "timeout: no response within 20.0s"), ("gemini", "server: HTTP 503")]
    )
    use_service(provider=FakeProvider(error=failure))

    # Act
    response = await post(client, question=MSA_QUESTION, stream=False)

    # Assert
    assert response.status_code == 502
    detail = response.json()["detail"]
    assert detail["error"] == "all_providers_failed"
    assert [attempt["provider"] for attempt in detail["attempts"]] == [
        "anthropic",
        "gemini",
    ]
    assert "timeout" in detail["attempts"][0]["reason"]


async def test_a_fatal_provider_error_is_not_dressed_up_as_an_answer(
    client, db_session, use_service
):
    # Arrange — a 401 is our bug, and FailoverProvider deliberately never fails
    # over on it; the route must not turn it into a 200.
    await seed_corpus(db_session)
    use_service(provider=FakeProvider(error=ProviderError("fake", ErrorKind.FATAL, "HTTP 401")))

    # Act / Assert
    with pytest.raises(ProviderError):
        await post(client, question=MSA_QUESTION, stream=False)


async def test_a_keyless_environment_returns_503_naming_the_missing_key(
    client, db_session
):
    # Arrange — no dependency override, so the real deps.get_service runs.
    reset_singletons()

    # Act
    response = await post(client, question=MSA_QUESTION, stream=False)

    # Assert — a configuration failure names the variable to set; it is not a
    # 500, and nothing was loaded to discover it (the embedder and reranker are
    # constructed first and neither touches its weights until it is used).
    assert response.status_code == 503
    assert "ANTHROPIC_API_KEY" in response.json()["detail"]
    reset_singletons()


# ---------------------------------------------------------------- validation


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        ({"question": ""}, "empty"),
        ({"question": "   "}, "whitespace only"),
        ({"question": "س" * 1001}, "over the length cap"),
        ({"question": MSA_QUESTION, "config": "magic"}, "unknown config"),
        ({}, "missing question"),
    ],
)
async def test_bad_requests_are_rejected_with_422(client, use_service, body, reason):
    # Arrange — no seeding: validation must reject before the DB is touched
    provider = FakeProvider()
    use_service(provider=provider)

    # Act
    response = await client.post("/ask", json={"stream": False, **body})

    # Assert
    assert response.status_code == 422, reason
    assert provider.calls == 0


async def test_the_422_for_an_unknown_config_lists_the_valid_ones(client, use_service):
    # Arrange
    use_service()

    # Act
    response = await client.post(
        "/ask", json={"question": MSA_QUESTION, "config": "magic", "stream": False}
    )

    # Assert
    message = json.dumps(response.json(), ensure_ascii=False)
    assert "hybrid+rerank" in message and "magic" in message
