"""RagService — the pipeline without HTTP.

Every collaborator here is a fake: no weights are loaded, no provider is called,
no API key exists. That is the point of injecting them — the orchestration is
what these tests are about, and it is the part that decides whether an LLM gets
called at all.

The fakes live in this module and are imported by ``tests/test_ask.py``; there is
no shared fixtures file to put them in (``conftest.py`` is owned elsewhere) and
two copies would drift.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from sqlalchemy import func, select

from app.config import settings
from app.constants import (
    DEFAULT_CONFIG,
    EVENT_CITATIONS,
    EVENT_DONE,
    EVENT_ERROR,
    EVENT_FINAL,
    EVENT_TOKEN,
    GARBLED,
    NOT_IN_CORPUS,
    REFUSALS,
    RERANK_SOURCE,
)
from app.data import Completion, Hit, Usage
from app.generation.base import ErrorKind, ProviderError
from app.generation.failover import AllProvidersFailed
from app.models.chunks import Chunk
from app.models.query_cache import QueryCache
from app.observability.cost import SpendCapExceeded, SpendTracker
from app.planning.planner import NoopPlanner, RuleBasedPlanner
from app.retrieval.cache import store as cache_store
from app.service import RagService, _cited_scores
from ingestion.normalize import normalize_for_index

REPO_ROOT = Path(__file__).resolve().parent.parent

DIM = 1024  # bge-m3, and the single width query_cache stores

MSA_QUESTION = "ما هي مدة الإشعار قبل إنهاء العقد؟"
GULF_QUESTION = "شكثر مدة الإشعار اللي لازم أعطيها قبل ما أطلع من الشغل؟"

NOTICE_TEXT = "على صاحب العمل إخطار العامل قبل إنهاء العقد بشهر واحد على الأقل"
LEAVE_TEXT = "للعامل الحق في إجازة سنوية مدفوعة الأجر لا تقل عن ثلاثة أسابيع"
ANSWER_TEXT = "مدة الإشعار شهر واحد [المادة 49]."


# --------------------------------------------------------------------- fakes


def unit_vector(axis: int = 0) -> list[float]:
    """One-hot, so two different axes are exactly orthogonal (cosine 0)."""
    vector = [0.0] * DIM
    vector[axis] = 1.0
    return vector


class FakeEmbedder:
    """Deterministic one-hot vectors. Records every batch it was asked for."""

    model_key = "bge"
    dim = DIM

    def __init__(self, axes: dict[str, int] | None = None) -> None:
        self.axes = axes or {}
        self.batches: list[list[str]] = []

    async def embed_queries(self, texts: list[str]) -> list[list[float]]:
        self.batches.append(list(texts))
        return [unit_vector(self.axes.get(text, 0)) for text in texts]

    async def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return await self.embed_queries(texts)


class FakeReranker:
    """Emits ``source="rerank"`` and a fixed 0-1 score — what the floor reads."""

    name = "fake"

    def __init__(self, score: float = 0.9) -> None:
        self.score = score
        self.queries: list[str] = []

    async def rerank(self, query: str, hits, top_k: int = 5) -> list[Hit]:
        self.queries.append(query)
        return [
            replace(hit, score=self.score, source=RERANK_SOURCE)
            for hit in list(hits)[:top_k]
        ]


class FakeProvider:
    """Counts calls, so "the provider was never called" is a real assertion."""

    name = "fake"
    model = "fake-model"

    def __init__(self, text: str = ANSWER_TEXT, error: Exception | None = None) -> None:
        self.text = text
        self.error = error
        self.usage = Usage(input_tokens=120, output_tokens=40, cost_usd=0.0004)
        self.complete_calls = 0
        self.stream_calls = 0

    @property
    def calls(self) -> int:
        return self.complete_calls + self.stream_calls

    def price(self) -> tuple[float, float]:
        return (1.0, 5.0)

    async def complete(self, system, messages, max_tokens=1024) -> Completion:
        self.complete_calls += 1
        if self.error is not None:
            raise self.error
        return Completion(
            text=self.text, usage=self.usage, provider=self.name, model=self.model
        )

    async def stream(self, system, messages, max_tokens=1024, *, on_usage=None):
        self.stream_calls += 1
        if self.error is not None:
            raise self.error
        for word in self.text.split(" "):
            yield word + " "
        if on_usage is not None:
            on_usage(self.usage)


def make_settings(**overrides):
    """The real Settings with a few fields moved — no hand-rolled stub to drift."""
    return settings.model_copy(update=overrides)


def make_service(
    *,
    embedder: FakeEmbedder | None = None,
    reranker: FakeReranker | None = None,
    planner=None,
    provider: FakeProvider | None = None,
    spend: SpendTracker | None = None,
    **setting_overrides,
) -> RagService:
    return RagService(
        embedder or FakeEmbedder(),
        reranker or FakeReranker(),
        planner or RuleBasedPlanner(),
        provider or FakeProvider(),
        spend or SpendTracker(5.0),
        make_settings(**setting_overrides),
    )


def make_chunk(chunk_id: str, text: str, article: str = "49", axis: int = 0) -> Chunk:
    return Chunk(
        id=chunk_id,
        doc_id=chunk_id.split(":")[0],
        article=article,
        seq=0,
        text=text,
        text_normalized=normalize_for_index(text),
        emb_bge=unit_vector(axis),
    )


async def seed_corpus(session) -> None:
    session.add_all(
        [
            make_chunk("law:49:0", NOTICE_TEXT, article="49", axis=0),
            make_chunk("law:79:0", LEAVE_TEXT, article="79", axis=1),
        ]
    )
    await session.commit()


async def collect(events) -> list[tuple[str, dict]]:
    return [event async for event in events]


# ---------------------------------------------------------------- happy path


async def test_answer_returns_generated_text_and_citations(db_session):
    # Arrange
    await seed_corpus(db_session)
    provider = FakeProvider()
    service = make_service(provider=provider)

    # Act
    answer = await service.answer(MSA_QUESTION)

    # Assert
    assert answer.text == ANSWER_TEXT
    assert answer.cached is False and answer.refused is False
    assert [citation.chunk_id for citation in answer.citations] == ["law:49:0", "law:79:0"]
    assert answer.citations[0].article == "49"
    assert answer.citations[0].excerpt == NOTICE_TEXT
    assert answer.usage == provider.usage
    assert provider.complete_calls == 1


async def test_every_pipeline_stage_is_timed(db_session):
    # Arrange
    await seed_corpus(db_session)
    service = make_service()

    # Act
    answer = await service.answer(MSA_QUESTION)

    # Assert — the stage list is the pipeline contract, not decoration
    assert set(answer.stages) == {
        "plan",
        "embed",
        "cache.lookup",
        "retrieve",
        "fuse",
        "rerank",
        "generate",
        "total",
    }
    assert all(value >= 0.0 for value in answer.stages.values())


async def test_the_gulf_rewrite_is_retrieved_with_alongside_the_original(db_session):
    # Arrange
    await seed_corpus(db_session)
    embedder = FakeEmbedder()
    service = make_service(embedder=embedder)

    # Act
    await service.answer(GULF_QUESTION)

    # Assert — one batch, original first, MSA rewrite appended (never substituted)
    assert len(embedder.batches) == 1
    batch = embedder.batches[0]
    assert batch[0] == GULF_QUESTION
    assert len(batch) == 2
    assert "كم" in batch[1] and "شكثر" not in batch[1]


async def test_rerank_scores_the_msa_rewrite_not_the_dialect_question(db_session):
    # Arrange — the cross-encoder's absolute score is not dialect-neutral, and the
    # refusal floor reads that absolute score. See app/retrieval/rerank.py.
    await seed_corpus(db_session)
    reranker = FakeReranker()
    service = make_service(reranker=reranker)

    # Act
    await service.answer(GULF_QUESTION)

    # Assert
    assert reranker.queries and "شكثر" not in reranker.queries[0]


# --------------------------------------------------------------------- cache


async def test_cache_hit_returns_the_stored_answer_without_calling_the_provider(
    db_session,
):
    # Arrange
    await seed_corpus(db_session)
    provider = FakeProvider()
    service = make_service(provider=provider)
    await cache_store(
        db_session,
        MSA_QUESTION,
        unit_vector(0),
        "bge",
        # The service's own key, not a literal: a hand-written copy would drift
        # and the test would pass by missing the cache instead of hitting it.
        service._pipeline_key(DEFAULT_CONFIG),
        "جواب محفوظ",
        ["law:49:0"],
    )

    # Act
    answer = await service.answer(MSA_QUESTION)

    # Assert
    assert answer.cached is True
    assert answer.text == "جواب محفوظ"
    assert answer.usage is None
    assert provider.calls == 0
    # Citations are rehydrated from the corpus, not returned as bare ids
    assert [citation.chunk_id for citation in answer.citations] == ["law:49:0"]
    assert answer.citations[0].excerpt == NOTICE_TEXT


def test_cited_scores_reads_both_stored_shapes():
    # Arrange: a bare id (written before scores were stored), a current scored
    # entry, and two rows this version cannot read.
    stored = ["law:49:0", {"chunk_id": "law:50:0", "score": 0.8125}, {"oops": 1}, 7]

    # Act
    pairs = _cited_scores(stored)

    # Assert: legacy reports the 0.0 it always did, junk is skipped rather than
    # crashing the hit, and order is preserved.
    assert pairs == [("law:49:0", 0.0), ("law:50:0", 0.8125)]


async def test_a_cache_hit_replays_the_stored_rerank_score(db_session):
    # Arrange
    await seed_corpus(db_session)
    service = make_service(provider=FakeProvider())
    await cache_store(
        db_session,
        MSA_QUESTION,
        unit_vector(0),
        "bge",
        service._pipeline_key(DEFAULT_CONFIG),
        "جواب محفوظ",
        [{"chunk_id": "law:49:0", "score": 0.8125}],
    )

    # Act
    answer = await service.answer(MSA_QUESTION)

    # Assert: a cached citation renders identically to a generated one, which is
    # the whole point — a 0.0 here made cached answers look unranked in any UI.
    assert answer.cached is True
    assert [citation.score for citation in answer.citations] == [0.8125]


async def test_a_generated_answer_is_cached_and_the_next_call_reuses_it(db_session):
    # Arrange
    await seed_corpus(db_session)
    provider = FakeProvider()
    service = make_service(provider=provider)

    # Act
    first = await service.answer(MSA_QUESTION)
    second = await service.answer(MSA_QUESTION)

    # Assert
    assert first.cached is False and second.cached is True
    assert second.text == first.text
    assert provider.complete_calls == 1


async def test_a_refusal_is_never_cached(db_session):
    # Arrange
    await seed_corpus(db_session)
    provider = FakeProvider()
    service = make_service(
        reranker=FakeReranker(score=0.01), provider=provider, rerank_min_score=0.15
    )

    # Act
    first = await service.answer(MSA_QUESTION)
    second = await service.answer(MSA_QUESTION)

    # Assert — a cached refusal would freeze a wrong "no" into the corpus
    assert first.refused is True
    assert second.refused is True and second.cached is False
    assert provider.calls == 0


# ------------------------------------------------------------- refusal gate


async def test_refusal_below_the_rerank_floor_skips_the_provider(db_session):
    # Arrange
    await seed_corpus(db_session)
    provider = FakeProvider()
    service = make_service(
        reranker=FakeReranker(score=0.02), provider=provider, rerank_min_score=0.15
    )

    # Act
    answer = await service.answer(MSA_QUESTION)

    # Assert
    assert answer.refused is True
    assert answer.text == NOT_IN_CORPUS
    assert answer.citations == []
    assert answer.usage is None
    assert provider.calls == 0


async def test_a_gulf_question_is_refused_in_gulf(db_session):
    # Arrange
    await seed_corpus(db_session)
    service = make_service(reranker=FakeReranker(score=0.0), rerank_min_score=0.15)

    # Act
    answer = await service.answer(GULF_QUESTION)

    # Assert
    assert answer.register == "gulf"
    assert answer.text == REFUSALS["gulf"] != NOT_IN_CORPUS


async def test_an_empty_corpus_is_a_refusal_not_a_crash(db_session):
    # Arrange — nothing seeded
    provider = FakeProvider()
    service = make_service(provider=provider)

    # Act
    answer = await service.answer(MSA_QUESTION)

    # Assert
    assert answer.refused is True
    assert provider.calls == 0


async def test_the_floor_is_not_applied_to_fusion_scores(db_session):
    # Arrange — RRF scores live around 1/61; comparing them to a 0.15 rerank
    # floor would refuse every un-reranked request.
    await seed_corpus(db_session)
    provider = FakeProvider()
    service = make_service(provider=provider)

    # Act
    answer = await service.answer(MSA_QUESTION, config="hybrid")

    # Assert
    assert answer.refused is False
    assert answer.citations[0].score < 0.15
    assert provider.complete_calls == 1


# ------------------------------------------------------------------ streaming


async def test_stream_emits_citations_first_then_tokens_then_final_then_done(db_session):
    # Arrange
    await seed_corpus(db_session)
    service = make_service()

    # Act
    events = await collect(service.stream(MSA_QUESTION))

    # Assert
    names = [name for name, _ in events]
    assert names[0] == EVENT_CITATIONS
    assert names[-2:] == [EVENT_FINAL, EVENT_DONE]
    assert names.index(EVENT_CITATIONS) < names.index(EVENT_TOKEN)
    assert "".join(payload["text"] for name, payload in events if name == EVENT_TOKEN) == (
        ANSWER_TEXT + " "
    )
    final = events[-2][1]
    assert final["usage"]["output_tokens"] == 40
    assert final["cost_usd"] == pytest.approx(0.0004)
    assert final["stages_ms"]["generate"] >= 0.0


async def test_stream_reports_a_generation_failure_as_an_error_event(db_session):
    # Arrange — the citations frame is already on the wire, so there is no status
    # code left to set; the failure has to arrive as data.
    await seed_corpus(db_session)
    failure = AllProvidersFailed(
        [("anthropic", "timeout: no response within 20.0s"), ("gemini", "server: HTTP 503")]
    )
    service = make_service(provider=FakeProvider(error=failure))

    # Act
    events = await collect(service.stream(MSA_QUESTION))

    # Assert
    names = [name for name, _ in events]
    assert names == [EVENT_CITATIONS, EVENT_ERROR, EVENT_DONE]
    attempts = events[1][1]["attempts"]
    assert [attempt["provider"] for attempt in attempts] == ["anthropic", "gemini"]


async def test_stream_serves_a_cache_hit_as_one_token_event(db_session):
    # Arrange
    await seed_corpus(db_session)
    provider = FakeProvider()
    service = make_service(provider=provider)
    await cache_store(
        db_session,
        MSA_QUESTION,
        unit_vector(0),
        "bge",
        # The service's own key, not a literal: a hand-written copy would drift
        # and the test would pass by missing the cache instead of hitting it.
        service._pipeline_key(DEFAULT_CONFIG),
        "جواب محفوظ",
        ["law:49:0"],
    )

    # Act
    events = await collect(service.stream(MSA_QUESTION))

    # Assert
    assert [name for name, _ in events] == [
        EVENT_CITATIONS,
        EVENT_TOKEN,
        EVENT_FINAL,
        EVENT_DONE,
    ]
    assert events[0][1]["cached"] is True
    assert events[1][1]["text"] == "جواب محفوظ"
    assert provider.calls == 0


# ----------------------------------------------------------------- spend cap


async def test_the_spend_cap_is_checked_before_the_provider_is_called(db_session):
    # Arrange
    await seed_corpus(db_session)
    provider = FakeProvider()
    service = make_service(provider=provider, spend=SpendTracker(0.0))

    # Act / Assert
    with pytest.raises(SpendCapExceeded):
        await service.answer(MSA_QUESTION)
    assert provider.calls == 0


async def test_the_spend_cap_stops_a_stream_before_its_first_event(db_session):
    # Arrange
    await seed_corpus(db_session)
    service = make_service(spend=SpendTracker(0.0))

    # Act / Assert — raising before the first yield is what lets /ask still 503
    with pytest.raises(SpendCapExceeded):
        await collect(service.stream(MSA_QUESTION))


async def test_a_generated_answer_is_recorded_against_the_days_spend(db_session):
    # Arrange
    await seed_corpus(db_session)
    spend = SpendTracker(5.0)
    service = make_service(spend=spend)

    # Act
    await service.answer(MSA_QUESTION)

    # Assert
    assert spend.today().calls == 1
    assert spend.today().usd == pytest.approx(0.0004)


# ---------------------------------------------------------------- validation


async def test_an_empty_question_is_rejected_at_the_service_boundary(db_session):
    # Arrange
    service = make_service()

    # Act / Assert
    with pytest.raises(ValueError, match="must not be empty"):
        await service.answer("   ")


async def test_an_unknown_config_names_the_valid_ones(db_session):
    # Arrange
    service = make_service()

    # Act / Assert
    with pytest.raises(ValueError, match="hybrid\\+rerank"):
        await service.answer(MSA_QUESTION, config="dense+magic")


async def test_a_lexical_only_config_still_retrieves(db_session):
    # Arrange
    await seed_corpus(db_session)
    service = make_service(planner=NoopPlanner())

    # Act
    answer = await service.answer("الإشعار قبل إنهاء العقد", config="lexical")

    # Assert
    assert answer.refused is False
    assert answer.citations


async def test_a_provider_error_propagates_from_answer(db_session):
    # Arrange
    await seed_corpus(db_session)
    service = make_service(
        provider=FakeProvider(error=ProviderError("fake", ErrorKind.FATAL, "HTTP 401"))
    )

    # Act / Assert — a 401 is our bug; it must not be dressed up as an answer
    with pytest.raises(ProviderError):
        await service.answer(MSA_QUESTION)


# ------------------------------------------------------------------ CI guard


def test_importing_app_main_loads_neither_torch_nor_sentence_transformers():
    """The whole reason deps.py builds nothing at import time.

    A subprocess, because this test session has almost certainly imported torch
    already for the rerank tests — asserting on *this* process would prove
    nothing.
    """
    # Arrange
    probe = (
        "import sys, app.main;"
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


# ------------------------------------------- refusals must not enter the cache


async def test_a_model_generated_refusal_is_never_cached(db_session):
    """Regression: only the score gate short-circuited, and it is off by default.

    With `rerank_min_score` at its measured 0.0 the model declining per the
    system prompt is the live refusal path. Caching it pinned the refusal: the
    cache is append-only with no TTL and no invalidation on re-ingestion, so
    ingesting the very article that was missing would never dislodge it — and it
    came back with `refused=false`.
    """
    # Arrange
    await seed_corpus(db_session)
    provider = FakeProvider(text=NOT_IN_CORPUS)
    service = make_service(provider=provider)

    # Act
    first = await service.answer(MSA_QUESTION)
    second = await service.answer(MSA_QUESTION)

    # Assert — nothing was stored, so the second request regenerates
    assert first.text == NOT_IN_CORPUS
    assert second.cached is False
    assert provider.complete_calls == 2
    assert await db_session.scalar(select(func.count(QueryCache.id))) == 0


async def test_a_real_answer_is_still_cached(db_session):
    """The refusal guard must not block the normal path."""
    # Arrange
    await seed_corpus(db_session)
    provider = FakeProvider()
    service = make_service(provider=provider)

    # Act
    await service.answer(MSA_QUESTION)
    second = await service.answer(MSA_QUESTION)

    # Assert
    assert second.cached is True
    assert provider.complete_calls == 1


async def test_the_same_question_under_a_different_config_is_not_served_from_cache(
    db_session,
):
    """The cache key carries the pipeline; see app/retrieval/cache.py."""
    # Arrange
    await seed_corpus(db_session)
    provider = FakeProvider()
    service = make_service(provider=provider)
    await service.answer(MSA_QUESTION, "hybrid+rerank")

    # Act
    other = await service.answer(MSA_QUESTION, "dense")

    # Assert — a second real generation, not the first config's answer
    assert other.cached is False
    assert provider.complete_calls == 2


# ------------------------------------------ spend accounting on a broken stream


async def test_a_client_disconnect_mid_stream_still_settles_the_spend(db_session):
    """Regression: `_record_usage` sat after the provider loop with no `finally`.

    A browser closing an SSE tab throws GeneratorExit at the token `yield`, so
    the accounting never ran: the vendor had billed real tokens and the tracker
    recorded $0. Repeat-disconnect traffic made the daily cap a no-op.
    """
    # Arrange
    await seed_corpus(db_session)
    service = make_service(provider=FakeProvider())

    # Act — consume one token frame, then abandon the generator as a client would
    events = service.stream(MSA_QUESTION)
    async for name, _payload in events:
        if name == EVENT_TOKEN:
            break
    held = service.spend.today().usd
    await events.aclose()

    # Assert — the hold is released, not merely "not negative". The old
    # `usd >= 0.0` passed whether or not the settle ran at all; these fail if the
    # reservation is still outstanding, which over repeat disconnects is what
    # pinned the cap shut.
    assert held > 0.0, "the reservation should still be held mid-stream"
    assert service.spend.today().usd == 0.0
    assert service.spend.today().calls == 1


# ------------------------------------------ a generation that came apart mid-answer


CORRUPT_TEXT = "صاحب العمل ملزم بضمان النظافة،&oその [&المادة 103]."


async def test_a_corrupt_generation_is_not_handed_to_the_reader(db_session):
    """The check used to live only in `cache.store`, so the user still got this.

    On the non-streaming path the whole body exists server-side before anything
    is sent, so corrupted legal text reaching the reader was a choice, not a
    constraint.
    """
    # Arrange
    await seed_corpus(db_session)
    provider = FakeProvider(text=CORRUPT_TEXT)
    service = make_service(provider=provider)

    # Act
    answer = await service.answer(MSA_QUESTION)

    # Assert
    assert answer.text == GARBLED
    assert CORRUPT_TEXT not in answer.text
    # the tokens were really billed, so usage is still reported
    assert answer.usage is not None


async def test_a_corrupt_generation_is_never_cached(db_session):
    # Arrange
    await seed_corpus(db_session)
    service = make_service(provider=FakeProvider(text=CORRUPT_TEXT))

    # Act
    await service.answer(MSA_QUESTION)

    # Assert
    assert (await db_session.execute(select(QueryCache))).first() is None


async def test_a_corrupt_generation_is_reported_even_with_the_cache_off(db_session, caplog):
    """The warning was the only signal the generator misfired, and it lived in
    the cache — so turning the cache off turned the signal off with it."""
    # Arrange
    await seed_corpus(db_session)
    service = make_service(provider=FakeProvider(text=CORRUPT_TEXT))
    service._cache_enabled = False

    # Act
    with caplog.at_level("WARNING"):
        answer = await service.answer(MSA_QUESTION)

    # Assert
    assert answer.text == GARBLED
    assert "another script" in caplog.text
