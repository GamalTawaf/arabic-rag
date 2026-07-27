"""Unit tests for cross-encoder reranking.

No weights are loaded: the cross-encoder's `_load()` is monkeypatched with a fake
scorer, so these run in milliseconds and in CI without a model cache. The one test
that touches the real BAAI/bge-reranker-v2-m3 is opt-in via RERANK_REAL_MODEL=1.
"""

from __future__ import annotations

import asyncio
import math
import os
import threading
import time

import pytest

from app.retrieval.rerank import (
    CrossEncoderReranker,
    Hit,
    NoopReranker,
    _sigmoid,
    get_reranker,
)


def _hit(chunk_id: str, text: str, score: float = 0.5, source: str = "dense") -> Hit:
    return Hit(
        chunk_id=chunk_id,
        doc_id="labour-law-2004",
        article="79",
        text=text,
        score=score,
        source=source,
    )


class FakeCrossEncoder:
    """Stands in for sentence_transformers.CrossEncoder: predict(pairs) -> logits."""

    def __init__(self, logits_by_text: dict[str, float]):
        self._logits_by_text = logits_by_text
        self.calls: list[list[tuple[str, str]]] = []
        self.thread_names: list[str] = []

    def predict(self, pairs, **_kwargs):
        self.calls.append(list(pairs))
        self.thread_names.append(threading.current_thread().name)
        return [self._logits_by_text[text] for _query, text in pairs]


def _fake_reranker(logits_by_text: dict[str, float]) -> tuple[CrossEncoderReranker, FakeCrossEncoder]:
    reranker = CrossEncoderReranker()
    fake = FakeCrossEncoder(logits_by_text)
    reranker._load = lambda: fake  # type: ignore[method-assign]
    return reranker, fake


# --- sigmoid ----------------------------------------------------------------


def test_sigmoid_maps_logits_into_the_unit_interval():
    # Arrange
    logits = [-1000.0, -8.0, 0.0, 8.0, 1000.0]

    # Act
    scores = [_sigmoid(logit) for logit in logits]

    # Assert
    assert all(0.0 <= score <= 1.0 for score in scores)
    assert scores == sorted(scores)
    assert scores[2] == pytest.approx(0.5)


def test_sigmoid_does_not_overflow_on_extreme_logits():
    # Arrange / Act / Assert — math.exp(710) raises OverflowError if done naively
    assert _sigmoid(-750.0) == pytest.approx(0.0, abs=1e-12)
    assert _sigmoid(750.0) == pytest.approx(1.0, abs=1e-12)


# --- CrossEncoderReranker ---------------------------------------------------


async def test_rerank_sorts_hits_by_cross_encoder_score_descending():
    # Arrange
    hits = [_hit("c1", "alpha"), _hit("c2", "beta"), _hit("c3", "gamma")]
    reranker, _fake = _fake_reranker({"alpha": -3.0, "beta": 5.0, "gamma": 1.0})

    # Act
    ranked = await reranker.rerank("q", hits, top_k=3)

    # Assert
    assert [hit.chunk_id for hit in ranked] == ["c2", "c3", "c1"]


async def test_rerank_replaces_score_with_sigmoid_of_the_logit_and_tags_the_source():
    # Arrange
    hits = [_hit("c1", "alpha", score=0.91, source="rrf")]
    reranker, _fake = _fake_reranker({"alpha": 2.0})

    # Act
    ranked = await reranker.rerank("q", hits)

    # Assert
    assert ranked[0].score == pytest.approx(1 / (1 + math.exp(-2.0)))
    assert ranked[0].source == "rerank"


async def test_rerank_scores_every_query_hit_pair():
    # Arrange
    hits = [_hit("c1", "alpha"), _hit("c2", "beta")]
    reranker, fake = _fake_reranker({"alpha": 0.0, "beta": 1.0})

    # Act
    await reranker.rerank("ما هي مدة الإجازة؟", hits)

    # Assert
    assert fake.calls == [[("ما هي مدة الإجازة؟", "alpha"), ("ما هي مدة الإجازة؟", "beta")]]


async def test_rerank_truncates_to_top_k():
    # Arrange
    hits = [_hit("c1", "alpha"), _hit("c2", "beta"), _hit("c3", "gamma")]
    reranker, _fake = _fake_reranker({"alpha": 1.0, "beta": 9.0, "gamma": 5.0})

    # Act
    ranked = await reranker.rerank("q", hits, top_k=2)

    # Assert
    assert [hit.chunk_id for hit in ranked] == ["c2", "c3"]


async def test_rerank_returns_everything_sorted_when_top_k_exceeds_hit_count():
    # Arrange
    hits = [_hit("c1", "alpha"), _hit("c2", "beta")]
    reranker, _fake = _fake_reranker({"alpha": 1.0, "beta": 2.0})

    # Act
    ranked = await reranker.rerank("q", hits, top_k=50)

    # Assert
    assert [hit.chunk_id for hit in ranked] == ["c2", "c1"]


async def test_rerank_returns_empty_list_for_empty_hits_without_loading_the_model():
    # Arrange
    reranker = CrossEncoderReranker()

    def explode():
        raise AssertionError("model must not load for empty hits")

    reranker._load = explode  # type: ignore[method-assign]

    # Act
    ranked = await reranker.rerank("q", [])

    # Assert
    assert ranked == []


async def test_rerank_does_not_mutate_the_input_hits():
    # Arrange
    hits = [_hit("c1", "alpha", score=0.11, source="dense"), _hit("c2", "beta", score=0.22, source="lexical")]
    reranker, _fake = _fake_reranker({"alpha": 1.0, "beta": 2.0})

    # Act
    ranked = await reranker.rerank("q", hits)

    # Assert
    assert [(h.score, h.source) for h in hits] == [(0.11, "dense"), (0.22, "lexical")]
    assert all(returned is not original for returned in ranked for original in hits)


async def test_rerank_keeps_original_order_for_tied_scores():
    # Arrange
    hits = [_hit("c1", "alpha"), _hit("c2", "beta"), _hit("c3", "gamma")]
    reranker, _fake = _fake_reranker({"alpha": 1.0, "beta": 1.0, "gamma": 1.0})

    # Act
    ranked = await reranker.rerank("q", hits)

    # Assert
    assert [hit.chunk_id for hit in ranked] == ["c1", "c2", "c3"]


async def test_rerank_runs_the_blocking_model_call_off_the_event_loop_thread():
    # Arrange
    hits = [_hit("c1", "alpha")]
    reranker, fake = _fake_reranker({"alpha": 1.0})
    caller_thread = threading.current_thread().name

    # Act
    await reranker.rerank("q", hits)

    # Assert
    assert fake.thread_names == [fake.thread_names[0]]
    assert fake.thread_names[0] != caller_thread


async def test_rerank_rejects_non_positive_top_k():
    # Arrange
    reranker, _fake = _fake_reranker({"alpha": 1.0})

    # Act / Assert
    with pytest.raises(ValueError, match="top_k"):
        await reranker.rerank("q", [_hit("c1", "alpha")], top_k=0)


def test_cross_encoder_caches_the_loaded_model_on_the_instance():
    # Arrange
    reranker = CrossEncoderReranker()
    loads: list[int] = []

    class Recorder:
        pass

    def fake_ctor(*_args, **_kwargs):
        loads.append(1)
        return Recorder()

    reranker._build_model = fake_ctor  # type: ignore[method-assign]

    # Act
    first = reranker._load()
    second = reranker._load()

    # Assert
    assert first is second
    assert loads == [1]


# --- NoopReranker (ablation baseline) ---------------------------------------


async def test_noop_reranker_returns_the_first_top_k_hits_untouched():
    # Arrange
    hits = [_hit("c1", "alpha", score=0.9), _hit("c2", "beta", score=0.1), _hit("c3", "gamma")]

    # Act
    ranked = await NoopReranker().rerank("q", hits, top_k=2)

    # Assert
    assert ranked == [hits[0], hits[1]]


async def test_noop_reranker_returns_empty_list_for_empty_hits():
    # Act
    ranked = await NoopReranker().rerank("q", [])

    # Assert
    assert ranked == []


# --- get_reranker -----------------------------------------------------------


def test_get_reranker_returns_the_named_implementations():
    # Act / Assert
    assert isinstance(get_reranker("bge"), CrossEncoderReranker)
    assert isinstance(get_reranker("noop"), NoopReranker)
    assert get_reranker().name == "bge"


def test_get_reranker_caches_instances_so_weights_load_once_per_process():
    # Act / Assert
    assert get_reranker("bge") is get_reranker("bge")


def test_get_reranker_rejects_an_unknown_name():
    # Act / Assert
    with pytest.raises(ValueError, match="unknown reranker"):
        get_reranker("cohere-rerank")


# --- opt-in: the real model -------------------------------------------------


@pytest.mark.skipif(
    os.getenv("RERANK_REAL_MODEL") != "1",
    reason="loads BAAI/bge-reranker-v2-m3 (~2GB, slow); set RERANK_REAL_MODEL=1 to run",
)
async def test_real_cross_encoder_ranks_the_relevant_arabic_chunk_first():
    # Arrange — one real question, its answer text, and two unrelated legal texts
    query = "ما هي مدة الإجازة السنوية للعامل؟"
    hits = [
        _hit("noise1", "يحظر تشغيل الأحداث في الأعمال الخطرة أو الضارة بالصحة."),
        _hit("noise2", "يجب على صاحب العمل توفير وسائل الوقاية من الحريق في مكان العمل."),
        _hit(
            "relevant",
            "يستحق العامل إجازة سنوية مدفوعة الأجر لا تقل عن ثلاثة أسابيع عن كل سنة "
            "إذا كانت مدة خدمته خمس سنوات فأكثر.",
        ),
    ]

    # Act
    ranked = await CrossEncoderReranker().rerank(query, hits, top_k=3)

    # Assert
    assert ranked[0].chunk_id == "relevant"
    assert all(0.0 <= hit.score <= 1.0 for hit in ranked)
    # Guards the double-sigmoid bug: sentence-transformers applies this model's
    # configured Sigmoid by default, and squashing again pins unrelated chunks at
    # ~0.5000 instead of ~0.0 — ordering survives, but every threshold breaks.
    assert ranked[0].score > 0.5
    assert ranked[-1].score < 0.1


async def test_concurrent_cold_starts_build_the_model_once():
    """Two requests arriving on a cold instance must not each build ~2 GB of weights.

    `_load` runs inside `to_thread`, so without a lock both worker threads can
    see `_model is None` and construct in parallel.
    """
    # Arrange — a slow build, so the second caller is guaranteed to arrive mid-build
    reranker = CrossEncoderReranker()
    builds = []

    def slow_build():
        builds.append(1)
        time.sleep(0.05)
        return FakeCrossEncoder({"alpha": 1.0})

    reranker._build_model = slow_build

    # Act
    await asyncio.gather(
        *(reranker.rerank("q", [_hit("c1", "alpha")]) for _ in range(4))
    )

    # Assert
    assert len(builds) == 1
