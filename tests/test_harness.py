import pytest

from app.models.chunks import EMBEDDING_DIMS, Chunk
from evals.harness import (
    CONFIGS,
    RetrievalConfig,
    build_configs,
    evaluate,
    report,
    run_config,
)
from evals.metrics import QueryResult
from evals.schema import EvalPair
from ingestion.normalize import normalize_for_index

DIM = EMBEDDING_DIMS["e5"]

WAGES_TEXT = "يَجِبُ عَلَى صَاحِبِ الْعَمَلِ أَنْ يَدْفَعَ الْأَجْرَ شَهْرِيًّا"
LEAVE_TEXT = "للعامل الحق في إجازة سنوية مدفوعة الأجر"
SAFETY_TEXT = "على المنشأة توفير وسائل الوقاية من مخاطر العمل"

# question -> (the chunk it is about, the axis its fake embedding points along)
GOLD = {
    "متى يدفع الأجر للعامل؟": ("law:1:0", 0),
    "شكثر مدة إجازة سنوية للعامل؟": ("law:2:0", 1),
    "ما هي وسائل الوقاية المطلوبة؟": ("reg:1:0", 2),
    "كم عدد سكان زيمبابوي؟": (None, 3),
}

PAIRS = (
    EvalPair("p1", "متى يدفع الأجر للعامل؟", "msa", "شهريا", "law", ["law:1:0"]),
    EvalPair("p2", "شكثر مدة إجازة سنوية للعامل؟", "gulf", "سنوية", "law", ["law:2:0"]),
    EvalPair("p3", "ما هي وسائل الوقاية المطلوبة؟", "msa", "وقاية", "reg", ["reg:1:0"]),
    EvalPair("p4", "كم عدد سكان زيمبابوي؟", "msa", "غير متوفر", "law", []),
)


def unit_vector(index: int) -> list[float]:
    vector = [0.0] * DIM
    vector[index] = 1.0
    return vector


def make_chunk(chunk_id: str, text: str, axis: int) -> Chunk:
    doc_id, article, seq = chunk_id.split(":")
    return Chunk(
        id=chunk_id,
        doc_id=doc_id,
        article=article,
        seq=int(seq),
        text=text,
        text_normalized=normalize_for_index(text),
        emb_e5=unit_vector(axis),
    )


class FakeEmbedder:
    """Records its calls, so the tests can prove batching and non-use."""

    def __init__(self, model_key: str = "e5") -> None:
        self.model_key = model_key
        self.dim = DIM
        self.calls: list[list[str]] = []

    async def embed_queries(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [unit_vector(GOLD[text][1]) for text in texts]

    async def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return await self.embed_queries(texts)


class FakeReranker:
    """Deterministic and weight-free: reverse chunk-id order."""

    name = "fake"

    async def rerank(self, query, hits, top_k: int = 5):
        import dataclasses

        ranked = sorted(hits, key=lambda hit: hit.chunk_id, reverse=True)
        return [
            dataclasses.replace(hit, score=1.0 / rank, source="rerank")
            for rank, hit in enumerate(ranked[:top_k], start=1)
        ]


@pytest.fixture()
async def corpus(db_session):
    db_session.add_all(
        [
            make_chunk("law:1:0", WAGES_TEXT, 0),
            make_chunk("law:2:0", LEAVE_TEXT, 1),
            make_chunk("reg:1:0", SAFETY_TEXT, 2),
        ]
    )
    await db_session.commit()
    return db_session


# ------------------------------------------------------------------- configs


def test_configs_are_the_four_standard_ablations():
    assert set(CONFIGS) == {"dense", "lexical", "hybrid", "hybrid+rerank"}
    assert CONFIGS["lexical"].dense is False
    assert CONFIGS["hybrid+rerank"].rerank is True
    assert build_configs("bge")["dense"].model_key == "bge"


def test_a_config_that_retrieves_nothing_is_rejected():
    with pytest.raises(ValueError, match="retrieves nothing"):
        RetrievalConfig("empty", "e5", dense=False, lexical=False, rerank=False)


def test_a_config_with_a_non_positive_top_k_is_rejected():
    with pytest.raises(ValueError, match="top_k must be"):
        RetrievalConfig("bad", "e5", dense=True, lexical=False, rerank=False, top_k=0)
    with pytest.raises(ValueError, match="rerank_top_k must be"):
        RetrievalConfig("bad", "e5", dense=True, lexical=False, rerank=True, rerank_top_k=0)


# ---------------------------------------------------------------- end to end


@pytest.mark.parametrize("config_name", ["dense", "lexical", "hybrid", "hybrid+rerank"])
async def test_every_config_runs_the_whole_dataset_end_to_end(corpus, config_name):
    # Arrange
    config = CONFIGS[config_name]

    # Act
    results = await evaluate(PAIRS, corpus, config, FakeEmbedder(), FakeReranker())

    # Assert — three answerable pairs scored, every gold chunk reachable
    assert results["config"] == config_name
    assert results["n_pairs"] == 4
    assert results["n"] == 3
    assert results["recall@10"] == 1.0
    assert set(results["by_dialect"]) == {"gulf", "msa"}


async def test_run_config_returns_one_query_result_per_pair_in_order(corpus):
    # Act
    results = await run_config(PAIRS, corpus, CONFIGS["hybrid"], FakeEmbedder())

    # Assert
    assert [r.pair_id for r in results] == ["p1", "p2", "p3", "p4"]
    assert all(isinstance(r, QueryResult) for r in results)
    assert results[0].relevant == ["law:1:0"]
    assert "law:1:0" in results[0].retrieved


async def test_dense_config_ranks_the_matching_vector_first(corpus):
    # Act
    results = await run_config(PAIRS, corpus, CONFIGS["dense"], FakeEmbedder())

    # Assert — the fake embedding points at the gold chunk's axis
    assert results[0].retrieved[0] == "law:1:0"
    assert results[2].retrieved[0] == "reg:1:0"


# --------------------------------------------------------- unanswerable pairs


async def test_unanswerable_pairs_are_excluded_from_recall_but_counted(corpus):
    # Act
    results = await evaluate(PAIRS, corpus, CONFIGS["hybrid"], FakeEmbedder())

    # Assert — p4 is retrieved for (so it has a top score) but never scored
    assert results["n"] == 3  # aggregate() saw only the answerable pairs
    assert results["unanswerable"]["n"] == 1
    assert results["unanswerable"]["with_hits"] == 1
    assert results["unanswerable"]["top_score_max"] > 0


async def test_unanswerable_pairs_still_appear_in_run_config_output(corpus):
    # Act
    results = await run_config(PAIRS, corpus, CONFIGS["dense"], FakeEmbedder())

    # Assert
    assert results[3].pair_id == "p4"
    assert results[3].relevant == []


# ---------------------------------------------------------------- by_doc


async def test_by_doc_splits_recall_by_source_doc(corpus):
    # Act
    results = await evaluate(PAIRS, corpus, CONFIGS["dense"], FakeEmbedder())

    # Assert — 2 answerable law pairs, 1 reg pair, the unanswerable one excluded
    assert set(results["by_doc"]) == {"law", "reg"}
    assert results["by_doc"]["law"]["n"] == 2
    assert results["by_doc"]["reg"]["n"] == 1
    assert results["by_doc"]["reg"]["recall@10"] == 1.0


async def test_by_doc_exposes_a_regression_confined_to_a_small_document(corpus):
    """The whole reason by_doc exists: aggregate recall stays high, one doc is dead."""
    # Arrange — p5 asks a labour-law question but is labelled against the reg doc,
    # so with top_k=1 the reg chunk is never returned while both law pairs still hit.
    broken = EvalPair("p5", "متى يدفع الأجر للعامل؟", "msa", "x", "reg", ["reg:1:0"])
    pairs = (*PAIRS[:2], broken)
    config = RetrievalConfig("dense-top1", "e5", dense=True, lexical=False, rerank=False, top_k=1)

    # Act
    results = await evaluate(pairs, corpus, config, FakeEmbedder())

    # Assert
    assert results["by_doc"]["law"]["recall@10"] == 1.0
    assert results["by_doc"]["reg"]["recall@10"] == 0.0
    assert results["recall@3"] < 1.0  # the aggregate only dips, the split is blunt


# --------------------------------------------------------------- embedder use


async def test_a_lexical_only_config_never_calls_the_embedder(corpus):
    # Arrange
    embedder = FakeEmbedder()

    # Act
    results = await evaluate(PAIRS, corpus, CONFIGS["lexical"], embedder)

    # Assert
    assert embedder.calls == []
    assert results["model_key"] is None


async def test_a_lexical_only_config_accepts_no_embedder_at_all(corpus):
    results = await run_config(PAIRS, corpus, CONFIGS["lexical"], None)
    assert len(results) == 4


async def test_every_question_is_embedded_in_a_single_batched_call(corpus):
    # Arrange
    embedder = FakeEmbedder()

    # Act
    await run_config(PAIRS, corpus, CONFIGS["hybrid"], embedder)

    # Assert
    assert len(embedder.calls) == 1
    assert embedder.calls[0] == [pair.question for pair in PAIRS]


async def test_a_dense_config_without_an_embedder_is_rejected(corpus):
    with pytest.raises(ValueError, match="embedder is required"):
        await run_config(PAIRS, corpus, CONFIGS["dense"], None)


async def test_an_embedder_for_the_wrong_model_is_rejected(corpus):
    with pytest.raises(ValueError, match="bge"):
        await run_config(PAIRS, corpus, CONFIGS["dense"], FakeEmbedder("bge"))


# ------------------------------------------------------------------- rerank


async def test_the_rerank_config_reorders_and_truncates(corpus):
    # Arrange
    config = RetrievalConfig(
        "hybrid+rerank", "e5", dense=True, lexical=True, rerank=True, rerank_top_k=2
    )

    # Act
    results = await run_config(PAIRS, corpus, config, FakeEmbedder(), FakeReranker())

    # Assert — reverse chunk-id order, capped at rerank_top_k
    assert results[0].retrieved == ["reg:1:0", "law:2:0"]


async def test_a_rerank_config_without_a_reranker_is_rejected(corpus):
    with pytest.raises(ValueError, match="reranker is required"):
        await run_config(PAIRS, corpus, CONFIGS["hybrid+rerank"], FakeEmbedder())


# -------------------------------------------------------------- determinism


async def test_evaluate_is_deterministic_apart_from_latency(corpus):
    # Act
    first = await evaluate(PAIRS, corpus, CONFIGS["hybrid"], FakeEmbedder(), FakeReranker())
    second = await evaluate(PAIRS, corpus, CONFIGS["hybrid"], FakeEmbedder(), FakeReranker())

    # Assert
    assert first.pop("latency").keys() == second.pop("latency").keys()
    assert first == second


async def test_run_config_returns_the_same_ranking_twice(corpus):
    first = await run_config(PAIRS, corpus, CONFIGS["hybrid"], FakeEmbedder())
    second = await run_config(PAIRS, corpus, CONFIGS["hybrid"], FakeEmbedder())
    assert first == second


# ------------------------------------------------------------------ latency


async def test_latency_is_reported_per_query_and_for_the_embed_batch(corpus):
    # Act
    results = await evaluate(PAIRS, corpus, CONFIGS["dense"], FakeEmbedder())

    # Assert
    latency = results["latency"]
    assert latency["mean_ms"] > 0
    assert latency["p95_ms"] >= latency["mean_ms"]
    assert latency["embed_batch_ms"] >= 0


async def test_a_lexical_config_reports_no_embed_time(corpus):
    results = await evaluate(PAIRS, corpus, CONFIGS["lexical"], None)
    assert results["latency"]["embed_batch_ms"] == 0.0


# ------------------------------------------------------------------- report


async def test_report_renders_every_section(corpus):
    # Act
    text = report(await evaluate(PAIRS, corpus, CONFIGS["hybrid"], FakeEmbedder()))

    # Assert
    assert "config: hybrid" in text
    assert "model: e5" in text
    assert "recall@10" in text
    assert "overall" in text and "gulf" in text and "msa" in text
    assert "by source_doc" in text and "law" in text and "reg" in text
    assert "unanswerable: 1 pairs" in text
    assert "latency: mean" in text


def test_report_survives_an_empty_result_dict():
    """A harness that crashes while printing a bad run is a harness nobody trusts."""
    assert "config: ?" in report({})
