"""Unit tests for the deterministic eval layer (schema validation + retrieval metrics)."""

from __future__ import annotations

import json

import pytest

from evals.metrics import (
    QueryResult,
    aggregate,
    compare_to_baseline,
    hit_rate_at_k,
    mrr,
    recall_at_k,
)
from evals.schema import EvalPair, load_pairs, save_pairs, validate_pair


def _raw(**overrides) -> dict:
    base = {
        "id": "q001",
        "question": "ما هي مدة الإجازة السنوية؟",
        "dialect_tag": "msa",
        "answer": "ثلاثة أسابيع.",
        "source_doc": "labour-law-2004",
        "source_chunk_ids": ["labour-law-2004:79:0"],
    }
    return {**base, **overrides}


# --- recall@k ---------------------------------------------------------------


def test_recall_is_one_when_all_relevant_chunks_retrieved():
    # Arrange
    retrieved = ["c1", "c2", "c3"]
    relevant = ["c1", "c3"]

    # Act
    score = recall_at_k(retrieved, relevant, k=3)

    # Assert
    assert score == 1.0


def test_recall_is_zero_when_nothing_relevant_retrieved():
    # Arrange
    retrieved = ["x1", "x2"]
    relevant = ["c1"]

    # Act
    score = recall_at_k(retrieved, relevant, k=10)

    # Assert
    assert score == 0.0


def test_recall_is_partial_when_only_some_relevant_chunks_retrieved():
    # Arrange
    retrieved = ["c1", "x1", "x2", "x3"]
    relevant = ["c1", "c2", "c3", "c4"]

    # Act
    score = recall_at_k(retrieved, relevant, k=4)

    # Assert
    assert score == 0.25


def test_recall_ignores_relevant_chunks_ranked_below_k():
    # Arrange
    retrieved = ["x1", "x2", "x3", "c1"]
    relevant = ["c1"]

    # Act & Assert
    assert recall_at_k(retrieved, relevant, k=3) == 0.0
    assert recall_at_k(retrieved, relevant, k=4) == 1.0


def test_recall_handles_k_larger_than_retrieved_list():
    # Arrange
    retrieved = ["c1"]
    relevant = ["c1", "c2"]

    # Act
    score = recall_at_k(retrieved, relevant, k=100)

    # Assert
    assert score == 0.5


def test_recall_counts_duplicate_retrieved_ids_only_once_but_keeps_their_rank():
    # Arrange
    retrieved = ["c1", "c1", "c2"]
    relevant = ["c1", "c2"]

    # Act & Assert
    assert recall_at_k(retrieved, relevant, k=2) == 0.5  # top-2 is ["c1", "c1"]
    assert recall_at_k(retrieved, relevant, k=3) == 1.0


def test_recall_raises_for_unanswerable_pair():
    # Arrange
    retrieved = ["c1"]

    # Act & Assert
    with pytest.raises(ValueError, match="unanswerable"):
        recall_at_k(retrieved, [], k=3)


def test_recall_raises_for_non_positive_k():
    # Act & Assert
    with pytest.raises(ValueError, match="k must be >= 1"):
        recall_at_k(["c1"], ["c1"], k=0)


# --- mrr / hit rate ---------------------------------------------------------


def test_mrr_is_half_when_first_relevant_hit_is_rank_two():
    # Arrange
    retrieved = ["x1", "c1", "c2"]
    relevant = ["c1", "c2"]

    # Act
    score = mrr(retrieved, relevant)

    # Assert
    assert score == 0.5


def test_mrr_is_one_for_a_top_ranked_hit_and_zero_when_nothing_hits():
    # Act & Assert
    assert mrr(["c1", "x1"], ["c1"]) == 1.0
    assert mrr(["x1", "x2"], ["c1"]) == 0.0


def test_mrr_returns_zero_for_unanswerable_pair():
    # Act & Assert
    assert mrr(["c1"], []) == 0.0


def test_hit_rate_is_one_when_any_relevant_chunk_is_in_top_k():
    # Arrange
    retrieved = ["x1", "x2", "c1"]
    relevant = ["c1", "c2"]

    # Act & Assert
    assert hit_rate_at_k(retrieved, relevant, k=3) == 1.0
    assert hit_rate_at_k(retrieved, relevant, k=2) == 0.0


def test_hit_rate_returns_zero_for_unanswerable_pair():
    # Act & Assert
    assert hit_rate_at_k(["c1"], [], k=3) == 0.0


# --- aggregate --------------------------------------------------------------


def test_aggregate_splits_metrics_by_dialect_tag():
    # Arrange
    results = [
        QueryResult("q1", "msa", ["c1", "c2", "c3"], ["c1"]),
        QueryResult("q2", "msa", ["c1", "c2", "c3"], ["c2"]),
        QueryResult("q3", "gulf", ["x1", "x2", "x3"], ["c9"]),
    ]

    # Act
    summary = aggregate(results, ks=(3,))

    # Assert
    assert summary["n"] == 3
    assert summary["recall@3"] == pytest.approx(2 / 3, abs=1e-4)
    assert summary["hit@3"] == pytest.approx(2 / 3, abs=1e-4)
    assert summary["mrr"] == pytest.approx(0.5, abs=1e-4)  # (1.0 + 0.5 + 0.0) / 3
    assert summary["by_dialect"]["msa"] == {"n": 2, "recall@3": 1.0, "hit@3": 1.0, "mrr": 0.75}
    assert summary["by_dialect"]["gulf"] == {"n": 1, "recall@3": 0.0, "hit@3": 0.0, "mrr": 0.0}


def test_aggregate_reports_default_ks_three_and_ten():
    # Arrange
    results = [QueryResult("q1", "msa", ["x1", "x2", "x3", "c1"], ["c1"])]

    # Act
    summary = aggregate(results)

    # Assert
    assert summary["recall@3"] == 0.0
    assert summary["recall@10"] == 1.0
    assert summary["hit@3"] == 0.0
    assert summary["hit@10"] == 1.0
    assert summary["mrr"] == 0.25


def test_aggregate_omits_dialect_tags_with_no_results():
    # Arrange
    results = [QueryResult("q1", "msa", ["c1"], ["c1"])]

    # Act
    summary = aggregate(results, ks=(3,))

    # Assert
    assert set(summary["by_dialect"]) == {"msa"}
    assert "gulf" not in summary["by_dialect"]


def test_aggregate_skips_unanswerable_results():
    # Arrange
    results = [
        QueryResult("q1", "msa", ["c1"], ["c1"]),
        QueryResult("q2", "gulf", ["x1"], []),  # unanswerable -> refusal scoring
    ]

    # Act
    summary = aggregate(results, ks=(3,))

    # Assert
    assert summary["n"] == 1
    assert summary["recall@3"] == 1.0
    assert set(summary["by_dialect"]) == {"msa"}


def test_aggregate_of_no_results_returns_zeros_not_nan():
    # Act
    summary = aggregate([], ks=(3,))

    # Assert
    assert summary == {"n": 0, "recall@3": 0.0, "hit@3": 0.0, "mrr": 0.0, "by_dialect": {}}


# --- baseline comparison ----------------------------------------------------


def test_compare_to_baseline_passes_at_exactly_the_tolerance_edge():
    # Arrange
    current = {"recall@10": 0.80}
    baseline = {"recall@10": 0.82}

    # Act
    passed, message = compare_to_baseline(current, baseline, tolerance=2.0)

    # Assert
    assert passed is True
    assert "PASS" in message


def test_compare_to_baseline_fails_just_past_the_tolerance_edge():
    # Arrange
    current = {"recall@10": 0.7979}
    baseline = {"recall@10": 0.82}

    # Act
    passed, message = compare_to_baseline(current, baseline, tolerance=2.0)

    # Assert
    assert passed is False
    assert "FAIL" in message
    assert "recall@10" in message


def test_compare_to_baseline_passes_on_improvement():
    # Act
    passed, message = compare_to_baseline({"mrr": 0.9}, {"mrr": 0.6}, metric="mrr")

    # Assert
    assert passed is True
    assert "+30.00 pts" in message


def test_compare_to_baseline_raises_when_metric_missing():
    # Act & Assert
    with pytest.raises(ValueError, match="baseline summary has no metric 'recall@10'"):
        compare_to_baseline({"recall@10": 0.8}, {"recall@3": 0.8})


# --- schema validation ------------------------------------------------------


def test_validate_pair_accepts_a_well_formed_pair():
    # Act
    pair = validate_pair(_raw())

    # Assert
    assert pair == EvalPair(
        id="q001",
        question="ما هي مدة الإجازة السنوية؟",
        dialect_tag="msa",
        answer="ثلاثة أسابيع.",
        source_doc="labour-law-2004",
        source_chunk_ids=["labour-law-2004:79:0"],
    )
    assert pair.answerable is True


def test_validate_pair_treats_empty_chunk_ids_as_unanswerable():
    # Act
    pair = validate_pair(_raw(source_chunk_ids=[]))

    # Assert
    assert pair.answerable is False


@pytest.mark.parametrize("field", ["question", "dialect_tag", "answer", "source_doc", "source_chunk_ids"])
def test_validate_pair_rejects_missing_field(field):
    # Arrange
    raw = _raw()
    del raw[field]

    # Act & Assert
    with pytest.raises(ValueError, match=f"missing field.*{field}"):
        validate_pair(raw)


def test_validate_pair_rejects_missing_or_blank_id():
    # Act & Assert
    with pytest.raises(ValueError, match="'id' must be a non-empty string"):
        validate_pair(_raw(id="  "))
    with pytest.raises(ValueError, match="'id' must be a non-empty string"):
        validate_pair({"question": "س", "dialect_tag": "msa"})


def test_validate_pair_rejects_unknown_dialect_tag():
    # Act & Assert
    with pytest.raises(ValueError, match="q001.*'dialect_tag'"):
        validate_pair(_raw(dialect_tag="egyptian"))


@pytest.mark.parametrize("field", ["question", "answer", "source_doc"])
def test_validate_pair_rejects_blank_text_field(field):
    # Act & Assert
    with pytest.raises(ValueError, match=f"q001.*{field}.*non-blank"):
        validate_pair(_raw(**{field: "   "}))


def test_validate_pair_rejects_non_string_question():
    # Act & Assert
    with pytest.raises(ValueError, match="non-blank string"):
        validate_pair(_raw(question=42))


def test_validate_pair_rejects_non_list_source_chunk_ids():
    # Act & Assert
    with pytest.raises(ValueError, match="'source_chunk_ids' must be a list"):
        validate_pair(_raw(source_chunk_ids="labour-law-2004:79:0"))


def test_validate_pair_rejects_blank_entry_in_source_chunk_ids():
    # Act & Assert
    with pytest.raises(ValueError, match="must contain non-blank strings"):
        validate_pair(_raw(source_chunk_ids=["c1", ""]))


def test_validate_pair_rejects_chunk_id_belonging_to_another_document():
    # Arrange - a typo'd doc prefix would silently score 0 recall forever
    raw = _raw(source_chunk_ids=["labour-law-2004:79:0", "domestic-workers-2017:12:0"])

    # Act & Assert
    with pytest.raises(ValueError, match="does not belong to source_doc"):
        validate_pair(raw)


def test_validate_pair_rejects_chunk_id_that_is_only_the_doc_id():
    # Arrange - ids are "doc:article:seq"; a bare doc id names no chunk
    raw = _raw(source_chunk_ids=["labour-law-2004"])

    # Act & Assert
    with pytest.raises(ValueError, match="does not belong to source_doc"):
        validate_pair(raw)


def test_validate_pair_rejects_non_object_input():
    # Act & Assert
    with pytest.raises(ValueError, match="must be a JSON object"):
        validate_pair(["not", "an", "object"])


# --- load / save ------------------------------------------------------------


def test_load_pairs_reads_jsonl_and_skips_blank_lines(tmp_path):
    # Arrange
    path = tmp_path / "pairs.jsonl"
    path.write_text(
        json.dumps(_raw(), ensure_ascii=False)
        + "\n\n"
        + json.dumps(_raw(id="q002", dialect_tag="gulf"), ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )

    # Act
    pairs = load_pairs(path)

    # Assert
    assert [p.id for p in pairs] == ["q001", "q002"]
    assert pairs[1].dialect_tag == "gulf"


def test_load_pairs_rejects_duplicate_ids_across_the_file(tmp_path):
    # Arrange
    path = tmp_path / "pairs.jsonl"
    path.write_text(
        json.dumps(_raw(), ensure_ascii=False) + "\n" + json.dumps(_raw(), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # Act & Assert
    with pytest.raises(ValueError, match="duplicate pair id 'q001'"):
        load_pairs(path)


def test_load_pairs_reports_line_number_on_invalid_json(tmp_path):
    # Arrange
    path = tmp_path / "pairs.jsonl"
    path.write_text(json.dumps(_raw(), ensure_ascii=False) + "\n{not json}\n", encoding="utf-8")

    # Act & Assert
    with pytest.raises(ValueError, match=r":2: invalid JSON"):
        load_pairs(path)


def test_load_pairs_reports_line_number_on_invalid_pair(tmp_path):
    # Arrange
    path = tmp_path / "pairs.jsonl"
    path.write_text(json.dumps(_raw(dialect_tag="urdu"), ensure_ascii=False) + "\n", encoding="utf-8")

    # Act & Assert
    with pytest.raises(ValueError, match=r":1: pair 'q001'"):
        load_pairs(path)


def test_save_pairs_round_trips_arabic_text_unescaped(tmp_path):
    # Arrange
    path = tmp_path / "pairs.jsonl"
    pairs = [validate_pair(_raw()), validate_pair(_raw(id="q002", source_chunk_ids=[]))]

    # Act
    save_pairs(pairs, path)
    reloaded = load_pairs(path)

    # Assert
    assert reloaded == pairs
    assert "ما هي مدة الإجازة السنوية؟" in path.read_text(encoding="utf-8")


def test_save_pairs_rejects_duplicate_ids(tmp_path):
    # Arrange
    pairs = [validate_pair(_raw()), validate_pair(_raw())]

    # Act & Assert
    with pytest.raises(ValueError, match="duplicate pair id 'q001'"):
        save_pairs(pairs, tmp_path / "pairs.jsonl")
