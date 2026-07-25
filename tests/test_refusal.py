"""Refusal-threshold calibration: sweep arithmetic, the recommendation policy, and the cache.

Everything here runs on synthetic scores. The expensive half of
:mod:`evals.refusal` — :func:`~evals.refusal.score_pairs` — needs a corpus and a
2 GB cross-encoder and is exercised by running the CLI for real; the point of
splitting the module the way it is split is that the decision logic is pure and
testable without either.
"""

from __future__ import annotations

import json
from itertools import pairwise

import pytest

from evals.refusal import (
    MAX_FALSE_REFUSAL_RATE,
    MIN_REFUSAL_PRECISION,
    DialectRefusal,
    RefusalPoint,
    RefusedPair,
    auc,
    load_scores,
    quantiles,
    recommend,
    refusal_by_dialect,
    render_audit,
    render_sweep,
    save_scores,
    sweep,
    threshold_grid,
)
from evals.schema import EvalPair


def pair(pair_id: str, dialect_tag: str = "msa", *, answerable: bool = True) -> EvalPair:
    return EvalPair(
        id=pair_id,
        question="س",
        dialect_tag=dialect_tag,
        answer="ج",
        source_doc="doc",
        source_chunk_ids=["doc:1:0"] if answerable else [],
    )


# --------------------------------------------------------------------------- #
# RefusalPoint arithmetic
# --------------------------------------------------------------------------- #


def test_refusal_point_reports_precision_recall_and_f1_for_the_refusal_decision():
    # Arrange: 3 of 5 unanswerable caught, 1 answerable wrongly refused of 10.
    point = RefusalPoint(
        threshold=0.3, true_refusals=3, false_refusals=1, true_answers=9, false_answers=2
    )

    # Assert
    assert point.n_answerable == 10
    assert point.n_unanswerable == 5
    assert point.precision == pytest.approx(0.75)  # 3 of 4 refusals deserved it
    assert point.recall == pytest.approx(0.6)  # 3 of 5 unanswerable caught
    assert point.f1 == pytest.approx(2 * 0.75 * 0.6 / 1.35)
    assert point.false_refusal_rate == pytest.approx(0.1)


def test_refusal_point_metrics_are_zero_rather_than_dividing_by_zero():
    empty = RefusalPoint(
        threshold=0.0, true_refusals=0, false_refusals=0, true_answers=0, false_answers=0
    )

    assert empty.precision == 0.0
    assert empty.recall == 0.0
    assert empty.f1 == 0.0
    assert empty.false_refusal_rate == 0.0
    assert empty.break_even_prevalence == 0.0


def test_break_even_prevalence_is_the_mix_at_which_refusals_are_right_half_the_time():
    # Arrange: recall 0.5, false-refusal rate 0.1 -> break-even at 0.1/(0.5+0.1).
    point = RefusalPoint(
        threshold=0.2, true_refusals=5, false_refusals=10, true_answers=90, false_answers=5
    )

    assert point.recall == pytest.approx(0.5)
    assert point.false_refusal_rate == pytest.approx(0.1)
    assert point.break_even_prevalence == pytest.approx(0.1 / 0.6)


# --------------------------------------------------------------------------- #
# sweep
# --------------------------------------------------------------------------- #


def test_sweep_counts_each_pair_into_exactly_one_confusion_cell():
    # Arrange: 3 answerable (0.9, 0.5, 0.1) and 2 unanswerable (0.4, 0.05).
    scored = [
        ("a1", True, 0.9),
        ("a2", True, 0.5),
        ("a3", True, 0.1),
        ("u1", False, 0.4),
        ("u2", False, 0.05),
    ]

    # Act
    (point,) = sweep(scored, [0.2])

    # Assert: refuse when score < 0.2 -> a3 and u2 refused.
    assert (point.true_refusals, point.false_refusals) == (1, 1)
    assert (point.true_answers, point.false_answers) == (2, 1)
    assert point.true_refusals + point.false_refusals + point.true_answers + point.false_answers == 5


def test_threshold_zero_refuses_nothing_and_threshold_one_refuses_everything():
    # Arrange: real sigmoid scores are strictly inside (0, 1).
    scored = [("a1", True, 0.999), ("a2", True, 0.0005), ("u1", False, 0.02)]

    # Act
    floor, ceiling = sweep(scored, [0.0, 1.0])

    # Assert
    assert (floor.true_refusals, floor.false_refusals) == (0, 0)
    assert (floor.true_answers, floor.false_answers) == (2, 1)
    assert (ceiling.true_refusals, ceiling.false_refusals) == (1, 2)
    assert (ceiling.true_answers, ceiling.false_answers) == (0, 0)


def test_sweep_uses_strict_less_than_so_a_score_equal_to_the_threshold_is_answered():
    scored = [("a1", True, 0.15), ("u1", False, 0.15)]

    (point,) = sweep(scored, [0.15])

    assert (point.false_refusals, point.true_refusals) == (0, 0)


def test_sweep_returns_one_point_per_threshold_in_the_order_given():
    scored = [("a1", True, 0.5)]

    points = sweep(scored, [0.6, 0.1, 0.4])

    assert [point.threshold for point in points] == [0.6, 0.1, 0.4]


def test_sweep_of_no_pairs_is_an_all_zero_point_rather_than_a_crash():
    (point,) = sweep([], [0.5])

    assert point.n_answerable == 0
    assert point.recall == 0.0


def test_threshold_grid_spans_zero_to_the_maximum_inclusive():
    grid = threshold_grid(0.02, 0.9)

    assert grid[0] == 0.0
    assert grid[-1] == pytest.approx(0.9)
    assert len(grid) == 46
    assert all(b > a for a, b in pairwise(grid))


@pytest.mark.parametrize(("step", "maximum"), [(0.0, 0.9), (-0.1, 0.9), (0.02, -1.0)])
def test_threshold_grid_rejects_a_grid_it_cannot_build(step, maximum):
    with pytest.raises(ValueError, match="must be"):
        threshold_grid(step, maximum)


# --------------------------------------------------------------------------- #
# recommend: the policy, and that it is the *stated* policy
# --------------------------------------------------------------------------- #


def _separable_scores() -> list[tuple[str, bool, float]]:
    """20 answerable at 0.8-0.99, 10 unanswerable at 0.01-0.1: cleanly separable."""
    answerable = [(f"a{i}", True, 0.8 + i * 0.01) for i in range(20)]
    unanswerable = [(f"u{i}", False, 0.01 + i * 0.01) for i in range(10)]
    return answerable + unanswerable


def test_recommend_picks_a_threshold_between_two_separable_populations():
    points = sweep(_separable_scores(), threshold_grid(0.02, 0.9))

    threshold, justification = recommend(points)

    # Every unanswerable score is below 0.1 and every answerable one at or above
    # 0.8, so the policy should catch all 10 at no cost to the 20.
    assert 0.1 <= threshold <= 0.8
    (chosen,) = sweep(_separable_scores(), [threshold])
    assert chosen.recall == 1.0
    assert chosen.false_refusals == 0
    assert "refusal recall 100.0%" in justification


def test_recommend_prefers_recall_over_f1_when_the_two_disagree():
    # Arrange: two eligible thresholds.
    #   0.2 -> catches 9 of 10 unanswerable at zero cost: precision 1.00, F1 0.947
    #   0.4 -> catches 10 of 10 for 2 wrong refusals: precision 0.83, F1 0.909
    # F1 prefers 0.2; the stated asymmetry prefers 0.4, which catches the last
    # uncovered question for two mildly inconvenienced users.
    scored = (
        [(f"a{i}", True, 0.9) for i in range(20)]
        + [("a20", True, 0.3), ("a21", True, 0.3)]
        + [(f"u{i}", False, 0.1) for i in range(9)]
        + [("u9", False, 0.3)]
    )
    low, high = sweep(scored, [0.2, 0.4])
    assert low.f1 > high.f1  # F1 would pick the low threshold
    assert high.recall > low.recall  # recall picks the high one
    assert high.false_refusal_rate <= MAX_FALSE_REFUSAL_RATE
    assert high.precision >= MIN_REFUSAL_PRECISION

    # Act
    threshold, _ = recommend([low, high])

    # Assert: the asymmetry wins -- catch the uncovered questions.
    assert threshold == 0.4


def test_recommend_refuses_to_buy_recall_by_exceeding_the_false_refusal_budget():
    # Arrange: catching the last unanswerable pair costs 30% of the answerable ones.
    scored = (
        [(f"a{i}", True, 0.5) for i in range(7)]
        + [(f"b{i}", True, 0.05) for i in range(3)]
        + [(f"u{i}", False, 0.02) for i in range(9)]
        + [("u9", False, 0.2)]
    )
    points = sweep(scored, threshold_grid(0.01, 0.9))

    # Act
    threshold, justification = recommend(points)

    # Assert: stops below 0.05, keeping recall 0.9 at zero cost.
    (chosen,) = sweep(scored, [threshold])
    assert chosen.false_refusal_rate <= MAX_FALSE_REFUSAL_RATE
    assert chosen.false_refusals == 0
    assert chosen.recall == pytest.approx(0.9)
    assert "budget" in justification


def test_recommend_turns_the_gate_off_and_says_so_when_the_populations_overlap():
    # Arrange: the shape of the real measurement -- answerable questions spread
    # across the whole range, unanswerable ones sitting inside that spread.
    scored = [(f"a{i}", True, i / 100) for i in range(100)] + [
        (f"u{i}", False, 0.1 + i / 100) for i in range(20)
    ]
    points = sweep(scored, threshold_grid(0.02, 0.9))
    assert max(point.precision for point in points) < MIN_REFUSAL_PRECISION

    # Act
    threshold, justification = recommend(points)

    # Assert: 0.0 -- gate off -- and the justification names the finding.
    assert threshold == 0.0
    assert "not separable" in justification
    assert "do not gate on the rerank score" in justification


def test_recommend_rejects_an_empty_sweep():
    with pytest.raises(ValueError, match="sweep something first"):
        recommend([])


# --------------------------------------------------------------------------- #
# dialect breakdown
# --------------------------------------------------------------------------- #


def test_refusal_by_dialect_counts_answerable_refusals_per_tag():
    # Arrange: 2 of 3 Gulf answerable pairs score below 0.15, 1 of 4 MSA ones.
    pairs = [
        pair("g1", "gulf"), pair("g2", "gulf"), pair("g3", "gulf"),
        pair("m1"), pair("m2"), pair("m3"), pair("m4"),
    ]
    scored = [
        ("g1", True, 0.01), ("g2", True, 0.10), ("g3", True, 0.90),
        ("m1", True, 0.05), ("m2", True, 0.80), ("m3", True, 0.95), ("m4", True, 0.99),
    ]

    # Act
    gulf, msa = refusal_by_dialect(scored, pairs, 0.15)

    # Assert
    assert (gulf.dialect_tag, gulf.n, gulf.refused) == ("gulf", 3, 2)
    assert gulf.rate == pytest.approx(2 / 3)
    assert (msa.dialect_tag, msa.n, msa.refused) == ("msa", 4, 1)
    assert msa.rate == pytest.approx(0.25)


def test_refusal_by_dialect_ignores_unanswerable_pairs():
    # A refused unanswerable question is the gate working; counting it would hide
    # the disparity this measures.
    pairs = [pair("g1", "gulf"), pair("u1", "gulf", answerable=False)]
    scored = [("g1", True, 0.9), ("u1", False, 0.01)]

    (gulf,) = refusal_by_dialect(scored, pairs, 0.15)

    assert (gulf.n, gulf.refused, gulf.rate) == (1, 0, 0.0)


def test_refusal_by_dialect_skips_scores_with_no_matching_pair():
    pairs = [pair("g1", "gulf")]
    scored = [("g1", True, 0.01), ("ghost", True, 0.01)]

    (gulf,) = refusal_by_dialect(scored, pairs, 0.15)

    assert gulf.n == 1


def test_dialect_refusal_rate_of_an_empty_split_is_zero():
    assert DialectRefusal("gulf", n=0, refused=0).rate == 0.0


# --------------------------------------------------------------------------- #
# cached scores
# --------------------------------------------------------------------------- #


def test_scores_survive_a_save_load_round_trip(tmp_path):
    # Arrange
    path = tmp_path / "scores.json"
    scored = [("a1", True, 0.123456), ("u1", False, 0.9), ("a2", True, 0.0)]
    meta = {"config": "hybrid+rerank", "model_key": "bge"}

    # Act
    save_scores(path, scored, meta)
    loaded, loaded_meta = load_scores(path)

    # Assert
    assert loaded == scored
    assert loaded_meta["config"] == "hybrid+rerank"
    assert loaded_meta["model_key"] == "bge"
    assert "_note" in loaded_meta  # the file explains how to regenerate itself


def test_saved_scores_are_readable_utf8_json_with_the_pair_ids_intact(tmp_path):
    path = tmp_path / "scores.json"
    save_scores(path, [("q-٠٠١", True, 0.5)], {})

    document = json.loads(path.read_text(encoding="utf-8"))

    assert document["scores"] == [
        {"pair_id": "q-٠٠١", "answerable": True, "top_score": 0.5}
    ]


def test_load_scores_names_the_fix_when_the_cache_is_missing(tmp_path):
    with pytest.raises(ValueError, match="evals.refusal"):
        load_scores(tmp_path / "nope.json")


@pytest.mark.parametrize(
    ("body", "match"),
    [
        ("{not json", "not valid JSON"),
        ('{"scores": {}}', "'scores' list"),
        ('{"scores": [1]}', "must be an object"),
        ('{"scores": [{"pair_id": "a"}]}', "missing answerable, top_score"),
        ('{"scores": [{"pair_id": "a", "answerable": true, "top_score": "x"}]}', "not a number"),
    ],
)
def test_load_scores_rejects_a_corrupt_cache_naming_the_record(tmp_path, body, match):
    path = tmp_path / "scores.json"
    path.write_text(body, encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        load_scores(path)


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #


def test_render_sweep_emits_one_markdown_row_per_threshold():
    points = sweep([("a1", True, 0.5), ("u1", False, 0.1)], [0.0, 0.2, 0.6])

    table = render_sweep(points)

    assert table.count("\n") == 4  # header, separator, 3 rows
    assert "| 0.20 | 1/1 | 0/1 |" in table


def test_render_sweep_can_thin_the_table():
    points = sweep([("a1", True, 0.5)], threshold_grid(0.02, 0.9))

    assert render_sweep(points, every=5).count("\n") == 2 + 10 - 1


def test_auc_is_one_when_every_unanswerable_scores_below_every_answerable():
    assert auc([0.8, 0.9, 1.0], [0.1, 0.2]) == 1.0


def test_auc_is_zero_point_five_for_identical_populations():
    # Every comparison is a tie, and ties count half.
    assert auc([0.5, 0.5], [0.5, 0.5]) == 0.5


def test_auc_counts_the_fraction_of_correctly_ordered_pairs():
    # 2 answerable x 2 unanswerable = 4 comparisons; 0.15 beats 0.1 but not 0.2.
    assert auc([0.15, 0.9], [0.1, 0.2]) == 0.75


def test_auc_of_an_empty_population_is_zero_rather_than_a_crash():
    assert auc([], [0.1]) == 0.0
    assert auc([0.1], []) == 0.0


def test_render_audit_reports_how_many_refusals_threw_away_a_retrieved_answer():
    audited = [
        RefusedPair("g1", "gulf", 0.01, gold_in_context=True, gold_at_rank_1=True),
        RefusedPair("g2", "gulf", 0.05, gold_in_context=True, gold_at_rank_1=False),
        RefusedPair("m1", "msa", 0.09, gold_in_context=False, gold_at_rank_1=False),
    ]

    report = render_audit(audited, 0.15)

    assert "3 answerable pairs refused" in report
    assert "top-k context: 2/3" in report
    assert "rank 1:            1/3" in report
    assert "gulf     2 refused, 2 with the gold chunk in context" in report


def test_render_audit_of_no_refusals_still_renders():
    assert "0 answerable pairs refused" in render_audit([], 0.0)


def test_quantiles_of_an_empty_population_are_zeros():
    assert quantiles([])["n"] == 0


def test_quantiles_bracket_the_data():
    stats = quantiles([0.1, 0.2, 0.3, 0.4, 0.5])

    assert (stats["min"], stats["median"], stats["max"]) == (0.1, 0.3, 0.5)
    assert stats["p25"] <= stats["median"] <= stats["p75"]
