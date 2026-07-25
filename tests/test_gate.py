"""The CI gate itself. Every test here is pure: no database, no model, no network.

The gate's job is to *fail*, so most of these assert on the failing side. A gate
that cannot fail manufactures false confidence, which is worse than no gate.
"""

import json

import pytest

from evals.gate import (
    BY_DOC_METRIC,
    BY_DOC_TOLERANCE_PTS,
    GATE_METRIC,
    TOLERANCE_PTS,
    baseline_entry,
    check,
    entry_key,
    load_baseline,
    render,
    summarise,
    write_baseline,
)
from evals.harness import build_configs

LABOUR = "qatar-labour-law-14-2004"
WAGE = "qatar-minimum-wage-law-17-2020"

# Shaped like a real baseline entry: one big document carrying the aggregate, one
# small one that the aggregate can hide.
BASELINE = {
    "config": "dense",
    "model": "e5",
    "n_pairs": 283,
    "n": 268,
    GATE_METRIC: 0.9272,
    "recall@3": 0.8545,
    "hit@3": 0.8993,
    "hit@10": 0.9478,
    "mrr": 0.8184,
    "by_doc": {
        LABOUR: {"n": 224, BY_DOC_METRIC: 0.9375},
        WAGE: {"n": 6, BY_DOC_METRIC: 0.8333},
    },
    "recorded_date": "2026-07-25",
    "recorded_commit": "6b70b6d",
}


def make_current(recall: float = 0.9272, by_doc: dict | None = None) -> dict:
    """A fresh run, identical to the baseline except where a test moves it."""
    docs = by_doc or {LABOUR: 0.9375, WAGE: 0.8333}
    return {
        **{key: value for key, value in BASELINE.items() if key != "by_doc"},
        GATE_METRIC: recall,
        "by_doc": {
            doc: {"n": BASELINE["by_doc"].get(doc, {}).get("n", 1), BY_DOC_METRIC: value}
            for doc, value in docs.items()
        },
    }


def test_identical_result_passes():
    passed, lines = check(make_current(), BASELINE)

    assert passed
    assert any("PASS" in line for line in lines)


def test_drop_within_tolerance_passes():
    # Arrange: 1.7 points down, inside the 2 point band.
    current = make_current(recall=BASELINE[GATE_METRIC] - 0.017)

    passed, lines = check(current, BASELINE)

    assert passed
    assert "-1.70 pts" in lines[0]


def test_drop_of_exactly_the_tolerance_passes():
    current = make_current(recall=BASELINE[GATE_METRIC] - TOLERANCE_PTS / 100)

    passed, _ = check(current, BASELINE)

    assert passed


def test_drop_beyond_tolerance_fails():
    # Arrange: 3 points down on the gated metric, every document still fine.
    current = make_current(recall=BASELINE[GATE_METRIC] - 0.03)

    passed, lines = check(current, BASELINE)

    assert not passed
    assert "FAIL" in lines[0]
    assert "-3.00 pts" in lines[0]


def test_improvement_passes():
    current = make_current(recall=0.97)

    passed, lines = check(current, BASELINE)

    assert passed
    assert "+4.28 pts" in lines[0]


def test_per_document_regression_fails_while_the_aggregate_is_fine():
    """The reason the second check exists: 6 pairs out of 268 cannot move the mean."""
    current = make_current(
        recall=BASELINE[GATE_METRIC] - 0.005,  # 0.5 pts: well inside the aggregate band
        by_doc={LABOUR: 0.9375, WAGE: 0.1667},  # the small document collapses
    )

    passed, lines = check(current, BASELINE)

    assert not passed
    assert "PASS" in lines[0], "the aggregate check must still pass — that is the point"
    assert any(WAGE in line and "FAIL" in line for line in lines)


def test_per_document_drop_within_the_wider_tolerance_passes():
    # 4 points down on one document: noisy, not a regression, at n=6.
    current = make_current(by_doc={LABOUR: 0.9375, WAGE: 0.8333 - 0.04})

    passed, _ = check(current, BASELINE)

    assert passed


def test_per_document_tolerance_is_wider_than_the_aggregate_one():
    assert BY_DOC_TOLERANCE_PTS > TOLERANCE_PTS


def test_document_missing_from_the_run_fails():
    """A document vanishing from the results is a corpus/ingestion break, not a pass."""
    current = make_current(by_doc={LABOUR: 0.9375})

    passed, lines = check(current, BASELINE)

    assert not passed
    assert any(WAGE in line and "not in this run" in line for line in lines)


def test_new_document_is_reported_but_not_gated():
    current = make_current(by_doc={LABOUR: 0.9375, WAGE: 0.8333, "new-decision-1-2026": 0.0})

    passed, lines = check(current, BASELINE)

    assert passed
    assert any("new-decision-1-2026" in line and "not in the baseline" in line for line in lines)


def test_changed_dataset_size_warns_without_failing():
    current = {**make_current(), "n": 300, "n_pairs": 315}

    passed, lines = check(current, BASELINE)

    assert passed
    assert any(line.strip().startswith("WARNING") for line in lines)


def test_empty_corpus_fails_loudly():
    """The smoke test for the gate itself: nothing retrieved must never be green."""
    current = make_current(recall=0.0, by_doc={LABOUR: 0.0, WAGE: 0.0})

    passed, _ = check(current, BASELINE)

    assert not passed


# --------------------------------------------------------------------------- #
# baseline file handling
# --------------------------------------------------------------------------- #


def test_missing_baseline_file_gives_an_actionable_error(tmp_path):
    missing = tmp_path / "baseline.json"

    with pytest.raises(ValueError, match="no baseline file"):
        load_baseline(missing)

    # The message has to say what to do, not just what went wrong.
    with pytest.raises(ValueError, match="--update-baseline"):
        load_baseline(missing)


def test_unknown_entry_lists_the_known_ones(tmp_path):
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps({"entries": {"lexical": BASELINE}}), encoding="utf-8")

    document = load_baseline(path)

    with pytest.raises(ValueError, match="no baseline entry 'e5:dense'.*lexical"):
        baseline_entry(document, "e5:dense")


def test_malformed_baseline_names_the_file(tmp_path):
    path = tmp_path / "baseline.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError, match="not valid JSON"):
        load_baseline(path)


def test_baseline_without_entries_is_rejected(tmp_path):
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps({"gate": {}}), encoding="utf-8")

    with pytest.raises(ValueError, match="'entries'"):
        load_baseline(path)


def test_write_baseline_leaves_other_entries_untouched(tmp_path):
    path = tmp_path / "baseline.json"
    write_baseline(path, "lexical", {GATE_METRIC: 0.3563, "by_doc": {}}, "aaaaaaa")

    write_baseline(path, "e5:dense", BASELINE, "bbbbbbb")

    document = load_baseline(path)
    assert set(document["entries"]) == {"lexical", "e5:dense"}
    assert document["entries"]["lexical"]["recorded_commit"] == "aaaaaaa"
    assert document["entries"]["e5:dense"][GATE_METRIC] == BASELINE[GATE_METRIC]


def test_written_baseline_round_trips_through_the_gate(tmp_path):
    """Record a run, re-gate the identical run against it: must pass."""
    path = tmp_path / "baseline.json"
    current = make_current()

    write_baseline(path, "e5:dense", current, "cccccc")
    recorded = baseline_entry(load_baseline(path), "e5:dense")

    passed, _ = check(current, recorded)
    assert passed


# --------------------------------------------------------------------------- #
# glue
# --------------------------------------------------------------------------- #


def test_entry_key_is_per_model_except_for_lexical():
    configs = build_configs("bge")

    assert entry_key(configs["dense"]) == "bge:dense"
    assert entry_key(configs["hybrid+rerank"]) == "bge:hybrid+rerank"
    # Lexical reads no embedding column, so one entry serves every model.
    assert entry_key(configs["lexical"]) == "lexical"
    assert entry_key(build_configs("e5")["lexical"]) == "lexical"


def test_summarise_keeps_the_gated_fields_and_drops_latency():
    results = {
        "config": "dense",
        "model_key": "e5",
        "n_pairs": 283,
        "n": 268,
        GATE_METRIC: 0.9272,
        "recall@3": 0.8545,
        "hit@3": 0.8993,
        "hit@10": 0.9478,
        "mrr": 0.8184,
        "by_doc": {LABOUR: {"n": 224, BY_DOC_METRIC: 0.9375}},
        "latency": {"mean_ms": 2.6, "p95_ms": 3.58},
        "unanswerable": {"n": 15},
    }

    summary = summarise(results)

    assert summary["model"] == "e5"
    assert summary[GATE_METRIC] == 0.9272
    assert summary["by_doc"] == results["by_doc"]
    # Machine-dependent numbers must not gate anything.
    assert "latency" not in summary
    assert "unanswerable" not in summary


def test_report_states_the_verdict_and_the_baseline_provenance():
    passed, text = render("e5:dense", make_current(recall=0.5), BASELINE, {"updated": "2026-07-25"})

    assert not passed
    assert "FAIL" in text
    assert "6b70b6d" in text, "a reader must see which commit the baseline came from"
    assert LABOUR in text
