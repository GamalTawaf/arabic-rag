"""The latency-budget gate — tested for the thing that makes it a gate: failing.

A budget check nobody has watched fail is a report with an exit code. So these
tests drive :mod:`benchmark.replay` past its allocations, past a stage the budget
has never heard of, and past a run with no generation provider, and assert on the
exit status and on what the operator is told each time.

Timings are injected. Nothing here loads a model or opens a database — the
pipeline itself is covered by tests/test_service.py, and what is under test here
is the arithmetic and the verdict.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from benchmark.replay import (
    BUDGET,
    GENERATION_ENV_HINT,
    TOTAL_BUDGET_MS,
    Run,
    check,
    main,
    percentile,
    render,
    render_budget,
    sample_questions,
    summarise,
)
from evals.schema import EvalPair

DOC = Path("docs/latency-budget.md")


def make_run(samples: dict[str, list[float]], *, generated: bool = False) -> Run:
    return Run(
        samples=samples,
        generated=generated,
        model_key="bge",
        cache_hits=0,
        refusals=0,
        warmup={"embed": 4297.0},
    )


def pair(pair_id: str, *, answerable: bool = True, dialect: str = "msa") -> EvalPair:
    return EvalPair(
        id=pair_id,
        question=f"سؤال {pair_id}",
        dialect_tag=dialect,
        answer="جواب",
        source_doc="qatar-labour-law-14-2004",
        source_chunk_ids=["law:1:0"] if answerable else [],
    )


# --- percentile ------------------------------------------------------------


def test_percentile_is_nearest_rank_with_no_interpolation():
    # Arrange — 1..10, so p50 is the 5th value and p95 the 10th
    values = [float(n) for n in range(1, 11)]

    # Act / Assert
    assert percentile(values, 50.0) == 5.0
    assert percentile(values, 95.0) == 10.0


def test_percentile_of_an_empty_sample_is_zero_not_an_error():
    assert percentile([], 95.0) == 0.0


# --- summarise -------------------------------------------------------------


def test_summarise_attaches_each_stage_its_allocation_and_keeps_budget_order():
    # Arrange — deliberately out of order
    samples = in_budget(rerank=[900.0], plan=[0.1], total=[950.0])

    # Act
    stats = summarise(samples)

    # Assert — BUDGET order, then total last
    assert [stage.name for stage in stats] == [*BUDGET, "total"]
    assert [stage.budget for stage in stats] == [*BUDGET.values(), TOTAL_BUDGET_MS]


def test_summarise_lists_every_budgeted_stage_even_with_no_samples():
    """An omitted stage is an unverified allocation the gate cannot see."""
    # Act
    stats = summarise({"plan": [0.1]})

    # Assert
    assert [stage.name for stage in stats] == list(BUDGET)
    silent = [stage.name for stage in stats if stage.n == 0]
    assert silent == [name for name in BUDGET if name != "plan"]


def test_summarise_reports_a_stage_the_budget_has_never_heard_of():
    # Arrange — a new pipeline stage nobody allocated for
    stats = summarise({"plan": [0.1], "translate": [40.0]})

    # Act
    unbudgeted = [stage for stage in stats if stage.budget is None]

    # Assert
    assert [stage.name for stage in unbudgeted] == ["translate"]



def in_budget(**overrides: list[float]) -> dict[str, list[float]]:
    """Every budgeted stage comfortably inside its allocation, then overrides.

    Tests state only the stage they are about. A partial dict is no longer a
    passing baseline: `summarise` now reports a budgeted stage with no samples
    as ``n=0`` and `check` fails it, because a replay that never exercised a
    stage cannot vouch for its allocation.
    """
    samples = {name: [value / 2] for name, value in BUDGET.items()}
    samples.update(overrides)
    return samples


# --- check -----------------------------------------------------------------


def test_check_passes_when_every_stage_is_inside_its_allocation():
    # Arrange
    stats = summarise(in_budget())

    # Act
    passed, problems = check(stats, generated=True)

    # Assert
    assert passed
    assert problems == []


def test_check_fails_and_names_the_stage_that_blew_its_allocation():
    # Arrange — rerank one millisecond over
    stats = summarise(in_budget(rerank=[BUDGET["rerank"] + 1.0]))

    # Act
    passed, problems = check(stats, generated=True)

    # Assert
    assert not passed
    assert len(problems) == 1
    assert "rerank" in problems[0]
    assert "exceeds" in problems[0]


def test_check_fails_on_an_unbudgeted_stage_so_the_budget_cannot_go_stale():
    # Arrange — a stage that is fast, but that the document does not describe
    stats = summarise(in_budget(translate=[1.0]))

    # Act
    passed, problems = check(stats, generated=True)

    # Assert
    assert not passed
    assert "no allocation in BUDGET" in problems[0]


def test_check_does_not_gate_the_total_when_generation_did_not_run():
    # Arrange — a "total" far over 3.5 s, but generation never happened
    stats = summarise(in_budget(total=[TOTAL_BUDGET_MS * 3]))

    # Act
    skipped_passed, _ = check(stats, generated=False)
    measured_passed, problems = check(stats, generated=True)

    # Assert — unchecked without a provider, enforced with one
    assert skipped_passed
    assert not measured_passed
    assert "total" in problems[0]


# --- render ----------------------------------------------------------------


def test_render_says_generation_was_skipped_and_names_the_env_var():
    # Arrange
    samples = in_budget(plan=[0.1], rerank=[900.0])
    samples.pop("generate")
    stats = summarise(samples)

    # Act
    report = render(stats, make_run(samples), n_questions=1, config="hybrid+rerank")

    # Assert — the skip is stated three ways: the row, the banner, and the fix
    assert "SKIPPED" in report
    assert "GENERATION NOT MEASURED" in report
    assert GENERATION_ENV_HINT in report
    assert "not a 3.5 s check" in report


def test_render_flags_the_over_budget_stage_in_the_table():
    # Arrange
    samples = in_budget(rerank=[BUDGET["rerank"] + 100.0])

    # Act
    report = render(stats := summarise(samples), make_run(samples), n_questions=1, config="dense")

    # Assert
    assert "OVER BUDGET" in report
    assert next(stage for stage in stats if stage.name == "rerank").over_budget


def test_render_reports_the_discarded_warmup_rather_than_hiding_it():
    # Arrange / Act
    samples = in_budget(plan=[0.1])
    report = render(summarise(samples), make_run(samples), n_questions=1, config="dense")

    # Assert — the lazy model load is a real cost, just not a per-request one
    assert "warm-up" in report
    assert "4297" in report


# --- sampling --------------------------------------------------------------


def test_sample_questions_never_replays_an_unanswerable_pair():
    # Arrange — unanswerable pairs have no gold chunks, so they always refuse,
    # and a refusal skips generation: they would flatter the budget.
    pairs = [pair("a"), pair("b", answerable=False), pair("c")]

    # Act
    sampled = sample_questions(pairs, 3)

    # Assert
    assert {item.id for item in sampled} == {"a", "c"}


def test_sample_questions_is_deterministic_across_runs():
    # Arrange
    pairs = [pair(str(n)) for n in range(50)]

    # Act
    first = [item.id for item in sample_questions(pairs, 10)]
    second = [item.id for item in sample_questions(pairs, 10)]

    # Assert
    assert first == second


def test_sample_questions_rejects_a_nonsense_n():
    with pytest.raises(ValueError, match="must be >= 1"):
        sample_questions([pair("a")], 0)


# --- CLI -------------------------------------------------------------------


def test_main_exits_non_zero_when_a_stage_is_over_budget(monkeypatch, capsys):
    # Arrange — a replay whose reranker takes twice its allocation
    async def blown(pairs, config, model_key=None):
        return make_run(in_budget(rerank=[BUDGET["rerank"] * 2] * 5))

    monkeypatch.setattr("benchmark.replay.replay", blown)

    # Act
    code = main(["--n", "5"])

    # Assert
    assert code == 1
    captured = capsys.readouterr()
    assert "OVER BUDGET" in captured.out
    assert "BUDGET EXCEEDED" in captured.err


def test_main_exits_zero_when_every_stage_fits(monkeypatch, capsys):
    # Arrange
    async def fast(pairs, config, model_key=None):
        return make_run({name: [value / 4] * 5 for name, value in BUDGET.items()})

    monkeypatch.setattr("benchmark.replay.replay", fast)

    # Act
    code = main(["--n", "5"])

    # Assert
    assert code == 0
    assert "OVER BUDGET" not in capsys.readouterr().out


def test_main_print_budget_emits_the_table_and_runs_nothing(monkeypatch, capsys):
    # Arrange — any attempt to replay is a failure of this flag
    async def explode(*args, **kwargs):  # pragma: no cover - must not be called
        raise AssertionError("--print-budget must not touch the database")

    monkeypatch.setattr("benchmark.replay.replay", explode)

    # Act
    code = main(["--print-budget"])

    # Assert
    assert code == 0
    assert capsys.readouterr().out.strip() == render_budget()


# --- the budget itself -----------------------------------------------------


def test_stage_allocations_fit_inside_the_end_to_end_target():
    # Arrange / Act
    allocated = sum(BUDGET.values())

    # Assert — the remainder is unallocated request overhead, and must exist
    assert allocated < TOTAL_BUDGET_MS


def test_budget_table_in_doc_matches_code():
    """docs/latency-budget.md quotes BUDGET. One definition, or none."""
    # Arrange
    doc = DOC.read_text(encoding="utf-8")
    quoted = re.search(
        r"<!-- BUDGET TABLE START -->\n(.*?)\n<!-- BUDGET TABLE END -->", doc, re.DOTALL
    )
    assert quoted is not None, f"{DOC} lost its budget-table markers"

    # Act / Assert
    assert quoted.group(1).strip() == render_budget().strip(), (
        f"{DOC} is out of date: regenerate with "
        "`PYTHONPATH=. python -m benchmark.replay --print-budget`"
    )


def test_check_fails_a_budgeted_stage_that_produced_no_samples():
    """Regression: an unexercised stage was omitted, so the gate passed on it.

    Not exotic — it is what the *second* run does. The semantic cache is
    append-only with no TTL, so replaying the same sampled questions serves
    every one from cache and `retrieve`, `fuse` and `rerank` never execute. The
    gate went green having measured nothing.
    """
    # Arrange — a cached replay: planning and the cache lookup ran, nothing else
    stats = summarise({"plan": [1.0], "cache.lookup": [3.0], "total": [10.0]})

    # Act
    passed, problems = check(stats, generated=True)

    # Assert — every silent stage is named
    assert not passed
    named = " ".join(problems)
    for stage in ("embed", "retrieve", "fuse", "rerank"):
        assert stage in named, stage
    assert "no samples" in problems[0]


def test_a_missing_generate_stage_is_not_a_problem_without_a_provider():
    """`generate` has no samples for the honest reason, and says so elsewhere."""
    # Arrange
    samples = in_budget()
    samples.pop("generate")

    # Act
    passed, problems = check(summarise(samples), generated=False)

    # Assert
    assert passed, problems
