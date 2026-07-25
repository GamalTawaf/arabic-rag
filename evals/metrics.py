"""Deterministic retrieval metrics — the layer that gates CI.

Ground truth is chunk ids, so nothing here needs a model, a DB or the network.
Unanswerable pairs (no relevant chunk ids) have *undefined* retrieval metrics:
``recall_at_k`` raises and ``aggregate`` skips them. Route those pairs to refusal
scoring instead.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass

_ROUND_DIGITS = 4
_FLOAT_SLACK = 1e-9  # so a drop of exactly `tolerance` points still passes


def _top_k(retrieved: Sequence[str], k: int) -> set[str]:
    if k <= 0:
        raise ValueError(f"k must be >= 1, got {k}")
    # Positional top-k: slice first, then dedupe. Duplicate ids in `retrieved`
    # are a retriever bug and are counted once but still occupy their rank.
    return set(retrieved[:k])


def recall_at_k(retrieved: Sequence[str], relevant: Collection[str], k: int) -> float:
    """Fraction of relevant chunk ids present in the top-k retrieved ids.

    Raises ValueError when `relevant` is empty: recall is undefined for
    unanswerable pairs, which belong in refusal scoring, not retrieval scoring.
    """
    relevant_ids = set(relevant)
    if not relevant_ids:
        raise ValueError(
            "recall_at_k is undefined for an unanswerable pair (no relevant ids); "
            "route unanswerable pairs to refusal scoring instead"
        )
    hits = relevant_ids & _top_k(retrieved, k)
    return len(hits) / len(relevant_ids)


def mrr(retrieved: Sequence[str], relevant: Collection[str]) -> float:
    """Reciprocal rank of the first relevant hit; 0.0 if there is none."""
    relevant_ids = set(relevant)
    if not relevant_ids:
        return 0.0
    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def hit_rate_at_k(retrieved: Sequence[str], relevant: Collection[str], k: int) -> float:
    """1.0 if any relevant id appears in the top-k retrieved ids, else 0.0."""
    relevant_ids = set(relevant)
    if not relevant_ids:
        return 0.0
    return 1.0 if relevant_ids & _top_k(retrieved, k) else 0.0


@dataclass(frozen=True)
class QueryResult:
    pair_id: str
    dialect_tag: str
    retrieved: list[str]
    relevant: list[str]


def _mean(values: Sequence[float]) -> float:
    return round(sum(values) / len(values), _ROUND_DIGITS) if values else 0.0


def _summarise(results: Sequence[QueryResult], ks: Sequence[int]) -> dict:
    summary: dict = {"n": len(results)}
    for k in ks:
        summary[f"recall@{k}"] = _mean([recall_at_k(r.retrieved, r.relevant, k) for r in results])
        summary[f"hit@{k}"] = _mean([hit_rate_at_k(r.retrieved, r.relevant, k) for r in results])
    summary["mrr"] = _mean([mrr(r.retrieved, r.relevant) for r in results])
    return summary


def aggregate(results: Sequence[QueryResult], ks: Sequence[int] = (3, 10)) -> dict:
    """Mean retrieval metrics overall and per dialect tag.

    Unanswerable results (empty `relevant`) are skipped — their retrieval metrics
    are undefined. Dialect tags with no scorable results are omitted from
    "by_dialect" rather than reported as 0.0/NaN; the dialect split is the
    headline of the benchmark, so an empty bucket must not look like a score.
    """
    scorable = [r for r in results if r.relevant]

    by_dialect: dict[str, dict] = {}
    for tag in sorted({r.dialect_tag for r in scorable}):
        by_dialect[tag] = _summarise([r for r in scorable if r.dialect_tag == tag], ks)

    return {**_summarise(scorable, ks), "by_dialect": by_dialect}


def compare_to_baseline(
    current: dict,
    baseline: dict,
    metric: str = "recall@10",
    tolerance: float = 2.0,
) -> tuple[bool, str]:
    """CI gate: has `metric` dropped more than `tolerance` PERCENTAGE POINTS?

    Metrics are 0-1 floats, so tolerance=2.0 means a drop of 0.02 is still a pass.
    Returns (passed, human-readable message).
    """
    for name, summary in (("current", current), ("baseline", baseline)):
        if metric not in summary:
            raise ValueError(f"{name} summary has no metric {metric!r}")

    current_value = float(current[metric])
    baseline_value = float(baseline[metric])
    drop_points = (baseline_value - current_value) * 100
    passed = drop_points <= tolerance + _FLOAT_SLACK

    verdict = "PASS" if passed else "FAIL"
    message = (
        f"{verdict}: {metric} {current_value:.4f} vs baseline {baseline_value:.4f} "
        f"({-drop_points:+.2f} pts, tolerance {tolerance:.2f} pts)"
    )
    return passed, message
