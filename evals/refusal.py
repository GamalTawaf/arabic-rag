"""Calibrate the "not in corpus" threshold from the eval set instead of from intuition.

    PYTHONPATH=. python -m evals.refusal --sweep
    PYTHONPATH=. python -m evals.refusal --sweep --scores evals/refusal_scores.json

``settings.rerank_min_score`` decides whether ``/ask`` answers or refuses: the
service refuses when the cross-encoder's top score is **below** it
(``app.service.RagService._is_refusal``). The shipped 0.15 was picked by hand and
refused 5 of 30 answerable questions in a real replay. The dataset settles it:
268 answerable pairs and 15 deliberately unanswerable ones.

Three separable pieces, so the expensive one runs once:

1. :func:`score_pairs` runs the **real** retrieval path — plan → embed → search
   per planned query → RRF → cross-encoder — over every pair and records the top
   rerank score. ~5 minutes with the models warm.
2. :func:`save_scores` / :func:`load_scores` persist those scores as JSON, so
   every later sweep is instant and needs no model in the process.
3. :func:`sweep` and :func:`recommend` are pure functions over the cached scores.

:func:`audit_refused` then answers the question that decides whether a gate is
worth having at all: of the answerable questions a threshold refuses, how many
already had the gold chunk in the context the generator would have read?

**The asymmetry.** Wrongly refusing a question the corpus can answer is a mild
failure; wrongly answering one it does not cover, on a legal-information service,
is a serious one. So :func:`recommend` does **not** maximise F1, which weighs
them identically — see its docstring, and docs/refusal-calibration.md for the
measured result and what it argues for.

# trade-off: the retrieval path here is a re-implementation of the four stages
# ``RagService._prepare`` runs before the gate, because ``_prepare`` also does a
# cache lookup, prompt assembly and a spend check, none of which may influence a
# calibration. The duplication is the trade, and it is guarded: this module
# reranks with ``plan.rewritten or plan.original`` and compares ``hits[0].score``
# exactly as ``_is_refusal`` does, so the number swept here is the number the
# service compares. Upgrade path: promote a public ``RagService.retrieve()`` and
# call it from here, from benchmark.replay, and from the service itself.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.constants import EMBEDDING_COLUMNS
from app.data import Hit
from app.planning.planner import Planner
from app.retrieval.embed import Embedder
from app.retrieval.rerank import Reranker
from app.retrieval.search import (
    dense_search,
    hybrid_search,
    lexical_search,
    rrf_fuse,
)
from evals.schema import EvalPair, load_pairs

_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PAIRS = _ROOT / "evals" / "data" / "eval_pairs.jsonl"
DEFAULT_SCORES = _ROOT / "evals" / "refusal_scores.json"

#: Same names as ``app.service.CONFIGS`` — imported lazily in the CLI so this
#: module stays importable without the service graph.
DEFAULT_CONFIG = "hybrid+rerank"

DEFAULT_STEP = 0.02
DEFAULT_MAX_THRESHOLD = 0.90

#: The most answerable questions the gate may refuse, as a fraction of all of
#: them. 10% is a budget, not a discovery: one in ten users of a labour-rights
#: service being refused something the corpus *does* cover is roughly where a
#: helpful service becomes a useless one. Move it and the recommendation moves.
MAX_FALSE_REFUSAL_RATE = 0.10

#: A refusal has to be right more often than it is wrong, or it is not a safety
#: feature. This is the constraint F1 cannot express — see :func:`recommend`, and
#: read it together with the prevalence caveat in that docstring.
MIN_REFUSAL_PRECISION = 0.50

#: Retrieval returned nothing. The service refuses that unconditionally, whatever
#: the threshold; it does not occur on this 233-chunk corpus.
NO_HITS_SCORE = 0.0

_ROUND_DIGITS = 4


# --------------------------------------------------------------------------- #
# pure: everything in this section is testable without a database or a model
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RefusalPoint:
    """One threshold's confusion matrix, with *refusal* as the positive class.

    Refuse when ``top_score < threshold``, which is the comparison
    ``RagService._is_refusal`` makes. So threshold 0.0 refuses nothing (a sigmoid
    score is never negative) and threshold 1.0 refuses everything.
    """

    threshold: float
    true_refusals: int  # unanswerable, correctly refused
    false_refusals: int  # answerable, wrongly refused — the cost paid by real users
    true_answers: int  # answerable, correctly answered
    false_answers: int  # unanswerable, wrongly answered — the serious failure

    @property
    def n_answerable(self) -> int:
        return self.false_refusals + self.true_answers

    @property
    def n_unanswerable(self) -> int:
        return self.true_refusals + self.false_answers

    @property
    def precision(self) -> float:
        """Of everything refused, how much deserved it. 0.0 when nothing is refused."""
        refused = self.true_refusals + self.false_refusals
        return self.true_refusals / refused if refused else 0.0

    @property
    def recall(self) -> float:
        """Of the questions the corpus cannot answer, how many are caught."""
        return self.true_refusals / self.n_unanswerable if self.n_unanswerable else 0.0

    @property
    def f1(self) -> float:
        """Reported for completeness only — see the module docstring on why it is
        not what :func:`recommend` optimises."""
        total = self.precision + self.recall
        return 2 * self.precision * self.recall / total if total else 0.0

    @property
    def false_refusal_rate(self) -> float:
        """The answerable-question cost: the share of real questions refused."""
        return self.false_refusals / self.n_answerable if self.n_answerable else 0.0

    @property
    def break_even_prevalence(self) -> float:
        """Share of unanswerable traffic at which refusals would be right half the time.

        :attr:`precision` depends on this dataset's answerable/unanswerable mix,
        which is an authoring decision; recall and false-refusal rate do not, since
        each is measured within one population. Comparing this number to a guess
        about real traffic is the only honest way to read a precision number off
        an eval set.
        """
        total = self.recall + self.false_refusal_rate
        return self.false_refusal_rate / total if total else 0.0


def threshold_grid(
    step: float = DEFAULT_STEP, maximum: float = DEFAULT_MAX_THRESHOLD
) -> list[float]:
    """``[0.0, step, 2*step, ...]`` up to and including ``maximum``."""
    if step <= 0:
        raise ValueError(f"step must be > 0, got {step}")
    if maximum < 0:
        raise ValueError(f"maximum must be >= 0, got {maximum}")
    count = round(maximum / step) + 1
    return [round(index * step, 6) for index in range(count)]


def sweep(
    scored: Sequence[tuple[str, bool, float]], thresholds: Sequence[float]
) -> list[RefusalPoint]:
    """One :class:`RefusalPoint` per threshold, in the order given."""
    points: list[RefusalPoint] = []
    for threshold in thresholds:
        true_refusals = false_refusals = true_answers = false_answers = 0
        for _pair_id, answerable, top_score in scored:
            refused = top_score < threshold
            if answerable and refused:
                false_refusals += 1
            elif answerable:
                true_answers += 1
            elif refused:
                true_refusals += 1
            else:
                false_answers += 1
        points.append(
            RefusalPoint(
                threshold=threshold,
                true_refusals=true_refusals,
                false_refusals=false_refusals,
                true_answers=true_answers,
                false_answers=false_answers,
            )
        )
    return points


def recommend(points: Sequence[RefusalPoint]) -> tuple[float, str]:
    """Pick a threshold under the stated asymmetry, and explain the pick.

    Two eligibility constraints, both of which encode the asymmetry rather than
    averaging it away the way F1 does:

    1. ``false_refusal_rate <= MAX_FALSE_REFUSAL_RATE``. The mild failure is
       mild, not free. A gate that fails answerable questions is not made
       acceptable by how many unanswerable ones it catches.
    2. ``precision >= MIN_REFUSAL_PRECISION``. A refusal that is wrong more often
       than right is not a safety feature — it is a second defect wearing one's
       coat. This is the constraint F1 cannot express: F1 will happily trade
       precision away for recall, and here that trade is exactly the mistake.

    Among eligible points, take the highest refusal recall (the serious failure
    is answering something uncovered, so catch as many as the budget allows),
    breaking ties on the lower false-refusal rate and then the lower threshold.

    When **nothing** is eligible the honest answer is that this score is not a
    usable refusal signal, and the justification says so instead of dressing the
    least-bad number up as a calibration; the returned threshold is then the one
    that costs answerable questions least — in a full sweep, 0.0, the gate off.

    Constraint 2 must be read with :attr:`RefusalPoint.break_even_prevalence`:
    precision depends on the dataset's 5.3% unanswerable share, which is not
    traffic.
    """
    if not points:
        raise ValueError("no threshold points to recommend from — sweep something first")

    eligible = [
        point
        for point in points
        if point.false_refusal_rate <= MAX_FALSE_REFUSAL_RATE
        and point.precision >= MIN_REFUSAL_PRECISION
    ]

    if not eligible:
        cheapest = min(
            points, key=lambda point: (point.false_refusal_rate, point.threshold)
        )
        best_precision = max(points, key=lambda point: point.precision)
        return cheapest.threshold, (
            f"No threshold refuses correctly more often than {MIN_REFUSAL_PRECISION:.0%} "
            f"of the time while staying inside the {MAX_FALSE_REFUSAL_RATE:.0%} "
            f"false-refusal budget — the best precision anywhere in the sweep is "
            f"{best_precision.precision:.1%} at {best_precision.threshold:.2f}, where "
            f"{best_precision.false_refusals} answerable questions are refused to catch "
            f"{best_precision.true_refusals} unanswerable ones. The two populations are "
            f"not separable on this score. Recommending {cheapest.threshold:.2f}: do not "
            f"gate on the rerank score at all, because at every threshold that catches "
            f"anything the gate destroys more good answers than bad ones. Refusal needs a "
            f"different signal — see the calibration doc's upgrade path."
        )

    best = min(
        eligible,
        key=lambda point: (-point.recall, point.false_refusal_rate, point.threshold),
    )
    return best.threshold, (
        f"Threshold {best.threshold:.2f} catches {best.true_refusals}/"
        f"{best.n_unanswerable} unanswerable questions "
        f"(refusal recall {best.recall:.1%}, precision {best.precision:.1%}) while "
        f"wrongly refusing {best.false_refusals}/{best.n_answerable} answerable ones "
        f"({best.false_refusal_rate:.1%}, inside the {MAX_FALSE_REFUSAL_RATE:.0%} "
        f"budget). Chosen by maximising refusal recall under that budget and a "
        f"{MIN_REFUSAL_PRECISION:.0%} precision floor, not by maximising F1 (which here "
        f"is {best.f1:.3f}): wrongly answering a question the corpus does not cover is "
        f"the serious failure, wrongly refusing one is the mild one, and F1 would weigh "
        f"them the same."
    )


@dataclass(frozen=True)
class DialectRefusal:
    """How one dialect's *answerable* questions fare at a threshold."""

    dialect_tag: str
    n: int
    refused: int

    @property
    def rate(self) -> float:
        return self.refused / self.n if self.n else 0.0


def refusal_by_dialect(
    scored: Sequence[tuple[str, bool, float]],
    pairs: Sequence[EvalPair],
    threshold: float,
) -> list[DialectRefusal]:
    """False-refusal rate per ``dialect_tag``, sorted by tag.

    Answerable pairs only: a refused unanswerable question is the gate working,
    and mixing the two would hide what this measures — whether the service fails
    dialect speakers disproportionately. Unscored pairs are skipped, not counted.
    """
    tags = {pair.id: pair.dialect_tag for pair in pairs}
    counts: dict[str, list[int]] = {}
    for pair_id, answerable, top_score in scored:
        if not answerable or pair_id not in tags:
            continue
        bucket = counts.setdefault(tags[pair_id], [0, 0])
        bucket[0] += 1
        bucket[1] += int(top_score < threshold)
    return [
        DialectRefusal(dialect_tag=tag, n=bucket[0], refused=bucket[1])
        for tag, bucket in sorted(counts.items())
    ]


def auc(answerable: Sequence[float], unanswerable: Sequence[float]) -> float:
    """P(a random unanswerable question scores below a random answerable one).

    Ties count half. The threshold-free answer to "is there a threshold at all":
    0.5 is a coin flip, 1.0 is perfect separation. Computed the direct way —
    268x15 is 4020 comparisons, and the rank formula only hides the meaning.
    Returns 0.0 when either population is empty.
    """
    if not answerable or not unanswerable:
        return 0.0
    wins = sum(
        (low < high) + 0.5 * (low == high) for high in answerable for low in unanswerable
    )
    return round(wins / (len(answerable) * len(unanswerable)), _ROUND_DIGITS)


def quantiles(values: Sequence[float]) -> dict[str, float]:
    """min / p25 / median / p75 / max, nearest-rank — same convention as the harness."""
    if not values:
        return {"n": 0, "min": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "max": 0.0}
    ordered = sorted(values)

    def at(fraction: float) -> float:
        index = min(int(fraction * (len(ordered) - 1) + 0.5), len(ordered) - 1)
        return round(ordered[index], _ROUND_DIGITS)

    return {
        "n": len(ordered),
        "min": round(ordered[0], _ROUND_DIGITS),
        "p25": at(0.25),
        "median": at(0.50),
        "p75": at(0.75),
        "max": round(ordered[-1], _ROUND_DIGITS),
    }


# --------------------------------------------------------------------------- #
# cached scores
# --------------------------------------------------------------------------- #


def save_scores(path: Path, scored: Sequence[tuple[str, bool, float]], meta: dict) -> None:
    """Write the scores as JSON so no later sweep has to load a 2 GB reranker."""
    document = {
        "_note": (
            "Top cross-encoder score per eval pair, produced by "
            "`python -m evals.refusal`. Regenerate with --rescore after any change "
            "to retrieval, the corpus, the planner or the reranker."
        ),
        **meta,
        "scores": [
            {"pair_id": pair_id, "answerable": answerable, "top_score": round(score, 6)}
            for pair_id, answerable, score in scored
        ],
    }
    Path(path).write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def load_scores(path: Path) -> tuple[list[tuple[str, bool, float]], dict]:
    """Read a cached score file. Returns ``(scored, meta)``.

    Trust boundary: the file is on disk and may be stale or hand-edited, so every
    record is checked and a ValueError names the fix.
    """
    try:
        raw = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ValueError(
            f"no cached scores at {path} — produce them with "
            f"`PYTHONPATH=. python -m evals.refusal --sweep` (runs the models once)"
        ) from exc
    except OSError as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc

    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc.msg} (line {exc.lineno})") from exc
    if not isinstance(document, dict) or not isinstance(document.get("scores"), list):
        raise ValueError(f"{path}: expected a JSON object with a 'scores' list")  # noqa: TRY004

    scored: list[tuple[str, bool, float]] = []
    for index, record in enumerate(document["scores"]):
        if not isinstance(record, dict):
            raise ValueError(f"{path}: scores[{index}] must be an object")  # noqa: TRY004
        missing = [key for key in ("pair_id", "answerable", "top_score") if key not in record]
        if missing:
            raise ValueError(f"{path}: scores[{index}] missing {', '.join(missing)}")
        try:
            score = float(record["top_score"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{path}: scores[{index}] top_score {record['top_score']!r} is not a number"
            ) from exc
        scored.append((str(record["pair_id"]), bool(record["answerable"]), score))

    meta = {key: value for key, value in document.items() if key != "scores"}
    return scored, meta


# --------------------------------------------------------------------------- #
# scoring: the one expensive step
# --------------------------------------------------------------------------- #


def _reranks(config: str) -> bool:
    """Whether ``config`` runs the cross-encoder, per ``RagService._rerank``."""
    return config.endswith("+rerank") and settings.rerank_enabled


def _check_config(config: str) -> str:
    """Reject a config this calibration cannot speak about. Mirrors the peers.

    ``benchmark.replay``, ``evals.gate`` and ``evals.harness`` all validate their
    ``--config``; this module did not, and ``_search`` falls through to
    ``hybrid_search`` for anything unrecognised — so a typo like ``hybrd+rerank``
    silently calibrated a different pipeline and then stamped the bogus string
    into the committed scores file as provenance.
    """
    from app.service import CONFIGS

    if config not in CONFIGS:
        raise ValueError(
            f"unknown retrieval config {config!r}; expected one of {list(CONFIGS)}"
        )
    if not _reranks(config):
        raise ValueError(
            f"config {config!r} does not run the cross-encoder, so there is no "
            "rerank score to calibrate rerank_min_score against "
            f"(rerank_enabled={settings.rerank_enabled}). Use a '+rerank' config."
        )
    return config


async def _search(
    session: AsyncSession,
    config: str,
    query: str,
    vector: Sequence[float] | None,
    model_key: str,
    limit: int,
) -> list[Hit]:
    """One retrieval leg. Mirrors ``RagService._search``."""
    if config == "lexical":
        return await lexical_search(session, query, limit=limit)
    if vector is None:
        raise ValueError(f"config {config!r} is dense but {query!r} was not embedded")
    if config == "dense":
        return await dense_search(session, vector, model_key, limit)
    return await hybrid_search(session, query, vector, model_key, limit=limit)


async def _context(
    pairs: Sequence[EvalPair],
    session: AsyncSession,
    embedder: Embedder,
    reranker: Reranker,
    planner: Planner,
    config: str,
    top_k_retrieve: int,
    top_k_context: int,
    progress: Any = None,
) -> list[list[Hit]]:
    """The reranked top-``top_k_context`` hits per pair — the generator's context.

    The one expensive function in this module: everything else reads its output.
    Query embedding is batched across the whole dataset for the same reason
    :mod:`evals.harness` batches it — 283 sequential embed calls cost minutes and
    buy nothing.
    """
    plans = [await planner.plan(pair.question) for pair in pairs]

    vectors: dict[str, list[float]] = {}
    if config != "lexical":
        texts = list(dict.fromkeys(query for plan in plans for query in plan.search_queries))
        vectors = dict(zip(texts, await embedder.embed_queries(texts)))

    contexts: list[list[Hit]] = []
    for index, plan in enumerate(plans, start=1):
        ranked_lists = [
            await _search(
                session, config, query, vectors.get(query), embedder.model_key, top_k_retrieve
            )
            for query in plan.search_queries
        ]
        fused = (
            ranked_lists[0]
            if len(ranked_lists) == 1
            else rrf_fuse(ranked_lists, limit=top_k_retrieve)
        )
        # Gated exactly as ``RagService._rerank`` gates it. Unconditional
        # reranking here would sweep a score the service never computes for
        # dense/lexical/hybrid — and for those configs ``_is_refusal`` does not
        # even reach the threshold comparison, because it first checks
        # ``hit.source == RERANK_SOURCE``. The swept number has to be the number
        # the service compares, or the calibration is of a pipeline nobody runs.
        if _reranks(config):
            contexts.append(
                await reranker.rerank(
                    plan.rewritten or plan.original, fused, top_k=top_k_context
                )
            )
        else:
            contexts.append(fused[:top_k_context])
        if progress is not None and index % 25 == 0:
            print(f"  scored {index}/{len(pairs)} pairs", file=progress, flush=True)
    return contexts


async def score_pairs(
    pairs: Sequence[EvalPair],
    session: AsyncSession,
    embedder: Embedder,
    reranker: Reranker,
    planner: Planner,
    config: str = DEFAULT_CONFIG,
    *,
    top_k_retrieve: int = settings.top_k_retrieve,
    top_k_context: int = settings.top_k_context,
    progress: Any = None,
) -> list[tuple[str, bool, float]]:
    """``(pair_id, is_answerable, top_rerank_score)`` for every pair, in input order.

    ``top_rerank_score`` is the value ``RagService._is_refusal`` compares against
    ``settings.rerank_min_score``: the score of the first hit after reranking,
    where reranking scores the MSA rewrite when the planner produced one.
    """
    if not pairs:
        return []
    contexts = await _context(
        pairs, session, embedder, reranker, planner, config,
        top_k_retrieve, top_k_context, progress,
    )
    return [
        (pair.id, pair.answerable, hits[0].score if hits else NO_HITS_SCORE)
        for pair, hits in zip(pairs, contexts)
    ]


@dataclass(frozen=True)
class RefusedPair:
    """One answerable question a threshold would refuse, and what it was refusing.

    ``gold_in_context`` is the question this exists to answer: was the correct
    article *already retrieved* into the top-k the generator would have read? If
    it was, the refusal is not "retrieval failed, so refuse" — it is the pipeline
    finding the answer and then discarding it.
    """

    pair_id: str
    dialect_tag: str
    top_score: float
    gold_in_context: bool
    gold_at_rank_1: bool


async def audit_refused(
    pairs: Sequence[EvalPair],
    session: AsyncSession,
    embedder: Embedder,
    reranker: Reranker,
    planner: Planner,
    config: str = DEFAULT_CONFIG,
    *,
    top_k_retrieve: int = settings.top_k_retrieve,
    top_k_context: int = settings.top_k_context,
    progress: Any = None,
) -> list[RefusedPair]:
    """Re-retrieve the given answerable pairs and check the gold chunk's position.

    Pass only the pairs a threshold would refuse — this runs the models, so
    handing it all 283 costs the full five minutes for no extra information.
    """
    answerable = [pair for pair in pairs if pair.answerable]
    if not answerable:
        return []
    contexts = await _context(
        answerable, session, embedder, reranker, planner, config,
        top_k_retrieve, top_k_context, progress,
    )
    audited: list[RefusedPair] = []
    for pair, hits in zip(answerable, contexts):
        gold = set(pair.source_chunk_ids)
        retrieved = [hit.chunk_id for hit in hits]
        audited.append(
            RefusedPair(
                pair_id=pair.id,
                dialect_tag=pair.dialect_tag,
                top_score=hits[0].score if hits else NO_HITS_SCORE,
                gold_in_context=bool(gold & set(retrieved)),
                gold_at_rank_1=bool(retrieved) and retrieved[0] in gold,
            )
        )
    return audited


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #


def render_sweep(points: Sequence[RefusalPoint], every: int = 1) -> str:
    """The ROC-like curve as a markdown table — what docs/refusal-calibration.md quotes."""
    lines = [
        (
            "| threshold | unanswerable refused | answerable refused | refusal recall "
            "| refusal precision | F1 | false-refusal rate |"
        ),
        "|---|---|---|---|---|---|---|",
    ]
    for index, point in enumerate(points):
        if index % every:
            continue
        lines.append(
            f"| {point.threshold:.2f} | {point.true_refusals}/{point.n_unanswerable} "
            f"| {point.false_refusals}/{point.n_answerable} | {point.recall:.3f} "
            f"| {point.precision:.3f} | {point.f1:.3f} | {point.false_refusal_rate:.3f} |"
        )
    return "\n".join(lines)


def _populations(
    scored: Sequence[tuple[str, bool, float]], pairs: Sequence[EvalPair]
) -> list[tuple[str, list[float], list[float]]]:
    """``(label, answerable scores, unanswerable scores)`` for "all" and per dialect."""
    tags = {pair.id: pair.dialect_tag for pair in pairs}
    groups = [("all", [None])] + [(tag, [tag]) for tag in sorted(set(tags.values()))]
    return [
        (
            label,
            [s for i, a, s in scored if a and (wanted == [None] or tags.get(i) in wanted)],
            [s for i, a, s in scored if not a and (wanted == [None] or tags.get(i) in wanted)],
        )
        for label, wanted in groups
    ]


def _distribution(populations: Sequence[tuple[str, list[float], list[float]]]) -> list[str]:
    header = f"    {'population':<20}{'n':>5}{'min':>10}{'p25':>10}{'median':>10}{'p75':>10}{'max':>10}"
    rows = [
        f"    {label + ' ' + kind:<20}{q['n']:>5}{q['min']:>10.4f}{q['p25']:>10.4f}"
        f"{q['median']:>10.4f}{q['p75']:>10.4f}{q['max']:>10.4f}"
        for label, ans, unans in populations
        for kind, q in (("answerable", quantiles(ans)), ("unanswerable", quantiles(unans)))
    ]
    return ["  top rerank score distribution", header, *rows]


def _separability(populations: Sequence[tuple[str, list[float], list[float]]]) -> list[str]:
    lines = [
        "  separability (AUC — 0.5 is a coin flip, 1.0 is a clean split)",
        *(
            f"    {label:<8} answerable vs unanswerable: {auc(ans, unans):.4f}  "
            f"(n {len(ans)} / {len(unans)})"
            for label, ans, unans in populations
        ),
    ]
    by_label = {label: ans for label, ans, _ in populations}
    if "msa" in by_label and "gulf" in by_label:
        lines.append(
            f"    the same score separates MSA-answerable from Gulf-answerable at "
            f"{auc(by_label['msa'], by_label['gulf']):.4f} — read that against the lines above"
        )
    return lines


def render(
    scored: Sequence[tuple[str, bool, float]],
    pairs: Sequence[EvalPair],
    points: Sequence[RefusalPoint],
    meta: dict,
    *,
    every: int,
    compare: float | None = None,
) -> str:
    populations = _populations(scored, pairs)
    _, answerable, unanswerable = populations[0]
    chosen, justification = recommend(points)
    # Answerable pairs only below: a refused unanswerable question is the gate working.
    breakdowns = (
        [chosen] if compare is None or abs(compare - chosen) < 1e-9 else [chosen, compare]
    )

    return "\n".join(
        [
            (
                f"refusal calibration: {len(scored)} pairs "
                f"({len(answerable)} answerable / {len(unanswerable)} unanswerable)"
            ),
            (
                f"  config={meta.get('config', '?')}  model={meta.get('model_key', '?')}  "
                f"reranker={meta.get('reranker', '?')}  planner={meta.get('planner', '?')}  "
                f"scored={meta.get('recorded', '?')}"
            ),
            "",
            *_distribution(populations),
            "",
            *_separability(populations),
            "",
            "  sweep",
            "    " + render_sweep(points, every).replace("\n", "\n    "),
            "",
            (
                f"  RECOMMENDED rerank_min_score = {chosen:.2f}  "
                f"(configured: {settings.rerank_min_score:.2f})"
            ),
            f"    {justification}",
            "",
            *[
                line
                for threshold in breakdowns
                for line in [
                    f"  false-refusal rate by dialect at {threshold:.2f} (answerable only)",
                    *(
                        f"    {entry.dialect_tag:<6}{entry.n:>5} answerable  "
                        f"{entry.refused:>4} refused  {entry.rate:>8.1%}"
                        for entry in refusal_by_dialect(scored, pairs, threshold)
                    ),
                ]
            ],
        ]
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _relative(path: Path) -> Path:
    """A repo-relative path when possible, so committed metadata is machine-neutral."""
    try:
        return Path(path).resolve().relative_to(_ROOT)
    except ValueError:
        return Path(path)


def _collaborators(model_key: str) -> tuple[Embedder, Reranker, Planner]:
    """(embedder, reranker, planner) — the service's own, imported here so that
    importing this module loads no model."""
    from app.deps import SERVICE_PLANNER, SERVICE_RERANKER
    from app.planning.planner import get_planner
    from app.retrieval.embed import get_embedder
    from app.retrieval.rerank import get_reranker

    return get_embedder(model_key), get_reranker(SERVICE_RERANKER), get_planner(SERVICE_PLANNER)


async def _run(args: argparse.Namespace, pairs: Sequence[EvalPair], *, audit: bool) -> Any:
    """One pass of the expensive path over ``pairs``: scores, or the refusal audit."""
    from app.db import engine, session_scope

    _check_config(args.config)  # before the models load, not after
    verb = "auditing" if audit else "scoring"
    print(
        f"{verb} {len(pairs)} pairs through {args.config} (model={args.model}) "
        "— this loads the models",
        file=sys.stderr,
        flush=True,
    )
    run = audit_refused if audit else score_pairs
    try:
        async with session_scope() as session:
            return await run(
                pairs, session, *_collaborators(args.model), args.config,
                progress=None if audit else sys.stderr,
            )
    finally:
        await engine().dispose()


def _meta(args: argparse.Namespace) -> dict:
    """Provenance recorded alongside the cached scores."""
    from app.deps import SERVICE_PLANNER, SERVICE_RERANKER

    return {
        "config": args.config,
        "model_key": args.model,
        "reranker": SERVICE_RERANKER,
        "planner": SERVICE_PLANNER,
        "top_k_retrieve": settings.top_k_retrieve,
        "top_k_context": settings.top_k_context,
        # Relative so the committed file does not carry one machine's home directory.
        "pairs": str(_relative(args.pairs)),
        "recorded": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def render_audit(audited: Sequence[RefusedPair], threshold: float) -> str:
    """Was the answer already in the context of the questions this threshold refuses?"""
    in_context = sum(entry.gold_in_context for entry in audited)
    at_one = sum(entry.gold_at_rank_1 for entry in audited)
    lines = [
        "",
        f"  refusal audit at {threshold:.2f}: {len(audited)} answerable pairs refused",
        f"    gold chunk already in the top-k context: {in_context}/{len(audited)}",
        f"    gold chunk already at rank 1:            {at_one}/{len(audited)}",
    ]
    for tag in sorted({entry.dialect_tag for entry in audited}):
        split = [entry for entry in audited if entry.dialect_tag == tag]
        lines.append(
            f"    {tag:<6}{len(split):>4} refused, "
            f"{sum(entry.gold_in_context for entry in split)} with the gold chunk in context"
        )
    return "\n".join(lines)


def _dataset_drift(scored: Iterable[tuple[str, bool, float]], pairs: Sequence[EvalPair]) -> str | None:
    """A warning when the cached scores no longer describe the dataset."""
    cached = {pair_id for pair_id, _, _ in scored}
    current = {pair.id for pair in pairs}
    if cached == current:
        return None
    return (
        f"WARNING: cached scores cover {len(cached)} pairs, the dataset has "
        f"{len(current)} ({len(current - cached)} unscored, {len(cached - current)} stale) "
        "— rerun with --rescore"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m evals.refusal",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="sweep thresholds and recommend one — the default action",
    )
    parser.add_argument(
        "--scores",
        type=Path,
        default=DEFAULT_SCORES,
        help="cached per-pair scores; reused when present (default: %(default)s)",
    )
    parser.add_argument(
        "--rescore",
        action="store_true",
        help="re-run the models even if the cache exists, and overwrite it",
    )
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--model", default="bge", choices=sorted(EMBEDDING_COLUMNS))
    parser.add_argument("--step", type=float, default=DEFAULT_STEP)
    parser.add_argument("--max", type=float, default=DEFAULT_MAX_THRESHOLD)
    parser.add_argument(
        "--compare",
        type=float,
        help="also break the false-refusal rate down at this threshold",
    )
    parser.add_argument(
        "--audit",
        type=float,
        metavar="THRESHOLD",
        help="re-retrieve the pairs this threshold refuses and report whether the "
        "gold chunk was already in the context (runs the models)",
    )
    parser.add_argument(
        "--every",
        type=int,
        default=1,
        help="print every Nth sweep row (default: every row)",
    )
    args = parser.parse_args(argv)

    try:
        pairs = load_pairs(args.pairs)
        if args.rescore or not Path(args.scores).exists():
            scored = asyncio.run(_run(args, pairs, audit=False))
            meta = _meta(args)
            save_scores(args.scores, scored, meta)
            print(f"wrote {args.scores}", file=sys.stderr)
        else:
            scored, meta = load_scores(args.scores)
            print(f"reusing cached scores from {args.scores}", file=sys.stderr)

        warning = _dataset_drift(scored, pairs)
        if warning:
            print(warning, file=sys.stderr)

        points = sweep(scored, threshold_grid(args.step, args.max))
        audit_report = ""
        if args.audit is not None:
            by_id = {pair.id: pair for pair in pairs}
            refused = [
                by_id[pair_id]
                for pair_id, answerable, score in scored
                if answerable and score < args.audit and pair_id in by_id
            ]
            audit_report = render_audit(
                asyncio.run(_run(args, refused, audit=True)) if refused else [], args.audit
            )
        print(
            render(
                scored, pairs, points, meta,
                every=max(args.every, 1), compare=args.compare,
            )
            + audit_report
        )
        return 0
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (SQLAlchemyError, OSError) as exc:
        print(
            f"database error: {type(exc).__name__}: {str(exc).splitlines()[0]}\n"
            "  is Postgres up (docker compose up -d db) and migrated "
            "(alembic -c config/alembic.ini upgrade head) with the corpus ingested and backfilled?",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
