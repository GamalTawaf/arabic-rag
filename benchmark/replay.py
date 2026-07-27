"""Replay eval questions through the real /ask pipeline and fail on a blown latency budget.

    PYTHONPATH=. python -m benchmark.replay --n 30

One job: catch a latency regression before a user does. It runs N labelled eval
questions through :class:`app.service.RagService` — the same object the HTTP route
uses, not a re-implementation of it — collects the per-stage millisecond timings
the service already records, and compares each stage's p95 against its allocation
in :data:`BUDGET`. Exit 0 if every stage is inside its allocation, 1 if any is not.

:data:`BUDGET` is the **only** definition of the allocations. ``docs/latency-budget.md``
quotes the output of ``--print-budget`` and ``tests/test_replay.py`` fails if the
two ever disagree, so the number in the doc cannot drift away from the number the
gate enforces.

**Generation is skipped when no provider is configured, loudly.** With no API key
the expensive stage — the one that dominates the budget — cannot run, so the
script prints ``SKIPPED`` for it, marks the end-to-end total as unchecked, and
says which environment variable would enable it. A budget check that quietly
omitted the dominant stage would report green while measuring a third of the
request.

# trade-off: this reads ``RagService._prepare`` directly on the no-provider path.
# It is private, and that is the trade: the alternative is a second copy of the
# stage sequence in this file, which is exactly the thing that drifts. Upgrade
# path: if a third caller ever needs retrieval-without-generation, promote it to a
# public ``RagService.retrieve()`` and call that from both.
"""

from __future__ import annotations

import argparse
import asyncio
import math
import random
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy.exc import SQLAlchemyError

from app.service import CONFIGS, DEFAULT_CONFIG
from evals.schema import EvalPair, load_pairs

DEFAULT_PAIRS = Path("evals/data/eval_pairs.jsonl")
DEFAULT_N = 30
SAMPLE_SEED = 0  # fixed, so two runs replay the same questions
PERCENTILES = (50.0, 95.0)

#: p95 milliseconds allowed per pipeline stage. The stage names are the ones
#: ``RagService`` records, so adding a stage to the pipeline without allocating
#: for it here fails this script rather than silently going unmeasured.
#:
#: Every retrieval figure is a laptop measurement against a local Postgres with
#: headroom on top (see docs/latency-budget.md for the derivation and for how
#: much headroom). ``generate`` is the one entry that is **not** measured — no
#: API key exists in this environment — so it is an assumption, and it is the
#: largest line in the budget.
BUDGET: dict[str, float] = {
    "plan": 5.0,
    "embed": 120.0,
    "cache.lookup": 25.0,
    "retrieve": 60.0,
    "fuse": 5.0,
    "rerank": 1200.0,
    "generate": 2000.0,
}

#: End-to-end p95 target from the design spec (§7). The stage allocations sum to
#: less than this on purpose — the difference is request overhead that belongs to
#: no single stage (FastAPI, connection checkout, SSE framing).
TOTAL_BUDGET_MS = 3500.0

GENERATE = "generate"
TOTAL = "total"
#: Which env var to set to make the generate stage measurable.
GENERATION_ENV_HINT = "ANTHROPIC_API_KEY or GOOGLE_API_KEY"


@dataclass(frozen=True)
class StageStats:
    """Observed timings for one stage across the replayed questions."""

    name: str
    n: int
    p50: float
    p95: float
    budget: float | None

    @property
    def over_budget(self) -> bool:
        return self.budget is not None and self.p95 > self.budget


def percentile(values: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile, no interpolation.

    Same definition as :mod:`evals.harness`, deliberately: a stage p95 printed by
    this script and a retrieval p95 printed by the eval harness must mean the
    same thing. At n=30 the 95th percentile is the second-largest sample, so it
    is close to a max — treat it as "the slow request", not as a smooth quantile.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    index = math.ceil(pct / 100 * len(ordered)) - 1
    return ordered[min(max(index, 0), len(ordered) - 1)]


def summarise(samples: dict[str, list[float]]) -> list[StageStats]:
    """Per-stage p50/p95, budgeted stages first and in budget order.

    A measured stage with no entry in :data:`BUDGET` gets ``budget=None`` and is
    reported (and treated as a failure by :func:`check`) rather than dropped:
    an unallocated stage means the budget document no longer describes the
    pipeline.

    The reverse also holds: a **budgeted** stage that produced no samples is
    reported with ``n=0`` rather than omitted. Omitting it let the gate pass on a
    pipeline it never exercised — and that is not exotic, it is what the second
    run does. The semantic cache is append-only, so a replay of the same
    questions serves every one from cache and `retrieve`, `fuse` and `rerank`
    never execute. :func:`check` fails on ``n=0``.
    """
    ordered = list(BUDGET)
    ordered += sorted(name for name in samples if name not in BUDGET and name != TOTAL)
    if TOTAL in samples:
        ordered.append(TOTAL)

    stats: list[StageStats] = []
    for name in ordered:
        values = samples.get(name, [])
        stats.append(
            StageStats(
                name=name,
                n=len(values),
                p50=round(percentile(values, PERCENTILES[0]), 2),
                p95=round(percentile(values, PERCENTILES[1]), 2),
                budget=TOTAL_BUDGET_MS if name == TOTAL else BUDGET.get(name),
            )
        )
    return stats


def check(stats: Sequence[StageStats], *, generated: bool) -> tuple[bool, list[str]]:
    """Verdict plus one line per problem. ``generated`` gates the end-to-end check.

    Without a generation provider the ``total`` line is missing the stage that
    dominates it, so comparing it to the 3.5 s target would pass for the wrong
    reason. It is reported and explicitly not checked, and neither is the absence
    of ``generate`` samples.
    """
    problems: list[str] = []
    for stage in stats:
        if stage.name in (TOTAL, GENERATE) and not generated:
            continue
        if stage.n == 0:
            problems.append(
                f"{stage.name}: no samples — the replay never exercised this stage, "
                "so its allocation is unverified (a cached replay skips retrieval; "
                "re-run against a clean query_cache or with the cache disabled)"
            )
        elif stage.budget is None:
            problems.append(
                f"{stage.name}: no allocation in BUDGET "
                f"(p95 {stage.p95:.2f} ms) — the pipeline gained a stage the budget does not know about"
            )
        elif stage.over_budget:
            problems.append(
                f"{stage.name}: p95 {stage.p95:.2f} ms exceeds its {stage.budget:.0f} ms allocation "
                f"by {stage.p95 - stage.budget:.2f} ms"
            )
    return not problems, problems


def render_budget() -> str:
    """The allocations as a markdown table — what docs/latency-budget.md quotes."""
    lines = [
        "| stage | p95 allocation (ms) |",
        "|---|---|",
        *(f"| `{name}` | {value:.0f} |" for name, value in BUDGET.items()),
        f"| **sum of stages** | **{sum(BUDGET.values()):.0f}** |",
        f"| unallocated overhead | {TOTAL_BUDGET_MS - sum(BUDGET.values()):.0f} |",
        f"| **end-to-end target** | **{TOTAL_BUDGET_MS:.0f}** |",
    ]
    return "\n".join(lines)


def render(
    stats: Sequence[StageStats], run: Run, *, n_questions: int, config: str
) -> str:
    generated = run.generated
    header = (
        f"replay: {n_questions} eval questions   config={config}   model={run.model_key}   "
        f"seed={SAMPLE_SEED}"
    )
    lines = [
        header,
        "",
        f"  {'stage':<14}{'n':>5}{'p50 ms':>11}{'p95 ms':>11}{'budget':>11}   status",
    ]
    for stage in stats:
        budget = "-" if stage.budget is None else f"{stage.budget:.0f}"
        if stage.name == TOTAL and not generated:
            status = "not checked (generation skipped)"
        elif stage.budget is None:
            status = "UNBUDGETED"
        else:
            status = "OVER BUDGET" if stage.over_budget else "ok"
        lines.append(
            f"  {stage.name:<14}{stage.n:>5}{stage.p50:>11.2f}{stage.p95:>11.2f}"
            f"{budget:>11}   {status}"
        )

    if not generated:
        lines.append(
            f"  {GENERATE:<14}{'-':>5}{'-':>11}{'-':>11}{BUDGET[GENERATE]:>11.0f}   "
            f"SKIPPED — no generation provider configured"
        )
    lines.append("")
    if not generated:
        lines.append(
            "  GENERATION NOT MEASURED: no provider is configured, so the stage that "
            "dominates the\n  budget did not run and the end-to-end total is not a "
            f"3.5 s check. Set {GENERATION_ENV_HINT}\n  to measure it."
        )
    warmup = "  ".join(f"{name} {ms:.0f} ms" for name, ms in run.warmup.items())
    lines.append(
        f"  warm-up request (discarded, lazy model load): {warmup or 'none'}"
    )
    lines.append(
        f"  cache hits: {run.cache_hits}/{n_questions}   refusals: {run.refusals}/{n_questions}"
    )
    return "\n".join(lines)


def sample_questions(pairs: Sequence[EvalPair], n: int) -> list[EvalPair]:
    """A seeded sample of answerable pairs, MSA and Gulf mixed.

    Not the first N: the dataset is ordered, the first questions are all MSA from
    one document, and a Gulf question is the *expensive* case — the planner adds
    an MSA rewrite, so retrieval runs twice and fusion has something to do. A
    budget replay that never saw one would under-measure the pipeline it guards.
    """
    if n < 1:
        raise ValueError(f"--n must be >= 1, got {n}")
    answerable = [pair for pair in pairs if pair.source_chunk_ids]
    if not answerable:
        raise ValueError("no answerable pairs in the dataset")
    if n >= len(answerable):
        return list(answerable)
    return random.Random(SAMPLE_SEED).sample(answerable, n)


class _NoProvider:
    """Stand-in for the generation provider when no API key is configured.

    Only ``price()`` is ever called (the pre-call spend estimate). Generation is
    never attempted through it — the replay takes the retrieval-only path instead
    and reports the stage as skipped.
    """

    name = "none"
    model = "none"

    def price(self) -> tuple[float, float]:
        return (0.0, 0.0)


def _build_service(model_key: str | None) -> tuple[Any, bool]:
    """The real service, plus whether a generation provider was available."""
    from app.config import settings
    from app.deps import (
        SERVICE_MODEL_KEY,
        build_embedder,
        build_planner,
        build_provider,
        build_reranker,
    )
    from app.observability.cost import SpendTracker
    from app.retrieval.embed import get_embedder
    from app.service import RagService

    embedder = (
        build_embedder()
        if model_key in (None, SERVICE_MODEL_KEY)
        else get_embedder(model_key)
    )

    provider: Any
    try:
        provider = build_provider()
        generated = True
    except (ValueError, RuntimeError):
        # No API key configured. Not an error here — it is the documented state of
        # this repo, and the caller reports the generate stage as skipped.
        provider = _NoProvider()
        generated = False

    service = RagService(
        embedder,
        build_reranker(),
        build_planner(),
        provider,
        # A fresh tracker: the replay must not be able to trip the day's real cap,
        # and a cap trip mid-replay would truncate the run instead of measuring it.
        SpendTracker(cap_usd=settings.daily_spend_cap_usd),
        settings,
    )
    return service, generated


async def _replay_one(
    service: Any, session: Any, question: str, config: str, generated: bool
) -> tuple[dict[str, float], bool, bool]:
    """One question through the pipeline. Returns (stage ms, cached, refused)."""
    if generated:
        answer = await service.answer(session, question, config)
        return answer.stages, answer.cached, answer.refused
    prepared = await service._prepare(session, question, config)
    return dict(prepared.stages), prepared.cached is not None, prepared.refused


@dataclass(frozen=True)
class Run:
    """Everything one replay produced, for rendering and for the verdict."""

    samples: dict[str, list[float]]
    generated: bool
    model_key: str
    cache_hits: int
    refusals: int
    warmup: dict[str, float]


async def replay(
    pairs: Sequence[EvalPair], config: str, model_key: str | None = None
) -> Run:
    """Run every pair through the pipeline once, after one discarded warm-up.

    The embedder and the cross-encoder load their weights on first use, so the
    first request through a fresh process pays seconds that no later request
    pays. In a 30-sample p95 that one request *is* the p95, which would make this
    gate measure process start-up instead of steady-state latency. So the first
    sampled question is replayed once with its timings thrown away, and reported
    separately — the cost is real, it is just a start-up cost, and hiding it
    entirely would be the other kind of dishonest.
    """
    from app.db import SessionLocal, engine

    service, generated = _build_service(model_key)
    samples: dict[str, list[float]] = {}
    cache_hits = 0
    refusals = 0

    try:
        async with SessionLocal() as session:
            warmup, _, _ = await _replay_one(
                service, session, pairs[0].question, config, generated
            )
            for pair in pairs:
                stages, cached, refused = await _replay_one(
                    service, session, pair.question, config, generated
                )
                cache_hits += int(cached)
                refusals += int(refused)
                for name, elapsed_ms in stages.items():
                    samples.setdefault(name, []).append(elapsed_ms)
    finally:
        await engine.dispose()

    return Run(
        samples=samples,
        generated=generated,
        model_key=service.embedder.model_key,
        cache_hits=cache_hits,
        refusals=refusals,
        warmup=warmup,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m benchmark.replay",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="questions to replay (default: 30)")
    parser.add_argument("--config", default=DEFAULT_CONFIG, choices=sorted(CONFIGS))
    parser.add_argument("--model", help="embedding model key (default: the service's)")
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument(
        "--print-budget",
        action="store_true",
        help="print the allocations as markdown and exit (what docs/latency-budget.md quotes)",
    )
    args = parser.parse_args(argv)

    if args.print_budget:
        print(render_budget())
        return 0

    try:
        pairs = sample_questions(load_pairs(args.pairs), args.n)
        run = asyncio.run(replay(pairs, args.config, args.model))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (SQLAlchemyError, OSError) as exc:
        print(
            f"database error: {type(exc).__name__}: {str(exc).splitlines()[0]}\n"
            "  is Postgres up (docker compose up -d db) and migrated "
            "(alembic upgrade head) with the corpus ingested (python -m ingestion ingest)?",
            file=sys.stderr,
        )
        return 1

    stats = summarise(run.samples)
    passed, problems = check(stats, generated=run.generated)
    print(render(stats, run, n_questions=len(pairs), config=args.config))
    if problems:
        print("\nBUDGET EXCEEDED:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
