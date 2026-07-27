"""CI regression gate: re-run one retrieval config and refuse a drop vs ``evals/baseline.json``.

    DATABASE_URL=... python -m evals.gate --config lexical
    DATABASE_URL=... python -m evals.gate --config hybrid+rerank --model bge
    DATABASE_URL=... python -m evals.gate --config dense --model e5 --update-baseline

Exit 0 = pass, exit 1 = regression (or a broken setup). Two checks, both on
recall@10:

1. **Aggregate recall@10, 2 pt tolerance.** The number the design spec gates on.
2. **Per-``source_doc`` recall@10, 5 pt tolerance.** 224 of the 268 answerable
   pairs come from the labour law, so the aggregate is essentially that one
   document — the dataset audit flagged that a small document can go to zero
   while the headline moves less than a point. The tolerance is wider on purpose:
   the smallest document is 6 pairs, so a single pair flipping is already 16.7
   pts and a 2 pt band there would fire on noise instead of on regressions.

The gate re-runs the harness; it never reads ``benchmark/results/results.json``.
The baseline was *seeded* from that file, but a gate that compares a stored
number to another stored number tests nothing.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy.exc import SQLAlchemyError

from app.models.chunks import EMBEDDING_COLUMNS
from evals.harness import BY_DOC_K, RetrievalConfig, build_configs, evaluate
from evals.metrics import compare_to_baseline
from evals.schema import load_pairs

_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASELINE = _ROOT / "evals" / "baseline.json"
DEFAULT_PAIRS = _ROOT / "evals" / "data" / "eval_pairs.jsonl"

GATE_METRIC = "recall@10"
BY_DOC_METRIC = f"recall@{BY_DOC_K}"  # the harness picks k=10 so both checks watch one number
TOLERANCE_PTS = 2.0
BY_DOC_TOLERANCE_PTS = 5.0

# Reported as context around the gated metric. Moving any of these is allowed;
# seeing them move without recall@10 moving is usually the interesting part.
CONTEXT_METRICS = ("recall@3", "hit@3", "hit@10", "mrr")

_BASELINE_NOTE = (
    "Measured numbers, not targets. Each entry was produced by "
    "`python -m evals.gate --config <c> --model <m> --update-baseline` against the "
    "corpus in the chunks table at the recorded commit. Update deliberately, in a "
    "commit of its own that says why."
)


# --------------------------------------------------------------------------- #
# pure: everything below is testable without a database or a model
# --------------------------------------------------------------------------- #


def entry_key(config: RetrievalConfig) -> str:
    """Baseline key for a config. Lexical search reads no embedding column, so it
    gets one entry rather than one per model — two identical rows invite averaging.
    """
    return f"{config.model_key}:{config.name}" if config.dense else config.name


def summarise(results: dict) -> dict:
    """The gate-relevant slice of :func:`evals.harness.evaluate` output.

    Latency and the refusal line are deliberately dropped: wall-clock is not
    reproducible across machines, and a gate that fails because CI is slower than
    a laptop gets disabled within a week.
    """
    return {
        "config": results["config"],
        "model": results["model_key"],
        "n_pairs": results["n_pairs"],
        "n": results["n"],
        GATE_METRIC: results[GATE_METRIC],
        **{key: results[key] for key in CONTEXT_METRICS},
        "by_doc": results["by_doc"],
    }


def load_baseline(path: Path) -> dict:
    """Read the whole baseline document. ValueError names the fix on any problem."""
    try:
        raw = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ValueError(
            f"no baseline file at {path} — record one with "
            f"`python -m evals.gate --config <config> --update-baseline` and commit it"
        ) from exc
    except OSError as exc:
        raise ValueError(f"cannot read baseline {path}: {exc}") from exc

    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"baseline {path} is not valid JSON: {exc.msg} (line {exc.lineno})") from exc

    # Every failure in this module is a bad *file*, not a bad call, so callers only
    # ever catch ValueError (same convention as evals.schema; noqa for isinstance).
    if not isinstance(document, dict) or not isinstance(document.get("entries"), dict):
        raise ValueError(f"baseline {path}: expected a JSON object with an 'entries' object")  # noqa: TRY004
    return document


def baseline_entry(document: dict, key: str) -> dict:
    """One recorded configuration out of a loaded baseline document."""
    entries = document["entries"]
    if key not in entries:
        known = ", ".join(sorted(entries)) or "none"
        raise ValueError(
            f"no baseline entry {key!r} (recorded: {known}) — "
            f"record it with --update-baseline, or gate a config that has one"
        )
    entry = entries[key]
    if not isinstance(entry, dict):
        raise ValueError(  # noqa: TRY004
            f"baseline entry {key!r} must be an object, got {type(entry).__name__}"
        )
    return entry


def _context_line(current: dict, baseline: dict) -> str:
    parts = [
        f"{key} {(float(current[key]) - float(baseline[key])) * 100:+.2f}"
        for key in CONTEXT_METRICS
        if key in current and key in baseline
    ]
    return "  context     " + " · ".join(parts) + " pts (not gated)"


def check(
    current: dict,
    baseline: dict,
    tolerance_pts: float = TOLERANCE_PTS,
    by_doc_tolerance_pts: float = BY_DOC_TOLERANCE_PTS,
) -> tuple[bool, list[str]]:
    """Compare a fresh :func:`summarise` result against a baseline entry.

    Returns ``(passed, lines)``. Both checks always run and always report, so a
    failing aggregate still prints which documents moved.
    """
    passed, message = compare_to_baseline(current, baseline, GATE_METRIC, tolerance_pts)
    lines = [f"  aggregate   {message}", _context_line(current, baseline)]

    if current.get("n") != baseline.get("n"):
        # trade-off: a warning, not a failure — the eval set is expected to grow, and
        # growth would otherwise fail the gate for every PR that adds a pair. Ceiling:
        # deleting hard pairs quietly raises recall. Upgrade path: gate per-pair ids.
        lines.append(
            f"  WARNING     scored {current.get('n')} answerable pairs, baseline scored "
            f"{baseline.get('n')} — the metrics below are over different datasets"
        )

    current_docs = current.get("by_doc", {})
    baseline_docs = baseline.get("by_doc", {})
    lines.append("")
    lines.append(f"  per-document {BY_DOC_METRIC} (tolerance {by_doc_tolerance_pts:.2f} pts):")

    for doc in sorted(baseline_docs):
        if doc not in current_docs:
            passed = False
            lines.append(f"    {doc}: FAIL: in the baseline but not in this run")
            continue
        doc_passed, doc_message = compare_to_baseline(
            current_docs[doc], baseline_docs[doc], BY_DOC_METRIC, by_doc_tolerance_pts
        )
        passed = passed and doc_passed
        n = current_docs[doc].get("n", "?")
        lines.append(f"    {doc}: n={n} {doc_message}")

    for doc in sorted(set(current_docs) - set(baseline_docs)):
        lines.append(f"    {doc}: new document, not in the baseline (not gated)")

    return passed, lines


def render(key: str, current: dict, baseline: dict, document: dict) -> tuple[bool, str]:
    """The whole gate report, verdict last. Returns ``(passed, text)``."""
    passed, lines = check(current, baseline)
    header = (
        f"gate: {key}  ({current['n']} answerable of {current['n_pairs']} pairs)\n"
        f"baseline: recorded {baseline.get('recorded_date', '?')} "
        f"at commit {baseline.get('recorded_commit', '?')} "
        f"(file updated {document.get('updated', '?')})"
    )
    verdict = "PASS — no regression" if passed else "FAIL — retrieval regressed"
    return passed, "\n".join([header, "", *lines, "", verdict])


def _git_commit() -> str | None:
    # trade-off: a copy of benchmark.run._git_commit rather than an import — benchmark
    # already imports evals, and a shared "repo metadata" module for eight lines of
    # subprocess is the kind of abstraction this repo is trying not to grow.
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        return None
    return out.stdout.strip() or None


def write_baseline(path: Path, key: str, entry: dict, commit: str | None = None) -> dict:
    """Record one entry, leaving every other entry byte-identical. Returns the document."""
    document = load_baseline(path) if Path(path).exists() else {"_note": _BASELINE_NOTE, "entries": {}}
    today = datetime.now(UTC).date().isoformat()
    updated = {
        **document,
        "_note": _BASELINE_NOTE,
        "gate": {
            "metric": GATE_METRIC,
            "tolerance_pts": TOLERANCE_PTS,
            "by_doc_metric": BY_DOC_METRIC,
            "by_doc_tolerance_pts": BY_DOC_TOLERANCE_PTS,
        },
        "updated": today,
        "entries": {
            **document["entries"],
            key: {**entry, "recorded_date": today, "recorded_commit": commit},
        },
    }
    Path(path).write_text(json.dumps(updated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return updated


_UPDATE_WARNING = """
================================================================================
BASELINE REWRITTEN — {path} entry {key!r}
This is a deliberate act, not a fix for a red gate. Commit it ON ITS OWN, with
the reason in the message. If the gate was failing, the retrieval change is what
needs explaining; moving the goalposts silently is how eval gates stop meaning
anything.
================================================================================
"""


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


async def _measure(args: argparse.Namespace, config: RetrievalConfig) -> dict:
    """Run the harness once against the configured database."""
    from app.db import SessionLocal, engine
    from app.retrieval.embed import get_embedder
    from app.retrieval.rerank import get_reranker

    pairs = load_pairs(args.pairs)
    embedder = get_embedder(config.model_key) if config.dense else None
    reranker = get_reranker("bge") if config.rerank else None

    try:
        async with SessionLocal() as session:
            return await evaluate(pairs, session, config, embedder, reranker)
    finally:
        await engine.dispose()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m evals.gate",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="lexical",
        choices=sorted(build_configs()),
        help="retrieval configuration to gate (default: lexical — the only one that needs no model download)",
    )
    parser.add_argument("--model", default="e5", choices=sorted(EMBEDDING_COLUMNS))
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="rewrite this config's baseline entry from the run instead of gating on it",
    )
    args = parser.parse_args(argv)

    config = build_configs(args.model)[args.config]
    key = entry_key(config)

    try:
        # Read the baseline *before* running anything: a missing entry should cost a
        # second, not the five minutes a reranked pass takes.
        document = None if args.update_baseline else load_baseline(args.baseline)
        baseline = None if document is None else baseline_entry(document, key)

        current = summarise(asyncio.run(_measure(args, config)))
        if document is None or baseline is None:
            write_baseline(args.baseline, key, current, _git_commit())
            print(json.dumps(current, ensure_ascii=False, indent=2))
            print(_UPDATE_WARNING.format(path=args.baseline, key=key), file=sys.stderr)
            return 0

        passed, report = render(key, current, baseline, document)
        print(report)
        return 0 if passed else 1
    except ValueError as exc:
        # Missing baseline, unknown entry, missing API key, malformed dataset.
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


if __name__ == "__main__":
    raise SystemExit(main())
