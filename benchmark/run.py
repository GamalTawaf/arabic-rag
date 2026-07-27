"""The benchmark matrix: every embedding model × every retrieval config × every dialect.

    DATABASE_URL=postgresql+asyncpg://... python -m benchmark.run
    DATABASE_URL=...                     python -m benchmark.run --models e5 --limit 20
    python -m benchmark.run --tables                 # markdown tables from an existing results.json

Each cell is one full :func:`evals.harness.evaluate` pass over a slice of the
labelled dataset, so a cell's latency is really measured on the pairs in that
cell and not projected from another run.

Two things the layout deliberately makes explicit rather than convenient:

1. **Lexical search does not use an embedding model.** Running it once per model
   would print four identical rows and invite a reader to average them as if they
   were four measurements. It runs *once*, is stored with ``"model": null``, and
   carries a ``shared`` note naming every model it applies to.
2. **Models with no API key are recorded, not omitted.** An absent row reads as
   "measured and uninteresting"; ``{"status": "not_run", "reason": ...}`` reads as
   what it is. Same for a local model whose vector column was never backfilled.

trade-off: no charts, no pandas, no statistics beyond means and a nearest-rank
p95 — the harness already computes every metric and n is a few hundred. Ceiling:
no confidence intervals, so the writeup states the Gulf subset (54 pairs) is too
small to separate close configs. Upgrade path: bootstrap the per-pair scores.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.chunks import EMBEDDING_COLUMNS, Chunk
from app.retrieval.embed import LOCAL_MODELS, available_embedders, get_embedder
from app.retrieval.rerank import get_reranker
from evals.harness import build_configs, evaluate
from evals.schema import EvalPair, load_pairs

_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PAIRS = _ROOT / "evals" / "data" / "eval_pairs.jsonl"
DEFAULT_OUT = _ROOT / "benchmark" / "results" / "results.json"

CONFIG_ORDER = ("dense", "lexical", "hybrid", "hybrid+rerank")
# Configs that ignore config.model_key entirely — run once, reused across models.
MODEL_FREE_CONFIGS = frozenset({"lexical"})
SPLITS = ("all", "msa", "gulf", "msa_matched")
# The controlled comparison. Every one of the 50 answerable Gulf questions is a
# rephrasing of an MSA question against *identical* gold chunks, so "msa" (218
# pairs, every target chunk in the corpus) and "gulf" (50 pairs, 47 target chunk
# sets) are not comparable populations — a difference between them can be chunk
# difficulty rather than dialect. "msa_matched" is the MSA pairs that aim at the
# same gold chunks as a Gulf pair, which holds the target fixed and varies only
# the register of the question.
MATCHED_SPLIT = "msa_matched"
METRIC_KEYS = ("recall@3", "recall@10", "mrr", "hit@3", "hit@10")
HEADLINE_METRIC = "recall@10"


def _device() -> str:
    """Same rule as the embedder/reranker: Apple Silicon first, CPU otherwise."""
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is installed in this venv
        return "unknown"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _git_commit() -> str | None:
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


def _safe_url() -> str:
    """The DB the numbers came from, without the password."""
    url = settings.database_url
    if "@" not in url:
        return url
    scheme, _, rest = url.partition("://")
    creds, _, host = rest.partition("@")
    user = creds.split(":", 1)[0]
    return f"{scheme}://{user}:***@{host}"


def _gold(pair: EvalPair) -> tuple[str, ...]:
    return tuple(sorted(pair.source_chunk_ids))


def split_pairs(pairs: Sequence[EvalPair], split: str) -> list[EvalPair]:
    """``all`` is every pair; ``msa``/``gulf`` filter on the dialect tag.

    ``msa_matched`` is the control group described at :data:`MATCHED_SPLIT`:
    answerable MSA pairs whose gold chunks are also targeted by a Gulf pair.
    Unanswerable pairs are excluded from it — they have no gold chunks to match on.
    """
    if split == "all":
        return list(pairs)
    if split in ("msa", "gulf"):
        return [pair for pair in pairs if pair.dialect_tag == split]
    if split == MATCHED_SPLIT:
        gulf_gold = {_gold(p) for p in pairs if p.dialect_tag == "gulf" and p.answerable}
        return [
            pair
            for pair in pairs
            if pair.dialect_tag == "msa" and pair.answerable and _gold(pair) in gulf_gold
        ]
    raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")


def _cell(
    model: str | None,
    config_name: str,
    split: str,
    result: dict,
    shared_with: list[str] | None = None,
) -> dict:
    """One row of results.json: what was run, on what, and what came out."""
    cell = {
        "model": model,
        "config": config_name,
        "split": split,
        "n_pairs": result["n_pairs"],  # everything fed in, unanswerable included
        "n": result["n"],  # answerable pairs, the denominator of every metric
        **{key: result[key] for key in METRIC_KEYS},
        "latency": result["latency"],
        "unanswerable": result["unanswerable"],
        "by_doc": result["by_doc"],
    }
    if shared_with is not None:
        cell["shared_with"] = shared_with
        cell["note"] = (
            "lexical search never touches an embedding column; this single "
            "measurement applies unchanged to every model listed in shared_with"
        )
    return cell


async def _corpus_stats(session: AsyncSession) -> dict:
    chunks = await session.scalar(select(func.count()).select_from(Chunk))
    documents = await session.scalar(select(func.count(func.distinct(Chunk.doc_id))))
    embedded = {
        key: await session.scalar(select(func.count(getattr(Chunk, column))))
        for key, column in EMBEDDING_COLUMNS.items()
    }
    return {"chunks": chunks or 0, "documents": documents or 0, "embedded": embedded}


def _dataset_stats(pairs: Sequence[EvalPair], path: Path) -> dict:
    return {
        "path": str(path.relative_to(_ROOT)) if path.is_relative_to(_ROOT) else str(path),
        "n_pairs": len(pairs),
        "answerable": sum(1 for pair in pairs if pair.answerable),
        "unanswerable": sum(1 for pair in pairs if not pair.answerable),
        "msa": sum(1 for pair in pairs if pair.dialect_tag == "msa"),
        "gulf": sum(1 for pair in pairs if pair.dialect_tag == "gulf"),
    }


def _model_status(requested: Sequence[str], coverage: dict[str, int]) -> dict[str, dict]:
    """Why every known model was or was not measured. Silence is not an answer."""
    status: dict[str, dict] = {}
    for key in sorted(EMBEDDING_COLUMNS):
        if key in requested:
            status[key] = {
                "status": "run",
                "name": LOCAL_MODELS.get(key, key),
                "chunks_embedded": coverage.get(key, 0),
            }
        elif key not in available_embedders():
            status[key] = {"status": "not_run", "reason": "no API key"}
        elif not coverage.get(key):
            status[key] = {
                "status": "not_run",
                "reason": f"no vectors in the chunks table (python -m ingestion backfill --model {key})",
            }
        else:
            status[key] = {"status": "not_run", "reason": "not requested on the command line"}
    return status


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


async def run_matrix(
    pairs: Sequence[EvalPair], model_keys: Sequence[str], session: AsyncSession
) -> list[dict]:
    """Every (model, config, split) cell. Lexical is measured once and shared."""
    # An empty slice would come back as a full row of 0.0000, which reads exactly
    # like a total retrieval failure. Drop it instead — see --limit smoke runs.
    slices = {split: split_pairs(pairs, split) for split in SPLITS}
    splits = [split for split in SPLITS if slices[split]]
    for split in SPLITS:
        if split not in splits:
            _log(f"skipping split {split!r}: no pairs")

    reranker = get_reranker("bge")
    cells: list[dict] = []

    for config_name in MODEL_FREE_CONFIGS:
        config = build_configs()[config_name]
        for split in splits:
            started = time.perf_counter()
            result = await evaluate(slices[split], session, config, None)
            _log(
                f"  [-]  {config_name:<14} {split:<5} n={result['n']:>3} "
                f"{HEADLINE_METRIC}={result[HEADLINE_METRIC]:.4f}  "
                f"({time.perf_counter() - started:.1f}s)"
            )
            cells.append(_cell(None, config_name, split, result, shared_with=list(model_keys)))

    for model_key in model_keys:
        embedder = get_embedder(model_key)
        configs = build_configs(model_key)
        for config_name in CONFIG_ORDER:
            if config_name in MODEL_FREE_CONFIGS:
                continue
            config = configs[config_name]
            for split in splits:
                started = time.perf_counter()
                result = await evaluate(slices[split], session, config, embedder, reranker)
                _log(
                    f"  [{model_key}] {config_name:<14} {split:<5} n={result['n']:>3} "
                    f"{HEADLINE_METRIC}={result[HEADLINE_METRIC]:.4f}  "
                    f"({time.perf_counter() - started:.1f}s)"
                )
                cells.append(_cell(model_key, config_name, split, result))

    return cells


async def run_benchmark(
    pairs: Sequence[EvalPair],
    model_keys: Sequence[str],
    session: AsyncSession,
    pairs_path: Path,
    smoke: bool = False,
) -> dict:
    """The whole matrix plus the metadata needed to reproduce or distrust it."""
    corpus = await _corpus_stats(session)
    usable = [key for key in model_keys if corpus["embedded"].get(key)]
    for key in model_keys:
        if key not in usable:
            _log(f"skipping {key}: no vectors in the chunks table")

    started = time.perf_counter()
    cells = await run_matrix(pairs, usable, session)
    wall_clock = time.perf_counter() - started

    return {
        "metadata": {
            "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
            "git_commit": _git_commit(),
            "device": _device(),
            "database_url": _safe_url(),
            "corpus": corpus,
            "dataset": _dataset_stats(pairs, pairs_path),
            "models": _model_status(usable, corpus["embedded"]),
            "configs": list(CONFIG_ORDER),
            "splits": list(SPLITS),
            "wall_clock_s": round(wall_clock, 1),
            "smoke_run": smoke,
        },
        "results": cells,
    }


# --------------------------------------------------------------------------- #
# markdown rendering — docs/benchmark.md pastes this verbatim, never retypes it
# --------------------------------------------------------------------------- #


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _index(results: Sequence[dict]) -> dict[tuple[str, str, str], dict]:
    return {(cell["model"] or "-", cell["config"], cell["split"]): cell for cell in results}


def _model_labels(data: dict) -> list[str]:
    return [key for key, info in data["metadata"]["models"].items() if info["status"] == "run"]


def _dialect_table(data: dict) -> str:
    """The headline. Matched MSA vs Gulf is the controlled column; full MSA is context."""
    cells = _index(data["results"])
    rows = []
    for model in _model_labels(data):
        for config in CONFIG_ORDER:
            key_model = "-" if config in MODEL_FREE_CONFIGS else model
            msa = cells.get((key_model, config, "msa"))
            matched = cells.get((key_model, config, MATCHED_SPLIT))
            gulf = cells.get((key_model, config, "gulf"))
            if not (msa and matched and gulf):
                continue
            gap = matched[HEADLINE_METRIC] - gulf[HEADLINE_METRIC]
            relative = gap / matched[HEADLINE_METRIC] * 100 if matched[HEADLINE_METRIC] else 0.0
            label = f"{config}*" if config in MODEL_FREE_CONFIGS else config
            rows.append(
                [
                    model,
                    label,
                    f"{msa[HEADLINE_METRIC]:.3f} (n={msa['n']})",
                    f"{matched[HEADLINE_METRIC]:.3f} (n={matched['n']})",
                    f"{gulf[HEADLINE_METRIC]:.3f} (n={gulf['n']})",
                    f"{-gap:+.3f}",
                    f"{-relative:.0f}%",
                ]
            )
    return _table(
        [
            "model",
            "config",
            "MSA all",
            "MSA matched",
            "Gulf",
            "Gulf − MSA matched",
            "relative",
        ],
        rows,
    )


def render_headline(data: dict, model: str, config: str = "dense") -> str:
    """The three-row highlight the writeup opens with, for one model and config.

    Same source as every other table. It exists so the lead paragraph of
    docs/benchmark.md quotes generated numbers rather than transcribed ones.
    """
    cells = _index(data["results"])
    matched = cells.get((model, config, MATCHED_SPLIT))
    gulf = cells.get((model, config, "gulf"))
    if not (matched and gulf):
        raise ValueError(f"no {model}/{config} cells for the matched and gulf splits")

    rows = [
        [
            metric,
            f"{matched[metric]:.3f}",
            f"{gulf[metric]:.3f}",
            f"**{(gulf[metric] - matched[metric]) * 100:+.1f} pts**",
        ]
        for metric in ("recall@10", "recall@3", "mrr")
    ]
    return _table(
        [
            f"{model}, {config} retrieval",
            f"MSA matched (n={matched['n']})",
            f"Gulf (n={gulf['n']})",
            "change",
        ],
        rows,
    )


def _matrix_table(data: dict, model: str) -> str:
    cells = _index(data["results"])
    rows = []
    for config in CONFIG_ORDER:
        key_model = "-" if config in MODEL_FREE_CONFIGS else model
        for split in SPLITS:
            cell = cells.get((key_model, config, split))
            if not cell:
                continue
            label = f"{config}*" if config in MODEL_FREE_CONFIGS else config
            rows.append(
                [
                    label,
                    split,
                    str(cell["n"]),
                    f"{cell['recall@3']:.3f}",
                    f"{cell['recall@10']:.3f}",
                    f"{cell['hit@3']:.3f}",
                    f"{cell['hit@10']:.3f}",
                    f"{cell['mrr']:.3f}",
                    f"{cell['latency']['mean_ms']:.1f}",
                    f"{cell['latency']['p95_ms']:.1f}",
                ]
            )
    return _table(
        [
            "config",
            "split",
            "n",
            "recall@3",
            "recall@10",
            "hit@3",
            "hit@10",
            "MRR",
            "mean ms",
            "p95 ms",
        ],
        rows,
    )


def _not_measured_table(data: dict) -> str:
    rows = [
        [key, info["reason"]]
        for key, info in data["metadata"]["models"].items()
        if info["status"] != "run"
    ]
    return _table(["model", "why it was not measured"], rows)


def _refusal_table(data: dict) -> str:
    cells = _index(data["results"])
    rows = []
    for model in _model_labels(data):
        for config in CONFIG_ORDER:
            key_model = "-" if config in MODEL_FREE_CONFIGS else model
            cell = cells.get((key_model, config, "all"))
            if not cell:
                continue
            refusals = cell["unanswerable"]
            label = f"{config}*" if config in MODEL_FREE_CONFIGS else config
            rows.append(
                [
                    model,
                    label,
                    str(refusals["n"]),
                    str(refusals["with_hits"]),
                    f"{refusals['top_score_mean']:.4f}",
                    f"{refusals['top_score_max']:.4f}",
                ]
            )
    return _table(
        [
            "model",
            "config",
            "unanswerable pairs",
            "returned hits",
            "mean top score",
            "max top score",
        ],
        rows,
    )


def render_tables(data: dict) -> str:
    """Every table in docs/benchmark.md, straight out of results.json."""
    meta = data["metadata"]
    dataset, corpus = meta["dataset"], meta["corpus"]
    header = (
        f"Run {meta['timestamp']} · commit `{meta['git_commit']}` · device `{meta['device']}` · "
        f"{corpus['chunks']} chunks / {corpus['documents']} documents · "
        f"{dataset['n_pairs']} pairs ({dataset['answerable']} answerable, "
        f"{dataset['msa']} MSA / {dataset['gulf']} Gulf) · "
        f"matrix wall clock {meta['wall_clock_s']}s"
    )
    gap_note = (
        "**MSA matched** is the control: the MSA pairs whose gold chunks are also targeted by a "
        "Gulf question, so the retrieval target is held fixed and only the register of the "
        "question changes. **MSA all** is every MSA pair and is *not* a like-for-like comparison. "
        "The last two columns are Gulf minus MSA-matched, in absolute recall@10 points and as a "
        "share of the matched score; negative means Gulf-dialect questions retrieve worse."
    )
    shared_note = (
        "`*` marks the lexical rows: full-text search uses no embedding model, so the same single "
        "measurement is repeated under each model for reading convenience. It is one run, not two."
    )
    refusal_note = (
        "Retrieval cannot refuse — it always returns its top-k — so what is measured is the score "
        "each config puts on its best *wrong* answer. A usable 'not in corpus' threshold has to "
        "separate this distribution from the answerable one."
    )

    parts = [
        "<!-- generated by `python -m benchmark.run --tables`; do not edit by hand -->",
        "",
        header,
        "",
        "### The dialect penalty",
        "",
        _dialect_table(data),
        "",
        gap_note,
        "",
        "### Full matrix",
        "",
    ]
    for model in _model_labels(data):
        parts += [
            f"**{model} — {meta['models'][model]['name']}**",
            "",
            _matrix_table(data, model),
            "",
        ]
    parts += [
        shared_note,
        "",
        "### Unanswerable questions (refusal signal)",
        "",
        _refusal_table(data),
        "",
        refusal_note,
        "",
        "### Not measured",
        "",
        _not_measured_table(data),
        "",
    ]
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _parse_models(raw: str | None) -> list[str]:
    if not raw:
        return [key for key in available_embedders()]
    keys = [key.strip() for key in raw.split(",") if key.strip()]
    unknown = [key for key in keys if key not in EMBEDDING_COLUMNS]
    if unknown:
        raise ValueError(f"unknown model key(s) {unknown}; expected {sorted(EMBEDDING_COLUMNS)}")
    return keys


async def _main(args: argparse.Namespace) -> int:
    from app.db import SessionLocal, engine

    pairs = load_pairs(args.pairs)
    if args.limit:
        pairs = pairs[: args.limit]
    model_keys = _parse_models(args.models)

    _log(f"benchmark: {len(pairs)} pairs × {len(CONFIG_ORDER)} configs × {len(SPLITS)} splits")
    _log(f"models: {model_keys or '(none)'}  device: {_device()}  db: {_safe_url()}")

    try:
        async with SessionLocal() as session:
            data = await run_benchmark(pairs, model_keys, session, args.pairs, smoke=bool(args.limit))
    finally:
        await engine.dispose()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(_merge_results(args.out, data), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _log(f"wrote {args.out} ({len(data['results'])} cells, {data['metadata']['wall_clock_s']}s)")
    return 0


def _merge_results(path: Path, data: dict) -> dict:
    """New run's ``metadata``/``results`` over whatever else the file already holds.

    A wholesale ``write_text`` destroyed the top-level ``planning`` section: the
    Gulf→MSA ablation and its depth-sensitivity tables are maintained by hand,
    no code in this repo regenerates them, and ``app/service.py``,
    ``app/planning/planner.py`` and the README all cite them as the provenance
    for the shipped planner defaults. Running the documented benchmark command
    once silently deleted 60+ lines of measured data.

    ``evals.gate.write_baseline`` does the same read-modify-write for the same
    reason; this is that pattern, applied where it was missing.
    """
    if not path.exists():
        return data
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        # Never silently: an unreadable file here means the merge cannot protect
        # anything, and the caller deserves to know before the write lands.
        _log(f"warning: could not read {path} to merge ({exc}); writing fresh")
        return data
    if not isinstance(existing, dict):
        return data
    preserved = sorted(set(existing) - set(data))
    if preserved:
        _log(f"preserved hand-maintained sections: {', '.join(preserved)}")
    return {**existing, **data}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m benchmark.run",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--models", help="comma-separated model keys (default: every usable one)")
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, help="smoke run: only the first N pairs")
    parser.add_argument(
        "--tables",
        action="store_true",
        help="skip the benchmark; print markdown tables from an existing --out file",
    )
    parser.add_argument(
        "--headline",
        metavar="MODEL:CONFIG",
        help="skip the benchmark; print the MSA-matched vs Gulf highlight table (e.g. e5:dense)",
    )
    args = parser.parse_args(argv)

    if args.tables or args.headline:
        data: dict[str, Any] = json.loads(args.out.read_text(encoding="utf-8"))
        if args.headline:
            model, _, config = args.headline.partition(":")
            print(render_headline(data, model, config or "dense"))
        else:
            print(render_tables(data))
        return 0

    try:
        return asyncio.run(_main(args))
    except ValueError as exc:
        # Unknown model key, missing API key, malformed dataset: one line, no traceback.
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
