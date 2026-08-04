"""Eval harness: run the labelled pairs through one retrieval configuration, score it.

One configuration in, one dict of numbers out. The harness owns three things the
raw metrics in :mod:`evals.metrics` deliberately do not:

1. **Batched query embedding.** Every question is embedded in a single
   ``embed_queries`` call before any search runs. Per-query embedding of 283
   questions on a local model costs minutes; one batch costs seconds.
2. **A ``by_doc`` breakdown.** 239 of the 283 pairs come from the labour law, so
   an aggregate recall number can hide a total regression on a 6-pair document.
   The dataset audit flagged exactly this, so the split is not optional.
3. **Refusal accounting.** ``recall_at_k`` raises on an unanswerable pair by
   design. Those 15 pairs are routed here to their own line instead, carrying the
   top retrieval score, which is what a later "not in corpus" threshold gets
   picked from.

Determinism: everything outside the ``"latency"`` key is a pure function of the
pairs, the corpus and the config — searches tie-break on chunk id, dialects and
documents are iterated in sorted order. A rerun must produce the same numbers.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.data import Hit
from app.retrieval.embed import Embedder
from app.retrieval.rerank import Reranker
from app.retrieval.search import dense_search, hybrid_search, lexical_search
from evals.metrics import QueryResult, aggregate, recall_at_k
from evals.schema import EvalPair

DEFAULT_MODEL_KEY = "e5"
DEFAULT_TOP_K = 20
BY_DOC_K = 10  # the CI gate metric, so the per-document split watches the same k
LATENCY_PERCENTILE = 95.0
_ROUND_DIGITS = 4

# The rerank ablation keeps 10, not the service's 5: recall@10 over a 5-item list
# measures the truncation, not the model, and the whole point of the config table
# is that one column is comparable down the rows. Production top-5 behaviour is
# what recall@3 tracks.
RERANK_TOP_K = 10


@dataclass(frozen=True)
class RetrievalConfig:
    """One row of the ablation table. ``model_key`` is ignored when ``dense`` is False."""

    name: str
    model_key: str
    dense: bool
    lexical: bool
    rerank: bool
    top_k: int = DEFAULT_TOP_K
    rerank_top_k: int = 5

    def __post_init__(self) -> None:
        if not self.dense and not self.lexical:
            raise ValueError(f"config {self.name!r} retrieves nothing: enable dense, lexical or both")
        if self.top_k < 1:
            raise ValueError(f"config {self.name!r}: top_k must be >= 1, got {self.top_k}")
        if self.rerank_top_k < 1:
            raise ValueError(
                f"config {self.name!r}: rerank_top_k must be >= 1, got {self.rerank_top_k}"
            )


def build_configs(model_key: str = DEFAULT_MODEL_KEY) -> dict[str, RetrievalConfig]:
    """The four standard ablations for one embedding model."""
    return {
        "dense": RetrievalConfig("dense", model_key, dense=True, lexical=False, rerank=False),
        "lexical": RetrievalConfig("lexical", model_key, dense=False, lexical=True, rerank=False),
        "hybrid": RetrievalConfig("hybrid", model_key, dense=True, lexical=True, rerank=False),
        "hybrid+rerank": RetrievalConfig(
            "hybrid+rerank",
            model_key,
            dense=True,
            lexical=True,
            rerank=True,
            rerank_top_k=RERANK_TOP_K,
        ),
    }


CONFIGS: dict[str, RetrievalConfig] = build_configs()


@dataclass(frozen=True)
class _Measured:
    """Per-pair bookkeeping evaluate() needs and QueryResult has no room for."""

    result: QueryResult
    source_doc: str
    elapsed_s: float
    top_score: float | None  # None when the config returned no hits at all


def _check_inputs(
    config: RetrievalConfig, embedder: Embedder | None, reranker: Reranker | None
) -> None:
    """Fail before embedding 283 questions, not after."""
    if config.dense:
        if embedder is None:
            raise ValueError(f"config {config.name!r} is dense: an embedder is required")
        if embedder.model_key != config.model_key:
            raise ValueError(
                f"config {config.name!r} searches the {config.model_key!r} column but the "
                f"embedder produces {embedder.model_key!r} vectors"
            )
    if config.rerank and reranker is None:
        raise ValueError(f"config {config.name!r} reranks: a reranker is required")


async def _search(
    session: AsyncSession,
    config: RetrievalConfig,
    query: str,
    vector: Sequence[float] | None,
    reranker: Reranker | None,
) -> list[Hit]:
    if config.dense and config.lexical:
        hits = await hybrid_search(session, query, vector, config.model_key, limit=config.top_k)
    elif config.dense:
        hits = await dense_search(session, vector, config.model_key, limit=config.top_k)
    else:
        hits = await lexical_search(session, query, limit=config.top_k)

    if config.rerank:
        hits = await reranker.rerank(query, hits, top_k=config.rerank_top_k)
    return hits


async def _measure(
    pairs: Sequence[EvalPair],
    session: AsyncSession,
    config: RetrievalConfig,
    embedder: Embedder | None,
    reranker: Reranker | None,
) -> tuple[list[_Measured], float]:
    """Run every pair once. Returns the per-pair records and the batch-embed seconds."""
    _check_inputs(config, embedder, reranker)

    embed_s = 0.0
    vectors: list[Sequence[float] | None] = [None] * len(pairs)
    if config.dense:
        started = time.perf_counter()
        # One call for the whole dataset — see the module docstring.
        vectors = list(await embedder.embed_queries([pair.question for pair in pairs]))
        embed_s = time.perf_counter() - started

    measured: list[_Measured] = []
    for pair, vector in zip(pairs, vectors):
        started = time.perf_counter()
        hits = await _search(session, config, pair.question, vector, reranker)
        elapsed = time.perf_counter() - started
        measured.append(
            _Measured(
                result=QueryResult(
                    pair_id=pair.id,
                    dialect_tag=pair.dialect_tag,
                    retrieved=[hit.chunk_id for hit in hits],
                    relevant=list(pair.source_chunk_ids),
                ),
                source_doc=pair.source_doc,
                elapsed_s=elapsed,
                top_score=hits[0].score if hits else None,
            )
        )
    return measured, embed_s


async def run_config(
    pairs: Sequence[EvalPair],
    session: AsyncSession,
    config: RetrievalConfig,
    embedder: Embedder | None,
    reranker: Reranker | None = None,
) -> list[QueryResult]:
    """Retrieve for every pair (answerable or not) under one configuration.

    Unanswerable pairs come back with an empty ``relevant`` list — ``aggregate``
    skips them and ``evaluate`` counts them separately.
    """
    measured, _ = await _measure(pairs, session, config, embedder, reranker)
    return [item.result for item in measured]


def _percentile(values: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile. trade-off: no interpolation, n is a few hundred."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = math.ceil(pct / 100 * len(ordered)) - 1
    return ordered[min(max(index, 0), len(ordered) - 1)]


def _by_doc(measured: Sequence[_Measured], k: int = BY_DOC_K) -> dict[str, dict]:
    """recall@k per source_doc, answerable pairs only, sorted for reproducible output."""
    scorable = [item for item in measured if item.result.relevant]
    breakdown: dict[str, dict] = {}
    for doc in sorted({item.source_doc for item in scorable}):
        recalls = [
            recall_at_k(item.result.retrieved, item.result.relevant, k)
            for item in scorable
            if item.source_doc == doc
        ]
        breakdown[doc] = {
            "n": len(recalls),
            f"recall@{k}": round(sum(recalls) / len(recalls), _ROUND_DIGITS),
        }
    return breakdown


def _refusals(measured: Sequence[_Measured]) -> dict:
    """The unanswerable pairs, which have no defined recall.

    Retrieval cannot refuse — it always returns its top-k — so what is tracked
    here is the *score* the config puts on its best wrong answer. That is the
    distribution ``settings.rerank_min_score`` has to separate from the
    answerable one; the refusal decision itself lives in generation.
    """
    unanswerable = [item for item in measured if not item.result.relevant]
    scores = [item.top_score for item in unanswerable if item.top_score is not None]
    return {
        "n": len(unanswerable),
        "with_hits": len(scores),
        "top_score_mean": round(sum(scores) / len(scores), _ROUND_DIGITS) if scores else 0.0,
        "top_score_max": round(max(scores), _ROUND_DIGITS) if scores else 0.0,
    }


async def evaluate(
    pairs: Sequence[EvalPair],
    session: AsyncSession,
    config: RetrievalConfig,
    embedder: Embedder | None,
    reranker: Reranker | None = None,
) -> dict:
    """Score one configuration over the dataset.

    ``aggregate()``'s output (overall + ``by_dialect``) plus ``by_doc``, the
    refusal line and wall-clock latency. Everything except ``"latency"`` is
    deterministic.
    """
    measured, embed_s = await _measure(pairs, session, config, embedder, reranker)
    elapsed = [item.elapsed_s for item in measured]

    return {
        "config": config.name,
        "model_key": config.model_key if config.dense else None,
        "top_k": config.top_k,
        "rerank_top_k": config.rerank_top_k if config.rerank else None,
        "n_pairs": len(measured),
        **aggregate([item.result for item in measured]),
        "by_doc": _by_doc(measured),
        "unanswerable": _refusals(measured),
        # Retrieval only: query embedding is batched away above, so it is reported
        # separately rather than smeared over the per-query numbers. The service
        # embeds one query at a time, so a request budget = per-query embed + this.
        "latency": {
            "mean_ms": round(sum(elapsed) / len(elapsed) * 1000, 2) if elapsed else 0.0,
            "p95_ms": round(_percentile(elapsed, LATENCY_PERCENTILE) * 1000, 2),
            "embed_batch_ms": round(embed_s * 1000, 2),
        },
    }


_METRIC_COLUMNS = ("recall@3", "recall@10", "hit@3", "hit@10", "mrr")


def _metric_row(label: str, summary: dict) -> str:
    cells = "".join(f"{summary.get(name, 0.0):>10.4f}" for name in _METRIC_COLUMNS)
    return f"  {label:<12}{summary.get('n', 0):>5}{cells}"


def report(results: dict) -> str:
    """Render :func:`evaluate` output as a terminal table."""
    latency = results.get("latency", {})
    refusals = results.get("unanswerable", {})
    model = results.get("model_key") or "-"

    lines = [
        (
            f"config: {results.get('config', '?')}   model: {model}   "
            f"top_k: {results.get('top_k', '?')}   pairs: {results.get('n_pairs', 0)}"
        ),
        "",
        f"  {'split':<12}{'n':>5}" + "".join(f"{name:>10}" for name in _METRIC_COLUMNS),
        _metric_row("overall", results),
    ]
    lines.extend(
        _metric_row(tag, summary) for tag, summary in results.get("by_dialect", {}).items()
    )

    lines.append("")
    lines.append(f"  by source_doc (recall@{BY_DOC_K}):")
    for doc, summary in results.get("by_doc", {}).items():
        lines.append(f"    {summary['n']:>5}{summary[f'recall@{BY_DOC_K}']:>10.4f}  {doc}")

    lines.append("")
    lines.append(
        f"  unanswerable: {refusals.get('n', 0)} pairs, {refusals.get('with_hits', 0)} with hits, "
        f"top score mean {refusals.get('top_score_mean', 0.0):.4f} / "
        f"max {refusals.get('top_score_max', 0.0):.4f}"
    )
    lines.append(
        f"  latency: mean {latency.get('mean_ms', 0.0):.2f} ms, "
        f"p95 {latency.get('p95_ms', 0.0):.2f} ms per query "
        f"(query embedding {latency.get('embed_batch_ms', 0.0):.2f} ms, batched)"
    )
    return "\n".join(lines) + "\n"
