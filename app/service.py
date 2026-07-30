"""The /ask pipeline, assembled — orchestration that a test can run without HTTP.

Everything the route needs happens here, so the route is a formatter and nothing
else. Two entry points over one pipeline:

``answer``  one :class:`Answer`.
``stream``  ``(event_name, payload)`` tuples. Transport-agnostic on purpose:
            the route turns them into SSE frames, the eval harness could turn
            them into anything else.

Stage order, each in its own OTel span::

    plan → embed → cache.lookup → retrieve → fuse → rerank → generate

Three of those stages can end the request before an LLM is ever called, and that
is the point of the design rather than an optimisation:

1. **Cache hit** — a semantically equivalent question was already answered
   (:mod:`app.retrieval.cache` owns the negation/digit guards that make that
   safe). Returns immediately, ``cached=True``, provider untouched.
2. **Refusal gate** — retrieval came back empty, or the top rerank score is below
   ``settings.rerank_min_score``. Returns "not in corpus" *in the user's
   register*, ``refused=True``, provider untouched. The score half of that gate
   is **off in the shipped config** (``rerank_min_score = 0.0``): swept over all
   283 eval pairs the cross-encoder's top score does not separate answerable
   from unanswerable well enough to gate on — see docs/refusal-calibration.md
   and :meth:`RagService._is_refusal`.
3. **Spend cap** — the estimate is *reserved* before the call and settled against
   the real cost after it, so concurrent requests cannot all pass one cap with
   room for one (:meth:`app.observability.cost.SpendTracker.reserve`).

**Multi-query retrieval.** ``RuleBasedPlanner`` returns both the user's original
question and its MSA rewrite; each is retrieved with, and the ranked lists are
fused with RRF. That is exactly the "fused" arm of the phase-3 measurement
(50 answerable Gulf pairs, dense, real corpus).

The ablation was run at two per-leg retrieval depths and the fused arm is
depth-sensitive, so the depth has to be named or the row is unreadable. **The
service retrieves ``settings.top_k_retrieve = 20`` per query**, so these are the
rows it actually runs::

    per-leg depth 20 (SHIPPED — settings.top_k_retrieve)
    bge   raw 0.79 / 0.93 / MRR 0.7444    fused 0.82 / 0.93 / 0.7722   (r@3/r@10)
    e5    raw 0.75 / 0.85 / MRR 0.6172    fused 0.76 / 0.89 / 0.6497

At per-leg depth 10 the fused arm gains more — bge 0.82 / 0.95 / 0.7737 and
e5 0.76 / 0.92 / 0.6512 — because RRF's top 10 is not yet diluted by deep
candidates from both legs. Same code, same corpus, same 50 pairs; only the depth
differs. Both tables are in ``benchmark/results/results.json`` under ``planning``
(``per_leg_depth`` and ``depth_sensitivity.per_leg_depth_20``), and
:class:`app.planning.planner.RuleBasedPlanner` quotes the depth-20 table with the
full arm list.

The rewrite is *added*, never substituted, so a bad lexicon entry can only
introduce candidates that rank badly — it can never remove a chunk the original
question would have found.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.generation.base import Provider, ProviderError, Usage, usage_cost_usd
from app.generation.budget import (
    NOT_IN_CORPUS,
    build_prompt,
    estimate_tokens,
    fit_context,
)
from app.generation.failover import AllProvidersFailed
from app.lib.sources import source_url
from app.models.chunks import Chunk
from app.models.query_cache import CACHE_DIM
from app.observability.cost import SpendTracker
from app.observability.tracing import (
    record_cache_lookup,
    record_llm_call,
    record_retrieval,
    span,
)
from app.planning.dialect import GULF
from app.planning.planner import Plan, Planner
from app.retrieval.cache import CachedAnswer, lookup, store
from app.retrieval.embed import Embedder
from app.retrieval.rerank import RERANK_SOURCE, Reranker
from app.retrieval.search import (
    Hit,
    dense_search,
    hybrid_search,
    lexical_search,
    rrf_fuse,
)

#: Retrieval configurations `/ask` accepts. Same names as the eval ablation
#: table (``evals.harness.build_configs``) so a request and a benchmark row mean
#: the same thing.
CONFIGS: tuple[str, ...] = ("dense", "lexical", "hybrid", "hybrid+rerank")
DEFAULT_CONFIG = "hybrid+rerank"

ANSWER_MAX_TOKENS = 1024
EXCERPT_CHARS = 240

# Event names yielded by :meth:`RagService.stream`.
EVENT_CITATIONS = "citations"
EVENT_TOKEN = "token"
EVENT_FINAL = "final"
EVENT_ERROR = "error"
EVENT_DONE = "done"

# The refusal, per register. Answering a Gulf question in MSA is a jarring
# register switch at exactly the moment the service is admitting it cannot help.
REFUSALS: dict[str, str] = {
    "msa": NOT_IN_CORPUS,
    GULF: "المواد المتوفرة ما فيها جواب عن هذا السؤال.",
}


def _is_refusal_text(text: str) -> bool:
    """Whether a generated answer is the model declining, in either register.

    Matched on the prompt's own refusal strings (``build_prompt`` instructs the
    model to reply with exactly ``NOT_IN_CORPUS``), because the score gate is no
    longer the live refusal path — see :meth:`RagService._store`.

    # trade-off: exact-ish string matching, so a model that paraphrases its
    # refusal is not recognised and gets cached. That is the same failure the
    # unmeasured-abstention gap in the README already names; the guard closes the
    # deterministic case. Upgrade path: have the model emit a structured
    # refusal flag rather than a sentence.
    """
    stripped = text.strip()
    return any(
        stripped.startswith(refusal.rstrip(".")) for refusal in REFUSALS.values()
    )


@dataclass(frozen=True)
class Citation:
    """One chunk the answer is allowed to rest on.

    ``score`` is the score of the last stage that ranked it — a 0-1
    cross-encoder value after rerank, an RRF or cosine score otherwise — so it is
    only comparable within one response.
    """

    chunk_id: str
    doc_id: str
    article: str | None
    score: float
    excerpt: str
    #: The law's page on the source portal, from the corpus manifest — what lets a
    #: reader check the excerpt instead of trusting it. None for a document that
    #: arrived over POST /ingest, which has no manifest entry (app/lib/sources.py).
    source_url: str | None = None


@dataclass(frozen=True)
class Answer:
    text: str
    citations: list[Citation]
    register: str  # "gulf" | "msa"
    cached: bool
    refused: bool
    usage: Usage | None  # None on a cache hit and on a refusal — nothing was billed
    stages: dict[str, float]  # per-stage wall time in milliseconds, plus "total"


@dataclass(frozen=True)
class _Prepared:
    """Everything the pipeline decided before generation. Internal."""

    plan: Plan
    citations: list[Citation]
    system: str
    user: str
    query_vec: list[float] | None
    cached: CachedAnswer | None
    refused: bool
    config: str = DEFAULT_CONFIG
    #: USD held against the daily cap by ``_reserve_spend``; 0.0 on the cached
    #: and refused paths, which never reach a provider. Handed to
    #: ``SpendTracker.settle`` exactly once, in a ``finally``.
    reserved_usd: float = 0.0
    stages: dict[str, float] = field(default_factory=dict)


@asynccontextmanager
async def _stage(
    stages: dict[str, float], name: str, **attributes: Any
) -> AsyncIterator[Any]:
    """One pipeline stage: an OTel span plus a millisecond entry in ``stages``.

    The timing is recorded in ``finally``, so a stage that raised still shows up
    in the per-stage breakdown of the request that failed.
    """
    started = time.perf_counter()
    try:
        async with span(name, **attributes) as current:
            yield current
    finally:
        stages[name] = round((time.perf_counter() - started) * 1000, 2)


def _excerpt(text: str) -> str:
    if len(text) <= EXCERPT_CHARS:
        return text
    return text[:EXCERPT_CHARS].rstrip() + "…"


def _citation(hit: Hit) -> Citation:
    return Citation(
        chunk_id=hit.chunk_id,
        doc_id=hit.doc_id,
        article=hit.article,
        score=round(hit.score, 6),
        excerpt=_excerpt(hit.text),
        source_url=source_url(hit.doc_id),
    )


def citation_payload(citations: Sequence[Citation]) -> list[dict]:
    return [
        {
            "chunk_id": citation.chunk_id,
            "doc_id": citation.doc_id,
            "article": citation.article,
            "score": citation.score,
            "excerpt": citation.excerpt,
            "source_url": citation.source_url,
        }
        for citation in citations
    ]


def error_payload(failure: AllProvidersFailed | ProviderError) -> dict:
    """A generation failure as data, for the one place a status code is gone.

    ``AllProvidersFailed`` already carries every attempt; a bare
    ``ProviderError`` is a single provider that failed *without* failing over
    (a 400 or a 401 — see app/generation/base.py), so it becomes one attempt.
    """
    if isinstance(failure, AllProvidersFailed):
        attempts = [
            {"provider": name, "reason": reason} for name, reason in failure.failures
        ]
    else:
        attempts = [{"provider": failure.provider, "reason": failure.reason}]
    return {"message": str(failure), "attempts": attempts}


def usage_payload(usage: Usage | None) -> dict | None:
    if usage is None:
        return None
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cost_usd": usage.cost_usd,
    }


def _cited_scores(cited: Sequence[object]) -> list[tuple[str, float]]:
    """``query_cache.citations`` as ``(chunk_id, score)``, in stored order.

    Two shapes live in that JSONB column. Entries written now are
    ``{"chunk_id", "score"}``; entries written before scores were stored are bare
    id strings, and there is no TTL or invalidation (see :func:`cache.store`), so
    they stay readable rather than being dropped or crashing the hit. A legacy
    entry scores 0.0 — the same value it has always reported.
    """
    pairs: list[tuple[str, float]] = []
    for entry in cited:
        if isinstance(entry, str):
            pairs.append((entry, 0.0))
        elif isinstance(entry, dict) and isinstance(entry.get("chunk_id"), str):
            score = entry.get("score")
            pairs.append(
                (
                    entry["chunk_id"],
                    float(score) if isinstance(score, int | float) else 0.0,
                )
            )
        # Anything else is a row this version did not write and cannot read; skip
        # it rather than fail the cache hit over one malformed citation.
    return pairs


async def _hydrate_citations(
    session: AsyncSession, cited: Sequence[object]
) -> list[Citation]:
    """Rebuild citations for a cache hit from what the cache stored.

    One indexed primary-key lookup. The alternative — returning bare ids — would
    make a cached response visibly worse than a generated one in any UI, which is
    also why the rerank score is stored and replayed rather than zeroed: a cached
    answer and a freshly generated one now render identically.
    """
    pairs = _cited_scores(cited)
    if not pairs:
        return []
    rows = (
        await session.execute(
            select(Chunk.id, Chunk.doc_id, Chunk.article, Chunk.text).where(
                Chunk.id.in_([chunk_id for chunk_id, _ in pairs])
            )
        )
    ).all()
    by_id = {row.id: row for row in rows}
    return [
        Citation(
            chunk_id=chunk_id,
            doc_id=by_id[chunk_id].doc_id,
            article=by_id[chunk_id].article,
            score=score,
            excerpt=_excerpt(by_id[chunk_id].text),
            source_url=source_url(by_id[chunk_id].doc_id),
        )
        for chunk_id, score in pairs
        if chunk_id in by_id
    ]


class RagService:
    """The pipeline. Every collaborator is injected, so tests need no weights.

    ``settings`` is a parameter rather than a module import for the same reason:
    the refusal threshold, the token budget and the cache switch are all things a
    test has to vary without touching a global.
    """

    def __init__(
        self,
        embedder: Embedder,
        reranker: Reranker,
        planner: Planner,
        provider: Provider,
        spend_tracker: SpendTracker,
        settings: Settings,
    ) -> None:
        self.embedder = embedder
        self.reranker = reranker
        self.planner = planner
        self.provider = provider
        self.spend = spend_tracker
        self.settings = settings
        # query_cache holds one vector width (see app/models/query_cache.py), so
        # a 3072-dim OpenAI embedder is un-cacheable rather than silently wrong.
        self._cache_enabled = bool(
            getattr(settings, "semantic_cache_enabled", False)
            and getattr(embedder, "dim", 0) == CACHE_DIM
        )

    # -- public ------------------------------------------------------------

    async def answer(
        self, session: AsyncSession, question: str, config: str = DEFAULT_CONFIG
    ) -> Answer:
        """Run the whole pipeline and return one finished answer."""
        started = time.perf_counter()
        prepared = await self._prepare(session, question, config)

        if prepared.cached is not None:
            return self._finish(prepared, prepared.cached.answer, None, started)
        if prepared.refused:
            return self._finish(prepared, self._refusal(prepared.plan), None, started)

        settled = False
        try:
            async with _stage(prepared.stages, "generate") as generate_span:
                completion = await self.provider.complete(
                    prepared.system,
                    [{"role": "user", "content": prepared.user}],
                    max_tokens=ANSWER_MAX_TOKENS,
                )
                self._record_usage(
                    generate_span,
                    completion.usage,
                    prepared.reserved_usd,
                    provider=completion.provider,
                    model=completion.model,
                )
                settled = True
        finally:
            # A provider failure leaves the reservation held otherwise, and the
            # cap would drift shut over a run of errors that cost nothing.
            if not settled:
                self.spend.settle(prepared.reserved_usd, 0.0)

        await self._store(session, prepared, completion.text)
        return self._finish(prepared, completion.text, completion.usage, started)

    async def stream(
        self, session: AsyncSession, question: str, config: str = DEFAULT_CONFIG
    ) -> AsyncIterator[tuple[str, dict]]:
        """Yield ``(event, payload)``: citations first, then tokens, then final, done.

        Citations lead so a UI can render its sources while the first token is
        still in flight — which on a cold provider is most of the wall time.

        Errors split by *when* they happen, because that is what the transport
        can still do about them. Anything raised before the first event (spend
        cap, a bad config, retrieval failure) propagates, so the route can still
        pick a status code. A generation failure lands *after* the citations
        frame is already on the wire, where no status code is available any more,
        so it arrives as an ``error`` event instead.
        """
        started = time.perf_counter()
        prepared = await self._prepare(session, question, config)

        yield (
            EVENT_CITATIONS,
            {
                "citations": citation_payload(prepared.citations),
                "register": prepared.plan.register,
                "cached": prepared.cached is not None,
                "refused": prepared.refused,
            },
        )

        if prepared.cached is not None or prepared.refused:
            text = (
                prepared.cached.answer
                if prepared.cached is not None
                else self._refusal(prepared.plan)
            )
            yield EVENT_TOKEN, {"text": text}
            yield EVENT_FINAL, self._final_payload(prepared, None, started)
            yield EVENT_DONE, {}
            return

        chunks: list[str] = []
        recorded: list[Usage] = []
        settled = False
        try:
            async with _stage(prepared.stages, "generate") as generate_span:
                try:
                    async for chunk in self.provider.stream(
                        prepared.system,
                        [{"role": "user", "content": prepared.user}],
                        ANSWER_MAX_TOKENS,
                        on_usage=recorded.append,
                    ):
                        chunks.append(chunk)
                        yield EVENT_TOKEN, {"text": chunk}
                except (AllProvidersFailed, ProviderError) as failure:
                    yield EVENT_ERROR, error_payload(failure)
                    yield EVENT_DONE, {}
                    return
                usage = recorded[0] if recorded else None
                self._record_usage(
                    generate_span,
                    usage,
                    prepared.reserved_usd,
                    # From the usage record, not from `self.provider`: that handle
                    # is the FailoverProvider wrapper, so its `.name` is
                    # "failover:anthropic,gemini" and its `.model` is the
                    # *primary's* even when the backup served this request. The
                    # non-streaming path already reads Completion.provider/model
                    # for exactly this reason.
                    provider=(usage.provider if usage else None) or self.provider.name,
                    model=(usage.model if usage else None) or self.provider.model,
                )
                settled = True
        finally:
            # The load-bearing `finally` on this path. A browser closing an SSE
            # tab throws GeneratorExit at the `yield` above, which used to skip
            # the accounting entirely: the vendor had billed real tokens and the
            # tracker recorded $0, so repeat-disconnect traffic made the daily cap
            # a no-op. Whatever the provider already reported is settled here.
            if not settled:
                spent = recorded[0].cost_usd if recorded else 0.0
                self.spend.settle(prepared.reserved_usd, spent)

        await self._store(session, prepared, "".join(chunks))
        yield EVENT_FINAL, self._final_payload(prepared, usage, started)
        yield EVENT_DONE, {}

    # -- pipeline ----------------------------------------------------------

    async def _prepare(
        self, session: AsyncSession, question: str, config: str
    ) -> _Prepared:
        """plan → embed → cache → retrieve → fuse → rerank → gate → budget → cap."""
        question = question.strip()
        if not question:
            raise ValueError("question must not be empty")
        if config not in CONFIGS:
            raise ValueError(
                f"unknown retrieval config {config!r}; expected one of {list(CONFIGS)}"
            )

        stages: dict[str, float] = {}
        async with _stage(
            stages, "plan", **{"app.retrieval.config": config}
        ) as planned:
            plan = await self.planner.plan(question)
            planned.set_attribute("app.planning.strategy", plan.strategy)
            planned.set_attribute("app.planning.register", plan.register)
            planned.set_attribute("app.planning.queries", len(plan.search_queries))

        vectors = await self._embed(plan, config, stages)
        query_vec = vectors.get(plan.original)

        cached = await self._lookup_cache(session, plan, query_vec, config, stages)
        if cached is not None:
            return _Prepared(
                plan=plan,
                citations=await _hydrate_citations(session, cached.citations),
                system="",
                user="",
                query_vec=query_vec,
                cached=cached,
                refused=False,
                config=config,
                stages=stages,
            )

        ranked = await self._retrieve(session, plan, config, vectors, stages)
        hits = await self._rerank(plan, config, ranked, stages)

        if self._is_refusal(hits):
            record_cache_lookup(hit=False)
            return _Prepared(
                plan=plan,
                citations=[],
                system="",
                user="",
                query_vec=query_vec,
                cached=None,
                refused=True,
                config=config,
                stages=stages,
            )

        kept, _ = fit_context(hits, self.settings.max_context_tokens)
        system, user = build_prompt(plan.original, kept)
        reserved = self._reserve_spend(system, user)
        return _Prepared(
            plan=plan,
            citations=[_citation(hit) for hit in kept],
            system=system,
            user=user,
            query_vec=query_vec,
            cached=None,
            refused=False,
            config=config,
            reserved_usd=reserved,
            stages=stages,
        )

    async def _embed(
        self, plan: Plan, config: str, stages: dict[str, float]
    ) -> dict[str, list[float]]:
        """Embed every query string once, in one batched call.

        The original is embedded first and unconditionally: it is the cache key,
        so a lexical-only request still pays one embedding when the cache is on.
        """
        if config == "lexical" and not self._cache_enabled:
            return {}
        # dict.fromkeys: original first, order preserved, duplicates dropped.
        texts = list(dict.fromkeys([plan.original, *plan.search_queries]))
        async with _stage(stages, "embed"):
            encoded = await self.embedder.embed_queries(texts)
        return dict(zip(texts, encoded))

    async def _lookup_cache(
        self,
        session: AsyncSession,
        plan: Plan,
        query_vec: list[float] | None,
        config: str,
        stages: dict[str, float],
    ) -> CachedAnswer | None:
        if not self._cache_enabled or query_vec is None:
            return None
        async with _stage(stages, "cache.lookup") as cache_span:
            cached = await lookup(
                session,
                plan.original,
                query_vec,
                self.embedder.model_key,
                self._pipeline_key(config),
                self.settings.semantic_cache_threshold,
            )
            cache_span.set_attribute("app.cache.hit", cached is not None)
            if cached is not None:
                cache_span.set_attribute("app.cache.similarity", cached.similarity)
        # Exactly one lookup is counted per request: here on a hit, in the
        # refusal branch, or inside record_llm_call(cached=False) on the
        # generate path. Counting it in more than one place doubles the
        # denominator of the hit ratio /stats reports.
        if cached is not None:
            record_cache_lookup(hit=True)
        return cached

    async def _retrieve(
        self,
        session: AsyncSession,
        plan: Plan,
        config: str,
        vectors: dict[str, list[float]],
        stages: dict[str, float],
    ) -> list[Hit]:
        limit = self.settings.top_k_retrieve
        async with _stage(stages, "retrieve") as retrieve_span:
            # Sequential, not gathered: hybrid_search already fans out two legs
            # onto their own sessions, and a second level of concurrency would
            # multiply connections per request for single-digit-ms queries.
            ranked_lists = [
                await self._search(session, config, query, vectors.get(query), limit)
                for query in plan.search_queries
            ]
            candidates = sum(len(hits) for hits in ranked_lists)
            record_retrieval(
                retrieve_span,
                config=config,
                model_key=self.embedder.model_key,
                n_candidates=candidates,
                n_returned=len({hit.chunk_id for hits in ranked_lists for hit in hits}),
                top_score=None,
            )

        async with _stage(stages, "fuse") as fuse_span:
            # One query means nothing to fuse — keep the leg's own scores rather
            # than overwriting them with RRF's positional ones.
            fused = (
                ranked_lists[0]
                if len(ranked_lists) == 1
                else rrf_fuse(ranked_lists, limit=limit)
            )
            fuse_span.set_attribute("app.fuse.inputs", len(ranked_lists))
            fuse_span.set_attribute("app.fuse.returned", len(fused))
        return fused

    async def _search(
        self,
        session: AsyncSession,
        config: str,
        query: str,
        vector: list[float] | None,
        limit: int,
    ) -> list[Hit]:
        if config == "lexical":
            return await lexical_search(session, query, limit=limit)
        if vector is None:  # pragma: no cover - _embed guarantees a vector here
            raise ValueError(
                f"config {config!r} is dense but {query!r} was not embedded"
            )
        if config == "dense":
            return await dense_search(session, vector, self.embedder.model_key, limit)
        return await hybrid_search(
            session, query, vector, self.embedder.model_key, limit=limit
        )

    async def _rerank(
        self, plan: Plan, config: str, hits: list[Hit], stages: dict[str, float]
    ) -> list[Hit]:
        top_k = self.settings.top_k_context
        async with _stage(stages, "rerank") as rerank_span:
            if config.endswith("+rerank") and self.settings.rerank_enabled:
                # Rerank with the MSA rewrite when there is one. The
                # cross-encoder's *ordering* is dialect-robust but its absolute
                # score is not: measured over 20 eval pairs the gold chunk's
                # score has median 0.92 for MSA questions and median 0.01 for
                # Gulf ones. Scoring the rewrite is what kept a global score
                # floor from refusing half the answerable Gulf questions; the
                # floor is now off entirely (rerank_min_score = 0.0) because at
                # n=283 it failed Gulf speakers 10x more often than MSA ones even
                # after the rewrite. See docs/refusal-calibration.md and
                # app/retrieval/rerank.py.
                ranked = await self.reranker.rerank(
                    plan.rewritten or plan.original, hits, top_k=top_k
                )
            else:
                ranked = list(hits[:top_k])
            record_retrieval(
                rerank_span,
                config=config,
                model_key=self.embedder.model_key,
                n_candidates=len(hits),
                n_returned=len(ranked),
                top_score=ranked[0].score if ranked else None,
            )
        return ranked

    def _is_refusal(self, hits: Sequence[Hit]) -> bool:
        """Nothing retrieved, or the cross-encoder says nothing is relevant enough.

        The ``rerank_min_score`` floor is only meaningful against a
        cross-encoder's 0-1 score, so it is applied only when one actually ran —
        ``hit.source`` says so. RRF scores live around 1/61 and cosine scores
        around 0.8; comparing either to a rerank threshold would refuse or accept
        everything.

        **The shipped floor is 0.0**, i.e. the score comparison never fires (a
        sigmoid score is never negative) and "retrieval returned nothing" is the
        only automatic refusal left. That is a measured decision, not an
        oversight: docs/refusal-calibration.md. The branch stays because the
        setting is still the right lever — it is the *value* that could not be
        justified, and a margin-based or dialect-aware replacement would land
        here.
        """
        if not hits:
            return True
        top = hits[0]
        return (
            top.source == RERANK_SOURCE and top.score < self.settings.rerank_min_score
        )

    def _reserve_spend(self, system: str, user: str) -> float:
        """Cap check *before* the call, on an estimate. Raises SpendCapExceeded.

        The estimate assumes the answer runs to ``ANSWER_MAX_TOKENS``, i.e. it
        over-estimates. A cap that only notices after the money is spent is a
        report, not a cap.
        """
        estimated_input = estimate_tokens(system) + estimate_tokens(user)
        return self.spend.reserve(
            usage_cost_usd(estimated_input, ANSWER_MAX_TOKENS, self.provider.price())
        )

    def _pipeline_key(self, config: str) -> str:
        """Everything except the question that determines the answer.

        The cache is keyed on (model_key, this, embedding, guard). Without it the
        key is the question alone, so a request under one config is served the
        answer another config produced — which makes every A/B through the HTTP
        API a measurement of the cache rather than of the config.

        The generating model belongs in here too: the same retrieval handed to a
        different model is a different answer, and swapping the model must not
        silently replay the old one's output.
        """
        return "|".join(
            (
                config,
                f"r{self.settings.top_k_retrieve}",
                f"c{self.settings.top_k_context}",
                f"rr{int(bool(self.settings.rerank_enabled))}",
                f"{self.provider.name}:{self.provider.model}",
            )
        )

    async def _store(
        self, session: AsyncSession, prepared: _Prepared, text: str
    ) -> None:
        """Cache a generated answer. Refusals are never stored.

        The score gate is not the only refusal path and, with
        ``rerank_min_score`` at its measured 0.0, it is not even the live one —
        the model declining per the system prompt is. Caching that would pin the
        refusal: the cache is append-only with no TTL and no invalidation on
        re-ingestion, so ingesting the very article that was missing would never
        dislodge it, and the answer would come back with ``refused=false``.
        """
        if not self._cache_enabled or prepared.query_vec is None or not text.strip():
            return
        if _is_refusal_text(text):
            return
        await store(
            session,
            prepared.plan.original,
            prepared.query_vec,
            self.embedder.model_key,
            self._pipeline_key(prepared.config),
            text,
            [
                {"chunk_id": citation.chunk_id, "score": citation.score}
                for citation in prepared.citations
            ],
        )

    # -- helpers -----------------------------------------------------------

    def _refusal(self, plan: Plan) -> str:
        return REFUSALS.get(plan.register, NOT_IN_CORPUS)

    def _record_usage(
        self,
        generate_span: Any,
        usage: Usage | None,
        reserved_usd: float,
        *,
        provider: str,
        model: str,
    ) -> None:
        if usage is None:  # pragma: no cover - providers always report usage
            self.spend.settle(reserved_usd, 0.0)
            record_cache_lookup(hit=False)
            return
        self.spend.settle(reserved_usd, usage.cost_usd)
        record_llm_call(
            generate_span,
            provider=provider,
            model=model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cost_usd=usage.cost_usd,
            cached=False,
        )

    def _finish(
        self, prepared: _Prepared, text: str, usage: Usage | None, started: float
    ) -> Answer:
        prepared.stages["total"] = round((time.perf_counter() - started) * 1000, 2)
        return Answer(
            text=text,
            citations=prepared.citations,
            register=prepared.plan.register,
            cached=prepared.cached is not None,
            refused=prepared.refused,
            usage=usage,
            stages=dict(prepared.stages),
        )

    def _final_payload(
        self, prepared: _Prepared, usage: Usage | None, started: float
    ) -> dict:
        prepared.stages["total"] = round((time.perf_counter() - started) * 1000, 2)
        return {
            "usage": usage_payload(usage),
            "cost_usd": usage.cost_usd if usage else 0.0,
            "stages_ms": dict(prepared.stages),
            "register": prepared.plan.register,
            "cached": prepared.cached is not None,
            "refused": prepared.refused,
            "provider": self.provider.name,
            "model": self.provider.model,
        }


__all__ = [
    "CONFIGS",
    "DEFAULT_CONFIG",
    "EVENT_CITATIONS",
    "EVENT_DONE",
    "EVENT_ERROR",
    "EVENT_FINAL",
    "EVENT_TOKEN",
    "Answer",
    "Citation",
    "RagService",
    "citation_payload",
    "error_payload",
    "usage_payload",
]
