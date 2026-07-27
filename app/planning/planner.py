"""Query planning: which strings to actually retrieve with, and in which register to answer.

Three planners behind one protocol, ordered by how much they can be trusted
today:

* :class:`NoopPlanner` — the ablation baseline. Detects the register (that costs
  nothing and drives the answer) and rewrites nothing.
* :class:`RuleBasedPlanner` — the shippable one. Lexicon-driven Gulf → MSA
  rewrite, measurable offline against the 283 labelled eval pairs with no API key
  and no model download. It also owns a very narrow two-part decomposition, which
  measurement then turned **off by default** — see the class docstring for the
  table that decided it.
* :class:`LLMPlanner` — the ceiling. Needs a generation provider, so on a machine
  with no keys it fails loudly at construction instead of at request time.

The interesting property is :class:`RuleBasedPlanner`'s: it never *replaces* the
user's question, it *adds* to it. See its docstring.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Protocol

from app.planning.dialect import GULF, MSA, detect_register, gulf_to_msa

logger = logging.getLogger(__name__)

STRATEGIES = ("noop", "rules", "llm")
MAX_SEARCH_QUERIES = 4  # original + rewrite + at most two sub-queries
MIN_SEGMENT_WORDS = 3  # a two-word fragment is not a question, it is a fragment


@dataclass(frozen=True)
class Plan:
    """What planning decided, in a form the retrieval and generation stages can read.

    ``search_queries`` is what to retrieve with (one search per entry, fused);
    ``register`` is what to answer in, and is independent of it — the corpus is
    MSA, so retrieval is always MSA-flavoured even when the reply is not.
    """

    original: str
    search_queries: list[str]
    register: str  # "gulf" | "msa"
    rewritten: str | None  # the MSA form, when a rewrite actually changed something
    strategy: str  # "noop" | "rules" | "llm"


class Planner(Protocol):
    async def plan(self, question: str) -> Plan: ...


class PlannerUnavailable(RuntimeError):
    """A planner was asked for that cannot run in this environment (no provider/key)."""


def _dedupe(queries: list[str]) -> list[str]:
    """Order-preserving dedupe on the stripped string, capped."""
    seen: set[str] = set()
    unique: list[str] = []
    for query in queries:
        text = query.strip()
        if text and text not in seen:
            seen.add(text)
            unique.append(text)
    return unique[:MAX_SEARCH_QUERIES]


# ── decomposition ─────────────────────────────────────────────────────────────
# Arabic writes a compound question as one sentence joined by a clitic و: "شكثر
# لازم ... ومتى يصير ...". Splitting on every و would shred normal prose, so the
# boundary has to be a و that *starts a new interrogative clause*: either after a
# question mark or comma, or glued directly onto an interrogative word.
_INTERROGATIVE_STEMS = (
    "كم", "ماذا", "ما", "من", "متى", "متي", "أين", "اين", "كيف", "لماذا", "هل",
    "شكثر", "شنو", "شلون", "منو", "وين", "ليش", "ش",
)
# ``\b`` after the stem is load-bearing: without it the one-letter stem "ش"
# (there for "وش") matches the start of any ordinary word beginning with ش, so
# "الأجر وشروط العمل" ("pay and conditions of work") split into two clauses.
_CLAUSE_BOUNDARY = re.compile(
    r"[؟?،]\s*(?=و)"  # "... ؟ ومتى ..." / "... ، ومتى ..."
    r"|\s+(?=و(?:" + "|".join(_INTERROGATIVE_STEMS) + r")\b)"  # "... وكم ..."
)
_SENTENCE_END = re.compile(r"[؟?]+")

# Normalized interrogative tokens, for deciding whether a segment is a question
# at all. Matched against ingestion.normalize output, so ى/أ are already folded.
# ``ما``, ``من`` and ``أي`` are excluded even though all three are real MSA
# interrogatives: they are also the negator, the preposition "from" and "any",
# and they occur in 78, 63 and 17 of the 229 MSA eval questions respectively.
# Including them made "وما أخذ ولا ريال" (a negated statement) look like a second
# question and split gulf-030 wrongly. Over-splitting is the failure mode that
# costs precision, so the ambiguous three are dropped and the questions that
# front them simply stay whole.
_INTERROGATIVE_TOKENS = frozenset(
    {"كم", "ماذا", "متي", "اين", "كيف", "لماذا", "هل", "شكثر",
     "شنو", "شلون", "منو", "وين", "ليش", "وش"}
)
_WORD = re.compile(r"\w+")

#: The two single-letter conjunctions Arabic glues to the next word: و ("and")
#: and ف ("so/then"). Only these may be stripped when looking for a glued-on
#: interrogative — see :func:`_interrogatives`.
_PROCLITIC_CONJUNCTIONS = frozenset({"و", "ف"})


def _interrogatives(segment: str) -> set[str]:
    from ingestion.normalize import normalize_query

    tokens = {normalize_query(token) for token in _WORD.findall(segment)}
    found = tokens & _INTERROGATIVE_TOKENS
    # "وكم"/"فماذا" — a proclitic conjunction is glued on; count the stem.
    # Restricted to the two single-letter conjunctions. Stripping the first
    # letter of *every* token turned ordinary words into interrogatives, most
    # damagingly حكم ("ruling", which this corpus is full of) → كم ("how many"),
    # inventing a second question and splitting the query.
    found |= {
        token[1:]
        for token in tokens
        if token[:1] in _PROCLITIC_CONJUNCTIONS and token[1:] in _INTERROGATIVE_TOKENS
    }
    return found


def decompose(question: str) -> list[str]:
    """Split a two-part question into sub-queries, or return ``[question]``.

    Fires only when *all* of: the text breaks at a coordinating و clause, at
    least two of the resulting segments carry an interrogative, and those
    segments ask *different* interrogatives. "كم ساعة وكم يوم" therefore stays one
    query (same interrogative, one topic) while "منو اللي يدفع؟ وراتبي شلون ينزل؟"
    becomes two.

    # trade-off: this catches the obvious two-part case and nothing else — no
    # coreference resolution ("وهل يشمل ذلك...?" keeps its dangling pronoun), no
    # implicit conjunction, no three-way splits beyond what the boundary regex
    # happens to produce. Real decomposition needs an LLM; that is LLMPlanner's
    # job and it is measurable the day a key exists. Over-splitting is the
    # failure mode that hurts, so the gates above are deliberately strict.
    """
    segments = [
        segment.strip(" \t\n،؛?؟")
        for chunk in _SENTENCE_END.split(question)
        for segment in _CLAUSE_BOUNDARY.split(chunk)
    ]
    segments = [s for s in segments if len(_WORD.findall(s)) >= MIN_SEGMENT_WORDS]
    if len(segments) < 2:
        return [question]

    asking = [(segment, _interrogatives(segment)) for segment in segments]
    questions = [segment for segment, found in asking if found]
    distinct = {word for _, found in asking for word in found}
    if len(questions) < 2 or len(distinct) < 2:
        return [question]
    return questions


# ── planners ──────────────────────────────────────────────────────────────────
class NoopPlanner:
    """Ablation baseline: retrieve with exactly what the user typed.

    Still detects the register — that is free, has no effect on retrieval, and
    keeps the "answer in the user's dialect" behaviour comparable across the
    planning ablation instead of confounding it.
    """

    strategy = "noop"

    async def plan(self, question: str) -> Plan:
        return Plan(
            original=question,
            search_queries=[question.strip()],
            register=detect_register(question),
            rewritten=None,
            strategy=self.strategy,
        )


class RuleBasedPlanner:
    """Lexicon Gulf → MSA rewrite, searching with **both** the original and the rewrite.

    This is the property that makes a hand-written dialect lexicon safe to ship:
    the rewrite is never substituted for the user's question, it is *appended* to
    the list of queries whose results get fused. A wrong rewrite can therefore
    only introduce candidates that rank badly and get out-ranked — it can never
    remove a chunk the original question would have found. The downside is bounded
    at "one extra vector search"; the upside is the dialect recall the phase-2
    benchmark measured as lost. Precision at the top is protected by the fusion
    ranking and, in the service, the reranker.

    The rewrite only runs when :func:`detect_register` says "gulf", so a question
    already in MSA is passed through untouched rather than nudged by a lexicon
    that was never meant for it.

    **Measured** — 50 answerable Gulf pairs, dense retrieval, real corpus, against
    the 50 MSA questions carrying identical ground truth ("msa" = the ceiling this
    is trying to reach). **Per-leg retrieval depth 20**, i.e.
    ``settings.top_k_retrieve``, which is what the service runs; the fused arm is
    depth-sensitive and scores *better* at depth 10 (bge 0.82 / 0.95 / 0.7737,
    e5 0.76 / 0.92 / 0.6512), so quoting it without the depth is meaningless.
    Both tables live in ``benchmark/results/results.json`` under ``planning``::

        bge-m3          recall@3  recall@10     MRR      e5-large   recall@3  recall@10     MRR
        raw               0.7900     0.9300  0.7444      raw          0.7500     0.8500  0.6172
        rewrite only      0.8400     0.9500  0.7632      rewrite      0.7300     0.9200  0.6917
        original+rewrite  0.8200     0.9300  0.7722      both         0.7600     0.8900  0.6497
        + decomposition   0.7700     0.9300  0.7350      + decomp     0.7200     0.8900  0.6164
        msa ceiling       0.8600     0.9500  0.8448      msa          0.9200     0.9700  0.8500

    Three things fall out of that table, and two of them set the defaults here:

    1. Fusing original+rewrite never loses on any aggregate metric for either
       model, so it is the default. Per-pair it is +3/-1 and +2/-1 at k=3 and
       +0/-0 and +2/-0 at k=10 — small, but one-sided.
    2. Rewrite *only* scores higher still on raw recall (e5 recall@10 0.85 → 0.92,
       58% of that model's dialect gap) but it is one-sided the wrong way at k=3:
       it loses 4 pairs to win 2. Discarding the user's own words is exactly the
       risk fusion exists to avoid, so recall alone does not buy the default.
    3. Decomposition costs 5 points of recall@3 on bge and 4 on e5 while adding
       nothing at k=10 — the same shape as the phase-2 finding that RRF hybrid
       hurts top-3 precision. It is therefore **off by default** and kept only as
       an ablation switch. n = 50, so one pair is two points: treat every delta
       here as directional, not significant.
    """

    strategy = "rules"

    def __init__(self, *, decompose_enabled: bool = False) -> None:
        self.decompose_enabled = decompose_enabled

    async def plan(self, question: str) -> Plan:
        original = question.strip()
        register = detect_register(original)

        rewritten: str | None = None
        if register == GULF:
            candidate = gulf_to_msa(original)
            if candidate != original:
                rewritten = candidate

        queries = [original]
        if rewritten:
            queries.append(rewritten)
        if self.decompose_enabled:
            # Split the ORIGINAL, then rewrite each part — not the other way round.
            # Measured: decomposing the rewrite fires on 1 of the 50 Gulf pairs
            # instead of 9, because the rewrite maps the unambiguous Gulf
            # interrogatives onto the MSA ones (منو→من, شنو→ما) that _decompose_
            # deliberately does not trust. The dialect form is the better signal.
            parts = decompose(original)
            if len(parts) > 1:
                queries.extend(
                    gulf_to_msa(part) if register == GULF else part for part in parts
                )

        return Plan(
            original=original,
            search_queries=_dedupe(queries),
            register=register,
            rewritten=rewritten,
            strategy=self.strategy,
        )


_SYSTEM = """أنت مساعد يعيد صياغة أسئلة قانون العمل القطري للبحث في نص عربي فصيح.
أعد كائن JSON فقط، بلا أي شرح، بهذا الشكل:
{"rewritten": "<السؤال بالفصحى>", "sub_queries": ["<سؤال فرعي>", ...]}
اترك sub_queries فارغة إذا كان السؤال يسأل عن شيء واحد."""

_PROMPT = "السؤال: {question}"

#: A plan is two short strings; the default 1024 would only ever pay for tokens
#: the parser throws away.
_PLANNER_MAX_TOKENS = 512

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(raw: str) -> dict[str, Any]:
    """Parse the first JSON object out of a model reply. Raises ValueError."""
    match = _JSON_OBJECT.search(raw)
    if match is None:
        raise ValueError("no JSON object in planner reply")
    parsed = json.loads(match.group())
    if not isinstance(parsed, dict):
        raise ValueError(f"planner reply is a {type(parsed).__name__}, expected an object")  # noqa: TRY004
    return parsed


class LLMPlanner:
    """Rewrite + decomposition through a generation provider.

    Unavailable without an API key, and says so at construction: an
    :class:`PlannerUnavailable` naming the missing key beats a traceback from
    inside the first request. Everything it can do, :class:`RuleBasedPlanner`
    approximates offline — which is why the rule-based path is the one that was
    actually measured.

    ``provider`` is an :class:`app.generation.base.Provider`. It is imported
    lazily rather than at module scope so this module stays importable without
    the vendor SDKs, but the *call* follows that protocol exactly — an earlier
    version duck-typed it and silently never worked (see :func:`_load_provider`).
    """

    strategy = "llm"

    def __init__(self, provider: Any | None = None) -> None:
        self._provider = provider if provider is not None else _load_provider()
        self._fallback = RuleBasedPlanner()

    async def plan(self, question: str) -> Plan:
        original = question.strip()
        register = detect_register(original)
        try:
            completion = await self._provider.complete(
                _SYSTEM,
                [{"role": "user", "content": _PROMPT.format(question=original)}],
                max_tokens=_PLANNER_MAX_TOKENS,
            )
            parsed = _extract_json(completion.text)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            # Not swallowed: logged with the reason, and the returned Plan says
            # "rules" so a caller reading strategy sees the LLM did not run.
            logger.warning("LLM planner reply unusable (%s); falling back to rules", exc)
            return await self._fallback.plan(original)

        rewritten = str(parsed.get("rewritten") or "").strip() or None
        sub_queries = [
            str(item).strip()
            for item in parsed.get("sub_queries") or []
            if str(item).strip()
        ]
        return Plan(
            original=original,
            search_queries=_dedupe([original, *( [rewritten] if rewritten else [] ), *sub_queries]),
            register=register,
            rewritten=rewritten if rewritten != original else None,
            strategy=self.strategy,
        )


def _load_provider() -> Any:
    """Build the same failover chain ``/ask`` generates with, or explain why not.

    This used to call ``get_provider()`` with no argument against
    ``get_provider(name: str)``, and the resulting TypeError was caught and
    re-labelled "set ANTHROPIC_API_KEY" — so LLMPlanner could never run, and said
    so in a way that sent the reader after a key that was already set. It builds
    ``FailoverProvider.from_settings()``, exactly as ``app.deps.build_provider``
    does, so the planner and the answer come from the same configured chain.
    """
    try:
        from app.generation.failover import FailoverProvider
    except ImportError as exc:
        raise PlannerUnavailable(
            "LLMPlanner needs app.generation; generation is not wired up yet. "
            "Use get_planner('rules') for the offline path."
        ) from exc
    try:
        return FailoverProvider.from_settings()
    except Exception as exc:
        raise PlannerUnavailable(
            f"LLMPlanner has no usable generation provider ({exc}). Set ANTHROPIC_API_KEY "
            "or GOOGLE_API_KEY, or use get_planner('rules')."
        ) from exc


def get_planner(strategy: str = "rules") -> Planner:
    """Build a planner. ``"llm"`` raises :class:`PlannerUnavailable` without a key."""
    if strategy == "noop":
        return NoopPlanner()
    if strategy == "rules":
        return RuleBasedPlanner()
    if strategy == "llm":
        return LLMPlanner()
    raise ValueError(f"unknown planning strategy {strategy!r}; expected one of {STRATEGIES}")


__all__ = [
    "GULF",
    "MSA",
    "LLMPlanner",
    "NoopPlanner",
    "Plan",
    "Planner",
    "PlannerUnavailable",
    "RuleBasedPlanner",
    "decompose",
    "get_planner",
]
