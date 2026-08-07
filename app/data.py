"""Every shape the service passes around: request bodies, and the values between stages.

Two kinds, and the split is the trust boundary. The pydantic models are what
arrives over HTTP, so they carry validation — that is the *only* place a
malformed document is cheap to reject. The frozen dataclasses are what the
pipeline hands stage to stage; they are already-valid, so they carry fields and
nothing else.

Top-level rather than under ``app/api/``, and importing only ``app.constants``,
so anything can import it: ``app.ingest_worker`` needs ``IngestDocument`` and
``app.retrieval`` needs ``Hit`` without either depending on a router. The type
graph closes on itself — ``Prepared`` refers to ``Plan``, ``Citation`` and
``CachedAnswer``, all of which live here — which is what keeps it cycle-free.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, Field, field_validator

from app.constants import (
    CONFIGS,
    DEFAULT_CONFIG,
    DOC_ID_RE,
    MAX_DOC_ID_CHARS,
    MAX_QUESTION_CHARS,
    MAX_TEXT_CHARS,
    MAX_TITLE_CHARS,
    MAX_URL_CHARS,
)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    config: str = DEFAULT_CONFIG
    stream: bool = True

    @field_validator("question")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be empty or whitespace")
        return value

    @field_validator("config")
    @classmethod
    def _known_config(cls, value: str) -> str:
        if value not in CONFIGS:
            raise ValueError(
                f"unknown retrieval config {value!r}; expected one of {list(CONFIGS)}"
            )
        return value


class IngestDocument(BaseModel):
    """One document to ingest. The trust boundary for both entrypoints.

    ``extra`` is left at pydantic's default (ignore) rather than ``forbid``:
    a push subscription that rejects unknown fields turns an additive change on
    the publisher side into a total ingestion outage, every message poisoned at
    once. Unknown fields are dropped instead.
    """

    doc_id: str = Field(min_length=1, max_length=MAX_DOC_ID_CHARS)
    title: str = Field(min_length=1, max_length=MAX_TITLE_CHARS)
    text: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)
    source_url: str | None = Field(default=None, max_length=MAX_URL_CHARS)

    @field_validator("doc_id")
    @classmethod
    def _usable_in_a_chunk_id(cls, value: str) -> str:
        if ":" in value:
            raise ValueError(
                "doc_id must not contain ':' — chunk ids are 'doc_id:article:seq', "
                "so a colon here splits into the wrong fields and silently breaks "
                "id parsing and every eval pair that names a chunk"
            )
        if not DOC_ID_RE.match(value):
            raise ValueError(
                "doc_id must start with a letter or digit and contain only "
                f"letters, digits, '.', '_' and '-' (got {value!r})"
            )
        return value

    @field_validator("title", "text")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty or whitespace")
        return value

    @field_validator("source_url")
    @classmethod
    def _http_url(cls, value: str | None) -> str | None:
        if value is not None and not value.startswith(("http://", "https://")):
            raise ValueError("source_url must be an http(s) URL")
        return value


# ---------------------------------------------------------------- app/generation/base.py

@dataclass(frozen=True)
class Usage:
    """Real token counts from the provider, priced with that provider's rates.

    ``provider``/``model`` name whoever *actually* produced the tokens. They live
    here rather than being read off the caller's provider handle because under
    failover that handle is the ``FailoverProvider`` wrapper: on the streaming
    path there is no ``Completion`` to read attribution from, so cost and traces
    were being filed under ``failover:anthropic,gemini`` and the *primary's*
    model even when the backup served the request. Carrying it on the immutable
    usage record keeps it correct without a mutable "last used" field that two
    concurrent streams would race over.
    """

    input_tokens: int
    output_tokens: int
    cost_usd: float
    provider: str | None = None
    model: str | None = None


@dataclass(frozen=True)
class Completion:
    """A finished generation. `provider`/`model` are recorded, not assumed.

    Under failover the answer may not come from the provider the caller asked
    for, so cost attribution and traces read these fields rather than config.
    """

    text: str
    usage: Usage
    provider: str
    model: str


# ---------------------------------------------------------------- app/observability/cost.py

@dataclass(frozen=True)
class Spend:
    """Accumulated spend for one UTC day."""

    date: str  # ISO date, e.g. "2026-07-25"
    usd: float
    calls: int


# ---------------------------------------------------------------- app/planning/planner.py

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


# ---------------------------------------------------------------- app/retrieval/search.py

@dataclass(frozen=True)
class Hit:
    """One retrieved chunk. ``score`` is only comparable within one ``source``."""

    chunk_id: str
    doc_id: str
    article: str | None
    text: str
    score: float
    source: str  # "dense" | "lexical" | "rrf"


# ---------------------------------------------------------------- app/retrieval/cache.py

@dataclass(frozen=True)
class CachedAnswer:
    answer: str
    #: Opaque here on purpose: the cache stores and returns whatever
    #: ``query_cache.citations`` holds, and
    #: :func:`app.service._cited_scores` owns knowing that older rows are bare id
    #: strings while current ones are ``{"chunk_id", "score"}`` objects.
    citations: list[object]
    similarity: float  # cosine, 1.0 == identical vector
    age_seconds: float  # since the entry was created, not since its last hit


# ---------------------------------------------------------------- app/ingest_worker.py

@dataclass(frozen=True)
class PushMessage:
    """A decoded Pub/Sub push delivery. ``message_id`` is for logs only."""

    document: IngestDocument
    message_id: str
    subscription: str


# ---------------------------------------------------------------- app/service.py

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
class Prepared:
    """Everything the pipeline decided before generation.

    Was ``_Prepared`` in ``app.service``; the underscore stopped being true once
    it lived in a module something else imports. Still nobody's but the
    pipeline's — it is never serialized and never reaches a response body.
    """

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
