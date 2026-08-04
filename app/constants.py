"""Every literal the service is tuned by, in one place.

Each block keeps the reasoning that was written next to it — a number without
its rationale is a number nobody dares change. Definitions only: nothing here
imports from ``app`` — the one dependency is ``ingestion.normalize``, a
stdlib-only leaf — so any module can import this without a cycle.

Four constants deliberately stayed where they were, because they are not
literals: ``app.generation.base.RETRYABLE_KINDS`` (built from ``ErrorKind`` in
that module), ``app.planning.lexicon.GULF_TO_MSA`` (merged from private tables),
``app.lib.sources.CORPUS_DIR`` (module-level so a test can repoint it) and
``app.generation.failover.T`` (a TypeVar).
"""

from __future__ import annotations

import re

from ingestion.normalize import normalize_query

# ---------------------------------------------------------------- app/models/chunks.py

# Benchmarked embedding models -> the column holding their vectors.
EMBEDDING_COLUMNS: dict[str, str] = {
    "e5": "emb_e5",
    "bge": "emb_bge",
    "openai": "emb_openai",
    "cohere": "emb_cohere",
}


EMBEDDING_DIMS: dict[str, int] = {
    "e5": 1024,  # intfloat/multilingual-e5-large
    "bge": 1024,  # BAAI/bge-m3
    "openai": 3072,  # text-embedding-3-large
    "cohere": 1536,  # cohere embed-v4
}


# trade-off: HNSW ceiling in pgvector is 2000 dims, so emb_openai (3072) gets no
# index — exact scan is fine at corpus scale (a few thousand chunks, single-digit
# ms). Upgrade path when the corpus grows: store it as halfvec(3072) and index
# with halfvec_cosine_ops, or reduce dimensions via the OpenAI `dimensions` param.
HNSW_INDEXED = ("emb_e5", "emb_bge", "emb_cohere")


# ---------------------------------------------------------------- app/models/query_cache.py

CACHE_DIM = 1024


# ---------------------------------------------------------------- app/planning/dialect.py

GULF = "gulf"


MSA = "msa"


# ---------------------------------------------------------------- app/planning/lexicon.py

# The definite-article demonstrative clitic: "هالمدة" = "هذه المدة". Handled as a
# prefix rule in dialect.py rather than a table, because it is productive.
HAL_PREFIX = "هال"


HAL_MIN_LEN = 5  # هال + at least two letters, so "هالة" (halo) is never touched


# Stems that must never be reached by stripping a leading و (see dialect._lookup).
# "ولو" is MSA "even if", not "و + لو"; rewriting it to "وإذا" flips the meaning.
# Measured: it is the only such collision across the 283 eval questions.
WAW_BLOCKLIST: frozenset[str] = frozenset({"لو"})


# Register detection is a different job from rewriting, so it gets its own set.
# Rule for membership: the token must be *unmistakably* Gulf. Tokens that are
# ordinary MSA in another sense are excluded even though they are rewritten —
# ``لو`` (MSA conditional), ``صار``/``يصير`` (MSA "became"), ``يقدر`` (MSA "is
# able"), ``ولا`` (MSA "and not"), ``راتب`` (used across the Gulf press). Answering
# an MSA question in dialect is a visible mistake; missing a marker is not.
REGISTER_MARKERS: frozenset[str] = frozenset(
    {
        # interrogatives / relatives / particles
        "شكثر", "شلون", "شنو", "وش", "وشو", "منو", "وين", "ليش", "كيفنا",
        "اللي", "عشان", "علشان", "لين", "لسا", "بس", "احنا", "شوي", "خلاص",
        "يعني", "مو", "مب", "الحين", "ببلاش", "جوه", "برا", "شي",
        # verbs
        "ابغي", "ابغا", "ابي", "يبغي", "يبي", "تبي", "بغيت", "ودي",
        "اسوي", "نسوي", "يسوون", "سواها", "يدش", "اطفش", "يطيروني",
        # nouns
        "فلوس", "فلوسي", "فلوسه", "كفيل", "الكفيل", "لكفيلي",
        "دريول", "الدريول", "خدامه", "الخدامه", "للخدامه",
        "سنين", "كاش", "يهالي", "دكتور",
        # شغل: colloquial for عمل. 0 of 229 MSA questions and 0 of 233 corpus
        # chunks use it; every one says عمل. Safe, and it is what catches the
        # otherwise MSA-looking "كم ساعة شغل باليوم".
        "شغل", "الشغل", "بالشغل", "شغلي",
    }
)


# ---------------------------------------------------------------- app/planning/planner.py

STRATEGIES = ("noop", "rules", "llm")


MAX_SEARCH_QUERIES = 4  # original + rewrite + at most two sub-queries


MIN_SEGMENT_WORDS = 3  # a two-word fragment is not a question, it is a fragment


# ---------------------------------------------------------------- app/generation/base.py

TOKENS_PER_MILLION = 1_000_000


# ---------------------------------------------------------------- app/generation/budget.py

# trade-off: Arabic runs about 3 characters per token on the Claude and Gemini
# tokenizers (a 4-letter word plus its space is typically one or two tokens),
# so length/3 is within ~10% and always rounds up. No tiktoken: it is the wrong
# tokenizer for both providers, it is a dependency and a model download, and the
# only consumer is a budget that already keeps a 512-token reserve. Ceiling:
# a Latin-heavy or digit-heavy question is over-estimated. Upgrade path when
# cost accounting needs exactness rather than safety — the provider's own
# counter (Anthropic's /messages/count_tokens, Gemini's count_tokens).
CHARS_PER_TOKEN = 3


DEFAULT_RESERVE_TOKENS = 512  # question + system prompt + room for the answer


SYSTEM_PROMPT = """\
أنت مساعد قانوني يجيب عن أسئلة قانون العمل القطري اعتماداً على مواد مرفقة فقط.

القواعد:
1. أجب من المواد المرفقة وحدها. لا تستعن بمعرفة خارجية ولا تستنتج ما ليس فيها.
2. اذكر رقم المادة بعد كل معلومة توردها، بهذه الصيغة: [المادة 12].
3. أجب بنفس أسلوب السؤال: إن سُئلت بالعامية الخليجية فأجب بالعامية الخليجية، \
وإن سُئلت بالفصحى فأجب بالفصحى.
4. إن لم تكن الإجابة موجودة في المواد المرفقة فقل بوضوح: \
"لا تتضمن المواد المتاحة إجابة عن هذا السؤال." ولا تضف أي تخمين.
5. اختصر: من جملة إلى أربع جمل."""


NOT_IN_CORPUS = "لا تتضمن المواد المتاحة إجابة عن هذا السؤال."


# ---------------------------------------------------------------- app/generation/failover.py

MAX_ATTEMPTS = 3  # per provider, including the first try


RETRY_MULTIPLIER_S = 0.5


RETRY_MAX_WAIT_S = 8.0


FAILOVER_EVENT = "provider.failover"


# ---------------------------------------------------------------- app/generation/providers.py

# model id -> (USD / 1M input tokens, USD / 1M output tokens). Checked 2026-07.
ANTHROPIC_PRICES: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-haiku-4-5-20251001": (1.00, 5.00),
}


# Checked 2026-07. Gemini 2.5 Flash text rates; output includes thinking tokens.
GEMINI_PRICES: dict[str, tuple[float, float]] = {
    "gemini-2.5-flash": (0.30, 2.50),
}


# ---------------------------------------------------------------- app/observability/cost.py

# Price per *million* tokens, keyed by model-id prefix so dated snapshots
# ("claude-haiku-4-5-20251001") resolve to their family price.
#
# Anthropic rows are the published first-party API rates (verified 2026-06-24).
# The Gemini row is Google's published list price and is NOT verifiable from
# this repo — there are no API keys here. Re-check it before quoting a cost per
# 1K queries in the writeup.
#
# trade-off: a hard-coded table, not a pricing API. Prices change a few times a
# year; a wrong number here shows up as a wrong dashboard, not a wrong answer.
# Upgrade path when that stops being acceptable: read it from a JSON file that
# CI refreshes.
PRICES_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    # model-id prefix: (input, output)
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-opus-5": (5.00, 25.00),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-pro": (1.25, 10.00),
}


# ---------------------------------------------------------------- app/observability/tracing.py

INSTRUMENTATION_NAME = "arabic-rag"


# ---------------------------------------------------------------- app/retrieval/embed.py

# model_key -> HuggingFace id, for the models that run locally.
LOCAL_MODELS: dict[str, str] = {
    "e5": "intfloat/multilingual-e5-large",
    "bge": "BAAI/bge-m3",
}


# model_key -> (query prefix, passage prefix). Part of the model, not a style
# choice: e5 is trained with these, bge-m3 is trained without and adding them
# only adds noise.
PREFIXES: dict[str, tuple[str, str]] = {
    "e5": ("query: ", "passage: "),
    "bge": ("", ""),
}


ST_BATCH = 16  # sentence-transformers batch; 16 x 512 tokens fits MPS comfortably


API_BATCH = 96  # both OpenAI and Cohere accept far more; 96 keeps payloads small


API_TIMEOUT_S = 60.0


MAX_TRIES = 3


# ---------------------------------------------------------------- app/retrieval/rerank.py

CROSS_ENCODER_MODEL = "BAAI/bge-reranker-v2-m3"


MAX_SEQUENCE_LENGTH = 512  # query + chunk; longer pairs are truncated by the tokenizer


RERANK_SOURCE = "rerank"


# ---------------------------------------------------------------- app/retrieval/search.py

DEFAULT_LIMIT = 20


RRF_K = 60  # the constant from Cormack et al. 2009; damps the top of each list


MAX_TOKEN_CHARS = 64  # pg errors above 2047 bytes; no real Arabic word is close


MAX_QUERY_TOKENS = 32  # longer than any question in the eval set


# ---------------------------------------------------------------- app/retrieval/cache.py

# Written in *normalized* form (see ingestion.normalize: ة->ه, ى->ي, hamza
# seats folded), because guard_key compares tokens of the normalized query.
# Includes Gulf negations (مو/مب/ماكو) — the dialect questions are the point.
NEGATION_PARTICLES = frozenset(
    normalize_query(word)
    for word in (
        "لا",
        "ما",
        "لم",
        "لن",
        "ليس",
        "ليست",
        "غير",
        "بدون",
        "دون",
        "مو",
        "مب",
        "ماكو",
    )
)


# ---------------------------------------------------------------- app/lib/rate_limit.py

RATE_WINDOW_S = 60.0


#: Sweep every key once the table passes this many, so a spray of one-shot
#: addresses cannot grow it without bound. A sweep is O(keys) and only runs when
#: the table is already this large, so it is amortised to nothing.
RATE_SWEEP_AT = 10_000


# ---------------------------------------------------------------- app/data.py

#: Long enough for any question in the eval set with room to spare, short enough
#: that a pathological body never reaches the embedder or the tokenizer.
MAX_QUESTION_CHARS = 1000


#: Chunk ids are ``doc_id:article:seq``. A doc_id containing ``:`` would split
#: into the wrong fields on every downstream parse, so the character is banned
#: rather than escaped — see :meth:`IngestDocument._usable_in_a_chunk_id`.
#: ``\A``/``\Z``, not ``^``/``$``: Python's ``$`` also matches immediately before
#: a trailing newline, so ``"my-doc\n"`` passed this check and the newline went
#: straight into the chunk primary key — where it is invisible in logs, breaks
#: exact-match lookups against the clean id, and makes re-ingesting the same
#: document write a second set of rows.
MAX_DOC_ID_CHARS = 128


DOC_ID_RE = re.compile(rf"\A[A-Za-z0-9][A-Za-z0-9._-]{{0,{MAX_DOC_ID_CHARS - 1}}}\Z")


MAX_TITLE_CHARS = 300


MAX_URL_CHARS = 2048


#: The largest document in the committed corpus (Law 14/2004) is ~67k
#: characters, so this is roughly 3x the real ceiling and still small enough
#: that a body of this size cannot exhaust a Cloud Run instance's memory.
MAX_TEXT_CHARS = 200_000


# ---------------------------------------------------------------- app/ingest_worker.py

#: Pub/Sub's own documented per-message limit. A push body larger than this did
#: not come from Pub/Sub, so it is rejected before any parsing happens.
MAX_PUSH_BODY_BYTES = 10 * 1024 * 1024


#: ``CorpusDoc.license`` is required by the manifest loader but the ``chunks``
#: table has no column for it, so a posted document carries a placeholder.
POSTED_LICENSE = "unspecified"


# ---------------------------------------------------------------- app/deps.py

#: The embedding model the *service* runs. Not ``settings.embedding_model``:
#: that field names an API model (`text-embedding-3-large`) this deployment has
#: no key for, and it is the benchmark's variable, not the service's.
#:
#: bge-m3 is the measured choice — recall@10 0.946 on the full eval set at
#: ~2.6 ms mean / 3.9 ms p95, and a Gulf-dialect penalty of -0.4 points against
#: e5's -9.9. It is also 1024-dim, which is what `query_cache` stores.
#:
#: # trade-off: a module constant, because config lives in app/config.py and that
#: # file is not mine to edit. Upgrade path: add `retrieval_model_key: str =
#: # "bge"` to Settings and read it here.
SERVICE_MODEL_KEY = "bge"


#: Cross-encoder reranker. "noop" keeps fusion order (the ablation baseline).
SERVICE_RERANKER = "bge"


#: Rule-based Gulf→MSA planning: measurable offline, needs no API key, and is
#: the arm the phase-3 numbers were produced with.
SERVICE_PLANNER = "rules"


# ---------------------------------------------------------------- app/service.py

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


#: Shown instead of a generation that came apart. Deliberately not one of
#: REFUSALS: the corpus did have an answer and the generator mangled it, so
#: "not in the materials" would be a false statement about the law.
GARBLED = "تعذّر إنتاج إجابة سليمة لهذا السؤال. من فضلك أعد المحاولة."


# ---------------------------------------------------------------- app/api/ask.py

SSE_MEDIA_TYPE = "text/event-stream"


# Proxies love to buffer event streams; both headers are the conventional opt-out.
SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
