# arabic-rag

Retrieval-augmented question answering over Qatari labour legislation, built
Arabic-first: the corpus is Modern Standard Arabic, the questions arrive in MSA
*or* Gulf dialect, and the point of the project is to measure what that
difference costs.

Status: phases 1–3 are done — corpus, ingestion, eval dataset, retrieval,
benchmark, CI regression gate, and now the service surface: `POST /ask` (SSE),
query planning, generation behind a provider-agnostic failover chain, OpenTelemetry
tracing, a semantic cache, a daily spend cap, and an enforced latency budget.
Phase 4 (Terraform, Cloud Run, Pub/Sub ingestion) is not started — see
[Not built yet](#not-built-yet).

**Generation has never run.** There is no API key in this environment, so every
provider adapter is unit-tested against mocked transports and `/ask` answers 503
until `ANTHROPIC_API_KEY` (or `GOOGLE_API_KEY`) is set. Everything up to the LLM
call — planning, retrieval, reranking, the refusal gate, the cache, the spend
check, the traces — runs and is measured. This is stated again wherever it
matters below rather than buried here.

## The headline number

Same question, same gold chunk, only the register changes. Gulf-dialect phrasings
of MSA questions, scored against identical ground truth:

| multilingual-e5-large, dense | MSA (matched control, n=68) | Gulf (n=50) | change |
|---|---|---|---|
| recall@10 | 0.949 | 0.850 | **-9.9 pts** |
| recall@3 | 0.882 | 0.750 | **-13.2 pts** |
| MRR | 0.815 | 0.617 | **-19.7 pts** |

The penalty lands on *ranking*, not coverage: the right article is usually still
in the top 10, much less often at the top. It is also model-dependent —
`BAAI/bge-m3` is nearly flat on the same comparison (-0.4 / -3.3 / -5.9), a
difference of four questions out of fifty, which at that n is suggestive and not
settled. Reranking shrinks the gap but does not close it.

Overall, over all 268 answerable pairs:

| config | recall@3 | recall@10 | MRR | mean latency |
|---|---|---|---|---|
| lexical (FTS, model-free) | 0.188 | 0.356 | 0.181 | 2.6 ms |
| e5 dense | 0.855 | 0.927 | 0.818 | 2.6 ms |
| bge dense | 0.864 | 0.946 | 0.829 | 2.6 ms |
| e5 hybrid+rerank | 0.910 | 0.953 | 0.881 | 985 ms |
| bge hybrid+rerank | 0.922 | 0.965 | 0.883 | 974 ms |

Two results worth reading the writeup for: unweighted RRF **hurts** the top of
the list (recall@3 collapses from 0.864 to 0.489 for bge — verified as correct
RRF arithmetic, not a bug), and the reranker buys +1.9 pts recall@10 for ~375x
the latency. Full method, confidence intervals, anomalies and limitations:
[docs/benchmark.md](docs/benchmark.md). Design spec:
[docs/specs/2026-07-24-arabic-rag-design.md](docs/specs/2026-07-24-arabic-rag-design.md).

Caveat stated up front: 233 chunks means top-10 is 4% of the index, and the MSA
questions were LLM-drafted from the chunks then reviewed by hand. These are not
query-log numbers.

## What query planning recovers

Phase 3's ablation, and the direct answer to the headline above. A hand-written
~30-entry Gulf→MSA lexicon (no model, no API key) rewrote 49 of the 50 Gulf
questions. Dense retrieval, 10 candidates per leg:

| arm | e5 recall@3 | e5 recall@10 | e5 MRR | bge recall@3 | bge recall@10 | bge MRR |
|---|---|---|---|---|---|---|
| raw | 0.75 | 0.85 | 0.6146 | 0.79 | 0.93 | 0.7432 |
| rewritten | 0.73 | **0.92** | 0.6917 | **0.84** | **0.95** | 0.7632 |
| fused (raw + rewritten, RRF) | **0.76** | **0.92** | 0.6512 | 0.82 | **0.95** | **0.7737** |

The lexicon recovers roughly **7 of e5's 9.9-point recall@10 dialect gap** and
improves MRR for both models. It does **not** improve e5's recall@3 — 0.75 raw,
0.73 rewritten, 0.76 fused — so for the model with the worst penalty the top of
the list, which is what a generator actually reads, is essentially unchanged. At
n=50 the interval is roughly ±8–10 points: trust the direction, not the size.

The service ships the **fused** arm even though "rewritten" scores higher on raw
recall, because searching with both queries means a bad rewrite can only *add*
candidates — it can never remove the chunk the user's own words would have found.
The rewrite is also depth-sensitive (the advantage shrinks at the depth the
service actually retrieves at), and the LLM rewriter is implemented but
**unmeasured** for want of a key. All of that, with the numbers:
[docs/benchmark.md § Query planning](docs/benchmark.md#query-planning-does-a-gulfmsa-rewrite-recover-the-dialect-penalty),
raw data under the `planning` key of
[`benchmark/results/results.json`](benchmark/results/results.json).

## Run it end to end

Requires Python 3.12 and Docker. The heavy model dependencies are a separate
requirements file — everything except embedding and reranking runs without them.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt          # service + tests, no torch
pip install -r requirements-models.txt       # sentence-transformers + torch (~2 GB)

cp .env.example .env
docker compose up -d db                      # pgvector/pgvector:pg17 on :5433
export DATABASE_URL=postgresql+asyncpg://rag_user:rag_pass@localhost:5433/rag_db

alembic upgrade head
python -m ingestion ingest                   # committed corpus -> 233 chunks
python -m ingestion stats                    # rows per document, embedding coverage

python -m ingestion backfill --model e5      # ~10 s on Apple Silicon
python -m ingestion backfill --model bge     # the model the service queries with

uvicorn app.main:app --port 8000
```

Then, in another shell:

```bash
curl -s localhost:8000/stats
```

```json
{"corpus":{"chunks":233,"documents":5,
 "embeddings":{"e5":{"chunks":233,"coverage":1.0},"bge":{"chunks":233,"coverage":1.0},
               "openai":{"chunks":0,"coverage":0.0},"cohere":{"chunks":0,"coverage":0.0}}},
 "cache":{"entries":0,"hits":0,"hit_ratio":0.0,"enabled":true,"threshold":0.95},
 "spend":{"date":"2026-07-25","usd":0.0,"calls":0,"cap_usd":5.0,"remaining_usd":5.0},
 "providers":{"configured":["anthropic","gemini"],"available":[],"missing_keys":["anthropic","gemini"]},
 "retrieval":{"model_key":"bge","reranker":"bge","planner":"rules","rerank_enabled":true,
              "rerank_min_score":0.15,"top_k_retrieve":20,"top_k_context":5}}
```

`providers.available: []` is the honest state of this checkout. Asking a question
in it returns a 503 that names the variable to set:

```console
$ curl -s -X POST localhost:8000/ask -H 'content-type: application/json' \
    -d '{"question":"كم مدة الإشعار قبل إنهاء العقد؟"}'
HTTP/1.1 503 Service Unavailable
{"detail":"generation is not configured: AnthropicProvider needs an API key:
 set ANTHROPIC_API_KEY in the environment or .env (see available_providers())"}
```

**Set `ANTHROPIC_API_KEY` (or `GOOGLE_API_KEY` for the Gemini adapter) and the
same request streams.** The frames below are the real output shape, captured from
this pipeline with a stub provider standing in for the LLM — retrieval,
citations, register detection and framing are genuine; the three `token` frames
are the stub saying so, because no key exists here to produce real ones.

```console
$ curl -sN -X POST localhost:8000/ask -H 'content-type: application/json' \
    -d '{"question":"شكثر مدة الإشعار قبل ما ينتهي العقد؟"}'

event: citations
data: {"citations": [{"chunk_id": "qatar-labour-law-14-2004:52:1", "doc_id": "qatar-labour-law-14-2004",
       "article": "52", "score": 0.894817, "excerpt": "المادة 52 - مكرر …"},
      {"chunk_id": "qatar-labour-law-14-2004:49:0", "…": "…"}],
       "register": "gulf", "cached": false, "refused": false}

event: token
data: {"text": "[stub provider — "}

event: token
data: {"text": "no API key configured] "}

event: token
data: {"text": "المادة 49"}

event: final
data: {"usage": {"input_tokens": 812, "output_tokens": 17, "cost_usd": 0.000897},
       "cost_usd": 0.000897,
       "stages_ms": {"plan": 0.07, "embed": 44.82, "cache.lookup": 10.8, "retrieve": 6.39,
                     "fuse": 0.04, "rerank": 907.99, "generate": 0.07, "total": 974.76},
       "register": "gulf", "cached": false, "refused": false,
       "provider": "stub", "model": "stub-model"}

event: done
data: {}
```

Note what the pipeline did with a dialect question with no key involved: it
detected the register as `gulf`, rewrote it to MSA, searched with both, and put
articles 52 and 49 — notice periods on termination — at the top. `"stream":false`
returns the same content as one JSON body; `"config"` selects
`dense | lexical | hybrid | hybrid+rerank` per request, which is how the
benchmark's ablations stay reachable from the running service.

Two more real responses from the same run, both of which reached no provider:

```jsonc
// refusal — top rerank score below settings.rerank_min_score, answered in the
// user's own register, provider never called, never cached
{"answer": "المواد المتوفرة ما فيها جواب عن هذا السؤال.", "citations": [], "register": "gulf",
 "cached": false, "refused": true, "usage": null, "cost_usd": 0.0,
 "stages_ms": {"plan": 0.07, "embed": 56.83, "cache.lookup": 4.09, "retrieve": 10.73,
               "fuse": 0.04, "rerank": 898.95, "total": 970.79}}

// semantic cache hit on a repeat question — 26 ms, no retrieval, no LLM, no cost
{"usage": null, "cost_usd": 0.0,
 "stages_ms": {"plan": 0.06, "embed": 17.09, "cache.lookup": 7.38, "total": 26.23},
 "register": "gulf", "cached": true, "refused": false}
```

`python -m ingestion fetch` re-downloads the corpus from its public sources; the
cleaned text is committed under `data/corpus/` so nothing above needs the network.

The benchmark and the evaluation suite run without the service:

```bash
python -m benchmark.run                      # ~21 min: 2 models x 4 configs x 4 splits
python -m benchmark.run --tables             # regenerate the docs/benchmark.md tables
python -m evals.gate --config lexical        # regression gate, exit 1 on a drop
python -m evals.gate --config dense --model bge
```

Tests and lint:

```bash
pytest            # 470 pass, 1 skipped (it loads the 2 GB reranker; set
                  # RERANK_REAL_MODEL=1 to run it). DB-backed tests need the
                  # pgvector container up, and use a separate rag_test database.
ruff check .
```

## The service

`POST /ask` is the whole surface. One request runs:

```
plan → embed → cache.lookup → retrieve → fuse → rerank → generate
```

Each stage is its own OTel span, each is timed into the `stages_ms` map above, and
**three of them can end the request before an LLM is ever called** — which is the
design, not an optimisation:

- **Semantic cache.** The query embedding is matched against previously answered
  questions in the same pgvector table (cosine ≥ `semantic_cache_threshold`, 0.95).
  A hit returns in ~26 ms with citations rehydrated and nothing billed. The cache
  carries digit and negation guards so "30 days" and "60 days", or a question and
  its negation, cannot collide.
- **Refusal gate.** If the cross-encoder's top score is below
  `rerank_min_score`, the service answers "not in corpus" *in the user's
  register* rather than guessing. The eval set contains 15 unanswerable questions
  specifically to measure this. Refusals are never cached.
- **Spend cap.** Estimated cost is checked *before* the call, against a daily USD
  cap; past it, `/ask` returns 503 with the day's spend, not a silent overrun.

Generation sits behind a two-method `Provider` protocol with Anthropic and Gemini
adapters. Each adapter classifies its own SDK's failures into one `ErrorKind`, and
the failover policy is a pure function of that: rate-limit and 5xx retry then fail
over, timeout and connection errors fail over immediately, and a 4xx/auth error
**does not** fail over — it would fail identically on the next provider, so
burying it behind "all providers failed" helps nobody. Failover is recorded as a
span event.

Also: `/stats` (corpus, cache, spend, providers, retrieval config), `/health`,
`/health/db`, and `/metrics` (Prometheus). Tracing exports only when
`OTEL_EXPORTER_OTLP_ENDPOINT` or `OTEL_CONSOLE_EXPORT=1` is set, so it costs
nothing by default.

## The latency budget

p95 ≤ 3.5 s end to end, allocated per stage in
[docs/latency-budget.md](docs/latency-budget.md) and enforced by a replay script:

```bash
PYTHONPATH=. python -m benchmark.replay --n 30
```

It runs 30 seeded eval questions through the same `RagService` the HTTP route
uses and exits non-zero if any stage's p95 exceeds its allocation. Measured on
this laptop (Apple Silicon, local Postgres, `hybrid+rerank`, bge):

| stage | p50 | p95 | allocation |
|---|---|---|---|
| plan | 0.05 ms | 0.07 ms | 5 ms |
| embed | 20.9 ms | 29.6 ms | 120 ms |
| cache.lookup | 1.8 ms | 2.1 ms | 25 ms |
| retrieve | 8.2 ms | 14.0 ms | 60 ms |
| fuse | 0.01 ms | 0.04 ms | 5 ms |
| rerank | 899.1 ms | 914.1 ms | 1200 ms |
| generate | — | — | 2000 ms |

**The reranker is the entire retrieval cost** — 914 ms of a ~960 ms path — and
**generation is the entire budget**, 57% of it, and it is an assumption rather
than a measurement because no key exists. The replay prints `SKIPPED` for that
stage and refuses to check the end-to-end total, because a budget check that
quietly omits the dominant stage reports green while measuring under half the
request. The allocations live in one place (`BUDGET` in `benchmark/replay.py`);
a test fails if the document's copy drifts from it.

## The regression gate

`python -m evals.gate` re-runs the eval harness against whatever is in the
database and compares it to [`evals/baseline.json`](evals/baseline.json), which
holds real measured numbers from a named commit — not a target. Exit 0 on pass,
1 on regression.

- **recall@10, 2 point tolerance** — the aggregate.
- **per-document recall@10, 5 point tolerance** — because 224 of the 268
  answerable pairs come from one document, so a 6-pair document can go to zero
  while the aggregate barely moves. Wider tolerance because at n=6 a single pair
  is worth 16.7 points.

`--update-baseline` rewrites one entry and shouts about it: that is a deliberate
act belonging in its own commit, never a fix for a red gate.

CI ([.github/workflows/ci.yml](.github/workflows/ci.yml)) runs the **lexical**
gate on every pull request — it needs no model download and still catches
regressions in normalization, chunking, chunk ids and the generated tsvector,
which is the surface the dense path shares. The dense gate needs a 2.2 GB model,
so it runs nightly and on `workflow_dispatch`. The trade-off is written out in
the workflow file.

## Layout

```
app/
  service.py          the /ask pipeline, assembled — runnable without HTTP
  deps.py             lazily built singletons; importing the app loads no model
  api/                ask.py (SSE + JSON), stats.py, health.py
  models/chunks.py    one table: text, normalized text, generated tsvector,
                      one nullable vector column per benchmarked model
  models/query_cache.py  the semantic cache, same pgvector table family
  planning/           dialect.py (Gulf→MSA lexicon), planner.py (noop/rules/llm)
  retrieval/          embed.py (4 models, 1 interface), search.py (dense /
                      lexical / RRF hybrid), rerank.py (bge cross-encoder),
                      cache.py (semantic cache with digit/negation guards)
  generation/         base.py (Provider protocol, error taxonomy), providers.py
                      (Anthropic, Gemini), failover.py, budget.py (context fit)
  observability/      tracing.py (OTel spans + Prometheus), cost.py (spend cap)
ingestion/            fetch -> normalize -> chunk -> load -> backfill, CLI in
                      __main__.py; chunks split on المادة boundaries, stable
                      ids "doc:article:seq"
evals/
  data/eval_pairs.jsonl  283 labelled pairs (229 MSA / 54 Gulf, 15 unanswerable)
  schema.py metrics.py   dataset validation; recall@k, MRR, hit rate
  harness.py             one config in, one dict of numbers out (+ by_doc split)
  gate.py baseline.json  the CI regression gate
benchmark/
  run.py                 the full matrix -> benchmark/results/results.json
  replay.py              the latency-budget gate
docs/                  benchmark writeup, latency budget, design spec
terraform/             empty; phase 4
```

Dependency direction: `evals/` and `benchmark/` use `app/` only through the
retrieval interface and the service object, so both stay independently runnable
and readable.

## Not built yet

Deliberately absent, in the order they are planned:

- **Phase 4 — Terraform.** Cloud Run (scale to zero), Cloud SQL with pgvector,
  a Pub/Sub topic for asynchronous ingestion, Artifact Registry, Cloud Trace.
  `terraform/` is an empty directory. Nothing has been deployed; the Cloud Run
  cold-start and CPU-only reranker costs in particular are un-measured, and the
  latency budget assumes MPS.
- **`POST /ingest`** and the Pub/Sub ingestion entrypoint. Ingestion runs as a
  local CLI only.
- **LLM-judged faithfulness/correctness evals.** The deterministic retrieval
  layer is what gates CI; the judged layer is designed (fixed judge model,
  versioned prompt, spend cap) and unwritten, and needs a key.
- **A Grafana dashboard JSON.** `/metrics` exports; nothing consumes it yet.
- **Anything that requires an API key**, which is a longer list than it looks:
  live generation end to end, the `LLMPlanner` ablation, judged evals, and the
  OpenAI/Cohere embedding columns (wired, unit-tested against mocks, unmeasured).

Also not done, and known: the Postgres FTS index uses the `simple` config
because Postgres ships no Arabic stemmer, which is most of why lexical retrieval
scores 0.356 recall@10; the rerank confidence floor is not dialect-neutral (see
the score-semantics note in `app/retrieval/rerank.py`), and it is why a
well-formed Gulf question can still be refused; and the semantic cache is keyed
on a 1024-dim embedding, so a 3072-dim OpenAI embedder disables caching rather
than silently mis-keying it.
