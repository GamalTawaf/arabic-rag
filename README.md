# arabic-rag

Retrieval-augmented question answering over Qatari labour legislation, built
Arabic-first: the corpus is Modern Standard Arabic, the questions arrive in MSA
*or* Gulf dialect, and the point of the project is to measure what that
difference costs.

All four phases are built — corpus and ingestion, eval dataset and benchmark,
the `/ask` service with planning/generation/tracing/caching/spend-cap, and a
Terraform stack for Cloud Run + Cloud SQL + Pub/Sub. 542 tests, a CI regression
gate on every PR, a nightly dense gate and latency replay.

**Two things have never run, and every claim below is written around that.**
There is no LLM API key in this environment, so generation is unit-tested
against mocked transports and `/ask` answers 503 until `ANTHROPIC_API_KEY` (or
`GOOGLE_API_KEY`) is set. And there are no GCP credentials, so the Terraform is
schema-validated against the real provider but has never been planned, applied or
destroyed. Everything else — planning, retrieval, reranking, the refusal gate,
the cache, the spend check, the traces, ingestion over HTTP and Pub/Sub — runs
and is measured on this machine.

| | |
|---|---|
| [The dialect penalty](#the-headline-number) and the full benchmark matrix | [docs/benchmark.md](docs/benchmark.md) |
| [What query planning recovers](#what-query-planning-recovers) | [docs/benchmark.md § Query planning](docs/benchmark.md#query-planning-does-a-gulfmsa-rewrite-recover-the-dialect-penalty) |
| [Why the refusal threshold is now off](#the-refusal-gate-that-had-to-be-switched-off) | [docs/refusal-calibration.md](docs/refusal-calibration.md) |
| [The latency budget](#the-latency-budget), MPS vs CPU | [docs/latency-budget.md](docs/latency-budget.md) |
| [The GCP stack](#the-gcp-stack-validated-never-applied) | [terraform/README.md](terraform/README.md) |
| [The dashboard](#the-dashboard) | [dashboards/README.md](dashboards/README.md) |
| Design spec | [docs/specs/2026-07-24-arabic-rag-design.md](docs/specs/2026-07-24-arabic-rag-design.md) |

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
[docs/benchmark.md](docs/benchmark.md).

Caveat stated up front: 233 chunks means top-10 is 4% of the index, and the MSA
questions were LLM-drafted from the chunks then reviewed by hand. These are not
query-log numbers.

## What query planning recovers

Phase 3's ablation, and the direct answer to the headline above. A hand-written
~30-entry Gulf→MSA lexicon (no model, no API key) rewrote 49 of the 50 Gulf
questions. Dense retrieval, **10 candidates per leg**:

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

**The depth matters and the shipped config is the weaker row.** The service
retrieves `top_k_retrieve = 20` per query, not 10, and re-run at that depth the
fused arm's recall@10 falls back to 0.89 (e5) and 0.93 (bge) — RRF's top 10 gets
diluted by deep candidates from both legs. Same code, same corpus, same 50 pairs.
Both tables are in `benchmark/results/results.json` under `planning`; the
depth-20 rows are quoted in `app/planning/planner.py` and `app/service.py`
because those are what the running service does.

The service ships the **fused** arm even though "rewritten" scores higher on raw
recall, because searching with both queries means a bad rewrite can only *add*
candidates — it can never remove the chunk the user's own words would have found.
The LLM rewriter is implemented but **unmeasured** for want of a key.
[docs/benchmark.md § Query planning](docs/benchmark.md#query-planning-does-a-gulfmsa-rewrite-recover-the-dialect-penalty).

## The refusal gate that had to be switched off

The most interesting negative result in the repo, and the reason
[docs/refusal-calibration.md](docs/refusal-calibration.md) exists.

`/ask` refuses to answer when the cross-encoder's top score falls below
`rerank_min_score`, which was a hand-picked 0.15. Swept over all 283 labelled
pairs, that floor was:

- refusing **39 answerable questions** to catch **7 of 15** unanswerable ones —
  and all 39 had the correct article *already retrieved into the context window*,
  30 of them at position one;
- refusing **54% of Gulf questions against 5.5% of MSA ones** — the same
  questions with the same correct answers, a 10x disparity that persists at every
  threshold that catches anything;
- unable to clear both a 10% false-refusal cap and a 50% refusal-precision floor
  at *any* value in a 0.00–0.90 sweep. Peak refusal precision is 0.171.

So `rerank_min_score` is now **0.0** — the score comparison is off, and an empty
result set is the only automatic refusal left. That is a measured decision with
its trade-off written down (wrongly answering a legal question is worse than
wrongly refusing one, which is precisely why a gate this biased could not be
kept), plus the prevalence arithmetic that would revisit it given a query log,
and three ranked replacements: model abstention, a top1-vs-top5 margin feature,
and a dialect-aware threshold. Reproduce the sweep in a second, from cached
scores:

```bash
python -m evals.refusal --sweep --compare 0.15
```

Visible consequence: the 30-question latency replay used to refuse 5 of 30
answerable questions. It now refuses 0.

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

`docker compose up -d db` creates **two** databases: `rag_db` for development and
`rag_test` for the suite, via `docker/init-rag-test-db.sql`. That file only runs
on an empty data volume, so if you already had this container before that file
existed, create it once by hand — otherwise 153 tests skip themselves and
`pytest` still exits 0:

```bash
docker compose exec db createdb -U rag_user rag_test
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
              "rerank_min_score":0.0,"top_k_retrieve":20,"top_k_context":5}}
```

(`cache.entries` is 0 on a fresh ingest; it grows as questions are answered.)

`providers.available: []` is the honest state of this checkout. Asking a question
in it returns a 503 that names the variable to set — this is real captured output,
not an illustration:

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

Two more response shapes from the same run, neither of which reached a provider:

```jsonc
// refusal — answered in the user's own register, provider never called, never
// cached. CAPTURED AT THE OLD rerank_min_score = 0.15. With the calibrated 0.0
// this envelope now appears only when retrieval returns nothing at all; see
// docs/refusal-calibration.md for why the score gate went away.
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
python -m evals.dataset_stats                # what is actually in the 283 pairs
python -m evals.gate --config lexical        # regression gate, exit 1 on a drop
python -m evals.gate --config dense --model bge
python -m evals.refusal --sweep              # the refusal threshold sweep
python -m benchmark.replay --n 30            # the latency-budget gate
```

Tests and lint:

```bash
pytest            # 541 pass, 1 skipped (it loads the 2 GB reranker; set
                  # RERANK_REAL_MODEL=1 to run it). Needs the rag_test database
                  # above — without it 153 more tests skip and pytest still
                  # exits 0.
ruff check .
```

## The service

`POST /ask` is the main surface. One request runs:

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
- **Refusal gate.** Answers "not in corpus" *in the user's register* rather than
  guessing. The score half of this gate is currently disabled on measured
  evidence — [see above](#the-refusal-gate-that-had-to-be-switched-off) — so it
  fires only when retrieval returns nothing. Refusals are never cached.
- **Spend cap.** Estimated cost is checked *before* the call, against a daily USD
  cap; past it, `/ask` returns 503 with the day's spend, not a silent overrun.

Generation sits behind a two-method `Provider` protocol with Anthropic and Gemini
adapters. Each adapter classifies its own SDK's failures into one `ErrorKind`, and
the failover policy is a pure function of that: rate-limit and 5xx retry then fail
over, timeout and connection errors fail over immediately, and a 4xx/auth error
**does not** fail over — it would fail identically on the next provider, so
burying it behind "all providers failed" helps nobody. Failover is recorded as a
span event.

### Ingestion over HTTP

`POST /ingest` takes one document and runs the same chunk → normalize → embed →
upsert pipeline the CLI does; `POST /ingest/pubsub` unwraps a Pub/Sub push
envelope around the identical payload. Real responses:

```console
$ curl -s -X POST localhost:8000/ingest -H 'content-type: application/json' \
    -d '{"doc_id":"readme-smoke-doc","title":"وثيقة تجريبية","text":"المادة 1 - …\n\nالمادة 2 - …"}'
{"doc_id":"readme-smoke-doc","documents":1,"chunks_written":2,"chunks_skipped":0,"model_key":"bge"}

$ curl -s -X POST localhost:8000/ingest/pubsub -H 'content-type: application/json' \
    -d '{"message":{"data":"<base64 of the same JSON>","messageId":"1"},"subscription":"…"}'
{"status":"ok","message_id":"1","doc_id":"readme-smoke-doc","documents":1,"chunks_written":1,…}

$ curl -s -X POST localhost:8000/ingest/pubsub -H 'content-type: application/json' \
    -d '{"message":{"data":"!!!not-base64!!!"}}'
HTTP/1.1 200 OK
{"status":"rejected","reason":"message.data is not valid base64: Only base64 data is allowed"}
```

That last one is the whole design. A permanently-malformed push gets **200**,
because Pub/Sub retries anything that is not 2xx and identical bytes fail
identically forever — the message becomes poison and the backlog never drains. A
dead database gets **5xx**, because that one really should be retried. Ingestion
is idempotent by construction (chunk ids are `doc:article:seq`, a pure function of
the text, upserted `ON CONFLICT DO UPDATE`), so at-least-once redelivery is safe,
and a test posts the same envelope twice and asserts the row count does not move.

Also: `/stats` (corpus, cache, spend, providers, retrieval config), `/health`,
`/health/db`, and `/metrics` (Prometheus). Tracing exports only when
`OTEL_EXPORTER_OTLP_ENDPOINT` or `OTEL_CONSOLE_EXPORT=1` is set, so it costs
nothing by default.

## The latency budget

p95 ≤ 3.5 s end to end, allocated per stage in
[docs/latency-budget.md](docs/latency-budget.md) and enforced by a replay script:

```bash
python -m benchmark.replay --n 30
```

It runs 30 seeded eval questions through the same `RagService` the HTTP route
uses and exits non-zero if any stage's p95 exceeds its allocation. Measured on
this laptop (Apple Silicon, local Postgres, `hybrid+rerank`, bge), and with MPS
forced off to see what the same code costs without an accelerator:

| stage | p50 (MPS) | p95 (MPS) | p95 (CPU) | allocation |
|---|---|---|---|---|
| plan | 0.05 ms | 0.07 ms | 0.07 ms | 5 ms |
| embed | 21.3 ms | 32.8 ms | 53.7 ms | 120 ms |
| cache.lookup | 2.2 ms | 2.5 ms | 3.0 ms | 25 ms |
| retrieve | 8.5 ms | 14.0 ms | 13.4 ms | 60 ms |
| fuse | 0.01 ms | 0.04 ms | 0.04 ms | 5 ms |
| rerank | 912.5 ms | 933.1 ms | **3972.4 ms** | 1200 ms |
| generate | — | — | — | 2000 ms |

**The reranker is the entire retrieval cost** — 933 ms of a ~980 ms path — and
**generation is the entire budget**, 57% of it, and it is an assumption rather
than a measurement because no key exists. The replay prints `SKIPPED` for that
stage and refuses to check the end-to-end total, because a budget check that
quietly omits the dominant stage reports green while measuring under half the
request.

The CPU column is the load-bearing one for deployment: everything except the
cross-encoder survives losing the GPU, and the cross-encoder is 4.3x over. That
is why the nightly CI replay gates `--config hybrid` and records
`hybrid+rerank` without gating it, and why `BUDGET` was **not** widened to fit
CPU. The allocations live in one place (`BUDGET` in `benchmark/replay.py`); a
test fails if the document's copy drifts from it.

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

CI ([.github/workflows/ci.yml](.github/workflows/ci.yml)) has three jobs:

| job | when | what |
|---|---|---|
| `ci` | push + PR | ruff, pytest against a real pgvector service, pip-audit |
| `eval-gate-lexical` | push + PR | ingest the committed corpus, gate **lexical** recall@10 |
| `eval-gate-dense` | nightly 03:17 + dispatch | gate **dense** recall@10, then the latency replay |

The lexical gate needs no model download and still catches regressions in
normalization, chunking, chunk ids and the generated tsvector — the surface the
dense path shares. The dense gate needs a 2.2 GB model, so it runs nightly, and
the latency replay rides along in that job because it needs the same corpus,
database and weights. The trade-offs are written out in the workflow file.

## The dashboard

`dashboards/grafana-arabic-rag.json` — ten panels over the Prometheus metrics
`app/observability/tracing.py` already exports, provisioned automatically:

```bash
docker compose -f docker-compose.yml -f docker-compose.observability.yml up -d
open http://localhost:3000/d/arabic-rag     # Grafana
open http://localhost:16686                 # Jaeger traces
```

Read [dashboards/README.md](dashboards/README.md) first: with no LLM key almost
every panel is empty, because a keyless `/ask` 503s in the dependency before the
route body runs and no counter ever moves. That file explains how the numbers in
it were produced without a key, and marks which panels are verified against a
live instance and which are not.

## The GCP stack (validated, never applied)

`terraform/` builds Cloud Run (gen2, scale-to-zero) + Cloud SQL Postgres 17 +
Pub/Sub topic/push-subscription/DLQ + Artifact Registry + Secret Manager + two
purpose-made service accounts, sized for a demo and designed to be destroyed the
same day.

**It has never been applied.** No GCP credentials, no billing account, no
project. `terraform fmt -check` and `terraform validate` pass against the real
`hashicorp/google` 7.41.0 schema — and the binary available here was OpenTofu
1.12.2, not `terraform`, which
[terraform/README.md § Validation](terraform/README.md#validation) states plainly
rather than glossing. `plan`, `apply` and `destroy` have never run, every cost
figure is list-price arithmetic with the working shown, and the README carries a
ranked list of what would break first.

## Layout

```
app/
  service.py          the /ask pipeline, assembled — runnable without HTTP
  deps.py             lazily built singletons; importing the app loads no model
  api/                ask.py (SSE + JSON), ingest.py, stats.py, health.py
  ingest_worker.py    framework-free ingestion + the Pub/Sub failure taxonomy
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
  refusal.py             the refusal-threshold sweep (cached scores, no model)
benchmark/
  run.py                 the full matrix -> benchmark/results/results.json
  replay.py              the latency-budget gate
terraform/            Cloud Run + Cloud SQL + Pub/Sub, validated, never applied
dashboards/           Grafana JSON + Prometheus config, provisioned
docs/                 benchmark, latency budget, refusal calibration, spec
```

Dependency direction: `evals/` and `benchmark/` use `app/` only through the
retrieval interface and the service object, so both stay independently runnable
and readable.

## Not built, and why

Accurate as of the current commit.

- **Live generation.** Both provider adapters, the failover policy, the context
  fitter and the spend cap are written and unit-tested against mocked transports,
  and none of them has ever spoken to an LLM. `/ask` 503s until a key is set.
- **LLM-judged faithfulness/correctness evals.** The deterministic retrieval
  layer is what gates CI; the judged layer is designed (fixed judge model,
  versioned prompt, spend cap) and unwritten, and needs a key. Measuring model
  abstention against the 15 unanswerable pairs — the first replacement for the
  disabled score gate — is the same blocker.
- **`terraform plan/apply/destroy`.** Never run. See
  [the section above](#the-gcp-stack-validated-never-applied) and
  terraform/README.md.
- **Model weights are not baked into the image.** The image installs CPU torch and
  `sentence-transformers` and imports cleanly (verified: 2.06 GB, `import app.main`
  succeeds), but bge-m3 and the cross-encoder — about 4.4 GB — download on first use
  rather than at build time. On Cloud Run that lands as a slow first request after
  each scale-to-zero cold start. The trade is noted in the `Dockerfile`.
- **The `LLMPlanner` ablation and the API-embedding rows** — OpenAI and Cohere
  embedding columns are wired and unit-tested against mocks, and both are 0% in
  `/stats` coverage because there is no key to backfill them with. Benchmark rows
  for them are recorded as `{"status": "not_run"}` rather than omitted.
- **Cloud Trace does not receive the app's spans.** The Terraform enables it and
  grants `cloudtrace.agent`, so Cloud Run's own request spans arrive; the
  per-stage plan/embed/retrieve/rerank spans do not, because
  `app/observability/tracing.py` exports OTLP/gRPC and that exporter cannot
  attach Google credentials. `otel_exporter_otlp_endpoint` defaults to `""` for
  that reason and terraform/README.md explains what would close the gap.

Also known and not fixed: the Postgres FTS index uses the `simple` config because
Postgres ships no Arabic stemmer, which is most of why lexical retrieval scores
0.356 recall@10; the semantic cache is keyed on a 1024-dim embedding, so a
3072-dim OpenAI embedder disables caching rather than silently mis-keying it; and
refusal, after the calibration above, has no positive signal at all — an
unanswerable question gets answered from the five nearest articles unless the
model itself declines, which is exactly the measurement that needs a key.
