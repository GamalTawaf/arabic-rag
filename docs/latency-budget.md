# Latency budget — `POST /ask`

Target: **p95 ≤ 3.5 s end to end**, from the design spec (§7). This document
allocates that target across the pipeline stages, says where each number came
from, and points at the script that fails when a stage outgrows its allocation.

Two things to read before the table:

- **Every retrieval number here is a laptop measurement** — Apple Silicon, MPS,
  Postgres in a local Docker container on the same machine, a 233-chunk corpus
  that fits entirely in cache. It is not a production p95. What it is good for is
  the *shape* of the budget: which stage dominates, and by how much.
- **The generation allocation is an assumption, not a measurement.** No API key
  exists in this environment, so the stage has never run. It is the largest line
  in the budget and the least evidenced one. It is labelled as such everywhere it
  appears.

## The allocations

Source of truth: the `BUDGET` dict in
[`benchmark/replay.py`](../benchmark/replay.py). The table below is the output of
`PYTHONPATH=. python -m benchmark.replay --print-budget`, and
`tests/test_replay.py::test_budget_table_in_doc_matches_code` fails if this copy
drifts from that dict.

<!-- BUDGET TABLE START -->
| stage | p95 allocation (ms) |
|---|---|
| `plan` | 5 |
| `embed` | 120 |
| `cache.lookup` | 25 |
| `retrieve` | 60 |
| `fuse` | 5 |
| `rerank` | 1200 |
| `generate` | 2000 |
| **sum of stages** | **3415** |
| unallocated overhead | 85 |
| **end-to-end target** | **3500** |
<!-- BUDGET TABLE END -->

The stage names are the ones `RagService` records, so a new pipeline stage that
nobody allocated for fails the replay instead of quietly going unmeasured.

## Where each number comes from

| stage | measured p95 | allocation | basis |
|---|---|---|---|
| `plan` | 0.07 ms | 5 | Rule-based Gulf→MSA rewrite: regex and a ~30-entry lexicon, no I/O. The allocation is granularity, not headroom. An `LLMPlanner` would move this to a full provider round-trip and the budget would have to be rewritten. |
| `embed` | 29.6 ms | 120 | One query through `bge-m3` on MPS, batch of 1. **~4× headroom on purpose**: this is the stage most sensitive to the host. A CPU-only Cloud Run container has no MPS, and batch-of-1 transformer inference is exactly where that hurts. |
| `cache.lookup` | 2.1 ms | 25 | One pgvector cosine query against the `query_cache` table on a local Postgres. Headroom is for a real network hop and a connection-pool wait. |
| `retrieve` | 14.0 ms | 60 | Covers the *whole* retrieve stage, which for a Gulf question is two searches (original + MSA rewrite), each of which is itself a dense leg and a lexical leg. Consistent with the phase-2 benchmark: dense alone 2.6 ms mean / 3.9 ms p95, hybrid ~4 ms per search. |
| `fuse` | 0.04 ms | 5 | RRF over ≤4 ranked lists of ≤20 hits. Pure Python, no I/O. |
| `rerank` | 914 ms | 1200 | `bge-reranker-v2-m3` cross-encoder over 20 candidates on MPS. **This is the entire retrieval cost**: it is ~20× everything else in the pipeline put together. |
| `generate` | **not measured** | 2000 | **Assumption** — see below. |

Measurements are the p50/p95 printed by `python -m benchmark.replay --n 30`
against the dev database (30 seeded eval questions, `hybrid+rerank`, `bge`); the
full output is reproduced at the bottom of this page.

## The generation assumption

The number to argue with. There is no API key in this environment, so nothing
here has been observed:

- Primary provider is `claude-haiku-4-5` (`settings.anthropic_model`), streamed.
- Assumed ~350 ms to first token and ~120 output tokens/s.
- A citation-grounded answer over ≤5 retrieved articles is assumed to run
  **150–250 output tokens**, which lands at roughly 1.6–2.4 s of streaming.
- 2000 ms is the middle of that, and it is the *whole stream*, not
  time-to-first-token — `RagService` times the generate stage across the entire
  token loop.

Two consequences worth stating plainly rather than discovering later:

1. **`ANSWER_MAX_TOKENS` is 1024 and a 1024-token answer does not fit this
   budget.** At the assumed rate that is ~8.5 s of streaming on its own. The cap
   is a ceiling against a runaway generation, not the expected length; if typical
   answers turn out to run long, the budget is wrong, not the cap.
2. **A user does not wait for the total.** `/ask` streams, and the `citations`
   event is emitted before the first token, so what a user actually experiences
   is ≈ 970 ms of retrieval plus time-to-first-token — around 1.3 s — with the
   rest arriving as text. The 3.5 s target governs the complete response, which
   is what matters for a timeout, a trace and a cost, not for perceived speed.

The failover chain makes the tail worse in a way this budget does not model: a
primary that times out at `generation_timeout_s` (20 s) and fails over to Gemini
spends the timeout *plus* the second provider's generation. That request blows
the budget by design — the alternative is returning nothing — and it is visible
in the trace as a span event rather than hidden.

## What dominates

- **Within retrieval, the reranker is everything.** 914 ms of a ~960 ms
  retrieval path. Everything else — planning, embedding, both searches, fusion —
  sums to under 50 ms. The phase-2 benchmark priced that: the cross-encoder buys
  +1.9 points of recall@10 (0.946 → 0.965) for roughly 375× the latency of dense
  search alone. It is worth it here because the corpus is small and the answers
  are legal text where a wrong article is expensive; on a latency-critical
  surface the honest move is `config=dense` (2.6 ms mean, recall@10 0.946), which
  `/ask` already accepts per request.
- **Across the whole request, generation dominates** — 2000 of 3415 allocated
  milliseconds, 57% of the budget, and it is the assumed number. If the assumption
  is wrong, the budget is wrong.
- **The 85 ms of unallocated overhead** is deliberate: FastAPI routing, request
  validation, connection checkout, SSE framing and JSON serialisation belong to
  no single stage. It is slack, not a measurement.

## Enforcing it

```bash
DATABASE_URL=postgresql+asyncpg://rag_user:rag_pass@localhost:5433/rag_db \
  PYTHONPATH=. python -m benchmark.replay --n 30
```

Replays 30 seeded eval questions through the same `RagService` object the HTTP
route uses, records the per-stage timings the service already produces, and exits
**non-zero** if any stage's p95 exceeds its allocation. `--config` and `--model`
select the ablation; `--print-budget` prints the table above.

Three properties that make it a gate rather than a report:

- **A stage with no allocation is a failure.** Adding a pipeline stage without
  adding it to `BUDGET` fails the run, so the budget cannot silently stop
  describing the pipeline.
- **The first request is a discarded warm-up.** `bge-m3` and the cross-encoder
  load their weights on first use — measured at 4.2 s and 2.8 s in the run below.
  In a 30-sample p95 that single request *is* the p95, and the gate would be
  measuring process start-up instead of steady-state latency. The warm-up cost is
  real, so it is printed rather than dropped; it belongs to a deploy, not to a
  request. (A production service should warm both models on startup.)
- **Generation is skipped loudly when no provider is configured.** The replay
  prints `SKIPPED` for that stage and marks the end-to-end total as *not checked*.
  A budget check that silently omitted the dominant stage would report green
  while measuring under half of the request.

### Actual run

30 questions, `hybrid+rerank`, `bge`, dev database, 233 chunks, Apple Silicon
(MPS), Postgres in Docker on the same host:

```
replay: 30 eval questions   config=hybrid+rerank   model=bge   seed=0

  stage             n     p50 ms     p95 ms     budget   status
  plan             30       0.05       0.07          5   ok
  embed            30      20.94      29.59        120   ok
  cache.lookup     30       1.78       2.09         25   ok
  retrieve         30       8.21      14.03         60   ok
  fuse             30       0.01       0.04          5   ok
  rerank           30     899.10     914.06       1200   ok
  generate          -          -          -       2000   SKIPPED — no generation provider configured

  GENERATION NOT MEASURED: no provider is configured, so the stage that dominates the
  budget did not run and the end-to-end total is not a 3.5 s check. Set ANTHROPIC_API_KEY or GOOGLE_API_KEY
  to measure it.
  warm-up request (discarded, lazy model load): plan 0 ms  embed 4234 ms  cache.lookup 30 ms  retrieve 26 ms  fuse 0 ms  rerank 2828 ms
  cache hits: 0/30   refusals: 5/30
```

Three consecutive runs agreed to within 3 ms on every stage except `embed`
(20.9–21.4 ms p50), so the p50s are stable; the p95s are one sample each and
should be read as "the slow request in thirty".

Measured retrieval path: **~960 ms p95** against an allocation of 1415 ms, so the
retrieval half of the budget has ~30% headroom on this hardware. `cache hits: 0`
is expected — the replay never generates, so it never stores, so every request
takes the cold path, which is the path a budget should be measured on. `refusals:
5/30` are questions whose top rerank score fell below `rerank_min_score`; those
requests skip generation by design, which is one more reason the total is not
checkable here.

At n=30 the nearest-rank p95 is the second-largest sample. Treat it as "the slow
request in thirty", not as a smooth quantile.

## Known gaps

- Generation is unmeasured, and it is 57% of the budget.
- The `total` line is therefore never checked against 3.5 s in CI. Only the
  per-stage retrieval allocations are enforced today.
- Numbers are single-host, single-client, no concurrency. Connection-pool
  contention, HNSW cold cache, and cross-AZ latency to Cloud SQL are all absent.
- The reranker allocation assumes MPS. On a CPU-only Cloud Run instance the
  cross-encoder is the first thing that will blow this budget, and the mitigation
  is already available per request: `config=hybrid` or `config=dense`.
