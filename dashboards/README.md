# The cost & latency dashboard

Prometheus + Grafana over the metrics the service already exports from
`app/observability/tracing.py`. One dashboard, ten panels, no clicking:
`grafana-arabic-rag.json` is provisioned into Grafana at boot, and the
Prometheus datasource with it.

**Read this first: with no LLM API key, almost every panel on this dashboard is
empty — not just the cost ones.** A keyless `/ask` fails in the `get_service`
dependency and returns 503 *before the route body runs*, so `record_request`
never fires, no pipeline stage ever opens a span, and no counter moves. The
dashboard is not broken in that state; the service genuinely did nothing worth
measuring. [Filling it without a key](#filling-it-without-an-api-key) below is
how the numbers in this file were produced.

## Run it

Everything in one project, the app inside Docker:

```bash
docker compose -f docker-compose.yml -f docker-compose.observability.yml up -d
open http://localhost:3000/d/arabic-rag     # anonymous admin, no login
open http://localhost:9090/targets          # scrape health
open http://localhost:16686                 # Jaeger, from docker-compose.yml
```

Both `-f` flags are required. Compose merges the two files into one project on
one network, which is what lets Prometheus resolve `app:8000`.

Or with `uvicorn app.main:app --port 8000` on the host and only the observability
stack in Docker:

```bash
docker compose -f docker-compose.observability.yml up -d
```

`prometheus.yml` scrapes **both** `app:8000` and `host.docker.internal:8000` in
one job, so either way works with no edit. That means `up{job="arabic-rag"}`
normally shows **one target up and one down** — that is the design, not a fault.
Every panel sums over `instance`, so the dead target contributes nothing, and the
`Instance` variable at the top lets you pin one explicitly.

Two details worth knowing before you debug a blank graph:

- The scrape path is `/metrics/` **with the trailing slash**. `app/main.py`
  mounts the exposition app at `/metrics`, and Starlette answers the bare path
  with a 307. Prometheus would follow it, at the cost of two requests per scrape.
- Editing `grafana-arabic-rag.json` is the whole edit loop: the repo directory is
  bind-mounted into Grafana and the file provider re-reads it within ~10 s. Grafana
  cannot save over it (`allowUiUpdates: false`) — change the file, not the UI.

## The metrics that actually exist

Curled from a running instance, not from the docs. Everything below is emitted by
`app/observability/tracing.py`; there is nothing else in the `rag_*` or `gen_ai_*`
namespaces.

| series | labels | written by |
|---|---|---|
| `rag_requests_total` | `route`, `status` ∈ `ok`\|`spend_cap`\|`providers_failed` | `record_request`, from `app/api/ask.py` |
| `rag_stage_duration_seconds_{bucket,sum,count}` | `stage` ∈ `plan`\|`embed`\|`cache.lookup`\|`retrieve`\|`fuse`\|`rerank`\|`generate`, `error` | the `span()` context manager, i.e. every stage |
| `rag_cache_lookups_total` | `hit` ∈ `true`\|`false` | `record_cache_lookup`, once per request |
| `rag_cost_usd_total` | `gen_ai_provider_name`, `gen_ai_request_model` | `record_llm_call` |
| `gen_ai_client_token_usage_total` | `gen_ai_provider_name`, `gen_ai_request_model`, `gen_ai_token_type` ∈ `input`\|`output` | `record_llm_call` |
| `target_info` | `service_name`, `service_instance_id`, … | the OTel Prometheus exporter |

Every series also carries `otel_scope_name="arabic-rag"` and two empty
`otel_scope_*` labels; the dashboard aggregates them away.

Three things a reviewer will look for and **not** find, because the service does
not emit them:

1. **No refusal counter.** Panel 4 derives it — see below.
2. **No failover counter.** `app/generation/failover.py` records each hand-off as
   an OpenTelemetry span *event* (`provider.failover`), which lives in Jaeger.
   Prometheus cannot show it. Panel 10 shows the consequences instead — which
   provider is answering, and whether the chain exhausted. Upgrade path, one line:
   a `rag.failovers` counter labelled `from`/`to`/`kind` next to the
   `add_event(FAILOVER_EVENT, …)` call in `_record_failover`.
3. **No `daily_spend_cap_usd` gauge.** The cap is config, so panel 5's gauge
   maximum is the hard-coded 5.00 USD default. `GET /stats` is authoritative.

## What each panel answers

| # | panel | question | empty without an API key? |
|---|---|---|---|
| 1 | /ask request rate | is anything using it | yes — no request completes |
| 2 | /ask error rate | what share of served requests ended in a 502/503 | yes |
| 3 | semantic cache hit ratio | how much traffic never reaches retrieval at all | yes |
| 4 | refusal rate (derived) | how often the rerank floor says "not in corpus" | yes |
| 5 | LLM spend (needs an API key) | how close is the day to the spend cap | **yes, structurally** |
| 6 | /ask requests by outcome | `ok` vs `spend_cap` vs `providers_failed`, over time | yes |
| 7 | stage mean vs its budget allocation | which stage is eating its allowance | partly — `generate` bar is absent |
| 8 | mean time per stage (stacked) | where a typical request actually goes | partly — same |
| 9 | token throughput and burn rate | tokens/s in and out, USD/hour at that rate | **yes, structurally** |
| 10 | provider failover | which provider is answering; did the chain exhaust | **yes, structurally** |

"Structurally" means the panel reads from `record_llm_call`, which only runs when
a provider actually answers. No key, no series, forever — not a scrape problem.

### Panel 4: the refusal rate is derived, and here is the arithmetic

The service counts exactly one cache lookup per request, and a *miss* is followed
by exactly one of two outcomes: the rerank floor refuses (`app/service.py`
`_is_refusal`), or the request reaches `generate`. So

```
refusals = cache misses − generate stages
```

is exact, not an estimate — including for spend-cap rejections, which happen after
the refusal branch and increment neither term.

Verified against counter deltas, not by argument: five questions, three of them
outside the corpus (pizza recipe, car engine, 2018 World Cup) and two inside it,
moved the counters by **+5 misses and +2 generates**. Derived refusals = 3;
questions that came back `"refused": true` = the same 3.

It is exact *for today's pipeline*. Adding a third way for a non-cached request to
end before `generate` would silently fold into "refused".

> Upgrade path, ~3 lines: give `rag.requests` a `refused` label in
> `app/api/ask.py`, or add a `rag.refusals` counter next to `record_cache_lookup`
> in `tracing.py`, and replace the expression with a ratio of two counters.

### Panels 7 and 8: the latency budget, and why neither panel shows p95

Both panels show the **mean**, from the histogram's `sum`/`count`. The budget is
written in p95, so that needs justifying.

The p95 was drawn first, and it lied on a rendered dashboard. It has to be
interpolated from the fixed buckets in `tracing.py` —
`0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0 s` — and not one
boundary sits near an allocation. The smallest is 5 ms, which is also the whole
allocation for `plan` and for `fuse`, so both stages report a p95 of 4.75 ms —
95% of budget, permanently orange — when their true p50s are ~0.05 ms and
~0.01 ms. Higher up it is no better; measured in one live window:

| stage | p95 as the buckets report it | exact mean (`sum`/`count`) |
|---|---|---|
| rerank | 2425 ms | 1787 ms |
| embed | 800 ms | 149 ms |

A budget panel that cries wolf on two stages and cannot resolve a third is worse
than one that shows a number it can state exactly. So:

- **Panel 7** is the mean against each stage's allocation from `BUDGET`
  (`benchmark/replay.py`): plan 5, embed 120, cache.lookup 25, retrieve 60,
  fuse 5, rerank 1200, generate 2000 ms, as that bar's maximum, orange at 80%,
  red at 100%. Read it honestly: a mean **over** the allocation is unambiguously
  bad; a mean under it is **not** proof of p95 compliance.
- **Panel 8** is the same means stacked over time — means add up, so stacking is
  arithmetic that holds, which stacked p95s would not be. Each stage averages
  over the requests that ran it (a cache hit never reaches `rerank`), so the
  stack is the profile of a typical request, not an exact end-to-end mean. The
  3500 ms line is `TOTAL_BUDGET_MS`.
- The exact per-stage **p95** gate is unchanged and lives where it is exact:
  `PYTHONPATH=. python -m benchmark.replay --n 30`.

Putting p95 back on the dashboard is one line in a file this dashboard does not
own: add `0.9, 1.2, 2.0, 3.5` to `_LATENCY_BUCKETS_S` in
`app/observability/tracing.py` so the boundaries straddle the allocations, then
switch these two panels to
`histogram_quantile(0.95, sum by (stage, le) (rate(rag_stage_duration_seconds_bucket[$__rate_interval])))`.

One thing the panel showed immediately, on this laptop, that the per-request
`stages_ms` map does not make obvious: **traffic containing Gulf-dialect questions
roughly doubles mean rerank time** (879 ms over MSA-only questions, 1787 ms over a
mix including Gulf). The mechanism is in `app/service.py` — the rules planner emits
a second, MSA-rewritten query, and the cross-encoder scores the candidates against
both — so the dialect work the README measures in *recall* has a visible latency
price too.

### Panel 2: what the error rate does not see

`rag_requests_total` is incremented inside the `/ask` route, so it counts what the
route decided: `ok`, `spend_cap` (503), `providers_failed` (502). A 503 raised by
the `get_service` dependency because no API key is configured, and a 422 from
request validation, never reach it. If you need true HTTP-level error rates, the
FastAPI OTel instrumentation in `setup_tracing` is the place that would provide
them (it only activates when `OTEL_EXPORTER_OTLP_ENDPOINT` is set, which the
compose file does set).

## Filling it without an API key

There is no key in this repo and no stub provider in `app/`, so to see the
dashboard with data in it, run the real app with a stub `Provider` injected over
the dependency. Save this outside the repo, e.g. `/tmp/verify_app.py`:

```python
import asyncio
from collections.abc import AsyncIterator, Callable

from app import deps
from app.config import settings
from app.generation.base import Completion, Usage, usage_cost_usd
from app.main import app
from app.observability.cost import get_spend_tracker
from app.service import RagService

PRICE = (1.0, 5.0)  # USD per 1M tokens — invented, no key exists here


class StubProvider:
    name, model = "stub", "stub-model"

    def _usage(self, system: str, messages: list[dict]) -> Usage:
        n_in = len(system.split()) + sum(len(m["content"].split()) for m in messages)
        return Usage(n_in, 17, usage_cost_usd(n_in, 17, PRICE))

    async def complete(self, system, messages, max_tokens=1024) -> Completion:
        await asyncio.sleep(0.05)
        return Completion("[stub]", self._usage(system, messages), self.name, self.model)

    async def stream(self, system, messages, max_tokens=1024, *, on_usage=None) -> AsyncIterator[str]:
        for chunk in ("[stub ", "answer]"):
            await asyncio.sleep(0.01)
            yield chunk
        if on_usage:
            on_usage(self._usage(system, messages))

    def price(self) -> tuple[float, float]:
        return PRICE


app.dependency_overrides[deps.get_service] = lambda: RagService(
    deps.build_embedder(), deps.build_reranker(), deps.build_planner(),
    StubProvider(), get_spend_tracker(), settings,
)
```

```bash
PYTHONPATH=.:/tmp .venv/bin/python -m uvicorn verify_app:app --port 8000
curl -s -X POST localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question":"كم مدة الإشعار قبل إنهاء العقد؟","stream":false}'
```

The cost and token panels then show *stub* numbers at an invented price, and
`gen_ai_provider_name="stub"` labels them as such on the graph. They are a wiring
check, not a cost measurement.

## What has been verified, and what has not

Verified on this machine (macOS, Docker 29.4.3, Compose v5.1.4, Prometheus
v3.1.0, Grafana 11.5.2), against the real service with the stub provider above:

- Prometheus scrapes the app: `http://host.docker.internal:8000/metrics/` → `up`,
  `http://app:8000/metrics/` → down with a DNS error, exactly as documented.
- Grafana provisions the datasource (`Prometheus`, uid `prometheus`, default) and
  the dashboard (`arabic-rag — cost & latency`) from the inline compose configs,
  with no provisioning errors in its log; all ten panels parse.
- Every panel query returns data through Grafana's own `/api/ds/query`, except
  `providers_failed` — which is correctly empty, because no provider failed.
  Sample values from one run: request rate 0.109 req/s, cache hit ratio 50%,
  refusal rate 33.3%, spend 0.0098 USD, input tokens 41.1/s, output 1.2/s,
  rerank mean 879 ms / p95 975 ms.
- `docker compose -f docker-compose.yml -f docker-compose.observability.yml config`
  merges into one project on one network.
- The dashboard was opened in a real browser at 1600×1200 and screenshotted, with
  data in every panel. That render is what caught the two defects since fixed: the
  stat row was two grid units too short and clipped its own sparklines, and the p95
  bars sat at 95% of budget for `plan` and `fuse` purely from bucket interpolation.
  Final render, traffic through the stub: request rate 0.55 req/s, error rate
  0.0%, cache hit ratio 10.0%, refusal rate 88.9%, spend 0.0286 USD, and every
  stage bar green — rerank 885.0 ms against its 1200 ms allocation, embed 16.2 ms
  against 120 ms. That refusal rate is not a panel bug: a third of the traffic was
  deliberately outside the corpus, and the rest ran into the then-uncalibrated
  `rerank_min_score` floor of 0.15. Making that visible is the point of panel 4 —
  and it worked: that 88.9% is what prompted the sweep in
  [docs/refusal-calibration.md](../docs/refusal-calibration.md), after which
  `rerank_min_score` is 0.0 and the same traffic would show a refusal rate near
  the unanswerable share alone. The render above is therefore a *pre-calibration*
  capture; panel 4 is still correct, its input changed.

Not verified: **the `app:8000` scrape target has never been up.** Every run above
used uvicorn on the host, because this checkout's populated Postgres (233 chunks,
e5 + bge backfilled) is a standalone container, not the compose `db` service, and
the in-container app would have come up against an empty database. The
`host.docker.internal:8000` path is proven; the `app:8000` path is a one-line DNS
name in the same job that Compose is confirmed to put on the same network. Also
not verified: any panel with real LLM numbers in it. There is no API key in this
environment, so every cost and token value ever displayed here came from the stub
provider at an invented price.
