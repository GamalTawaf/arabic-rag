# ROADMAP — Arabic-first RAG Service

**Updated:** 2026-07-25 · **Status:** all four phases built

## Why this project

Portfolio artifact for AI-engineer roles in the GCC market. The differentiator is not the chatbot — it is the eval harness, the benchmark numbers, the traces, and the cost controls around it. Converts 12 years of backend work plus native Arabic into signal most applicants cannot fake.

## Where it stands

575 tests pass, 1 skipped (opt-in real-cross-encoder test). `ruff check .` clean. The eval gate, the latency replay, and the benchmark all run green against real data.

Two things have **never** run, and every document in the repo says so: an actual LLM API call (no key on this machine), and `terraform apply` (no GCP credentials).

| Phase | State |
|---|---|
| 1 — corpus, normalization, chunking, chunk store, eval metrics | done |
| 2 — retrieval, embedders, reranker, benchmark, CI gate | done |
| 3 — planning, generation failover, tracing, cache, `/ask` | done |
| 4 — Terraform, Pub/Sub ingestion, refusal calibration, Grafana | done |

## The findings (all measured in this repo, reproducible)

**Dense beats lexical decisively on Arabic.** recall@10 0.93–0.95 versus 0.356 for Postgres full-text search. The `simple` tsvector config has no Arabic stemmer, and it shows.

**The Gulf-dialect penalty is real and model-dependent.** Against an MSA-matched control (same gold chunks, same questions, only the register changes), `multilingual-e5-large` loses 9.9 points of recall@10 and 19.7 points of MRR. `BAAI/bge-m3` is nearly flat (−0.4). The damage lands on ranking, not coverage.

**A 77-entry dialect lexicon recovers most of it.** Rule-based Gulf→MSA rewriting takes e5 from 0.85 to 0.92 recall@10 on the 50 Gulf pairs. The gain shrinks at retrieval depth 20 (what the service actually runs) — documented rather than glossed. n=50, so direction is more trustworthy than magnitude.

**RRF hybrid hurts top-3 precision.** Fusing a strong dense leg with a weak lexical one collapses recall@3 from 0.86 to 0.49. Traced to unweighted RRF arithmetic, confirmed not a bug. The reranker repairs it; without a reranker, dense-only is the right production config.

**Reranking costs 375× the latency for 1.9 points of recall@10.** ~975 ms/query on Apple Silicon MPS versus ~2.6 ms for dense alone, and 4.3× worse on CPU (measured: 3972 ms p95, blowing its 1200 ms allocation).

**The refusal gate had to be switched off — the strongest result in the repo.** `rerank_min_score = 0.15` was refusing 39 answerable questions to catch 7 unanswerable ones, and it was dialect-biased: **54% of Gulf questions refused versus 5.5% of MSA**. Swept over all 283 pairs, refusal precision peaks at 0.171 — the cross-encoder score simply does not separate the two populations. Threshold is now 0.0 and `docs/refusal-calibration.md` writes up the negative result plus ranked upgrade paths.

## What is in the repo

```
app/          service.py (/ask pipeline), deps.py, api/{ask,stats,ingest,health}.py
              retrieval/{embed,search,rerank,cache}.py  generation/{base,providers,failover,budget}.py
              planning/{planner,dialect,lexicon}.py     observability/{tracing,cost}.py
ingestion/    fetch, normalize, chunk, pipeline, backfill, CLI (fetch|ingest|backfill|stats)
evals/        schema, metrics, harness, gate, refusal, dataset_stats, data/eval_pairs.jsonl, baseline.json
benchmark/    run.py, replay.py (BUDGET = single source of truth for latency), results/results.json
terraform/    Cloud Run + Cloud SQL + Pub/Sub + Artifact Registry + Secret Manager, least-privilege IAM
dashboards/   Grafana dashboard + Prometheus scrape config
docs/         benchmark.md, refusal-calibration.md, latency-budget.md, specs/
```

Corpus: 5 real Qatari legal documents from Al Meezan (Labour Law 14/2004, Domestic Workers 15/2017, Minimum Wage 17/2020, Heat Stress Decision 17/2021, Dispute Committees Decision 6/2018) → 233 article-boundary chunks.

Dataset: 283 labelled pairs — 229 MSA, 54 Gulf, 268 answerable, 15 unanswerable. Zero dangling chunk citations. 79.8% corpus coverage.

## Local setup

```bash
docker compose up -d db                       # pgvector on 5433 (creates rag_db AND rag_test)
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements-dev.txt
uv pip install --python .venv/bin/python -r requirements-models.txt   # torch, for the local models
alembic upgrade head && python -m ingestion ingest
python -m ingestion backfill --model bge      # and --model e5
```
Dev database is `rag_db`; tests use `rag_test` and never touch dev data.

## Known gaps (all documented in-repo)

- Generation never executed. The 2000 ms `generate` allocation — 57% of the latency budget — is an assumption, so the 3.5 s end-to-end total is never actually checked.
- Refusal now has no positive signal at all. Switching off the score gate was right on the evidence, but an unanswerable question now gets answered from the five nearest articles unless the model itself declines — and model abstention is unmeasured.
- `terraform plan/apply/destroy` never run. Schema-validated against the real `hashicorp/google` 7.41.0 provider, but with OpenTofu 1.12.2 rather than the `terraform` binary, which is not installed here.
- Cloud Trace gets Cloud Run's request spans but not the app's per-stage spans — the OTLP/gRPC exporter cannot attach Google credentials.
- Model weights (~4.4 GB) are not baked into the image, so a cold Cloud Run instance downloads them on first request.
- OpenAI and Cohere embedding columns, the LLM-judged faithfulness suite, and the LLMPlanner ablation are all wired and mock-tested but unmeasured — one API key unblocks all of them.

## Next three things

1. **Get one API key and spend a day on everything it unblocks.** Model abstention against the 15 unanswerable pairs (the named replacement for the refusal gate), then the `generate` stage measurement, then the LLM-judged suite and the LLMPlanner ablation. One key retires most of the gaps list.
2. **Decide the reranker's fate on CPU.** It cannot meet its own budget without a GPU. Either re-derive the allocation from CPU numbers in its own commit, or ship `config=hybrid` at a measured cost of 1.9 points recall@10.
3. **Implement top1-vs-top5 margin refusal.** Cheapest item on the calibration doc's list — no new model, no key — and it directly tests whether the *gap* between candidates is dialect-neutral even though the absolute score demonstrably is not.
