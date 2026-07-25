# HANDOVER — Arabic-first RAG Service (hiring-portfolio artifact)

**Date:** 2026-07-24 · **Status:** design approved (sections 1–4), work starting · **Repo:** `~/projects/personal/arabic-rag`

## Why this project

Portfolio artifact for AI-engineer roles (GCC/Doha market). Differentiators most applicants lack: eval harness, latency budgets, cost dashboards, OTel tracing. Converts Gamal's 12 yrs backend + Arabic fluency into unfakeable signal. A second, smaller future project (agentic tool-calling with permission model + audit log) is out of scope here.

## Decisions locked (user-approved)

| Decision | Choice |
|---|---|
| Corpus | Qatar labor law + ministry regulations (MSA, public, verifiable answers) |
| Dialect angle | Gulf-dialect *questions* against MSA docs (eval subset, ~50 pairs) |
| Budget | Local-first, ~$0. GCP exposed only when needed (`terraform apply` before demo, `destroy` after; Cloud Run scale-to-zero) |
| Generation | Provider-agnostic: thin `Provider` protocol + 2 impls (Anthropic, Gemini), timeout/429 failover. **No LiteLLM.** |
| Scope | RAG + query planning (dialect→MSA rewrite, question decomposition). Not a full tool-calling agent. |
| Timeline | 6–8 weeks thorough: full benchmark matrix, ~250 eval pairs, polished writeup |
| Architecture | Single monorepo, clear boundaries. Rejected managed Vertex AI retrieval (kills benchmark story). |
| Foundation | Copy of `~/projects/personal/boilerplate` (async FastAPI, SQLAlchemy, Alembic, pytest, Docker, CI) |
| No fine-tuning | Explicitly skipped — retrieval/evals/reliability is the signal |

## Approved design (4 sections)

### 1. Repo layout & boundaries
```
arabic-rag/
├── app/            # FastAPI service: api/ (ask, ingest, health), retrieval/, planning/, generation/, observability/
├── ingestion/      # fetch → clean → chunk → embed → load (local CLI + Pub/Sub-triggered on GCP)
├── evals/          # labeled Q/A JSONL + harness; CI regression gate
├── benchmark/      # 4-embedding-model comparison → results.json → writeup numbers
├── terraform/      # Cloud Run + Cloud SQL(pgvector) + Pub/Sub; ephemeral (apply/destroy)
└── docs/           # benchmark writeup, architecture, eval methodology, latency budget
```
Boundary rule: `evals/` and `benchmark/` touch `app/` only via HTTP API + a shared retrieval interface — independently runnable/readable.

### 2. Eval harness & dataset (BUILT FIRST)
- ~250 pairs JSONL, versioned: `id, question, dialect_tag (msa|gulf), answer, source_doc, source_chunk_ids`.
- Pass 1 LLM-generate from chunks → Pass 2 manual review of all (~200 MSA) → Pass 3 hand-write ~50 Gulf-dialect rephrasings (same ground-truth chunks). Plus ~15 unanswerable questions (measures refusal).
- Metrics: **retrieval** (deterministic, gates CI): recall@3/10, MRR vs chunk IDs. **End-to-end** (LLM-judged, on-demand): faithfulness + correctness; fixed judge model+prompt, versioned; ~20 human-scored pairs to check judge agreement once.
- CI: GH Actions, dockerized pgvector + frozen corpus snapshot; PR fails if recall@10 drops >2 pts vs `evals/baseline.json` (updated deliberately, never auto). LLM-judged suite manual-trigger with spend cap.
- Skipped: Ragas/DeepEval — ~200 lines of owned Python instead.

### 3. Retrieval & benchmark
- Ingestion: Arabic normalization (strip tatweel/diacritics, normalize alef/ya/ta-marbuta) for index; original kept for display. Chunk on article (المادة) boundaries, ~500-token cap, stable IDs `doc:article:seq`.
- One Postgres table: raw text, normalized text, tsvector, **one vector column per benchmarked model**.
- Hybrid: pgvector HNSW cosine + Postgres FTS → RRF → top-20 → BGE-reranker-v2-m3 (local) → top-5. Each stage toggleable per-request for ablations.
- Benchmark matrix: {multilingual-e5-large, BGE-m3 (local)} × {OpenAI text-embedding-3-large, Cohere embed-v4 or Voyage (API)} × 4 configs (dense/lexical/hybrid/+rerank) × 2 dialect tags → recall@3/10, MRR, latency, $/1K queries. One script → `benchmark/results.json`.
- Query planning: cheap-LLM pre-step — (a) dialect→MSA rewrite for retrieval, answer in user's register; (b) multi-part decomposition, sub-query fusion before rerank. Measured as ablations.

### 4. Service, observability, failure engineering
- `POST /ask` SSE (citations event, then tokens; JSON flag), `POST /ingest` (BackgroundTasks local / Pub/Sub GCP), `/health`, `/metrics` (Prometheus).
- Citations by chunk ID; explicit "not in corpus" path on low rerank confidence.
- Generation adapter: `Provider` protocol, Anthropic + Gemini impls (~60 lines each, official SDKs), failover on p95-timeout/429 → span event.
- OTel: one trace per /ask, spans plan→retrieve.dense→retrieve.lexical→fuse→rerank→generate; attrs: model, tokens, USD cost, cache hit, config. Jaeger locally, Cloud Trace on GCP. Latency budget doc: p95 ≤ 3.5s with per-stage allocations + replay script (30 eval Qs) failing blown budgets.
- Cost/failure: semantic cache (query-embedding similarity in same pgvector table, `cache` namespace; hit rate on spans), prompt token cap (trim lowest-ranked chunks whole), tenacity retry/backoff, per-day USD kill-switch (503 past cap), `/stats` + one committed Grafana dashboard JSON.

### 5. GCP/Terraform + phasing (folded in, not yet user-reviewed in detail)
- Terraform: Cloud Run (scale-to-zero) + Cloud SQL Postgres w/ pgvector (spun up for demo windows only) + Pub/Sub ingestion topic + Artifact Registry + Cloud Trace. Ephemeral by design.
- Local demo alternative: `cloudflared tunnel` to local docker-compose.
- Rough phases (6–8 wks): 1) corpus + ingestion + eval dataset; 2) retrieval + benchmark + writeup numbers; 3) service + planning + OTel + caching/failover; 4) Terraform + CI polish + published writeup w/ charts.

## Next steps (where we stopped)
1. Write full design spec → `docs/specs/2026-07-24-arabic-rag-design.md`, commit.
2. superpowers:writing-plans → phased implementation plan.
3. Phase 1 start: copy boilerplate in, strip example Item CRUD, pgvector in docker-compose, corpus fetch script.

## Process notes
- Ponytail mode active (lazy/minimal). Never push to main — branch + PR (user's global rule).
- Git repo already initialized (`main`) at `~/projects/personal/arabic-rag`, nothing committed yet.
