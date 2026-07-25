# Arabic-first RAG Service — Design Spec

**Date:** 2026-07-24 · **Status:** approved · **Author:** Gamal Tawaf

## 1. Purpose

A production-shaped Arabic retrieval service, built as a hiring artifact for AI-engineer roles in the GCC market. The differentiator is not the chatbot — it is the eval harness, the benchmark numbers, the traces, and the cost controls around it. The deliverable that gets read in an interview is `docs/benchmark.md`; the code exists to make those numbers real and reproducible.

Explicit non-goals: fine-tuning any model, a polished web UI, multi-tenancy, and a general-purpose agent framework. A separate future project covers agentic tool-calling with a permission model and audit log.

## 2. Domain and corpus

Qatari public legal texts — Labour Law No. 14 of 2004 plus related ministerial decisions. Modern Standard Arabic, article-structured (`المادة`), publicly available, and answerable with verifiable ground truth. Article structure gives natural chunk boundaries and stable chunk identifiers.

The dialect dimension enters through questions, not documents: Gulf-dialect phrasings of questions whose answers live in MSA text. This is the real production failure mode — users do not write like legislation — and it is what the benchmark measures.

## 3. Architecture

Single repository, module boundaries enforced by dependency direction. `evals/` and `benchmark/` depend on the service only through its HTTP API and a shared retrieval interface, so each is independently runnable and readable.

```
app/          FastAPI service
  api/          /ask (SSE), /ingest, /health, /stats, /metrics
  planning/     dialect→MSA rewrite, question decomposition
  retrieval/    embeddings, dense + lexical search, RRF fusion, rerank
  generation/   Provider protocol, Anthropic + Gemini adapters, failover
  observability/ OTel spans, cost accounting, spend cap
ingestion/    fetch → normalize → chunk → embed → load (CLI + Pub/Sub entrypoint)
evals/        labelled Q/A set, metrics, harness, CI regression gate
benchmark/    embedding-model comparison → results.json
terraform/    ephemeral GCP: Cloud Run, Cloud SQL, Pub/Sub
docs/         benchmark writeup, architecture, latency budget, eval methodology
```

### Data model

One `chunks` table. Original text and index-normalized text are stored separately; the tsvector is a generated column over the normalized form. One nullable vector column per benchmarked embedding model (`emb_e5`, `emb_bge`, `emb_openai`, `emb_cohere`), so the benchmark varies exactly one factor while chunks, queries, and ground truth stay fixed.

Chunk identifiers are `doc_id:article:seq` and must be stable across re-ingestion — evaluation ground truth references them directly.

### Request flow (`POST /ask`)

```
question
  → plan          (normalize; rewrite Gulf dialect → MSA; decompose multi-part questions)
  → retrieve      (dense pgvector cosine ‖ lexical Postgres FTS, per sub-query)
  → fuse          (Reciprocal Rank Fusion → top-20)
  → rerank        (cross-encoder → top-5, with a confidence floor)
  → generate      (streamed answer in the user's register, citing chunk ids)
```

Every stage is toggleable per request, which is what makes the ablation study possible: dense-only, lexical-only, hybrid, hybrid+rerank, with and without planning.

Below the rerank confidence floor the service answers "not in corpus" rather than guessing. The eval set contains unanswerable questions specifically to measure this.

## 4. Evaluation

Built before retrieval, because a retrieval system without a scoreboard is a demo.

**Dataset** — roughly 250 pairs in versioned JSONL: `id, question, dialect_tag, answer, source_doc, source_chunk_ids`. Empty `source_chunk_ids` marks an unanswerable question. Built in three passes: LLM-generated candidates from corpus chunks, full manual review of every pair, then hand-written Gulf-dialect rephrasings of existing MSA questions against identical ground truth.

**Two metric layers.** Retrieval metrics — recall@3, recall@10, MRR, hit rate — are deterministic, cost nothing, and gate every pull request. End-to-end metrics — faithfulness and answer correctness — require an LLM judge, so they run on demand with a fixed judge model, a versioned prompt, and a spend cap. Judge agreement is validated once against roughly twenty human-scored pairs.

**Regression gate.** CI runs the retrieval suite against a dockerized pgvector instance seeded from a frozen corpus snapshot. A pull request fails if recall@10 falls more than two percentage points below `evals/baseline.json`. The baseline is only ever updated deliberately, in its own commit.

No eval framework dependency. The metrics are a few hundred lines of plain Python; owning them is worth more in an interview than importing them.

## 5. Benchmark

Four embedding models — `multilingual-e5-large` and `BGE-m3` running locally, `text-embedding-3-large` and Cohere `embed-v4` via API — across four retrieval configurations and both dialect tags. Reported per cell: recall@3, recall@10, MRR, p50/p95 latency, and cost per thousand queries.

The headline result is the dialect penalty: how much recall each model loses when the same question arrives in Gulf dialect instead of MSA, and how much of that loss query planning recovers.

One script regenerates `benchmark/results.json` from a clean checkout. Charts in the writeup are generated from that file, never hand-drawn.

## 6. Reliability and cost

Generation sits behind a two-method `Provider` protocol with Anthropic and Gemini implementations. On timeout or exhausted rate-limit retries, the service fails over to the next configured provider and records a span event — so failover is visible in traces rather than hidden in logs.

Cost controls: a semantic cache keyed on the normalized query embedding (cosine similarity above threshold serves the cached answer), a hard context-token budget that trims lowest-ranked chunks whole rather than truncating mid-chunk, jittered retry/backoff on 429 and 5xx, and a daily USD spend cap that returns an explanatory 503 rather than quietly burning budget.

## 7. Observability

One OpenTelemetry trace per request, spanning `plan → retrieve.dense → retrieve.lexical → fuse → rerank → generate`. Span attributes carry model name, input and output token counts, computed USD cost, cache hit or miss, and the active retrieval configuration. Jaeger locally via docker-compose; Cloud Trace when deployed.

A documented latency budget — p95 at or under 3.5 seconds, allocated per stage — is enforced by a replay script over thirty evaluation questions that fails when any stage exceeds its allocation. Prometheus metrics and a committed Grafana dashboard JSON provide the cost and latency views reproduced in the writeup.

## 8. Deployment

Local-first: docker-compose brings up the service, pgvector Postgres, and Jaeger. Everything — ingestion, evaluation, benchmark — runs offline against committed corpus data.

GCP is ephemeral by design. Terraform provisions Cloud Run (scale to zero), Cloud SQL with pgvector, a Pub/Sub topic for asynchronous ingestion, Artifact Registry, and Cloud Trace. Applied before a demo, destroyed after, so steady-state cost is zero.

## 9. Phasing

1. Corpus, Arabic normalization, article-aware chunking, chunk store, eval metrics.
2. Retrieval pipeline, embedding backfill, eval dataset construction, benchmark run.
3. Service surface, query planning, generation with failover, tracing, caching, spend controls.
4. Terraform, CI regression gate, latency budget enforcement, published writeup.

## 10. Success criteria

The project is done when a reviewer can clone the repository, run one command to reproduce the benchmark numbers, read a writeup whose charts come from committed data, open a trace showing per-stage latency and cost for a single question, and see a CI run where an evaluation regression blocks a merge.
