# arabic-rag

Retrieval-augmented question answering over Qatari labour legislation, built
Arabic-first: the corpus is Modern Standard Arabic, the questions arrive in MSA
*or* Gulf dialect, and the point of the project is to measure what that
difference costs.

Status: phases 1 and 2 are done — corpus, ingestion, eval dataset, retrieval,
benchmark, CI regression gate. The FastAPI `/ask` surface, query planning and
generation are **not built yet** (see [Not built yet](#not-built-yet)).

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
python -m ingestion backfill --model bge

python -m benchmark.run                      # ~21 min: 2 models x 4 configs x 4 splits
python -m benchmark.run --tables             # regenerate the docs/benchmark.md tables

python -m evals.gate --config lexical        # regression gate, exit 1 on a drop
python -m evals.gate --config dense --model bge
```

`python -m ingestion fetch` re-downloads the corpus from its public sources; the
cleaned text is committed under `data/corpus/` so nothing above needs the network.

No API keys are required. `openai` and `cohere` are implemented and unit-tested
against mocks, but with no key set they report `{"status": "not_run", "reason":
"no API key"}` in the results rather than failing the run.

Tests and lint:

```bash
pytest            # 226 pass, 1 skipped (it loads the 2 GB reranker; set
                  # RERANK_REAL_MODEL=1 to run it). DB-backed tests need the
                  # pgvector container up, and use a separate rag_test database.
ruff check .
```

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
  models/chunks.py     one table: text, normalized text, generated tsvector,
                       one nullable vector column per benchmarked model
  retrieval/           embed.py (4 models, 1 interface), search.py (dense /
                       lexical / RRF hybrid), rerank.py (bge cross-encoder)
  api/health.py        /health, /health/db
ingestion/             fetch -> normalize -> chunk -> load -> backfill, CLI in
                       __main__.py; chunks split on المادة boundaries, stable
                       ids "doc:article:seq"
evals/
  data/eval_pairs.jsonl  283 labelled pairs (229 MSA / 54 Gulf, 15 unanswerable)
  schema.py metrics.py   dataset validation; recall@k, MRR, hit rate
  harness.py             one config in, one dict of numbers out (+ by_doc split)
  gate.py baseline.json  the CI regression gate
benchmark/run.py       the full matrix -> benchmark/results/results.json
docs/                  benchmark writeup, design spec
terraform/             empty; phase 4
```

Dependency direction: `evals/` and `benchmark/` use `app/` only through the
retrieval interface, so both stay independently runnable and readable.

## Not built yet

Deliberately absent, in the order they are planned:

- `POST /ask` (SSE), `/stats`, `/metrics` — only `/health` exists today.
- Query planning: dialect→MSA rewrite and multi-part decomposition. This is the
  intended fix for the dialect penalty above, and it is measured as an ablation,
  not assumed to work.
- Generation: `Provider` protocol with Anthropic and Gemini adapters, timeout/429
  failover, semantic cache, daily spend cap.
- OpenTelemetry tracing and the p95 ≤ 3.5 s latency budget with its replay script.
- LLM-judged faithfulness/correctness evals (the deterministic retrieval layer is
  what gates CI; the judged layer is on-demand with a spend cap).
- Terraform for ephemeral GCP (Cloud Run + Cloud SQL + Pub/Sub).
- OpenAI and Cohere embedding columns are wired but unmeasured — no API keys.

Also not done, and known: the Postgres FTS index uses the `simple` config
because Postgres ships no Arabic stemmer, which is most of why lexical retrieval
scores 0.356 recall@10; and the rerank confidence floor is not dialect-neutral
(see the score-semantics note in `app/retrieval/rerank.py`).
