## Multi-stage production build.
## asyncpg is a pure-Python-wheel driver — no libpq needed, unlike psycopg2.
##
## Python 3.12, not 3.14: torch has no 3.14 wheels, and app/deps.py serves /ask with a
## sentence-transformers embedder (SERVICE_MODEL_KEY = "bge"). An image without torch
## boots and serves /health and /stats, then ImportErrors on the first /ask — so the
## model deps are not optional for a working service, only for the CLI paths.

FROM python:3.12-slim-bookworm AS builder
WORKDIR /app
COPY requirements.txt requirements-models.txt ./
## CPU wheels explicitly: the default linux torch wheel is the CUDA build and Cloud Run
## has no GPU. The "+cpu" local version is the load-bearing part — it exists only on the
## pytorch index, so the resolver cannot quietly swap in the CUDA wheel of the same
## version number when sentence-transformers pulls torch in as a dependency. Installing
## CPU torch in a separate earlier step does NOT work: a --prefix install is invisible to
## the next pip run's resolver, which then reinstalls the CUDA build over it (measured:
## a 9 GB image instead of the 2.06 GB the CPU wheels build).
RUN pip install --upgrade pip \
 && pip install --prefix=/install \
      --extra-index-url https://download.pytorch.org/whl/cpu \
      "torch==2.13.0+cpu" -r requirements.txt -r requirements-models.txt

FROM python:3.12-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
## Weights land here. The embedder is baked in below; the ~2.2 GB cross-encoder is not,
## because a Cloud Run instance has no GPU and reranking there measured 3972 ms p95
## against a 1200 ms allocation (docs/latency-budget.md), so the deployed service runs
## with RERANK_ENABLED=false and never loads it. Set it true and the first /ask on a
## cold instance pays that download.
ENV HF_HOME=/app/.cache/huggingface
WORKDIR /app
COPY --from=builder /install /usr/local
## Bake bge-m3 (~2.3 GB) into the image. Every /ask embeds the query, so without this
## the first request on a cold instance downloads it from Hugging Face before it can
## answer — with min_instances = 0 that is every demo's first question. Image size is
## paid once at deploy; the download would be paid on every scale-from-zero.
## Its own layer, before `COPY . .`, so editing app code does not re-download it.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"
COPY . .
## Non-root. `appuser` needs a writable /app because HF_HOME points inside it — the
## baked embedder is read from there, and anything not baked is downloaded into it.
RUN useradd --create-home --uid 10001 appuser \
 && mkdir -p "$HF_HOME" \
 && chown -R appuser:appuser /app
USER appuser
EXPOSE 8000
## Built from the repository root, not from this directory:
##   docker build -f docker/Dockerfile .
## The COPY paths below are context-relative, so they do not change with this file.
##
## The entrypoint runs `alembic upgrade head` and then execs uvicorn with
## --proxy-headers (Cloud Run terminates TLS and puts the caller in
## X-Forwarded-For; without it /ask's per-IP rate limit degrades into one shared
## bucket) and --forwarded-allow-ips=* (the only peer that can reach this
## container is the platform front end). Read the comment in that file before
## copying this pattern — migrating from the entrypoint is a demo shortcut with
## known costs, not a recommendation.
ENTRYPOINT ["/app/docker/entrypoint.sh"]
