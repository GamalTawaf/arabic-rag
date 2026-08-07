## Multi-stage production build.
## asyncpg is a pure-Python-wheel driver — no libpq needed, unlike psycopg2.
##
## Python 3.12, not 3.14: torch has no 3.14 wheels, and app/deps.py serves /ask with a
## sentence-transformers embedder (SERVICE_MODEL_KEY = "bge"). An image without torch
## boots and serves /health and /stats, then ImportErrors on the first /ask — so the
## model deps are not optional for a working service, only for the CLI paths.

FROM python:3.12-slim-bookworm AS builder
WORKDIR /app
COPY config/requirements/ config/requirements/
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
      "torch==2.13.0+cpu" -r config/requirements/base.txt -r config/requirements/models.txt

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
## Non-root, and BEFORE the model bake, not after. `appuser` needs a writable /app
## because HF_HOME points inside it — the baked embedder is read from there, and
## anything not baked is downloaded into it.
##
## The ordering is load-bearing, not tidiness. `chown -R` rewrites the owner of every
## file it walks, and an overlay layer records a full copy of every file it modifies —
## so running this after the bake put a second complete copy of the 2.3 GB model in
## the image, and put it in a layer sitting after `COPY . .`, which changes on every
## commit. That is ~2.3 GB rebuilt, re-pushed and re-pulled per deploy, and no build
## cache could help because the layer's input changed each time. Chowning here instead
## walks an empty /app, and the bake below writes as appuser so the weights are owned
## correctly on the way in. (Measured with a 200 MB stand-in: the chown layer weighed
## the same as the payload layer.)
RUN useradd --create-home --uid 10001 appuser \
 && mkdir -p "$HF_HOME" \
 && chown -R appuser:appuser /app
USER appuser
## Bake bge-m3 (~2.3 GB) into the image. Every /ask embeds the query, so without this
## the first request on a cold instance downloads it from Hugging Face before it can
## answer — with min_instances = 0 that is every demo's first question. Image size is
## paid once at deploy; the download would be paid on every scale-from-zero.
## Its own layer, before `COPY . .`, so editing app code does not re-download it.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"
## --chown, because the copy runs as appuser but COPY defaults to root-owned files.
COPY --chown=appuser:appuser . .
EXPOSE 8000
## The entrypoint runs `alembic upgrade head` and then execs uvicorn with
## --proxy-headers (Cloud Run terminates TLS and puts the caller in
## X-Forwarded-For; without it /ask's per-IP rate limit degrades into one shared
## bucket) and --forwarded-allow-ips=* (the only peer that can reach this
## container is the platform front end). Read the comment in that file before
## copying this pattern — migrating from the entrypoint is a demo shortcut with
## known costs, not a recommendation.
ENTRYPOINT ["/app/config/docker-entrypoint.sh"]
## The default command the entrypoint execs. Kept as a CMD so it can be overridden
## (`docker run <image> python -m ingestion stats`) while migrations still run first.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--proxy-headers", "--forwarded-allow-ips", "*"]
