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
## 9 GB image instead of 3 GB).
RUN pip install --upgrade pip \
 && pip install --prefix=/install \
      --extra-index-url https://download.pytorch.org/whl/cpu \
      "torch==2.13.0+cpu" -r requirements.txt -r requirements-models.txt

FROM python:3.12-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
## Weights land here. Left OUT of the image on purpose: bge-m3 + the cross-encoder are
## ~4.4 GB, which triples the image and slows every deploy, so they download on first
## use instead. The cost is a slow first request after a scale-to-zero cold start.
## ponytail: fine for an ephemeral demo. To trade image size for cold-start latency,
## add a build step here that runs SentenceTransformer("BAAI/bge-m3") to bake them in.
ENV HF_HOME=/app/.cache/huggingface
WORKDIR /app
COPY --from=builder /install /usr/local
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
