"""The ASGI app: routers, tracing, metrics. Nothing heavy at import time.

``app.deps`` builds the embedder, reranker and provider lazily, so importing
this module loads neither ``torch`` nor ``sentence_transformers`` — a guard the
test suite asserts on, because CI and the ``/health`` path must not pay for a
2 GB ML stack.
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api import ask, health, ingest, stats
from app.config import settings
from app.observability.logs import setup_logging
from app.observability.tracing import metrics_app, setup_tracing

# First, before anything can log: uvicorn installs its own handlers, and a
# handler added after the first log line means that line is formatted differently
# from every line after it.
setup_logging()

app = FastAPI(title=settings.service_name)

app.include_router(health.router)
app.include_router(ask.router)
app.include_router(ingest.router)
app.include_router(stats.router)

# No-op unless an exporter is configured (OTEL_EXPORTER_OTLP_ENDPOINT or
# OTEL_CONSOLE_EXPORT=1); see app/observability/tracing.py.
setup_tracing(app)

# Prometheus exposition. Mounting is what installs the meter provider, so
# instruments created earlier in the process start reporting from here on.
app.mount("/metrics", metrics_app())

# The demo page, last: a mount at "/" swallows every path not already claimed, so
# it must come after the routers and /metrics or it would shadow them. One static
# file, no build step, served from the app itself — which is also what keeps it
# same-origin, so /ask needs no CORS middleware.
app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True))
