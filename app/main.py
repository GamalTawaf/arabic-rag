"""The ASGI app: routers, tracing, metrics. Nothing heavy at import time.

``app.deps`` builds the embedder, reranker and provider lazily, so importing
this module loads neither ``torch`` nor ``sentence_transformers`` — a guard the
test suite asserts on, because CI and the ``/health`` path must not pay for a
2 GB ML stack.
"""

from fastapi import FastAPI

from app.api import ask, health, stats
from app.config import settings
from app.observability.tracing import metrics_app, setup_tracing

app = FastAPI(title=settings.service_name)

app.include_router(health.router)
app.include_router(ask.router)
app.include_router(stats.router)

# No-op unless an exporter is configured (OTEL_EXPORTER_OTLP_ENDPOINT or
# OTEL_CONSOLE_EXPORT=1); see app/observability/tracing.py.
setup_tracing(app)

# Prometheus exposition. Mounting is what installs the meter provider, so
# instruments created earlier in the process start reporting from here on.
app.mount("/metrics", metrics_app())
