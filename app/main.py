"""The ASGI app: routers, tracing, metrics. Nothing heavy at import time.

``app.deps`` builds the embedder, reranker and provider lazily, so importing
this module loads neither ``torch`` nor ``sentence_transformers`` — a guard the
test suite asserts on, because CI and the ``/health`` path must not pay for a
2 GB ML stack.
"""

from pathlib import Path

from fastapi import FastAPI, Response
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


# …and the same thing at the bare path. A Starlette Mount at "/metrics" matches
# only "/metrics/…", so "/metrics" fell through to the StaticFiles mount at "/"
# and answered 404 — which is the path every scraper uses by default, including
# the Managed Prometheus sidecar's RunMonitoring config (terraform/prometheus.tf)
# and the dashboard page's fetch. Measured against the deployed service: bare
# /metrics 404, /metrics/ 200.
@app.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

class RevalidatedStatic(StaticFiles):
    """StaticFiles that makes the browser check before reusing a page.

    StaticFiles sends ETag and Last-Modified but no Cache-Control, and a response
    without Cache-Control is *heuristically* cacheable: Chrome will re-serve the
    page from disk cache without asking, so an edit here can stay invisible in the
    browser through several reloads. "no-cache" does not mean don't store — it means
    revalidate, so the ETag still answers 304 and nothing is re-downloaded unless it
    actually changed. Costs one conditional request per load; buys never shipping a
    stale shell that reads as "my change did not deploy".
    """

    def file_response(self, *args: object, **kwargs: object) -> Response:
        response = super().file_response(*args, **kwargs)  # type: ignore[arg-type]
        response.headers.setdefault("Cache-Control", "no-cache")
        return response


# The demo page, last: a mount at "/" swallows every path not already claimed, so
# it must come after the routers and /metrics or it would shadow them. One static
# file, no build step, served from the app itself — which is also what keeps it
# same-origin, so /ask needs no CORS middleware.
app.mount("/", RevalidatedStatic(directory=Path(__file__).parent / "static", html=True))
