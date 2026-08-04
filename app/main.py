"""The ASGI app: routers, tracing, metrics. Nothing heavy at import time.

``app.deps`` builds the embedder, reranker and provider lazily, so importing
this module loads neither ``torch`` nor ``sentence_transformers`` — a guard the
test suite asserts on, because CI and the ``/health`` path must not pay for a
2 GB ML stack.
"""

from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api import ask, health, ingest, stats
from app.config import settings
from app.generation.base import ProviderError
from app.generation.failover import AllProvidersFailed
from app.ingest_worker import StorageUnavailable
from app.observability.cost import SpendCapExceeded
from app.observability.logs import setup_logging
from app.observability.tracing import metrics_app, record_request, setup_tracing
from app.service import error_payload

# First, before anything can log: uvicorn installs its own handlers, and a
# handler added after the first log line means that line is formatted differently
# from every line after it.
setup_logging()

app = FastAPI(title=settings.service_name)

app.include_router(health.router)
app.include_router(ask.router)
app.include_router(ingest.router)
app.include_router(stats.router)


# Pipeline failures → status codes. All of it lives here because the services
# raise domain errors and the routers are formatters; neither should own the
# mapping. A handler only fires while a status line is still available, which on
# the streaming path means "before the first frame" — after that a failure is an
# `error` event instead (see app.service.RagService.stream).


@app.exception_handler(StorageUnavailable)
async def storage_unavailable(request: Request, exc: StorageUnavailable) -> Response:
    """A dead database is transient, so it must answer 5xx — never 2xx.

    503 is what tells a client to retry and, more importantly, what stops Pub/Sub
    from acking a message whose document was never written (app.api.ingest).
    """
    record_request(request.url.path, "db_error")
    return JSONResponse({"detail": str(exc)}, status_code=503)


@app.exception_handler(SpendCapExceeded)
async def spend_cap_exceeded(request: Request, exc: SpendCapExceeded) -> Response:
    record_request(request.url.path, "spend_cap")
    return JSONResponse(
        {
            "detail": {
                "error": "daily_spend_cap_exceeded",
                "message": str(exc),
                "cap_usd": exc.cap_usd,
                "spent_usd": round(exc.spend.usd, 6),
                "calls": exc.spend.calls,
                "date": exc.spend.date,
                "remaining_usd": round(exc.remaining, 6),
            }
        },
        status_code=503,
    )


@app.exception_handler(AllProvidersFailed)
async def all_providers_failed(request: Request, exc: AllProvidersFailed) -> Response:
    record_request(request.url.path, "providers_failed")
    return JSONResponse(
        {"detail": {"error": "all_providers_failed", **error_payload(exc)}},
        status_code=502,
    )


@app.exception_handler(ProviderError)
async def provider_failed(request: Request, exc: ProviderError) -> Response:
    """One provider failing fatally — a revoked key is the common one.

    ``FailoverProvider`` re-raises a FATAL error rather than trying the next
    provider (the next would fail identically), so it arrives as a bare
    ``ProviderError``. Without this it escaped as a 500 with no body, while the
    streaming path reported the same cause as a structured frame.
    """
    record_request(request.url.path, "providers_failed")
    return JSONResponse(
        {"detail": {"error": "provider_failed", **error_payload(exc)}},
        status_code=502,
    )

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
@app.api_route("/metrics", methods=["GET", "HEAD"], include_in_schema=False)
async def metrics(request: Request) -> Response:
    """Exposition at the bare path, negotiated the same way the mount does.

    Two things this does not do by hand, because doing them by hand is what was
    wrong before. The encoder comes from ``choose_encoder(Accept)`` rather than
    the body being ``generate_latest()`` under a hardcoded
    ``CONTENT_TYPE_LATEST``: in the installed prometheus_client that constant
    advertises OpenMetrics 1.0.0 while ``generate_latest`` emits classic 0.0.4
    text, so the header described a body this endpoint does not produce. And
    ``REGISTRY`` is named rather than left to a default, so this path and the
    mounted app above are visibly reading the same collector — they were coupled
    only by both happening to default to it.

    HEAD is registered alongside GET because an uptime monitor that probes with
    HEAD read a GET-only route as a missing endpoint.
    """
    from prometheus_client import REGISTRY
    from prometheus_client.exposition import choose_encoder

    encoder, content_type = choose_encoder(request.headers.get("Accept", ""))
    return Response(encoder(REGISTRY), media_type=content_type)

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
