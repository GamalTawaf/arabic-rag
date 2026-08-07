"""OpenTelemetry tracing + Prometheus metrics for the /ask pipeline.

Design rules this file follows:

- **Nothing here may break a request.** With no exporter configured the tracer is
  the API's no-op, ``span()`` costs a context-manager and a ``perf_counter``, and
  the metric instruments are proxies that drop their measurements on the floor.
- **SDK imports are function-local.** Importing ``opentelemetry.sdk`` costs real
  time; a process that never calls ``setup_tracing`` (the eval harness, the
  benchmark) should not pay it.
- **Attribute names come from the GenAI semantic conventions** where a convention
  exists; everything we invented lives under ``app.``. See ``_GEN_AI_*`` below.

Exporter selection, in ``setup_tracing``:

===========================================  =========================================
``OTEL_EXPORTER_OTLP_ENDPOINT`` is set       OTLP/gRPC + BatchSpanProcessor (Jaeger)
``OTEL_CONSOLE_EXPORT=1``                    ConsoleSpanExporter + SimpleSpanProcessor
neither                                      no provider at all — spans are no-ops
===========================================  =========================================
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from opentelemetry import metrics, trace
from opentelemetry.trace import Span, Status, StatusCode, Tracer

from app.config import settings
from app.constants import INSTRUMENTATION_NAME

log = logging.getLogger(__name__)


# --- Span attribute names -------------------------------------------------
#
# The four GenAI names below are verified against the installed
# `opentelemetry-semantic-conventions` package (0.65b0), module
# `opentelemetry.semconv._incubating.attributes.gen_ai_attributes`. They are
# hard-coded rather than imported because in 0.65b0 the whole `gen_ai.*`
# namespace is marked "Deprecated: moved to the OpenTelemetry GenAI semantic
# conventions repository" — the *string values* are still the conventional
# names, but the Python constants are on their way out of this package.
#
# `gen_ai.system` is superseded by `gen_ai.provider.name` in current semconv; we
# emit both so a backend on either vintage can group by provider.
_GEN_AI_SYSTEM = "gen_ai.system"
_GEN_AI_PROVIDER_NAME = "gen_ai.provider.name"
_GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
_GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
_GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
_GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"

# Non-standard, so namespaced under "app." as the convention requires.
_APP_COST_USD = "app.cost_usd"
_APP_CACHE_HIT = "app.cache.hit"
_APP_RETRIEVAL_CONFIG = "app.retrieval.config"
_APP_RETRIEVAL_MODEL_KEY = "app.retrieval.model_key"
_APP_RETRIEVAL_CANDIDATES = "app.retrieval.candidates"
_APP_RETRIEVAL_RETURNED = "app.retrieval.returned"
_APP_RETRIEVAL_TOP_SCORE = "app.retrieval.top_score"

_LATENCY_BUCKETS_S = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)

_provider: Any = None  # SDK TracerProvider once configured; None -> no-op tracer
_metrics_reader: Any = None  # PrometheusMetricReader; also the "configured" flag


# --------------------------------------------------------------------------
# Tracing
# --------------------------------------------------------------------------


def setup_tracing(app: Any = None, service_name: str | None = None) -> None:
    """Configure tracing from the environment. Safe to call more than once.

    Passing ``app`` (a FastAPI instance) also instruments it, but only when an
    exporter is actually configured — instrumenting against a no-op provider
    just adds middleware that produces nothing.
    """
    if _provider is None:
        provider = _build_provider(service_name or settings.service_name)
        if provider is None:
            return  # no exporter configured: stay a no-op, don't instrument
        install_provider(provider)

    if app is not None:
        _instrument_app(app)


def _build_provider(service_name: str) -> Any:
    """Build an SDK TracerProvider from env, or None when no exporter is set."""
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    console = os.getenv("OTEL_CONSOLE_EXPORT") == "1"
    if not endpoint and not console:
        return None

    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
        SimpleSpanProcessor,
    )

    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    if endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )

            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        except Exception:
            log.exception("OTLP exporter unavailable; continuing without it")
    if console:
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    return provider


def install_provider(provider: Any) -> None:
    """Install ``provider`` as the tracer source for this module.

    Also published globally (best effort) so third-party instrumentation finds
    it. The global can only be set once per process, so a second call keeps the
    first global while still switching *our* tracer — which is exactly what the
    tests need.
    """
    global _provider
    _provider = provider
    if type(trace.get_tracer_provider()).__name__ == "ProxyTracerProvider":
        trace.set_tracer_provider(provider)


def reset_tracing() -> None:
    """Drop the configured provider. For tests; not used by the service."""
    global _provider
    _provider = None


def _instrument_app(app: Any) -> None:
    if getattr(app, "_is_instrumented_by_opentelemetry", False):
        return
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app, tracer_provider=_provider)
    except Exception:
        log.exception("FastAPI instrumentation failed; continuing without it")


def tracer() -> Tracer:
    """The tracer to emit spans on — a no-op tracer until ``setup_tracing`` runs."""
    return trace.get_tracer(INSTRUMENTATION_NAME, tracer_provider=_provider)


@asynccontextmanager
async def span(name: str, **attributes: Any) -> AsyncIterator[Span]:
    """One pipeline stage: a span plus a latency observation.

    ``None`` attribute values are dropped (OTel rejects them). Exceptions are
    recorded on the span, the span is marked ERROR, and the exception is
    re-raised — this wrapper never swallows anything.
    """
    started = time.perf_counter()
    failed = False
    with tracer().start_as_current_span(name) as current:
        _set_attributes(current, attributes)
        try:
            yield current
        except Exception as exc:
            failed = True
            current.record_exception(exc)
            current.set_status(Status(StatusCode.ERROR, str(exc)))
            raise
        finally:
            _stage_latency().record(
                time.perf_counter() - started,
                {"stage": name, "error": str(failed).lower()},
            )


def _set_attributes(target: Span, attributes: dict[str, Any]) -> None:
    for key, value in attributes.items():
        if value is not None:
            target.set_attribute(key, value)


def record_llm_call(
    span: Span,
    *,
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
    cached: bool,
) -> None:
    """Annotate a generation span with GenAI-convention usage attributes."""
    _set_attributes(
        span,
        {
            _GEN_AI_SYSTEM: provider,
            _GEN_AI_PROVIDER_NAME: provider,
            _GEN_AI_OPERATION_NAME: "chat",
            _GEN_AI_REQUEST_MODEL: model,
            _GEN_AI_USAGE_INPUT_TOKENS: input_tokens,
            _GEN_AI_USAGE_OUTPUT_TOKENS: output_tokens,
            _APP_COST_USD: cost_usd,
            _APP_CACHE_HIT: cached,
        },
    )
    labels = {_GEN_AI_PROVIDER_NAME: provider, _GEN_AI_REQUEST_MODEL: model}
    _tokens().add(input_tokens, {**labels, "gen_ai.token.type": "input"})
    _tokens().add(output_tokens, {**labels, "gen_ai.token.type": "output"})
    _cost().add(cost_usd, labels)
    record_cache_lookup(cached)


def record_retrieval(
    span: Span,
    *,
    config: str,
    model_key: str,
    n_candidates: int,
    n_returned: int,
    top_score: float | None,
) -> None:
    """Annotate a retrieval span. No GenAI convention covers these — all ``app.``."""
    _set_attributes(
        span,
        {
            _APP_RETRIEVAL_CONFIG: config,
            _APP_RETRIEVAL_MODEL_KEY: model_key,
            _APP_RETRIEVAL_CANDIDATES: n_candidates,
            _APP_RETRIEVAL_RETURNED: n_returned,
            _APP_RETRIEVAL_TOP_SCORE: top_score,
        },
    )


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
#
# Instruments are created lazily off the *global* meter provider. Before
# `metrics_app()` installs the real one, `get_meter` hands back a proxy whose
# instruments forward once a provider appears — so import order does not matter
# and an unconfigured process records into nothing.

_instruments: dict[str, Any] = {}


def _instrument(key: str, factory_name: str, *args: Any, **kwargs: Any) -> Any:
    if key not in _instruments:
        meter = metrics.get_meter(INSTRUMENTATION_NAME)
        _instruments[key] = getattr(meter, factory_name)(*args, **kwargs)
    return _instruments[key]


def _requests() -> Any:
    return _instrument(
        "requests", "create_counter", "rag.requests", unit="1", description="Requests served"
    )


def _stage_latency() -> Any:
    return _instrument(
        "stage_latency",
        "create_histogram",
        "rag.stage.duration",
        unit="s",
        description="Per-stage wall time",
        explicit_bucket_boundaries_advisory=list(_LATENCY_BUCKETS_S),
    )


def _tokens() -> Any:
    return _instrument(
        "tokens", "create_counter", "gen_ai.client.token.usage", unit="1", description="LLM tokens"
    )


def _cost() -> Any:
    # No unit: the Prometheus exporter appends it to the series name, and
    # "rag_cost_usd_USD_total" helps nobody.
    return _instrument(
        "cost", "create_counter", "rag.cost.usd", description="Estimated LLM spend in USD"
    )


def _cache() -> Any:
    return _instrument(
        "cache",
        "create_counter",
        "rag.cache.lookups",
        unit="1",
        description="Semantic cache lookups, labelled hit=true|false",
    )


def record_request(route: str, status: str = "ok") -> None:
    """Count one served request. Ratio math (error rate) happens in Prometheus."""
    _requests().add(1, {"route": route, "status": status})


def record_cache_lookup(hit: bool) -> None:
    """Count one semantic-cache lookup; hit ratio = hit=true / sum over hit."""
    _cache().add(1, {"hit": str(hit).lower()})


def metrics_app() -> Any:
    """An ASGI app exposing the Prometheus exposition format; mount at ``/metrics``.

    First call installs the meter provider, so instruments created earlier start
    reporting from here on. Idempotent — mounting twice re-uses one reader
    (registering a second one would duplicate every series).
    """
    global _metrics_reader
    if _metrics_reader is None:
        from opentelemetry.exporter.prometheus import PrometheusMetricReader
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.resources import Resource

        _metrics_reader = PrometheusMetricReader()
        metrics.set_meter_provider(
            MeterProvider(
                resource=Resource.create({"service.name": settings.service_name}),
                metric_readers=[_metrics_reader],
            )
        )

    from prometheus_client import make_asgi_app

    return make_asgi_app()
