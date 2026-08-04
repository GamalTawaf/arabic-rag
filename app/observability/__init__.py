"""Tracing, metrics and cost accounting.

Two independent pieces:

- :mod:`app.observability.tracing` — OpenTelemetry spans + Prometheus metrics.
- :mod:`app.observability.cost`    — USD estimation and the daily spend cap.

They stay decoupled on purpose: the spend cap must work in a process that never
configured tracing (the eval harness, a CLI backfill), and the tracing helpers
must never depend on a spend tracker existing.
"""

from app.constants import PRICES_USD_PER_MTOK
from app.data import Spend
from app.observability.cost import (
    SpendCapExceeded,
    SpendTracker,
    estimate_cost_usd,
    get_spend_tracker,
    reset_spend_tracker,
)
from app.observability.tracing import (
    install_provider,
    metrics_app,
    record_cache_lookup,
    record_llm_call,
    record_request,
    record_retrieval,
    reset_tracing,
    setup_tracing,
    span,
    tracer,
)

__all__ = [
    "PRICES_USD_PER_MTOK",
    "Spend",
    "SpendCapExceeded",
    "SpendTracker",
    "estimate_cost_usd",
    "get_spend_tracker",
    "install_provider",
    "metrics_app",
    "record_cache_lookup",
    "record_llm_call",
    "record_request",
    "record_retrieval",
    "reset_spend_tracker",
    "reset_tracing",
    "setup_tracing",
    "span",
    "tracer",
]
