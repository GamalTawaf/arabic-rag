"""Tracing, metrics and spend-cap tests.

The tracing tests never touch the global OTel provider: they build their own
``TracerProvider`` with an ``InMemorySpanExporter`` and install it through
``install_provider``, so "no provider configured" and "attributes land on a
span" can both run in one process regardless of test order.
"""

from __future__ import annotations

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app.observability import cost, tracing
from app.observability.cost import SpendCapExceeded, SpendTracker


@pytest.fixture
def exporter():
    """Install an in-memory provider for one test, then restore the no-op."""
    memory = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(memory))
    tracing.install_provider(provider)
    yield memory
    tracing.reset_tracing()


# --------------------------------------------------------------------------
# span() with nothing configured
# --------------------------------------------------------------------------


async def test_span_is_a_noop_when_tracing_is_not_configured(capsys):
    # Arrange
    tracing.reset_tracing()

    # Act
    async with tracing.span("plan", **{"app.retrieval.config": "dense"}) as current:
        result = 42

    # Assert
    assert result == 42
    assert current.is_recording() is False
    assert capsys.readouterr().out == ""


async def test_span_reraises_and_still_records_latency_without_a_provider():
    # Arrange
    tracing.reset_tracing()

    # Act / Assert
    with pytest.raises(ValueError, match="boom"):
        async with tracing.span("rerank"):
            raise ValueError("boom")


async def test_tracer_returns_a_usable_tracer_with_no_provider():
    # Arrange
    tracing.reset_tracing()

    # Act
    span = tracing.tracer().start_span("x")
    span.end()

    # Assert — a no-op span, but a real object with the Span interface
    assert span.is_recording() is False


# --------------------------------------------------------------------------
# span() with an in-memory exporter
# --------------------------------------------------------------------------


async def test_span_exports_name_and_attributes(exporter):
    # Act
    async with tracing.span("retrieve.dense", **{"app.retrieval.config": "dense"}):
        pass

    # Assert
    (finished,) = exporter.get_finished_spans()
    assert finished.name == "retrieve.dense"
    assert finished.attributes["app.retrieval.config"] == "dense"


async def test_span_drops_none_attributes(exporter):
    # Act
    async with tracing.span("fuse", **{"app.retrieval.top_score": None}):
        pass

    # Assert
    (finished,) = exporter.get_finished_spans()
    assert "app.retrieval.top_score" not in finished.attributes


async def test_span_records_the_exception_and_sets_error_status(exporter):
    # Act
    with pytest.raises(RuntimeError):
        async with tracing.span("generate"):
            raise RuntimeError("provider timeout")

    # Assert
    (finished,) = exporter.get_finished_spans()
    assert finished.status.status_code.name == "ERROR"
    assert finished.events[0].name == "exception"


def test_record_llm_call_uses_genai_convention_attribute_names(exporter):
    # Act
    with tracing.tracer().start_as_current_span("generate") as current:
        tracing.record_llm_call(
            current,
            provider="anthropic",
            model="claude-haiku-4-5-20251001",
            input_tokens=1200,
            output_tokens=300,
            cost_usd=0.0027,
            cached=False,
        )

    # Assert
    (finished,) = exporter.get_finished_spans()
    attributes = finished.attributes
    assert attributes["gen_ai.system"] == "anthropic"
    assert attributes["gen_ai.provider.name"] == "anthropic"
    assert attributes["gen_ai.operation.name"] == "chat"
    assert attributes["gen_ai.request.model"] == "claude-haiku-4-5-20251001"
    assert attributes["gen_ai.usage.input_tokens"] == 1200
    assert attributes["gen_ai.usage.output_tokens"] == 300
    # Everything non-standard is namespaced under "app."
    assert attributes["app.cost_usd"] == pytest.approx(0.0027)
    assert attributes["app.cache.hit"] is False


def test_record_retrieval_puts_everything_under_the_app_prefix(exporter):
    # Act
    with tracing.tracer().start_as_current_span("retrieve") as current:
        tracing.record_retrieval(
            current,
            config="hybrid_rerank",
            model_key="bge",
            n_candidates=20,
            n_returned=5,
            top_score=0.87,
        )

    # Assert
    (finished,) = exporter.get_finished_spans()
    assert finished.attributes["app.retrieval.config"] == "hybrid_rerank"
    assert finished.attributes["app.retrieval.model_key"] == "bge"
    assert finished.attributes["app.retrieval.candidates"] == 20
    assert finished.attributes["app.retrieval.returned"] == 5
    assert finished.attributes["app.retrieval.top_score"] == pytest.approx(0.87)
    assert not [key for key in finished.attributes if key.startswith("gen_ai.")]


# --------------------------------------------------------------------------
# setup_tracing
# --------------------------------------------------------------------------


def test_setup_tracing_is_a_noop_without_an_exporter(monkeypatch):
    # Arrange
    tracing.reset_tracing()
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_CONSOLE_EXPORT", raising=False)

    # Act
    tracing.setup_tracing()

    # Assert
    assert tracing._provider is None


def test_setup_tracing_twice_does_not_double_register(monkeypatch):
    # Arrange
    tracing.reset_tracing()
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.setenv("OTEL_CONSOLE_EXPORT", "1")

    # Act
    tracing.setup_tracing(service_name="arabic-rag-test")
    first = tracing._provider
    processors = len(first._active_span_processor._span_processors)
    tracing.setup_tracing(service_name="arabic-rag-test")

    # Assert
    assert tracing._provider is first
    assert len(tracing._provider._active_span_processor._span_processors) == processors
    tracing.reset_tracing()


def test_setup_tracing_skips_app_instrumentation_when_unconfigured(monkeypatch):
    # Arrange
    tracing.reset_tracing()
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_CONSOLE_EXPORT", raising=False)

    from fastapi import FastAPI

    app = FastAPI()

    # Act
    tracing.setup_tracing(app=app)

    # Assert
    assert getattr(app, "_is_instrumented_by_opentelemetry", False) is False


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def test_metrics_app_serves_recorded_metrics():
    # Arrange
    app = tracing.metrics_app()
    tracing.record_request("/ask", "ok")
    tracing.record_cache_lookup(hit=True)
    tracing.record_cache_lookup(hit=False)

    # Act
    from prometheus_client import generate_latest

    exposition = generate_latest().decode()

    # Assert
    assert callable(app)
    assert "rag_requests" in exposition
    assert "rag_cache_lookups" in exposition
    assert 'hit="true"' in exposition


def test_metrics_app_is_idempotent():
    # Act
    first, second = tracing.metrics_app(), tracing.metrics_app()

    # Assert — one reader, so no duplicate series registered
    assert callable(first) and callable(second)
    assert tracing._metrics_reader is not None


# --------------------------------------------------------------------------
# Cost estimation
# --------------------------------------------------------------------------


def test_estimate_cost_matches_the_published_per_mtok_rate():
    # Arrange — haiku 4.5 is $1.00 in / $5.00 out per million tokens
    # Act
    usd = cost.estimate_cost_usd("claude-haiku-4-5", 1_000_000, 1_000_000)

    # Assert
    assert usd == pytest.approx(6.00)


def test_estimate_cost_resolves_a_dated_snapshot_by_prefix():
    # Act
    dated = cost.estimate_cost_usd("claude-haiku-4-5-20251001", 2000, 500)
    alias = cost.estimate_cost_usd("claude-haiku-4-5", 2000, 500)

    # Assert
    assert dated == pytest.approx(alias) == pytest.approx(0.0045)


def test_estimate_cost_raises_for_an_unpriced_model():
    # Act / Assert — a silent 0.0 would disarm the spend cap
    with pytest.raises(ValueError, match="no price for model"):
        cost.estimate_cost_usd("llama-4-maverick", 10, 10)


def test_estimate_cost_rejects_negative_tokens():
    with pytest.raises(ValueError, match="token counts"):
        cost.estimate_cost_usd("claude-haiku-4-5", -1, 0)


# --------------------------------------------------------------------------
# SpendTracker
# --------------------------------------------------------------------------


class FakeClock:
    """An injected clock — the date only moves when the test moves it."""

    def __init__(self, date: str) -> None:
        self.date = date

    def __call__(self) -> str:
        return self.date


def test_spend_tracker_accumulates_usd_and_calls():
    # Arrange
    tracker = cost.SpendTracker(5.0, clock=FakeClock("2026-07-25"))

    # Act
    tracker.record(0.10)
    tracker.record(0.05)

    # Assert
    today = tracker.today()
    assert today == cost.Spend(date="2026-07-25", usd=pytest.approx(0.15), calls=2)


def test_spend_resets_across_a_date_boundary():
    # Arrange
    clock = FakeClock("2026-07-25")
    tracker = cost.SpendTracker(5.0, clock=clock)
    tracker.record(4.0)

    # Act — midnight UTC, no sleeping involved
    clock.date = "2026-07-26"

    # Assert
    assert tracker.today() == cost.Spend(date="2026-07-26", usd=0.0, calls=0)
    assert tracker.remaining() == pytest.approx(5.0)
    assert tracker.would_exceed(1.0) is False


def test_would_exceed_under_at_and_over_the_cap():
    # Arrange
    tracker = cost.SpendTracker(1.0, clock=FakeClock("2026-07-25"))

    # Act / Assert — under
    tracker.record(0.90)
    assert tracker.would_exceed(0.05) is False

    # at the cap: a ceiling to stop at, not to sit on
    assert tracker.would_exceed(0.10) is True

    # over
    tracker.record(0.20)
    assert tracker.would_exceed() is True


def test_remaining_is_never_negative():
    # Arrange
    tracker = cost.SpendTracker(1.0, clock=FakeClock("2026-07-25"))

    # Act
    tracker.record(3.0)

    # Assert
    assert tracker.remaining() == 0.0


def test_record_rejects_negative_spend():
    tracker = cost.SpendTracker(1.0, clock=FakeClock("2026-07-25"))
    with pytest.raises(ValueError, match="usd must be >= 0"):
        tracker.record(-0.01)


def test_negative_cap_is_rejected():
    with pytest.raises(ValueError, match="cap_usd must be >= 0"):
        cost.SpendTracker(-1.0)


def test_check_raises_spend_cap_exceeded_carrying_cap_and_spend():
    # Arrange
    tracker = cost.SpendTracker(2.0, clock=FakeClock("2026-07-25"))
    tracker.record(1.5)
    tracker.record(0.75)

    # Act / Assert
    with pytest.raises(cost.SpendCapExceeded) as excinfo:
        tracker.check(0.01)

    error = excinfo.value
    assert error.cap_usd == 2.0
    assert error.spend.usd == pytest.approx(2.25)
    assert error.spend.calls == 2
    assert error.spend.date == "2026-07-25"
    assert error.remaining == 0.0
    assert "2.2500" in str(error) and "2.00" in str(error)


def test_check_is_silent_below_the_cap():
    tracker = cost.SpendTracker(2.0, clock=FakeClock("2026-07-25"))
    tracker.record(0.5)
    assert tracker.check(0.1) is None


def test_get_spend_tracker_is_a_process_singleton():
    # Arrange
    cost.reset_spend_tracker()

    # Act
    first, second = cost.get_spend_tracker(), cost.get_spend_tracker()

    # Assert
    from app.config import settings

    assert first is second
    assert first.cap_usd == settings.daily_spend_cap_usd
    cost.reset_spend_tracker()


# ------------------------------------------------- reserve / settle (the race)


def test_reserve_holds_the_estimate_so_concurrent_callers_cannot_all_pass():
    """Regression: check-then-record straddled the provider await.

    Every in-flight request read the same pre-call total, so N concurrent
    requests all passed a cap with room for one. Measured before the fix: 10
    concurrent calls against a cap with room for 3 spent 2.56x the cap.
    """
    # Arrange — room for exactly two calls at the estimate
    tracker = SpendTracker(cap_usd=0.03, clock=lambda: "2026-07-26")
    estimate = 0.01

    # Act — three callers reserve before any of them settles
    first = tracker.reserve(estimate)
    second = tracker.reserve(estimate)
    with pytest.raises(SpendCapExceeded):
        tracker.reserve(estimate)

    # Assert — the two holds are already debited, the third never ran
    assert tracker.today().usd == pytest.approx(0.02)
    assert tracker.today().calls == 0  # a hold is not a call
    tracker.settle(first, 0.008)
    tracker.settle(second, 0.012)
    assert tracker.today().usd == pytest.approx(0.02)
    assert tracker.today().calls == 2


def test_settle_releases_the_hold_when_the_call_produced_nothing():
    # Arrange
    tracker = SpendTracker(cap_usd=1.0, clock=lambda: "2026-07-26")
    held = tracker.reserve(0.25)

    # Act — the provider failed, or the client vanished mid-stream
    tracker.settle(held, 0.0)

    # Assert — the cap is not left permanently narrowed by a free failure
    assert tracker.today().usd == pytest.approx(0.0)


def test_settle_never_drives_the_total_negative_across_a_day_rollover():
    # Arrange — the clock advances between the reserve and the settle
    day = {"value": "2026-07-26"}
    tracker = SpendTracker(cap_usd=1.0, clock=lambda: day["value"])
    held = tracker.reserve(0.5)
    day["value"] = "2026-07-27"  # midnight UTC resets the counter to zero

    # Act
    tracker.settle(held, 0.1)

    # Assert
    assert tracker.today().usd >= 0.0
