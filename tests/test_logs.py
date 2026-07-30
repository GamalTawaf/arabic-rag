"""Structured logging tests.

The unit under test is the formatter, not the root logger: a formatter takes a
``LogRecord`` and returns a string, so every assertion here is `json.loads` on
that string. ``setup_logging`` is exercised separately for the two things it can
get wrong — installing twice, and picking the wrong formatter.

Trace correlation is tested against a real span from an in-memory
``TracerProvider`` (same fixture shape as tests/test_observability.py), because
the ids have to come from the SDK's context, not from something we pass in.
"""

from __future__ import annotations

import json
import logging

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app.observability import logs, tracing


@pytest.fixture
def tracer():
    """A recording provider for one test, then back to the no-op."""
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(InMemorySpanExporter()))
    tracing.install_provider(provider)
    yield provider
    tracing.reset_tracing()


def record(msg: str = "hello", level: int = logging.INFO, **extra) -> logging.LogRecord:
    """A LogRecord as `log.info(msg, extra=extra)` would produce one."""
    rec = logging.LogRecord(
        name="app.test", level=level, pathname=__file__, lineno=1, msg=msg, args=(), exc_info=None
    )
    for key, value in extra.items():
        setattr(rec, key, value)
    return rec


def emit(rec: logging.LogRecord, **kwargs) -> dict:
    return json.loads(logs.JsonFormatter(**kwargs).format(rec))


# --------------------------------------------------------------------------
# The line itself
# --------------------------------------------------------------------------


def test_formats_one_json_object_per_record():
    # Arrange
    rec = record("retrieved 8 chunks")

    # Act
    line = logs.JsonFormatter().format(rec)

    # Assert
    assert "\n" not in line  # one line per record, or a log pipeline splits it
    assert json.loads(line)["message"] == "retrieved 8 chunks"


def test_carries_severity_logger_and_timestamp():
    # Arrange / Act
    payload = emit(record(level=logging.WARNING))

    # Assert
    assert payload["severity"] == "WARNING"  # Cloud Logging reads this key by name
    assert payload["logger"] == "app.test"
    assert payload["time"].endswith("Z")


def test_message_is_interpolated_not_the_template():
    # Arrange
    rec = logging.LogRecord(
        name="app.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="answered in %d ms",
        args=(2500,),
        exc_info=None,
    )

    # Act / Assert
    assert emit(rec)["message"] == "answered in 2500 ms"


def test_extra_fields_are_merged_as_top_level_keys():
    # Arrange / Act
    payload = emit(record(register="gulf", chunks=8))

    # Assert
    assert payload["register"] == "gulf"
    assert payload["chunks"] == 8


def test_extra_cannot_overwrite_severity_or_message():
    # Arrange: a caller passing extra={"severity": ...} must not rewrite the level
    payload = emit(record(severity="DEBUG", message="spoofed"))

    # Assert
    assert payload["severity"] == "INFO"
    assert payload["message"] == "hello"


def test_unserialisable_extra_becomes_a_string_instead_of_failing():
    # Arrange: logging must never raise. An object json cannot encode is repr'd.
    payload = emit(record(provider=object()))

    # Assert
    assert isinstance(payload["provider"], str)


def test_exception_is_reported_as_a_stack_trace_field():
    # Arrange
    try:
        raise ValueError("no embedder")
    except ValueError:
        rec = record("generation failed", level=logging.ERROR)
        import sys

        rec.exc_info = sys.exc_info()

    # Act
    payload = emit(rec)

    # Assert
    assert payload["severity"] == "ERROR"
    assert "ValueError: no embedder" in payload["stack_trace"]
    assert payload["message"] == "generation failed"  # message stays the message


# --------------------------------------------------------------------------
# Trace correlation
# --------------------------------------------------------------------------


async def test_trace_and_span_ids_come_from_the_active_span(tracer):
    # Arrange / Act
    async with tracing.span("retrieve"):
        payload = emit(record())

    # Assert
    assert len(payload["trace_id"]) == 32  # hex, not the int OTel stores
    assert len(payload["span_id"]) == 16
    assert int(payload["trace_id"], 16) != 0


def test_no_trace_fields_without_an_active_span():
    # Arrange
    tracing.reset_tracing()

    # Act
    payload = emit(record())

    # Assert
    assert "trace_id" not in payload
    assert "span_id" not in payload


async def test_cloud_logging_trace_field_when_a_project_is_configured(tracer):
    # Arrange / Act
    async with tracing.span("retrieve"):
        payload = emit(record(), project="arabic-rag-demo-gt2")

    # Assert: this exact key is what joins a log line to its trace in Cloud Logging
    trace_field = payload["logging.googleapis.com/trace"]
    assert trace_field == f"projects/arabic-rag-demo-gt2/traces/{payload['trace_id']}"
    assert payload["logging.googleapis.com/spanId"] == payload["span_id"]


async def test_no_cloud_logging_field_without_a_project(tracer):
    # Arrange / Act
    async with tracing.span("retrieve"):
        payload = emit(record())

    # Assert: off GCP the key would be noise, and a wrong project id is worse
    assert "logging.googleapis.com/trace" not in payload


# --------------------------------------------------------------------------
# setup_logging
# --------------------------------------------------------------------------


def test_setup_logging_installs_exactly_one_handler_however_often_it_runs():
    # Arrange
    root = logging.getLogger()
    before = list(root.handlers)

    # Act
    try:
        logs.setup_logging(json_output=True)
        logs.setup_logging(json_output=True)

        # Assert
        ours = [h for h in root.handlers if isinstance(h.formatter, logs.JsonFormatter)]
        assert len(ours) == 1
        assert root.level == logging.INFO
    finally:
        root.handlers = before


def test_setup_logging_uses_a_plain_formatter_when_json_is_off():
    # Arrange
    root = logging.getLogger()
    before = list(root.handlers)

    # Act
    try:
        logs.setup_logging(json_output=False)

        # Assert: local runs stay readable; JSON is for a log backend
        assert not any(isinstance(h.formatter, logs.JsonFormatter) for h in root.handlers)
        assert root.handlers
    finally:
        root.handlers = before
