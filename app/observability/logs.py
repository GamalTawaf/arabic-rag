"""Structured logging, correlated to the trace.

One JSON object per line, with the active span's ids on it. The point is the
join: a request that took 4 s shows up in Cloud Trace as a span tree and in
Cloud Logging as a handful of lines, and without a shared id those two views are
two unrelated stories about the same request.

Field names are chosen for the backend, not for us:

``severity``                        Cloud Logging's own level field; `level` is ignored
``message``                         the payload it displays
``logging.googleapis.com/trace``    what joins the line to its trace, needs the project id
``logging.googleapis.com/spanId``   which span inside that trace
``trace_id`` / ``span_id``          the plain hex, for any backend that is not Google

Rules this module keeps:

- **Logging never raises.** An `extra=` value json cannot encode is repr'd, not
  raised — a crash in the log call would take out the request it was describing.
- **JSON is opt-in** (``LOG_JSON=true``, set on Cloud Run). A local run gets the
  readable one-line format, because nobody greps their own terminal with jq.
- **No trace fields when there is no span.** Emitting zeros would make every
  unrelated line look like it belongs to trace 000…0.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any

from opentelemetry import trace

from app.config import settings

# LogRecord's own attributes. Anything on a record that is not in here came from
# `extra=` and belongs in the JSON. Derived from logging.LogRecord rather than
# retyped by hand, plus the three attrs the stdlib adds during formatting.
_RESERVED = frozenset(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__
) | {
    "message",
    "asctime",
    "taskName",
    "severity",
    "time",
    "logger",
    "stack_trace",
    "trace_id",
    "span_id",
}

_PLAIN_FORMAT = "%(levelname)-5.5s [%(name)s] %(message)s"


class JsonFormatter(logging.Formatter):
    """Renders a record as one line of JSON, with trace ids when a span is live.

    ``project`` is the GCP project id. Given one, the formatter also emits the
    ``logging.googleapis.com/*`` keys that make Cloud Logging link the line to
    its trace; without one it emits only the plain hex ids, because a *wrong*
    project in that field is worse than no field at all — the console resolves it
    to a trace that does not exist.
    """

    def __init__(self, project: str | None = None) -> None:
        super().__init__()
        self._project = project if project is not None else settings.gcp_project

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "time": _rfc3339(record.created),
            "severity": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            # Its own field, not appended to the message: a multi-line traceback
            # inside `message` is what turns one log entry into forty.
            payload["stack_trace"] = self.formatException(record.exc_info)

        payload.update(_trace_fields(self._project))

        # `extra=` last, but it cannot overwrite the keys above — a caller that
        # passes extra={"severity": "DEBUG"} gets a `severity` that still
        # reflects the level the line was logged at.
        for key, value in record.__dict__.items():
            if key not in _RESERVED and key not in payload:
                payload[key] = value

        return json.dumps(payload, ensure_ascii=False, default=repr)


def setup_logging(level: str | None = None, json_output: bool | None = None) -> None:
    """Install one root handler on stderr. Safe to call more than once.

    Idempotent by replacing our own handler rather than adding another: uvicorn
    reloads, and a module-level call that appends would print every line twice.
    """
    resolved_level = (level or settings.log_level).upper()
    as_json = settings.log_json if json_output is None else json_output

    root = logging.getLogger()
    for existing in [h for h in root.handlers if getattr(h, "_arabic_rag", False)]:
        root.removeHandler(existing)

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter() if as_json else logging.Formatter(_PLAIN_FORMAT))
    handler._arabic_rag = True  # type: ignore[attr-defined]  # marks it as ours to replace
    root.addHandler(handler)
    root.setLevel(resolved_level)


def _trace_fields(project: str) -> dict[str, str]:
    """Ids from the current span, or nothing at all when none is recording."""
    span = trace.get_current_span()
    context = span.get_span_context()
    if not context.is_valid:
        return {}

    trace_id = trace.format_trace_id(context.trace_id)
    span_id = trace.format_span_id(context.span_id)
    fields = {"trace_id": trace_id, "span_id": span_id}
    if project:
        fields["logging.googleapis.com/trace"] = f"projects/{project}/traces/{trace_id}"
        fields["logging.googleapis.com/spanId"] = span_id
    return fields


def _rfc3339(created: float) -> str:
    """UTC, milliseconds, trailing Z — the format every log backend parses."""
    base = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(created))
    return f"{base}.{int((created % 1) * 1000):03d}Z"
