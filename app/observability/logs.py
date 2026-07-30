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
import re
import sys
import time
from typing import Any

from opentelemetry import trace

from app.config import settings

# LogRecord's own attributes. Anything on a record that is not in here came from
# `extra=` and belongs in the JSON. Derived from logging.LogRecord rather than
# retyped by hand, plus the three attrs the stdlib adds during formatting.
_RESERVED = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
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

# --- Redaction ------------------------------------------------------------
#
# A backstop, NOT a policy. The policy is "do not log personal data", and the
# code follows it: nothing logs a question, an answer or a connection string.
# These patterns exist for the leak nobody writes on purpose — a driver error
# that quotes the DSN back with its password in it, or a traceback carrying the
# argument that caused it.
#
# Ceiling, stated so nobody trusts this more than it deserves: regexes match
# shapes, not meaning. A name, an address, or a Qatari ID in a format not listed
# here passes straight through, and a 9-digit order number gets redacted as if it
# were a SIN. Anything more sensitive than this should not reach a log line in the
# first place — see _DENY_KEYS for the by-name half of that rule.
#
# TODO(security/compliance): for anything handling real personal data, replace the
# regex pass with Sensitive Data Protection (DLP) masking. It is the difference
# between matching a shape and classifying a value, and it is what an audit
# actually asks for.
#
# What that looks like: build a DeidentifyTemplate with the infoTypes that apply
# here (EMAIL_ADDRESS, PHONE_NUMBER, PERSON_NAME, STREET_ADDRESS, IBAN_CODE,
# CREDIT_CARD_NUMBER, and the region's own — QATAR_ID_NUMBER, CANADA_SIN), pick a
# transform per type (CharacterMaskConfig for display, CryptoDeterministicConfig
# where two occurrences still need to match), and call
# `DlpServiceClient.deidentify_content` on the payload.
#
# Two ways to wire it, with the trade stated:
#   - In this formatter, before the write. Complete coverage of our own lines, but
#     a network round-trip inside a log call, which is exactly where a hang hurts:
#     it would need a short deadline, a circuit breaker, and a fallback to these
#     regexes when DLP is slow or down.
#   - Out of band: Log Router sink -> Pub/Sub -> a service that de-identifies and
#     writes to a separate bucket, with the raw sink excluded. Nothing in the
#     request path, at the cost of the raw entry existing for the seconds before
#     the pipeline catches up.
# Both bill per unit inspected, which is why neither is wired for a demo whose
# whole log volume is a few thousand lines.
_REDACTIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    # First: a URL's password, keeping scheme, user and host so the line is still
    # diagnosable. Before the generic rules, which would eat the whole DSN.
    (
        re.compile(r"\b([a-zA-Z][\w+.\-]*://[^:/?#\s]+):[^@\s]+@"),
        r"\1:[redacted:password]@",
    ),
    (re.compile(r"\b[\w.+\-]+@[\w\-]+\.[\w.\-]{2,}\b"), "[redacted:email]"),
    # A labelled secret: `token=…`, `Authorization: Bearer …`, `api_key: …`. The
    # optional `bearer` is what stops the label eating the scheme and leaving the
    # credential behind it in the clear.
    (
        re.compile(
            r"(?i)\b(bearer|api[_\-]?key|apikey|token|password|passwd|secret|authorization)\b"
            r"[\"'\s:=]+(?:bearer[\"'\s:=]+)?([^\s\"',}]+)"
        ),
        r"\1 [redacted]",
    ),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[redacted:id]"),  # SSN
    (re.compile(r"\b(?:\d{4}[ \-]?){3}\d{4}\b"), "[redacted:card]"),  # 16-digit PAN
    (
        re.compile(r"(?:\+\d{1,3}[ .\-]?)?\(?\d{3}\)?[ .\-]\d{3}[ .\-]\d{4}\b"),
        "[redacted:phone]",
    ),
    (re.compile(r"\b\d{3}[ \-]?\d{3}[ \-]?\d{3}\b"), "[redacted:id]"),  # SIN, 9 digits
)

# Redacted by field name, whatever the value looks like — the half that does not
# depend on guessing a format. `question` is here because in this domain a
# question is a personal circumstance ("my employer withheld my salary since…"),
# and `answer` because the model quotes it back.
_DENY_KEYS = frozenset(
    {
        "password",
        "passwd",
        "secret",
        "token",
        "apikey",
        "authorization",
        "cookie",
        "email",
        "mail",
        "ssn",
        "sin",
        "nationalid",
        "qid",
        "passport",
        "phone",
        "telephone",
        "mobile",
        "dob",
        "birthdate",
        "address",
        "iban",
        "creditcard",
        "card",
        "databaseurl",
        "dsn",
        "question",
        "answer",
    }
)


def redact(text: str) -> str:
    """Replace anything that looks like a personal identifier or a credential."""
    for pattern, replacement in _REDACTIONS:
        text = pattern.sub(replacement, text)
    return text


def _is_denied(key: str) -> bool:
    """`x-api-key`, `API_KEY` and `apiKey` are all the same field name."""
    return re.sub(r"[^a-z0-9]", "", key.lower()).removeprefix("x") in _DENY_KEYS


def _scrub(value: Any) -> Any:
    """Redact strings, walk containers, and stringify anything else first.

    An arbitrary object reaches the log line as its ``repr`` (json's ``default``),
    so it is repr'd here instead — otherwise a model or an exception could carry
    an identifier past the patterns inside a field json serialises later.
    """
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, bool) or value is None or isinstance(value, (int, float)):
        return value
    if isinstance(value, dict):
        return {
            k: "[redacted]" if _is_denied(str(k)) else _scrub(v)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_scrub(v) for v in value]
    return redact(repr(value))


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
            "message": redact(record.getMessage()),
        }

        if record.exc_info:
            # Its own field, not appended to the message: a multi-line traceback
            # inside `message` is what turns one log entry into forty.
            payload["stack_trace"] = redact(self.formatException(record.exc_info))

        # After redaction, never through it: a span id is 16 hex digits and can be
        # all decimal, so a scrub over the finished line would occasionally eat the
        # one field the whole trace-correlation feature depends on.
        payload.update(_trace_fields(self._project))

        # `extra=` last, but it cannot overwrite the keys above — a caller that
        # passes extra={"severity": "DEBUG"} gets a `severity` that still
        # reflects the level the line was logged at.
        for key, value in record.__dict__.items():
            if key not in _RESERVED and key not in payload:
                payload[key] = "[redacted]" if _is_denied(key) else _scrub(value)

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
    handler.setFormatter(
        JsonFormatter() if as_json else logging.Formatter(_PLAIN_FORMAT)
    )
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
