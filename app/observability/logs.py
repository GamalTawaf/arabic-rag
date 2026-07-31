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
from dataclasses import fields, is_dataclass
from math import isfinite
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

# uvicorn's own loggers, which it detaches from the root by default. `uvicorn.error`
# propagates to `uvicorn`, so resetting the two parents is enough — it is listed
# anyway, because relying on that inheritance is one refactor away from silence.
_UVICORN_LOGGERS = ("uvicorn", "uvicorn.error", "uvicorn.access")

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
    # `(?<!//)` keeps this off a passwordless DSN's userinfo. `rag_user@10.8.0.3`
    # is the shape of an email address and is not one, and redacting it destroyed
    # both the user and the host — the very context the password rule above goes
    # out of its way to preserve. Cloud SQL IAM auth and local trust both produce
    # exactly that URL.
    (re.compile(r"(?<!//)\b[\w.+\-]+@[\w\-]+\.[\w.\-]{2,}\b"), "[redacted:email]"),
    # Labelled secrets (`token=…`, `Authorization: Bearer …`) are handled by
    # _redact_labelled below, which checks the value's shape first.
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[redacted:id]"),  # SSN
    # Separated groups only — 4-4-4-4 is unambiguous. A run of 16 bare digits is
    # not (a request id, a nanosecond timestamp), so that case goes through Luhn
    # below instead of being redacted on length alone.
    (re.compile(r"\b\d{4}[ \-]\d{4}[ \-]\d{4}[ \-]\d{4}\b"), "[redacted:card]"),
    (
        re.compile(r"(?:\+\d{1,3}[ .\-]?)?\(?\d{3}\)?[ .\-]\d{3}[ .\-]\d{4}\b"),
        "[redacted:phone]",
    ),
    # SIN, separated only. `\b\d{9}\b` matched "served 123456789 bytes" and every
    # other nine-digit count in the logs; a false positive there deletes the
    # measurement someone was reading.
    (re.compile(r"\b\d{3}[ \-]\d{3}[ \-]\d{3}\b"), "[redacted:id]"),
)

# Labelled secrets are handled separately from _REDACTIONS: the value has to look
# like a credential before it is destroyed. `(label)\s+(\S+)` alone turned
# "Authorization header missing" into "Authorization [redacted] missing" —
# swallowing the diagnosis in the one line someone is reading to find the cause.
_LABELLED = re.compile(
    r"(?i)\b(bearer|api[_\-]?key|apikey|token|password|passwd|secret|authorization)\b"
    r"([\"'\s:=]+)(?:(bearer)([\"'\s:=]+))?([^\s\"',}]+)"
)

# What "looks like a credential": long enough to be one, and not a word. 12 is
# under every real key prefix (sk-…, hf_…, ghp_…, a JWT) and over "header",
# "missing", "expired", "manager".
_CREDENTIAL_MIN = 12
_WORDLIKE = re.compile(r"(?i)^[a-z]+$")

# A GCP resource name, not a key: four or more short lowercase words joined by
# hyphens. Without this, "secret arabic-rag-database-url access denied" redacted
# the secret's *name* and left an error that no longer says which of the five
# secrets is missing — the same "the redacted word was the diagnosis" failure
# _WORDLIKE exists to prevent, moved from values to names. Everything terraform/
# creates is named this way (arabic-rag-database-url, arabic-rag-hf-api-key).
#
# Deliberately narrow, because the cost of getting this wrong is a leaked
# credential rather than a lost diagnostic:
#   - hyphens only, so `hf_abcdefghijklmnop` and `ghp_…` are not exempt;
#   - >= 4 segments, so `my-secret-value` and any other short hand-written
#     password stays redacted; every secret this stack creates has four or more
#     (arabic-rag-database-url, arabic-rag-hf-api-key);
#   - segments of 2-12 letters, so the long random run in a real key never fits.
# Every vendor key also carries a digit or a capital (sk-ant-api03-…, a JWT),
# which fails the [a-z]-only class on its own.
#
# trade-off: a four-word lowercase passphrase ("correct-horse-battery-staple")
# would pass through. Nothing here produces one — random_password in terraform/
# is alphanumeric — and a passphrase is not a shape this service handles.
_NAMELIKE = re.compile(r"^[a-z]{2,12}(?:-[a-z]{2,12}){3,}$")


def _looks_like_credential(value: str) -> bool:
    if _WORDLIKE.match(value) or _NAMELIKE.match(value):
        return False  # "header", "missing", "required", "arabic-rag-database-url"
    return len(value) >= _CREDENTIAL_MIN or bool(
        re.search(r"[_\-.]", value) and len(value) >= 8
    )


def _redact_labelled(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        label, gap, bearer, bearer_gap, value = match.groups()
        if not _looks_like_credential(value):
            return match.group(0)  # prose, not a key
        prefix = f"{label}{gap}" + (f"{bearer}{bearer_gap}" if bearer else "")
        return f"{prefix}[redacted]"

    return _LABELLED.sub(replace, text)


def _luhn(digits: str) -> bool:
    """Card-number checksum. The only thing that tells a PAN from a 16-digit id."""
    total, parity = 0, len(digits) % 2
    for index, char in enumerate(digits):
        digit = int(char)
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def _redact_bare_cards(text: str) -> str:
    return re.sub(
        r"\b\d{13,19}\b",
        lambda m: "[redacted:card]" if _luhn(m.group(0)) else m.group(0),
        text,
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
    text = _redact_labelled(text)
    for pattern, replacement in _REDACTIONS:
        text = pattern.sub(replacement, text)
    return _redact_bare_cards(text)


def _key_parts(key: str) -> set[str]:
    """`access_token` -> {accesstoken, access, token}; `x-api-key` -> {…, apikey}.

    Parts and adjacent pairs, not a substring test: `cardinality` must not read as
    `card`, and `cache_key` must not read as a key. Whole-key matching alone missed
    `access_token`, `client_secret` and `x-api-key`, whose values have no shape any
    regex knows — so they were logged in full.
    """
    # The camelCase branch must come *before* the run-of-letters branch. Written
    # the other way round (`[A-Za-z0-9]+|[A-Z][a-z0-9]*`) the first alternative
    # swallowed the whole token and the second was unreachable, so `accessToken`
    # stayed one word and never met `token` in _DENY_KEYS — every camelCase
    # spelling of a credential was logged in full while its snake_case twin was
    # redacted. `apiKey` passed the suite only because the collapsed whole-key
    # form, `apikey`, is a literal deny entry.
    # `[A-Z]+(?![a-z])` keeps acronyms whole: HTTPHeader -> {http, header}.
    words = [
        w.lower()
        for w in re.findall(r"[A-Z]+(?![a-z])|[A-Z][a-z0-9]*|[a-z0-9]+", key)
        if w
    ]
    words = [w for part in words for w in re.findall(r"[a-z]+|[0-9]+", part.lower())]
    pairs = {words[i] + words[i + 1] for i in range(len(words) - 1)}
    return {re.sub(r"[^a-z0-9]", "", key.lower()), *words, *pairs}


def _is_denied(key: str) -> bool:
    return bool(_key_parts(key) & _DENY_KEYS)


_MAX_DEPTH = 6


def _scrub(value: Any, depth: int = 0) -> Any:
    """Redact strings, walk containers and objects, and never raise.

    Objects are walked rather than repr'd: a pydantic model or a dataclass reaching
    the repr fallback printed ``question='…'`` in full, which is the one field this
    module promises never to log. ``_MAX_DEPTH`` is what makes a cyclic structure
    terminate — a RecursionError here is caught by logging.Handler and the whole
    record is dropped, so the failure mode is a line that silently never appears.
    """
    if depth > _MAX_DEPTH:
        return "[nested too deep]"
    if isinstance(value, str):
        return redact(value)
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        # NaN and Infinity are not JSON (RFC 8259). json.dumps emits them bare,
        # Cloud Logging then fails to parse the line and files it as text —
        # losing `severity` and the trace link, the two fields this module is for.
        return value if isfinite(value) else str(value)
    if isinstance(value, dict):
        return {
            str(k): "[redacted]" if _is_denied(str(k)) else _scrub(v, depth + 1)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_scrub(v, depth + 1) for v in value]

    fields = _object_fields(value)
    if fields is not None:
        return _scrub(fields, depth + 1)
    try:
        return redact(repr(value))
    except Exception:  # noqa: BLE001 - a __repr__ that raises must not lose the line
        return f"[unrepresentable {type(value).__name__}]"


def _object_fields(value: Any) -> dict | None:
    """A model's or dataclass's fields, so _DENY_KEYS applies to them by name."""
    dump = getattr(value, "model_dump", None)  # pydantic v2
    if callable(dump):
        try:
            return dict(dump())
        except Exception:  # noqa: BLE001,S110 - a model that will not dump falls through
            pass  # to __dict__ / repr below; raising here would drop the line
    if is_dataclass(value) and not isinstance(value, type):
        # getattr per field, not `value.__dict__`: @dataclass(slots=True) has no
        # __dict__ at all, and the AttributeError propagated out of format() —
        # logging.Handler.emit then printed "--- Logging error ---" with a raw
        # traceback into the JSON stream and dropped the record, which is the
        # one thing this module promises never to do.
        return {f.name: getattr(value, f.name, None) for f in fields(value)}
    if hasattr(value, "__dict__") and not isinstance(value, type) and vars(value):
        return dict(vars(value))
    return None


class PlainFormatter(logging.Formatter):
    """The readable one-line format, scrubbed the same way the JSON one is.

    Redaction used to live only inside :class:`JsonFormatter`, so with
    ``LOG_JSON=false`` — the default, and what every local and docker-compose run
    uses — none of it ran: the driver-error-quotes-the-DSN leak this whole
    section exists for was written out verbatim, and every safeguard in
    tests/test_logs.py was exercising a formatter that was not installed.

    Extras are not printed by this format, so there is nothing for _DENY_KEYS to
    do here; the message and the traceback are the whole surface.
    """

    def __init__(self) -> None:
        super().__init__(_PLAIN_FORMAT)

    def format(self, record: logging.LogRecord) -> str:
        return redact(super().format(record))


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
    as_json = settings.log_json if json_output is None else json_output

    root = logging.getLogger()
    for existing in [h for h in root.handlers if getattr(h, "_arabic_rag", False)]:
        root.removeHandler(existing)

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter() if as_json else PlainFormatter())
    handler._arabic_rag = True  # type: ignore[attr-defined]  # marks it as ours to replace
    root.addHandler(handler)
    root.setLevel(_resolved_level(level or settings.log_level))

    # Hand uvicorn's loggers to the root handler. uvicorn's LOGGING_CONFIG sets
    # propagate = False on `uvicorn` and `uvicorn.access` and attaches its own
    # stream handlers, so without this every request line, every startup line and
    # every traceback uvicorn prints stays plain text with no severity and no trace
    # ids — which on Cloud Run is very nearly every line the service emits, because
    # the app itself logs on only a handful of paths.
    for name in _UVICORN_LOGGERS:
        logger = logging.getLogger(name)
        logger.handlers = []
        logger.propagate = True


def _resolved_level(level: str) -> int:
    """A typo in LOG_LEVEL must not take the service down.

    ``setup_logging`` runs at ``app.main`` import time, so a raise here means
    uvicorn never binds a port and the operator reads a startup-probe timeout that
    says nothing about a log level.
    """
    resolved = logging.getLevelName(level.upper())
    if isinstance(resolved, int):
        return resolved
    logging.getLogger(__name__).warning(
        "unknown LOG_LEVEL %r; falling back to INFO", level
    )
    return logging.INFO


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
