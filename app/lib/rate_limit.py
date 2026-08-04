"""In-process sliding-window rate limit, used as a FastAPI dependency."""

from __future__ import annotations

import threading
import time
from collections import deque

from fastapi import HTTPException, Request, status

from app.config import settings
from app.observability.tracing import record_request

RATE_WINDOW_S = 60.0

#: Sweep every key once the table passes this many, so a spray of one-shot
#: addresses cannot grow it without bound. A sweep is O(keys) and only runs when
#: the table is already this large, so it is amortised to nothing.
RATE_SWEEP_AT = 10_000

#: client IP -> request timestamps inside the window. Module-level so it survives
#: between requests; tests clear it. A plain dict, not a defaultdict: reading a
#: missing key must not create one, or the sweep below can never shrink anything.
_hits: dict[str, deque[float]] = {}

#: ``rate_limit`` is a sync dependency, so FastAPI runs it in the threadpool and
#: two requests really do touch ``_hits`` on different OS threads. Every read and
#: write below is under this lock — the trim-then-check sequence is not atomic
#: without it, and an interleaved popleft turns a healthy request into a 500.
_hits_lock = threading.Lock()


def _trim(seen: deque[float], now: float) -> None:
    """Drop timestamps that have aged out of the window. Caller holds the lock."""
    while seen and now - seen[0] >= RATE_WINDOW_S:
        seen.popleft()


def _sweep(now: float) -> None:
    """Forget every address with nothing left in the window. Caller holds the lock."""
    for key in [key for key, seen in _hits.items() if not seen or now - seen[-1] >= RATE_WINDOW_S]:
        del _hits[key]


def rate_limit(request: Request) -> None:
    """Sliding-window cap on ``/ask``, keyed on client IP. 0 disables it.

    A /ask costs an embedding, a rerank and an LLM call, so an unthrottled route
    is both a spend and a CPU amplifier. This runs before the pipeline, so a
    rejected request never reaches the model.

    # trade-off: in-process, so the real budget is the limit times the instance
    # count and it resets on every deploy or scale event — the same known ceiling
    # the spend cap carries (app/observability/cost.py). The key is
    # ``request.client.host``, which is the caller's address only because both
    # Dockerfiles start uvicorn with --proxy-headers; it is still an address, not
    # an identity, and a NAT or a shared egress shares one bucket. Upgrade path
    # for anything real: Cloud Armor, or API Gateway quotas keyed on a real
    # credential.
    """
    limit = settings.ask_rate_limit_per_minute
    if limit <= 0:
        return

    key = request.client.host if request.client else "unknown"
    now = time.monotonic()

    with _hits_lock:
        if len(_hits) >= RATE_SWEEP_AT:
            _sweep(now)
        seen = _hits.get(key)
        if seen is None:
            seen = _hits[key] = deque()
        _trim(seen, now)
        if len(seen) >= limit:
            over_limit = True
        else:
            over_limit = False
            seen.append(now)

    if over_limit:
        record_request("/ask", "rate_limited")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"rate limit exceeded: {limit} requests per minute",
            headers={"Retry-After": str(int(RATE_WINDOW_S))},
        )
