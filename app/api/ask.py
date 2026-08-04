"""``POST /ask`` — SSE by default, JSON on request.

This module is a *formatter*: SSE framing and nothing else. The answer is
decided in :mod:`app.service`, the request shape in :mod:`app.data`, and the
mapping from a pipeline exception to a status code in :mod:`app.main`:

============================  ======  =========================================
``SpendCapExceeded``          503     the daily USD cap is spent
``AllProvidersFailed``        502     every configured provider failed, named
bad question / bad config     422     rejected before the DB or an LLM is touched
============================  ======  =========================================

**Why the citations event goes first.** Retrieval finishes in single-digit
milliseconds; the first generated token arrives hundreds of milliseconds later.
Emitting sources up front lets a UI render them during that gap, and it means a
reader can check what the answer rests on before the answer exists.

**Where streaming stops being able to use status codes.** The route pulls the
first event out of the pipeline *before* returning a response, so anything that
fails ahead of generation (spend cap, validation, retrieval) still becomes a
status code. Once that first frame is committed the status line is gone, so a
generation failure arrives as an ``error`` event instead — see
:meth:`app.service.RagService.stream`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from app.constants import SSE_HEADERS, SSE_MEDIA_TYPE
from app.data import AskRequest
from app.deps import ServiceDep
from app.lib.rate_limit import rate_limit
from app.lib.sse import sse_frame
from app.observability.tracing import record_request
from app.service import answer_body

router = APIRouter(tags=["ask"])


@router.post("/ask", dependencies=[Depends(rate_limit)])
async def ask(payload: AskRequest, service: ServiceDep):
    if payload.stream:
        return await _stream(payload, service)

    answer = await service.answer(payload.question, payload.config)
    record_request("/ask", "ok")
    return answer_body(answer)


async def _stream(payload: AskRequest, service: ServiceDep) -> StreamingResponse:
    events = service.stream(payload.question, payload.config)
    try:
        # Pulled before a response exists, so a spend cap or a dead provider is
        # still a status code — ``app.main`` turns each into one. After this the
        # status line is committed and a failure can only be an `error` frame.
        first = await anext(events)
    except StopAsyncIteration as exc:  # pragma: no cover - stream always emits
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="pipeline produced no events",
        ) from exc

    async def frames() -> AsyncIterator[str]:
        yield sse_frame(*first)
        try:
            async for event, data in events:
                yield sse_frame(event, data)
        finally:
            # A disconnected browser throws GeneratorExit in here; closing the
            # pipeline generator explicitly releases its DB session and any open
            # provider response instead of waiting for the GC.
            await events.aclose()

    # "ok" here means the request was accepted and its citations produced — the
    # last point at which the outcome is still knowable from the route. A
    # generation failure after this shows up as an `error` frame and in the
    # generate span, not in this counter.
    record_request("/ask", "ok")
    return StreamingResponse(
        frames(), media_type=SSE_MEDIA_TYPE, headers=SSE_HEADERS
    )
