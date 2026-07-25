"""``POST /ask`` — SSE by default, JSON on request.

This module is a *formatter*. Every decision the answer depends on lives in
:mod:`app.service`; what happens here is validation, SSE framing, and the
mapping from pipeline exceptions to HTTP status codes:

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

import json
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.deps import ServiceDep
from app.generation.failover import AllProvidersFailed
from app.observability.cost import SpendCapExceeded
from app.observability.tracing import record_request
from app.service import (
    CONFIGS,
    DEFAULT_CONFIG,
    Answer,
    citation_payload,
    error_payload,
    usage_payload,
)

router = APIRouter(tags=["ask"])

DbSession = Annotated[AsyncSession, Depends(get_db)]

#: Long enough for any question in the eval set with room to spare, short enough
#: that a pathological body never reaches the embedder or the tokenizer.
MAX_QUESTION_CHARS = 1000

SSE_MEDIA_TYPE = "text/event-stream"
# Proxies love to buffer event streams; both headers are the conventional opt-out.
SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    config: str = DEFAULT_CONFIG
    stream: bool = True

    @field_validator("question")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be empty or whitespace")
        return value

    @field_validator("config")
    @classmethod
    def _known_config(cls, value: str) -> str:
        if value not in CONFIGS:
            raise ValueError(
                f"unknown retrieval config {value!r}; expected one of {list(CONFIGS)}"
            )
        return value


def sse_frame(event: str, payload: dict) -> str:
    """One SSE frame. ``ensure_ascii=False`` so Arabic stays readable on the wire.

    JSON escapes any newline inside the payload, so a frame is always exactly
    one ``data:`` line and the terminating blank line is unambiguous.
    """
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _answer_body(answer: Answer) -> dict:
    return {
        "answer": answer.text,
        "citations": citation_payload(answer.citations),
        "register": answer.register,
        "cached": answer.cached,
        "refused": answer.refused,
        "usage": usage_payload(answer.usage),
        "cost_usd": answer.usage.cost_usd if answer.usage else 0.0,
        "stages_ms": answer.stages,
    }


def _spend_cap_error(exc: SpendCapExceeded) -> HTTPException:
    record_request("/ask", "spend_cap")
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={
            "error": "daily_spend_cap_exceeded",
            "message": str(exc),
            "cap_usd": exc.cap_usd,
            "spent_usd": round(exc.spend.usd, 6),
            "calls": exc.spend.calls,
            "date": exc.spend.date,
            "remaining_usd": round(exc.remaining, 6),
        },
    )


def _providers_failed_error(exc: AllProvidersFailed) -> HTTPException:
    record_request("/ask", "providers_failed")
    return HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail={"error": "all_providers_failed", **error_payload(exc)},
    )


@router.post("/ask")
async def ask(payload: AskRequest, db: DbSession, service: ServiceDep):
    if payload.stream:
        return await _stream(payload, db, service)

    try:
        answer = await service.answer(db, payload.question, payload.config)
    except SpendCapExceeded as exc:
        raise _spend_cap_error(exc) from exc
    except AllProvidersFailed as exc:
        raise _providers_failed_error(exc) from exc
    record_request("/ask", "ok")
    return _answer_body(answer)


async def _stream(
    payload: AskRequest, db: AsyncSession, service: ServiceDep
) -> StreamingResponse:
    events = service.stream(db, payload.question, payload.config)
    try:
        first = await anext(events)
    except SpendCapExceeded as exc:
        await events.aclose()
        raise _spend_cap_error(exc) from exc
    except AllProvidersFailed as exc:  # pragma: no cover - generation is not open yet
        await events.aclose()
        raise _providers_failed_error(exc) from exc
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
