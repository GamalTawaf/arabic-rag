"""Shared-secret gate for the write path, used as a FastAPI dependency."""

from __future__ import annotations

import hmac
from typing import Annotated

from fastapi import Header, HTTPException, status

from app.config import settings


def require_ingest_key(x_api_key: Annotated[str | None, Header()] = None) -> None:
    """Shared-secret gate on ``/ingest``, off unless ``INGEST_API_KEY`` is set.

    Read at request time, not import time, so tests and deployments can flip it.
    ``compare_digest`` rather than ``==`` because a shared secret compared with
    an early-exit string comparison is timing-attackable.

    # trade-off: one static key, no rotation, no per-caller identity, and it is
    # the only thing between the open internet and a write path when Cloud Run is
    # deployed with allow_unauthenticated (terraform's default). Adequate for a
    # demo; the upgrade path is Cloud Run IAM + a service account, which is what
    # /ingest/pubsub already uses (terraform/pubsub.tf) and needs no app code.
    """
    expected = settings.ingest_api_key
    if not expected:
        return
    # Compared as bytes: compare_digest raises TypeError on a str containing any
    # non-ASCII codepoint, and the header is attacker-controlled — an unauthorized
    # caller could otherwise turn the auth gate into a 500 at will.
    if x_api_key is None or not hmac.compare_digest(
        x_api_key.encode("utf-8"), expected.encode("utf-8")
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid or missing x-api-key"
        )
