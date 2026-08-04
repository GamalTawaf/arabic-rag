"""Server-sent event framing."""

from __future__ import annotations

import json


def sse_frame(event: str, payload: dict) -> str:
    """One SSE frame. ``ensure_ascii=False`` so Arabic stays readable on the wire.

    JSON escapes any newline inside the payload, so a frame is always exactly
    one ``data:`` line and the terminating blank line is unambiguous.
    """
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
