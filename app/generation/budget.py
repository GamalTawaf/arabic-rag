"""Context-token budget and the answer prompt.

Two jobs, kept together because they have to agree: whatever
:func:`fit_context` counts is exactly what :func:`build_prompt` renders. If the
budget measured raw chunk text while the prompt shipped a formatted block with
article headers, the budget would be wrong by a few percent in the direction
that causes a provider 400 at the worst moment.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.constants import CHARS_PER_TOKEN, DEFAULT_RESERVE_TOKENS, SYSTEM_PROMPT
from app.data import Hit


def estimate_tokens(text: str) -> int:
    """Approximate token count for Arabic text: characters / 3, rounded up."""
    if not text:
        return 0
    return -(-len(text) // CHARS_PER_TOKEN)  # ceil division, no float rounding


def render_hit(hit: Hit) -> str:
    """One retrieved chunk as it appears in the prompt, header included.

    The header is what the model cites, so it carries the article number when
    there is one and falls back to the chunk id when there is not — a citation
    the caller cannot resolve is worse than an ugly one.
    """
    label = f"المادة {hit.article}" if hit.article else hit.chunk_id
    return f"[{label}]\n{hit.text}"


def fit_context(
    hits: Sequence[Hit],
    max_tokens: int,
    reserve: int = DEFAULT_RESERVE_TOKENS,
) -> tuple[list[Hit], int]:
    """Keep the highest-ranked hits that fit in ``max_tokens - reserve``.

    Hits are assumed ranked best-first (what rerank/RRF return). Chunks are
    dropped **whole**, from the bottom, and the walk stops at the first chunk
    that does not fit. A chunk is never truncated: half an article reads like a
    complete rule with its exception missing, and the model will answer from it
    confidently and wrongly. An oversized top chunk therefore yields an empty
    context — the caller's "not in corpus" path, which is the honest outcome.

    Returns ``(kept hits in original order, tokens used)``.

    # trade-off: stops at the first chunk that does not fit instead of skipping
    # it and trying smaller lower-ranked ones. Keeping the prefix means the
    # context is always the top-N by rank, which is what the eval measures.
    # Upgrade path if chunk sizes ever get very uneven: continue past the
    # oversized chunk and take whatever else fits.
    """
    if reserve < 0:
        raise ValueError(f"reserve must be >= 0, got {reserve}")

    budget = max_tokens - reserve
    kept: list[Hit] = []
    used = 0
    for hit in hits:
        cost = estimate_tokens(render_hit(hit))
        if used + cost > budget:
            break
        kept.append(hit)
        used += cost
    return kept, used


def build_prompt(question: str, hits: Sequence[Hit]) -> tuple[str, str]:
    """``(system, user)`` in Arabic. ``hits`` should already be budget-fitted.

    An empty ``hits`` still produces a valid prompt: the material section says so
    explicitly, which routes rule 4 rather than leaving the model to invent one.
    """
    if hits:
        blocks = "\n\n".join(render_hit(hit) for hit in hits)
    else:
        blocks = "(لا توجد مواد مسترجعة)"
    user = f"المواد المرفقة:\n\n{blocks}\n\nالسؤال: {question}"
    return SYSTEM_PROMPT, user
