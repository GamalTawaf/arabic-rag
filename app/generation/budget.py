"""Context-token budget and the answer prompt.

Two jobs, kept together because they have to agree: whatever
:func:`fit_context` counts is exactly what :func:`build_prompt` renders. If the
budget measured raw chunk text while the prompt shipped a formatted block with
article headers, the budget would be wrong by a few percent in the direction
that causes a provider 400 at the worst moment.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.retrieval.search import Hit

# trade-off: Arabic runs about 3 characters per token on the Claude and Gemini
# tokenizers (a 4-letter word plus its space is typically one or two tokens),
# so length/3 is within ~10% and always rounds up. No tiktoken: it is the wrong
# tokenizer for both providers, it is a dependency and a model download, and the
# only consumer is a budget that already keeps a 512-token reserve. Ceiling:
# a Latin-heavy or digit-heavy question is over-estimated. Upgrade path when
# cost accounting needs exactness rather than safety — the provider's own
# counter (Anthropic's /messages/count_tokens, Gemini's count_tokens).
CHARS_PER_TOKEN = 3

DEFAULT_RESERVE_TOKENS = 512  # question + system prompt + room for the answer

SYSTEM_PROMPT = """\
أنت مساعد قانوني يجيب عن أسئلة قانون العمل القطري اعتماداً على مواد مرفقة فقط.

القواعد:
1. أجب من المواد المرفقة وحدها. لا تستعن بمعرفة خارجية ولا تستنتج ما ليس فيها.
2. اذكر رقم المادة بعد كل معلومة توردها، بهذه الصيغة: [المادة 12].
3. أجب بنفس أسلوب السؤال: إن سُئلت بالعامية الخليجية فأجب بالعامية الخليجية، \
وإن سُئلت بالفصحى فأجب بالفصحى.
4. إن لم تكن الإجابة موجودة في المواد المرفقة فقل بوضوح: \
"لا تتضمن المواد المتاحة إجابة عن هذا السؤال." ولا تضف أي تخمين.
5. اختصر: من جملة إلى أربع جمل."""

NOT_IN_CORPUS = "لا تتضمن المواد المتاحة إجابة عن هذا السؤال."


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
