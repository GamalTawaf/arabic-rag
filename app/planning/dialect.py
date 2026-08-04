"""Gulf-dialect detection and rule-based Gulf → MSA rewriting.

Two functions, both pure, both stdlib-only. They exist because the phase-2
benchmark measured a real retrieval penalty on dialect: e5 loses 9.9 points of
recall@10 (13.2 of recall@3) when the same question arrives in Gulf Arabic
instead of MSA, against an MSA corpus. The lexicon in :mod:`app.planning.lexicon`
is the recovery attempt, and it is deliberately measurable without an LLM.

Rewriting is conservative by construction:

* **Whole tokens only.** ``re.sub`` over ``\\w+`` — no substring surgery, so a
  mapping can never eat the middle of an unrelated word.
* **Matched normalized, emitted MSA.** The look-up key is the normalized token
  (hamza-insensitive, so it matches however the user typed it) but the value is
  properly spelled MSA, because the result gets embedded and the corpus was
  embedded from raw text.
* **Untouched tokens keep their original bytes.** Punctuation, spacing and every
  word without a lexicon entry pass through unchanged, so ``gulf_to_msa`` on an
  MSA question is the identity function.
"""

from __future__ import annotations

import re

from app.constants import (
    GULF,
    HAL_MIN_LEN,
    HAL_PREFIX,
    MSA,
    REGISTER_MARKERS,
    WAW_BLOCKLIST,
)
from app.planning.lexicon import GULF_TO_MSA
from ingestion.normalize import normalize_query

_WORD = re.compile(r"\w+")
_WAW = "و"  # the coordinating clitic: "وشكثر" = و + شكثر


def _direct(key: str) -> str | None:
    """Replacement for one already-normalized token, table then prefix rule."""
    if key in GULF_TO_MSA:
        return GULF_TO_MSA[key]
    if key.startswith(HAL_PREFIX) and len(key) >= HAL_MIN_LEN:
        # trade-off: "هالمدة" → "المدة", dropping the demonstrative rather than
        # guessing "هذا" vs "هذه" — gender agreement needs morphology the lexicon
        # does not have, and the demonstrative carries almost no retrieval signal
        # while a wrong one carries noise. Upgrade path: camel-tools morphology,
        # or let the LLM planner handle it. Emits normalized text (ة is already
        # folded to ه by the caller); harmless, the embedder tolerates it.
        return "ال" + key[len(HAL_PREFIX) :]
    return None


def _lookup(token: str) -> str | None:
    """MSA replacement for a raw token, or None to leave it exactly as it is."""
    key = normalize_query(token)
    replacement = _direct(key)
    if replacement is not None:
        return replacement
    # Arabic glues the conjunction to the next word. Only strip it when what
    # remains is itself a known Gulf form, so "وقت" and "ولد" are never split.
    if key.startswith(_WAW) and len(key) > 2 and key[1:] not in WAW_BLOCKLIST:
        rest = _direct(key[1:])
        if rest is not None:
            return _WAW + rest
    return None


def _tokens(question: str) -> list[str]:
    return [normalize_query(token) for token in _WORD.findall(question)]


def detect_register(question: str) -> str:
    """``"gulf"`` if the question carries a Gulf marker, else ``"msa"``.

    Drives the register the *answer* is written in, never what is retrieved —
    retrieval always goes to an MSA corpus. Biased towards "msa": a marker has to
    be unmistakable to be in :data:`REGISTER_MARKERS`, because answering an MSA
    question in dialect is a visible mistake while missing a marker just means a
    formal answer to an informal question.

    Measured on the 283 labelled eval pairs: 53/54 Gulf questions detected,
    0/229 MSA questions misfired (recall 0.981, precision 1.000).
    """
    for key in _tokens(question):
        if key in REGISTER_MARKERS:
            return GULF
        if key.startswith(HAL_PREFIX) and len(key) >= HAL_MIN_LEN:
            return GULF
        if key.startswith(_WAW) and key[1:] in REGISTER_MARKERS:
            return GULF
    return MSA


def gulf_to_msa(question: str) -> str:
    """Rewrite Gulf tokens to their MSA equivalents. Identity on MSA input.

    Idempotent: no value in the lexicon is also a key, so a second pass over the
    output changes nothing (asserted by the test suite, not by hope).
    """
    return _WORD.sub(lambda match: _lookup(match.group()) or match.group(), question)
