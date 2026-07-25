"""Article-aware chunking for Arabic legal documents.

Arabic statutes are organised by article (المادة), so the article boundary is the
natural retrieval unit: one article ~= one answerable rule. We split on article
headings first and only fall back to a size-based split inside an article that is
too long for the embedding model's context.

Chunk ids are `doc_id:article:seq` and are a pure function of the input text, so
re-ingesting the same document yields the same ids (eval ground truth depends on it).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ingestion.normalize import normalize_digits

# ponytail: chunk size is a plain character budget, not a real tokenizer. Arabic
# averages ~3 chars/token, so max_chars=1400 approximates the ~500-token target from
# the design. Ceiling: chunk sizes drift by +/-20% across scripts and digit-heavy
# text. Upgrade path: swap _split_with_overlap's len() for tiktoken / the embedding
# model's own tokenizer if benchmark results turn out sensitive to chunk size.
DEFAULT_MAX_CHARS = 1400
DEFAULT_OVERLAP_CHARS = 150

# "المادة (12)" / "المادة ٥٧" / "مادة 57" / "المادّة رقم 3" at the start of a line.
# [^\S\n] is "whitespace but not a newline" - headings must start their own line.
_ARTICLE_HEADING = re.compile(
    r"(?m)^[^\S\n]*(?:ال)?مادّ?ة[^\S\n]*(?:رقم[^\S\n]*)?"
    r"\(?[^\S\n]*([0-9٠-٩۰-۹]+)[^\S\n]*\)?"
)

_PREAMBLE_MARKER = "p"


@dataclass(frozen=True)
class TextChunk:
    """One retrievable unit. `text` is the original, un-normalized document text."""

    id: str
    doc_id: str
    article: str | None
    seq: int
    text: str


def split_articles(text: str) -> list[tuple[str | None, str]]:
    """Split a document into (article_number, body) pairs in document order.

    The article heading itself stays at the head of its body, so concatenating the
    bodies reconstructs the document. Text before the first heading is returned with
    article=None. Empty/whitespace-only sections are dropped.
    """
    if not text or not text.strip():
        return []

    matches = list(_ARTICLE_HEADING.finditer(text))
    if not matches:
        return [(None, text)]

    sections: list[tuple[str | None, str]] = []
    preamble = text[: matches[0].start()]
    if preamble.strip():
        sections.append((None, preamble))

    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        # A heading with nothing under it carries no retrievable content.
        if not text[match.end() : end].strip():
            continue
        # Article numbers go into chunk ids, so they must be ASCII whatever the
        # source used — same digit fold the index normalizer applies.
        sections.append((normalize_digits(match.group(1)), text[match.start() : end]))

    return sections


def chunk_document(
    doc_id: str,
    text: str,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> list[TextChunk]:
    """Chunk a document on article boundaries, then on size within long articles.

    `seq` is 0-based within each article (within the preamble, when article is None).
    """
    if max_chars < 1:
        raise ValueError("max_chars must be >= 1")
    if overlap_chars < 0:
        raise ValueError("overlap_chars must be >= 0")

    chunks: list[TextChunk] = []
    for article, body in split_articles(text):
        pieces = _split_with_overlap(body.strip(), max_chars, overlap_chars)
        marker = article if article is not None else _PREAMBLE_MARKER
        for seq, piece in enumerate(pieces):
            chunks.append(
                TextChunk(
                    id=f"{doc_id}:{marker}:{seq}",
                    doc_id=doc_id,
                    article=article,
                    seq=seq,
                    text=piece,
                )
            )
    return chunks


def _split_with_overlap(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    """Split text into <=max_chars pieces on word boundaries, with overlap."""
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    # Overlap larger than half the budget would make windows crawl forward.
    overlap = min(overlap_chars, max_chars // 2)
    pieces: list[str] = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + max_chars, length)
        if end < length:
            cut = _last_whitespace(text, start, end)
            if cut > start:
                end = cut
            # else: a single "word" longer than max_chars - hard split, rare enough
            # in legal prose that a URL-style blob is the only realistic trigger.
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= length:
            break
        start = _next_word_start(text, max(start + 1, end - overlap))

    return pieces


def _last_whitespace(text: str, start: int, end: int) -> int:
    """Index of the last whitespace char in text[start:end], or -1."""
    for index in range(end - 1, start, -1):
        if text[index].isspace():
            return index
    return -1


def _next_word_start(text: str, index: int) -> int:
    """Move index forward to the next word start, so overlap never begins mid-word."""
    length = len(text)
    if index <= 0 or index >= length:
        return index
    if text[index - 1].isspace():
        return index
    while index < length and not text[index].isspace():
        index += 1
    while index < length and text[index].isspace():
        index += 1
    return index
