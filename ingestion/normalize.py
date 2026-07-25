"""Arabic text normalization for INDEX and QUERY forms only.

The functions here are lossy on purpose: they fold orthographic variants that
Arabic writers use interchangeably (hamza seats, alef maqsura vs ya, ta marbuta
vs ha, diacritics, tatweel, Arabic-Indic digits) so that a query and a document
that mean the same thing hash/match the same way.

**Callers MUST keep the original text.** Never store a normalized string as the
displayable/citable text of a chunk - it is a retrieval key, not content. The
pipeline stores raw text (display, citations, embedding input) alongside the
normalized text, which backs the tsvector and therefore lexical retrieval only -
see ``ingestion.pipeline`` for why embeddings read the raw text.

Stdlib only (``unicodedata`` + ``str.translate`` + ``re``). Every function
returns a new string and never mutates its input.
"""

from __future__ import annotations

import re
import unicodedata

# Harakat (fathatan..sukun), dagger alef, tatweel.
_DIACRITIC_CODEPOINTS = (
    *range(0x064B, 0x0653),  # ً ٌ ٍ َ ُ ِ ّ ْ
    0x0670,  # ٰ  dagger / superscript alef
    0x0640,  # ـ  tatweel (kashida)
)
_DIACRITICS_TABLE = dict.fromkeys(_DIACRITIC_CODEPOINTS)

# ponytail: flat one-to-one folds, no morphology and no dialect-specific letters
# (Persian/Urdu ک گ پ, Egyptian ی). Ceiling: fails on non-Arabic-language text in
# Arabic script. Upgrade path if that ever matters: camel-tools' normalizer.
_LETTERS_TABLE = str.maketrans(
    {
        "أ": "ا",  # أ -> ا
        "إ": "ا",  # إ -> ا
        "آ": "ا",  # آ -> ا
        "ٱ": "ا",  # ٱ -> ا
        "ى": "ي",  # ى -> ي
        "ة": "ه",  # ة -> ه  (index form only)
        "ؤ": "و",  # ؤ -> و
        "ئ": "ي",  # ئ -> ي
    }
)

_DIGITS_TABLE = str.maketrans(
    {
        **{chr(0x0660 + i): str(i) for i in range(10)},  # ٠-٩ Arabic-Indic
        **{chr(0x06F0 + i): str(i) for i in range(10)},  # ۰-۹ Eastern Arabic-Indic
    }
)

_WHITESPACE_RUN = re.compile(r"\s+")


def strip_diacritics(text: str) -> str:
    """Remove harakat (U+064B-U+0652), dagger alef (U+0670) and tatweel (U+0640)."""
    return text.translate(_DIACRITICS_TABLE)


def normalize_letters(text: str) -> str:
    """Fold orthographic letter variants to a single index form.

    Alef variants (أ إ آ ٱ) -> ا, alef maqsura (ى) -> ي, ta marbuta (ة) -> ه,
    hamza seats (ؤ -> و, ئ -> ي). The ta marbuta fold is index-only: it destroys
    a real morphological distinction, so never show the result to a user.
    """
    return text.translate(_LETTERS_TABLE)


def normalize_digits(text: str) -> str:
    """Map Arabic-Indic (٠-٩) and Eastern Arabic-Indic (۰-۹) digits to ASCII 0-9."""
    return text.translate(_DIGITS_TABLE)


def normalize_for_index(text: str) -> str:
    """Full index-form pipeline.

    NFKC -> strip diacritics -> fold letters -> ASCII digits -> collapse
    whitespace runs to a single space -> strip. Arabic punctuation (، ؛ ؟) and
    Latin/ASCII content pass through untouched. Idempotent.
    """
    folded = unicodedata.normalize("NFKC", text)
    folded = strip_diacritics(folded)
    folded = normalize_letters(folded)
    folded = normalize_digits(folded)
    return _WHITESPACE_RUN.sub(" ", folded).strip()


def normalize_query(text: str) -> str:
    """Normalize a user query. Identical to :func:`normalize_for_index`.

    Kept as a separate name so call sites read honestly: query and index forms
    MUST stay byte-identical or retrieval silently degrades. If they ever need to
    diverge, this is the seam to change.
    """
    return normalize_for_index(text)
