"""Unit tests for Arabic index/query normalization. Pure logic - no DB, no network."""

import pytest

from ingestion.normalize import (
    normalize_digits,
    normalize_for_index,
    normalize_letters,
    normalize_query,
    strip_diacritics,
)


def test_strip_diacritics_removes_harakat():
    # Arrange - "the article" fully vocalized (fatha, sukun, shadda, damma)
    vocalized = "الْمَادَّةُ"

    # Act
    result = strip_diacritics(vocalized)

    # Assert
    assert result == "المادة"


def test_strip_diacritics_removes_tatweel():
    # Arrange - kashida-stretched "المادة"
    stretched = "الــمــادة"

    # Act
    result = strip_diacritics(stretched)

    # Assert
    assert result == "المادة"


def test_strip_diacritics_removes_dagger_alef():
    # Arrange - "هٰذا" carries U+0670 superscript alef
    with_dagger = "هٰذا"

    # Act
    result = strip_diacritics(with_dagger)

    # Assert
    assert result == "هذا"


def test_strip_diacritics_does_not_mutate_input():
    # Arrange
    original = "الْمَادَّةُ"

    # Act
    strip_diacritics(original)

    # Assert
    assert original == "الْمَادَّةُ"


def test_normalize_letters_unifies_alef_variants():
    # Arrange - hamza-above, hamza-below, madda, wasla
    variants = "أ إ آ ٱ"

    # Act
    result = normalize_letters(variants)

    # Assert
    assert result == "ا ا ا ا"


def test_normalize_letters_unifies_alef_maqsura_with_ya():
    # Arrange - "مصطفى" (name) and "على" (preposition spelled with maqsura)
    text = "مصطفى على"

    # Act
    result = normalize_letters(text)

    # Assert
    assert result == "مصطفي علي"


def test_normalize_letters_folds_ta_marbuta_to_ha():
    # Arrange - index form only: "المادة" -> "الماده"
    text = "المادة"

    # Act
    result = normalize_letters(text)

    # Assert
    assert result == "الماده"


def test_normalize_letters_unifies_hamza_seats():
    # Arrange - "مسؤول" (responsible) and "قائمة" (list)
    text = "مسؤول قائمة"

    # Act
    result = normalize_letters(text)

    # Assert
    assert result == "مسوول قايمه"


def test_normalize_digits_converts_arabic_indic():
    # Arrange
    text = "المادة ٥٧"

    # Act
    result = normalize_digits(text)

    # Assert
    assert result == "المادة 57"


def test_normalize_digits_converts_eastern_arabic_indic():
    # Arrange - U+06F0-U+06F9 (Persian/Urdu forms)
    text = "۰۱۲۳۴۵۶۷۸۹"

    # Act
    result = normalize_digits(text)

    # Assert
    assert result == "0123456789"


def test_normalize_for_index_runs_full_pipeline():
    # Arrange - diacritics + tatweel + hamza + Arabic-Indic digits + ragged spacing
    raw = "الْمَادَّةُ   ٥٧  مِنْ قَانُونِ الْعَمَلِ"

    # Act
    result = normalize_for_index(raw)

    # Assert
    assert result == "الماده 57 من قانون العمل"


def test_normalize_for_index_preserves_arabic_punctuation():
    # Arrange - meaningful punctuation: comma, semicolon, question mark
    raw = "ما هي المادة ٥٧، أو ٥٨؛ في القانون؟"

    # Act
    result = normalize_for_index(raw)

    # Assert
    assert result == "ما هي الماده 57، او 58؛ في القانون؟"


def test_normalize_for_index_collapses_whitespace_runs():
    # Arrange - tabs, newlines and repeated spaces between words
    raw = "  قانون \t\n  العمل   القطري  "

    # Act
    result = normalize_for_index(raw)

    # Assert
    assert result == "قانون العمل القطري"


def test_normalize_for_index_keeps_mixed_latin_and_ascii_digits_intact():
    # Arrange
    raw = "Article ٥٧ of Qatar Labour Law No. 14 of 2004 - المادة ٥٧"

    # Act
    result = normalize_for_index(raw)

    # Assert
    assert result == "Article 57 of Qatar Labour Law No. 14 of 2004 - الماده 57"


def test_normalize_for_index_is_idempotent():
    # Arrange
    raw = "الْمَادَّةُ ٥٧: مَسْؤُولِيَّةُ صَاحِبِ الْعَمَلِ — Article 57"

    # Act
    once = normalize_for_index(raw)
    twice = normalize_for_index(once)

    # Assert
    assert twice == once


def test_normalize_for_index_returns_empty_string_for_empty_input():
    # Arrange / Act
    result = normalize_for_index("")

    # Assert
    assert result == ""


def test_normalize_for_index_returns_empty_string_for_whitespace_only_input():
    # Arrange - spaces, tab, newline, non-breaking space
    raw = "  \t\n   "

    # Act
    result = normalize_for_index(raw)

    # Assert
    assert result == ""


def test_normalize_query_matches_normalize_for_index():
    # Arrange - a Gulf-dialect style question against MSA index form
    question = "شنو تقول المادة ٥٧ عن الإجازة؟"

    # Act / Assert
    assert normalize_query(question) == normalize_for_index(question)


def test_normalize_query_makes_orthographic_variants_match_the_index():
    # Arrange - same phrase, different but equally valid spellings
    indexed = normalize_for_index("المادَّة ٥٧ مِن قانون العمل")
    asked = normalize_query("الماده ٥٧ من قانون العمل")

    # Assert
    assert asked == indexed


# ------------------------------------------------ bidi / zero-width controls


@pytest.mark.parametrize(
    ("name", "char"),
    [
        ("RLM", "\u200f"),
        ("LRM", "\u200e"),
        ("ZWNJ", "\u200c"),
        ("ZWSP", "\u200b"),
        ("BOM", "\ufeff"),
        ("RLE", "\u202b"),
        ("soft hyphen", "\u00ad"),
    ],
)
def test_formatting_characters_never_reach_the_index_form(name, char):
    """Regression: these survived NFKC, are not `isspace()`, and landed in the tsvector.

    ``_tsquery`` builds query terms with ``\\w+``, which drops them — so an
    affected chunk indexed a token no query could ever produce and disappeared
    from the lexical leg silently. ``ingestion.fetch`` parses with
    ``convert_charrefs=True``, so the ``&rlm;``/``&lrm;`` entities in RTL legal
    markup arrive as exactly these.
    """
    # Arrange / Act
    normalized = normalize_for_index(f"المادة{char} 12")

    # Assert
    assert char not in normalized, name
    assert normalized == normalize_for_index("المادة 12")


def test_a_decorated_query_and_a_clean_document_still_normalize_alike():
    """Index and query forms must stay byte-identical or retrieval degrades."""
    assert normalize_query("\u200fكم\u200c مدة الإشعار؟") == normalize_for_index(
        "كم مدة الإشعار؟"
    )
