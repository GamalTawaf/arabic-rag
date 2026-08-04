import pytest

from app.constants import CHARS_PER_TOKEN, NOT_IN_CORPUS, SYSTEM_PROMPT
from app.data import Hit
from app.generation.budget import (
    build_prompt,
    estimate_tokens,
    fit_context,
    render_hit,
)

ARABIC_ARTICLE = "يجب على صاحب العمل أن يدفع الأجر شهرياً وفق أحكام هذا القانون"


def make_hit(chunk_id: str, text: str, article: str | None = "12") -> Hit:
    return Hit(
        chunk_id=chunk_id,
        doc_id=chunk_id.split(":")[0],
        article=article,
        text=text,
        score=1.0,
        source="rrf",
    )


def sized_hit(chunk_id: str, chars: int, article: str | None = "12") -> Hit:
    """A hit whose *rendered* block costs a predictable number of tokens."""
    return make_hit(chunk_id, "ن" * chars, article)


# ------------------------------------------------------------- estimate_tokens


def test_estimate_tokens_returns_zero_for_empty_text():
    assert estimate_tokens("") == 0


def test_estimate_tokens_counts_three_characters_per_token():
    assert estimate_tokens("ا" * 30) == 30 // CHARS_PER_TOKEN


def test_estimate_tokens_rounds_partial_tokens_up():
    # Arrange: 31 chars is 10 full tokens plus a remainder.
    # Act / Assert: never under-count, or the provider rejects the request.
    assert estimate_tokens("ا" * 31) == 11


def test_estimate_tokens_handles_real_arabic_text():
    assert estimate_tokens(ARABIC_ARTICLE) == pytest.approx(
        len(ARABIC_ARTICLE) / CHARS_PER_TOKEN, abs=1
    )


# ----------------------------------------------------------------- fit_context


def test_fit_context_returns_empty_for_no_hits():
    kept, used = fit_context([], max_tokens=6000)

    assert kept == []
    assert used == 0


def test_fit_context_keeps_every_hit_when_the_budget_is_ample():
    hits = [sized_hit(f"d:1:{i}", 300) for i in range(3)]

    kept, used = fit_context(hits, max_tokens=6000, reserve=512)

    assert kept == hits
    assert used == sum(estimate_tokens(render_hit(hit)) for hit in hits)


def test_fit_context_keeps_the_highest_ranked_hits_and_drops_the_tail():
    # Arrange: three identical-cost hits, budget for exactly two.
    hits = [sized_hit(f"d:1:{i}", 300) for i in range(3)]
    cost = estimate_tokens(render_hit(hits[0]))

    # Act
    kept, used = fit_context(hits, max_tokens=2 * cost + 512, reserve=512)

    # Assert: the two best, in their original order.
    assert [hit.chunk_id for hit in kept] == ["d:1:0", "d:1:1"]
    assert used == 2 * cost


def test_fit_context_never_truncates_a_chunk():
    hits = [sized_hit("d:1:0", 300), sized_hit("d:1:1", 300)]
    cost = estimate_tokens(render_hit(hits[0]))

    # Budget for one and a half chunks.
    kept, used = fit_context(hits, max_tokens=cost + cost // 2, reserve=0)

    assert kept == [hits[0]]
    assert used == cost
    assert kept[0].text == hits[0].text  # whole, not a fragment


def test_fit_context_returns_empty_for_a_single_oversized_chunk():
    # A half-article is worse than no article: the model answers confidently
    # from the fragment. Refusing is the correct outcome here.
    kept, used = fit_context([sized_hit("d:1:0", 90_000)], max_tokens=6000)

    assert kept == []
    assert used == 0


def test_fit_context_honours_the_reserve():
    hits = [sized_hit("d:1:0", 300)]
    cost = estimate_tokens(render_hit(hits[0]))

    # Arrange: max_tokens alone would fit the chunk; the reserve must not.
    kept, _ = fit_context(hits, max_tokens=cost + 10, reserve=100)

    assert kept == []


def test_fit_context_fills_the_budget_exactly():
    hits = [sized_hit("d:1:0", 300)]
    cost = estimate_tokens(render_hit(hits[0]))

    kept, used = fit_context(hits, max_tokens=cost, reserve=0)

    assert kept == hits
    assert used == cost


def test_fit_context_rejects_a_negative_reserve():
    with pytest.raises(ValueError, match="reserve"):
        fit_context([], max_tokens=6000, reserve=-1)


# ---------------------------------------------------------------- build_prompt


def test_build_prompt_instructs_the_model_to_cite_article_numbers():
    system, _ = build_prompt("كم مدة الإجازة؟", [make_hit("d:12:0", ARABIC_ARTICLE)])

    assert "[المادة 12]" in system  # the exact citation format, shown by example
    assert "رقم المادة" in system


def test_build_prompt_instructs_the_model_to_refuse_outside_the_corpus():
    system, _ = build_prompt("سؤال", [])

    assert NOT_IN_CORPUS in system


def test_build_prompt_instructs_the_model_to_answer_in_the_users_register():
    system, _ = build_prompt("سؤال", [])

    assert "الخليجية" in system
    assert "الفصحى" in system


def test_build_prompt_grounds_the_model_in_the_provided_articles_only():
    system, _ = build_prompt("سؤال", [])

    assert "وحدها" in system
    assert "معرفة خارجية" in system
    assert system == SYSTEM_PROMPT


def test_build_prompt_user_message_carries_the_question_and_every_article():
    hits = [make_hit("d:12:0", ARABIC_ARTICLE, "12"), make_hit("d:5:0", "نص آخر", "5")]

    _, user = build_prompt("كم مدة الإجازة؟", hits)

    assert "كم مدة الإجازة؟" in user
    assert ARABIC_ARTICLE in user
    assert "نص آخر" in user
    assert "[المادة 12]" in user
    assert "[المادة 5]" in user


def test_build_prompt_labels_an_articleless_chunk_with_its_chunk_id():
    _, user = build_prompt("سؤال", [make_hit("d:preamble:0", "ديباجة", article=None)])

    assert "[d:preamble:0]" in user


def test_build_prompt_says_plainly_when_nothing_was_retrieved():
    _, user = build_prompt("سؤال", [])

    assert "لا توجد مواد مسترجعة" in user
    assert "سؤال" in user
