import math

import pytest
from sqlalchemy import select

from app.models.query_cache import CACHE_DIM, QueryCache
from app.retrieval.cache import guard_key, lookup, store

QUESTION = "كم مدة الإجازة السنوية للعامل؟"
ANSWER = "للعامل إجازة سنوية مدفوعة الأجر [المادة 79]."
PIPELINE = "hybrid+rerank|r20|c5|rr1|fake:fake-1"
CITATIONS = ["labor:79:0", "labor:80:0"]


def unit_vector(index: int) -> list[float]:
    """A one-hot vector: cosine similarity between two distinct axes is 0."""
    vector = [0.0] * CACHE_DIM
    vector[index] = 1.0
    return vector


def vector_at_cosine(cosine: float) -> list[float]:
    """A unit vector whose cosine similarity with ``unit_vector(0)`` is ``cosine``."""
    vector = [0.0] * CACHE_DIM
    vector[0] = cosine
    vector[1] = math.sqrt(1.0 - cosine**2)
    return vector


# --------------------------------------------------------------- guard_key


def test_guard_key_captures_digits_in_order():
    assert guard_key("المادة 5 بعد 3 سنوات")[0] == ("5", "3")
    assert guard_key("المادة 3 بعد 5 سنوات")[0] == ("3", "5")


def test_guard_key_normalizes_arabic_indic_digits_to_ascii():
    assert guard_key("بعد ٣ سنوات")[0] == guard_key("بعد 3 سنوات")[0] == ("3",)


def test_guard_key_captures_negation_particles():
    assert "لا" in guard_key("هل لا يجوز الفصل؟")[1]
    assert guard_key("هل يجوز الفصل؟")[1] == frozenset()


def test_guard_key_does_not_catch_spelled_out_numbers():
    """Pins the guard's real ceiling so nobody assumes it covers this.

    "سنة واحدة" and "سنتين" carry no digits, so the guard sees one key. Measured
    with bge-m3 the pair scores 0.9448 cosine — rejected by the 0.95 threshold
    with only 0.006 to spare, and by nothing else. Lowering
    ``semantic_cache_threshold`` below 0.94 without first folding Arabic number
    words into the digit key would serve the wrong notice period.
    """
    assert guard_key("بعد سنة واحدة") == guard_key("بعد سنتين")


# ------------------------------------------------------------------ lookup


async def test_lookup_returns_none_when_the_cache_is_empty(db_session):
    assert await lookup(db_session, QUESTION, unit_vector(0), "bge", PIPELINE, 0.95) is None


async def test_store_then_lookup_returns_the_cached_answer(db_session):
    # Arrange
    await store(db_session, QUESTION, unit_vector(0), "bge", PIPELINE, ANSWER, CITATIONS)

    # Act
    cached = await lookup(db_session, QUESTION, unit_vector(0), "bge", PIPELINE, 0.95)

    # Assert
    assert cached is not None
    assert cached.answer == ANSWER
    assert cached.citations == CITATIONS
    assert cached.similarity == pytest.approx(1.0, abs=1e-5)
    assert 0 <= cached.age_seconds < 60


async def test_lookup_misses_when_the_question_is_unrelated(db_session):
    await store(db_session, QUESTION, unit_vector(0), "bge", PIPELINE, ANSWER, CITATIONS)

    # An orthogonal vector: cosine 0.0, nowhere near the threshold.
    assert await lookup(db_session, "سؤال مختلف", unit_vector(7), "bge", PIPELINE, 0.95) is None


async def test_lookup_hits_just_above_the_threshold_and_misses_just_below(db_session):
    # Arrange: one entry, and a query vector at a known cosine of 0.95.
    await store(db_session, QUESTION, unit_vector(0), "bge", PIPELINE, ANSWER, CITATIONS)
    near = vector_at_cosine(0.95)

    # Act / Assert: the same pair flips purely on the threshold.
    assert await lookup(db_session, QUESTION, near, "bge", PIPELINE, 0.94) is not None
    assert await lookup(db_session, QUESTION, near, "bge", PIPELINE, 0.96) is None


async def test_lookup_never_matches_a_vector_from_another_model(db_session):
    # A vector from another model is in an unrelated space; a "match" there is a
    # confidently wrong answer, so model_key isolation is not optional.
    await store(db_session, QUESTION, unit_vector(0), "e5", PIPELINE, ANSWER, CITATIONS)

    assert await lookup(db_session, QUESTION, unit_vector(0), "bge", PIPELINE, 0.95) is None
    assert await lookup(db_session, QUESTION, unit_vector(0), "e5", PIPELINE, 0.95) is not None


async def test_lookup_rejects_a_semantic_match_whose_digits_differ(db_session):
    # Arrange: identical vectors (similarity 1.0) but a different number — the
    # exact collision a similarity threshold alone cannot catch.
    await store(
        db_session, "ما مدة الإشعار بعد 3 سنوات؟", unit_vector(0), "bge", PIPELINE, ANSWER, []
    )

    cached = await lookup(
        db_session, "ما مدة الإشعار بعد 5 سنوات؟", unit_vector(0), "bge", PIPELINE, 0.95
    )

    assert cached is None


async def test_lookup_rejects_a_semantic_match_whose_negation_differs(db_session):
    # The case the guard exists for: measured with bge-m3, inserting "لا" into
    # "هل يجوز لصاحب العمل فصل العامل أثناء الإجازة؟" still scores 0.9870 cosine,
    # over any sane threshold, with the opposite answer.
    await store(db_session, "هل يجوز فصل العامل؟", unit_vector(0), "bge", PIPELINE, ANSWER, [])

    cached = await lookup(
        db_session, "هل لا يجوز فصل العامل؟", unit_vector(0), "bge", PIPELINE, 0.95
    )

    assert cached is None


async def test_lookup_accepts_a_paraphrase_with_the_same_digits_and_negation(db_session):
    # The guard must not reject everything: same number, same (absent) negation.
    await store(
        db_session, "كم مدة الإشعار بعد 3 سنوات؟", unit_vector(0), "bge", PIPELINE, ANSWER, []
    )

    cached = await lookup(
        db_session, "كم مدة الإنذار بعد 3 سنوات؟", unit_vector(0), "bge", PIPELINE, 0.95
    )

    assert cached is not None


async def test_the_guard_over_rejects_interrogative_ma(db_session):
    """Pins a known, deliberate cost of the guard.

    "ما" is both the commonest MSA interrogative ("ما مدة...") and a Gulf
    negation ("ما يجوز..."). Telling them apart needs morphology, so the guard
    treats every "ما" as a negation. Consequence: an interrogative "ما" phrasing
    never matches a "كم" phrasing of the same question — measured at 0.9652
    cosine with bge-m3, so this is a real hit we give up. The failure direction
    is deliberate: this costs one generation, while the opposite error
    (accepting "ما يجوز" as "يجوز") returns the inverted legal answer.
    """
    await store(db_session, "ما مدة الإشعار؟", unit_vector(0), "bge", PIPELINE, ANSWER, [])

    assert await lookup(db_session, "كم مدة الإشعار؟", unit_vector(0), "bge", PIPELINE, 0.95) is None


async def test_a_hit_increments_hits_and_stamps_last_hit_at(db_session):
    # Arrange
    await store(db_session, QUESTION, unit_vector(0), "bge", PIPELINE, ANSWER, CITATIONS)
    row = (await db_session.execute(select(QueryCache))).scalar_one()
    assert row.hits == 0
    assert row.last_hit_at is None

    # Act
    await lookup(db_session, QUESTION, unit_vector(0), "bge", PIPELINE, 0.95)
    await lookup(db_session, QUESTION, unit_vector(0), "bge", PIPELINE, 0.95)

    # Assert
    refreshed = (await db_session.execute(select(QueryCache))).scalar_one()
    await db_session.refresh(refreshed)
    assert refreshed.hits == 2
    assert refreshed.last_hit_at is not None


async def test_a_guard_rejection_does_not_count_as_a_hit(db_session):
    await store(db_session, "بعد 3 سنوات", unit_vector(0), "bge", PIPELINE, ANSWER, [])

    await lookup(db_session, "بعد 5 سنوات", unit_vector(0), "bge", PIPELINE, 0.95)

    row = (await db_session.execute(select(QueryCache))).scalar_one()
    await db_session.refresh(row)
    assert row.hits == 0


# ------------------------------------------------------------------- store


async def test_store_normalizes_the_query_it_persists(db_session):
    await store(db_session, "الإجازةُ السَّنوية", unit_vector(0), "bge", PIPELINE, ANSWER, [])

    row = (await db_session.execute(select(QueryCache))).scalar_one()
    assert row.query_normalized == "الاجازه السنويه"  # diacritics/hamza folded


async def test_store_ignores_an_empty_answer(db_session):
    await store(db_session, QUESTION, unit_vector(0), "bge", PIPELINE, "   ", [])

    assert (await db_session.execute(select(QueryCache))).first() is None


# The answer that prompted this guard, verbatim from the live service: Qwen
# emitted two hiragana and a stray ampersand once, `store` took it, and every
# semantically-equivalent question served those bytes back forever.
POISONED = "صاحب العمل ملزم بضمان النظافة والتهوية في أماكن العمل،&oその [&المادة 103]."


async def test_store_ignores_an_answer_with_characters_from_another_script(db_session):
    await store(db_session, QUESTION, unit_vector(0), "bge", PIPELINE, POISONED, [])

    assert (await db_session.execute(select(QueryCache))).first() is None


@pytest.mark.parametrize(
    "answer",
    [
        "الحد الأقصى ثماني ساعات [المادة 73].",  # the ordinary case
        "تنص الاتفاقية (ILO C189) على ذلك [المادة 4].",  # Latin acronyms are legitimate
        "لا تقل عن 3 أسابيع — أي ما يعادل ٢١ يوماً… [المادة 79].",  # digits, dashes, ellipsis
        "الأجر 50% من الأساسي [المادة 100].",
    ],
)
async def test_store_keeps_answers_that_are_merely_arabic_with_punctuation(db_session, answer):
    await store(db_session, QUESTION, unit_vector(0), "bge", PIPELINE, answer, [])

    row = (await db_session.execute(select(QueryCache))).scalar_one()
    assert row.answer == answer


# -------------------------------------------------------------- validation


async def test_lookup_rejects_a_model_whose_vectors_do_not_fit_the_column(db_session):
    with pytest.raises(ValueError, match="1024-dim"):
        await lookup(db_session, QUESTION, unit_vector(0), "openai", PIPELINE, 0.95)


async def test_lookup_rejects_a_vector_of_the_wrong_length(db_session):
    with pytest.raises(ValueError, match="expected 1024"):
        await lookup(db_session, QUESTION, [0.1, 0.2], "bge", PIPELINE, 0.95)


async def test_store_rejects_a_missing_model_key(db_session):
    with pytest.raises(ValueError, match="model_key"):
        await store(db_session, QUESTION, unit_vector(0), "", PIPELINE, ANSWER, [])


async def test_lookup_never_matches_an_answer_from_another_pipeline(db_session):
    """Regression: the key was (model_key, embedding, guard) only.

    Asking the same question under ``config=lexical`` returned the answer
    ``config=hybrid+rerank`` had produced — cached=true, provider never called,
    retrieval never run. Any A/B through the HTTP API measured the cache.
    """
    # Arrange — one answer, stored under one pipeline
    await store(db_session, QUESTION, unit_vector(0), "bge", PIPELINE, ANSWER, CITATIONS)

    # Act — identical question and vector, different pipeline
    other = await lookup(
        db_session, QUESTION, unit_vector(0), "bge", "lexical|r20|c5|rr0|fake:fake-1", 0.95
    )
    same = await lookup(db_session, QUESTION, unit_vector(0), "bge", PIPELINE, 0.95)

    # Assert
    assert other is None
    assert same is not None and same.answer == ANSWER
