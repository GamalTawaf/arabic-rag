import pytest

from app.constants import EMBEDDING_DIMS
from app.data import Hit
from app.models.chunks import Chunk
from app.retrieval.search import (
    dense_search,
    hybrid_search,
    lexical_search,
    rrf_fuse,
)
from ingestion.normalize import normalize_for_index

DIM = EMBEDDING_DIMS["e5"]

# Same sentence twice: stored with diacritics, queried without (and vice versa).
WAGES_TEXT = "يَجِبُ عَلَى صَاحِبِ الْعَمَلِ أَنْ يَدْفَعَ الْأَجْرَ شَهْرِيًّا"
LEAVE_TEXT = "للعامل الحق في إجازة سنوية مدفوعة الأجر"


def unit_vector(index: int) -> list[float]:
    """A one-hot vector, so cosine similarity between axes is exactly 0."""
    vector = [0.0] * DIM
    vector[index] = 1.0
    return vector


def make_chunk(chunk_id: str, text: str, article: str = "1", **overrides) -> Chunk:
    fields = {
        "id": chunk_id,
        "doc_id": chunk_id.split(":")[0],
        "article": article,
        "seq": 0,
        "text": text,
        "text_normalized": normalize_for_index(text),
    }
    return Chunk(**{**fields, **overrides})


def make_hit(chunk_id: str, score: float = 1.0, source: str = "dense") -> Hit:
    return Hit(
        chunk_id=chunk_id,
        doc_id=chunk_id.split(":")[0],
        article="1",
        text=f"text of {chunk_id}",
        score=score,
        source=source,
    )


async def seed(session, *chunks: Chunk) -> None:
    session.add_all(chunks)
    await session.commit()


# --------------------------------------------------------------------- dense


async def test_dense_search_returns_chunks_ordered_by_cosine_similarity(db_session):
    # Arrange
    await seed(
        db_session,
        make_chunk("law:1:0", WAGES_TEXT, emb_e5=unit_vector(0)),
        make_chunk("law:2:0", LEAVE_TEXT, article="2", emb_e5=unit_vector(1)),
    )

    # Act — query leans mostly on axis 0
    query_vec = [0.9, 0.1] + [0.0] * (DIM - 2)
    hits = await dense_search(db_session, query_vec, "e5")

    # Assert
    assert [hit.chunk_id for hit in hits] == ["law:1:0", "law:2:0"]
    assert hits[0].score > hits[1].score
    assert hits[0].source == "dense"
    assert hits[0].text == WAGES_TEXT  # original diacritics, not the index form
    assert hits[0].doc_id == "law" and hits[0].article == "1"


async def test_dense_search_scores_an_exact_vector_match_at_one(db_session):
    # Arrange
    await seed(db_session, make_chunk("law:1:0", WAGES_TEXT, emb_e5=unit_vector(0)))

    # Act
    hits = await dense_search(db_session, unit_vector(0), "e5")

    # Assert
    assert hits[0].score == pytest.approx(1.0, abs=1e-6)


async def test_dense_search_excludes_rows_with_a_null_vector(db_session):
    # Arrange — law:2:0 was never embedded with e5
    await seed(
        db_session,
        make_chunk("law:1:0", WAGES_TEXT, emb_e5=unit_vector(0)),
        make_chunk("law:2:0", LEAVE_TEXT, article="2", emb_e5=None),
        make_chunk("law:3:0", LEAVE_TEXT, article="3", emb_bge=unit_vector(0)),
    )

    # Act
    hits = await dense_search(db_session, unit_vector(5), "e5")

    # Assert — NULLs are out of the index entirely, not ranked last
    assert [hit.chunk_id for hit in hits] == ["law:1:0"]


async def test_dense_search_respects_the_limit(db_session):
    # Arrange
    await seed(
        db_session,
        *(
            make_chunk(f"law:{i}:0", WAGES_TEXT, article=str(i), emb_e5=unit_vector(i))
            for i in range(5)
        ),
    )

    # Act
    hits = await dense_search(db_session, unit_vector(0), "e5", limit=2)

    # Assert
    assert len(hits) == 2


async def test_dense_search_rejects_an_unknown_model_key(db_session):
    with pytest.raises(ValueError, match="unknown model_key"):
        await dense_search(db_session, unit_vector(0), "nope")


async def test_dense_search_rejects_a_wrongly_sized_vector(db_session):
    with pytest.raises(ValueError, match="dims"):
        await dense_search(db_session, [0.1, 0.2], "e5")


def test_dense_query_sql_orders_by_the_cosine_operator():
    """The emitted SQL must be the pgvector <=> operator so HNSW can serve it."""
    # Arrange / Act
    from sqlalchemy.dialects import postgresql

    from app.retrieval.search import Chunk as _Chunk

    column = _Chunk.__table__.c.emb_e5
    sql = str(column.cosine_distance([0.0] * DIM).compile(dialect=postgresql.dialect()))

    # Assert
    assert "<=>" in sql


# ------------------------------------------------------------------- lexical


async def test_lexical_search_finds_a_chunk_despite_different_diacritics(db_session):
    # Arrange — stored text is fully diacritised
    await seed(db_session, make_chunk("law:1:0", WAGES_TEXT))

    # Act — user types it bare, with a hamza spelling the stored text lacks
    hits = await lexical_search(db_session, "الأجر")

    # Assert — proves normalize_query runs before plainto_tsquery
    assert [hit.chunk_id for hit in hits] == ["law:1:0"]
    assert hits[0].source == "lexical"
    assert hits[0].score > 0


async def test_lexical_search_ranks_the_better_match_first(db_session):
    # Arrange
    await seed(
        db_session,
        make_chunk("law:1:0", WAGES_TEXT),
        make_chunk("law:2:0", LEAVE_TEXT, article="2"),
    )

    # Act — both chunks contain الأجر, only one contains إجازة
    hits = await lexical_search(db_session, "إجازة سنوية")

    # Assert
    assert hits[0].chunk_id == "law:2:0"


async def test_lexical_search_matches_a_question_whose_terms_are_not_all_present(
    db_session,
):
    """Regression: plainto_tsquery ANDs terms and scored 0.000 recall on the eval set.

    "كم" and "للعامل" appear nowhere in the chunk; the query must still find it.
    """
    # Arrange
    await seed(db_session, make_chunk("law:2:0", LEAVE_TEXT, article="2"))

    # Act
    hits = await lexical_search(db_session, "كم مدة الإجازة السنوية للعامل؟")

    # Assert
    assert [hit.chunk_id for hit in hits] == ["law:2:0"]


async def test_lexical_search_ignores_tsquery_operator_characters(db_session):
    # Arrange
    await seed(db_session, make_chunk("law:1:0", WAGES_TEXT))

    # Act — raw tsquery syntax in user input must not reach the parser
    hits = await lexical_search(db_session, "الأجر & !(العمل) | *:B <-> ';DROP")

    # Assert
    assert [hit.chunk_id for hit in hits] == ["law:1:0"]


async def test_lexical_search_survives_a_pathologically_long_query(db_session):
    # Arrange
    await seed(db_session, make_chunk("law:1:0", WAGES_TEXT))

    # Act — 500 terms plus one 5000-character token
    noisy = " ".join(["الأجر", "ا" * 5000, *[f"كلمة{n}" for n in range(500)]])
    hits = await lexical_search(db_session, noisy)

    # Assert
    assert [hit.chunk_id for hit in hits] == ["law:1:0"]


async def test_lexical_search_returns_empty_for_an_unmatchable_query(db_session):
    # Arrange
    await seed(db_session, make_chunk("law:1:0", WAGES_TEXT))

    # Act
    hits = await lexical_search(db_session, "زيمبابوي")

    # Assert
    assert hits == []


async def test_lexical_search_returns_empty_for_a_blank_query(db_session):
    # Arrange
    await seed(db_session, make_chunk("law:1:0", WAGES_TEXT))

    # Act / Assert — punctuation-only normalizes to nothing matchable
    assert await lexical_search(db_session, "   ") == []
    assert await lexical_search(db_session, "؟؟؟") == []


# ----------------------------------------------------------------------- rrf


def test_rrf_fuse_scores_are_the_sum_of_reciprocal_ranks():
    # Arrange
    dense = [make_hit("a"), make_hit("b")]
    lexical = [make_hit("b", source="lexical"), make_hit("c", source="lexical")]

    # Act
    fused = rrf_fuse([dense, lexical], k=60)

    # Assert
    scores = {hit.chunk_id: hit.score for hit in fused}
    assert scores["a"] == pytest.approx(1 / 61)
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["c"] == pytest.approx(1 / 62)


def test_rrf_fuse_prefers_a_chunk_ranked_well_in_both_lists():
    # Arrange — "b" is 2nd twice; "a" is 1st once and absent from the other list
    dense = [make_hit("a"), make_hit("b")]
    lexical = [make_hit("c", source="lexical"), make_hit("b", source="lexical")]

    # Act
    fused = rrf_fuse([dense, lexical], k=60)

    # Assert
    assert fused[0].chunk_id == "b"


def test_rrf_fuse_breaks_ties_deterministically_by_chunk_id():
    # Arrange — three chunks each ranked 1st in exactly one list
    lists = [[make_hit("zz")], [make_hit("aa")], [make_hit("mm")]]

    # Act
    fused = rrf_fuse(lists)
    again = rrf_fuse(list(reversed(lists)))

    # Assert
    assert [hit.chunk_id for hit in fused] == ["aa", "mm", "zz"]
    assert [hit.chunk_id for hit in again] == ["aa", "mm", "zz"]


def test_rrf_fuse_respects_the_limit_and_relabels_the_source():
    # Arrange
    dense = [make_hit(name) for name in "abcde"]

    # Act
    fused = rrf_fuse([dense], limit=2)

    # Assert
    assert [hit.chunk_id for hit in fused] == ["a", "b"]
    assert all(hit.source == "rrf" for hit in fused)


def test_rrf_fuse_preserves_the_payload_of_the_first_list_a_hit_came_from():
    # Arrange
    dense = [make_hit("a")]
    lexical = [
        Hit("a", "other", "99", "lexical copy", 0.5, "lexical"),
        make_hit("b", source="lexical"),
    ]

    # Act
    fused = rrf_fuse([dense, lexical])

    # Assert
    top = next(hit for hit in fused if hit.chunk_id == "a")
    assert top.text == "text of a" and top.doc_id == "a" and top.article == "1"


def test_rrf_fuse_handles_empty_input():
    assert rrf_fuse([]) == []
    assert rrf_fuse([[], []]) == []


def test_rrf_fuse_rejects_a_non_positive_k():
    with pytest.raises(ValueError, match="k must be"):
        rrf_fuse([[make_hit("a")]], k=0)


# -------------------------------------------------------------------- hybrid


async def test_hybrid_search_merges_dense_only_and_lexical_only_matches(db_session):
    # Arrange — law:1:0 is reachable both ways, law:2:0 only lexically,
    # law:3:0 only densely (its text shares no tokens with the query).
    await seed(
        db_session,
        make_chunk("law:1:0", WAGES_TEXT, emb_e5=unit_vector(0)),
        make_chunk("law:2:0", LEAVE_TEXT, article="2", emb_e5=None),
        make_chunk("law:3:0", "نص اخر تماما", article="3", emb_e5=unit_vector(1)),
    )

    # Act
    query_vec = [0.9, 0.1] + [0.0] * (DIM - 2)
    hits = await hybrid_search(db_session, "إجازة سنوية مدفوعة الأجر", query_vec, "e5")

    # Assert
    assert [hit.chunk_id for hit in hits] == ["law:1:0", "law:2:0", "law:3:0"]
    assert all(hit.source == "rrf" for hit in hits)
    assert hits[0].text == WAGES_TEXT
    assert hits[0].score > hits[-1].score


async def test_hybrid_search_respects_the_limit(db_session):
    # Arrange
    await seed(
        db_session,
        make_chunk("law:1:0", WAGES_TEXT, emb_e5=unit_vector(0)),
        make_chunk("law:2:0", LEAVE_TEXT, article="2", emb_e5=unit_vector(1)),
    )

    # Act
    hits = await hybrid_search(db_session, "الأجر", unit_vector(0), "e5", limit=1)

    # Assert
    assert len(hits) == 1


async def test_hybrid_search_works_when_one_leg_finds_nothing(db_session):
    # Arrange
    await seed(db_session, make_chunk("law:1:0", WAGES_TEXT, emb_e5=unit_vector(0)))

    # Act — no lexical match at all, dense still carries the result
    hits = await hybrid_search(db_session, "زيمبابوي", unit_vector(0), "e5")

    # Assert
    assert [hit.chunk_id for hit in hits] == ["law:1:0"]


async def test_hybrid_search_propagates_a_failing_leg_without_leaking(db_session, recwarn):
    # Arrange — a short query_vec makes the dense leg raise while lexical still runs.
    await seed(db_session, make_chunk("law:1:0", WAGES_TEXT, emb_e5=unit_vector(0)))

    # Act / Assert — the real error surfaces, not an "operation on closed session"
    with pytest.raises(ValueError, match="dims"):
        await hybrid_search(db_session, "الأجر", [1.0, 0.0], "e5")

    # And the sibling leg finished on a live session rather than being torn out
    # from under itself, so nothing warns about an unawaited or orphaned task.
    assert [str(w.message) for w in recwarn] == []
