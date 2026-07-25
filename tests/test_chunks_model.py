from sqlalchemy import func, select

from app.models.chunks import EMBEDDING_COLUMNS, EMBEDDING_DIMS, Chunk

# Arabic with diacritics; normalized form has them stripped and alef folded.
ARABIC_WITH_DIACRITICS = "يَجِبُ عَلَى صَاحِبِ الْعَمَلِ أَنْ يَدْفَعَ الْأَجْرَ"
ARABIC_NORMALIZED = "يجب علي صاحب العمل ان يدفع الاجر"


def make_chunk(chunk_id: str = "law1:5:0", **overrides) -> Chunk:
    fields = {
        "id": chunk_id,
        "doc_id": "law1",
        "article": "5",
        "seq": 0,
        "text": ARABIC_WITH_DIACRITICS,
        "text_normalized": ARABIC_NORMALIZED,
    }
    return Chunk(**{**fields, **overrides})


def test_embedding_columns_and_dims_cover_the_same_models():
    # Arrange / Act / Assert
    assert EMBEDDING_COLUMNS.keys() == EMBEDDING_DIMS.keys()


def test_embedding_column_names_exist_on_the_model():
    for column in EMBEDDING_COLUMNS.values():
        assert column in Chunk.__table__.columns


async def test_insert_and_read_back_preserves_original_diacritics(db_session):
    # Arrange
    db_session.add(make_chunk())
    await db_session.commit()

    # Act
    stored = await db_session.get(Chunk, "law1:5:0")

    # Assert
    assert stored is not None
    assert stored.text == ARABIC_WITH_DIACRITICS  # diacritics survive the round trip
    assert stored.text_normalized == ARABIC_NORMALIZED
    assert stored.doc_id == "law1"
    assert stored.created_at is not None


async def test_tsv_is_generated_and_matches_a_normalized_arabic_word(db_session):
    # Arrange
    db_session.add(make_chunk())
    await db_session.commit()

    # Act
    match = func.plainto_tsquery("simple", "الاجر")
    found = await db_session.scalars(select(Chunk.id).where(Chunk.tsv.op("@@")(match)))
    tsv = await db_session.scalar(select(Chunk.tsv).where(Chunk.id == "law1:5:0"))

    # Assert
    assert found.all() == ["law1:5:0"]
    assert "يجب" in tsv  # auto-populated from text_normalized, no trigger needed


async def test_tsv_does_not_match_a_word_outside_the_chunk(db_session):
    # Arrange
    db_session.add(make_chunk())
    await db_session.commit()

    # Act
    match = func.plainto_tsquery("simple", "الاجازه")
    found = await db_session.scalars(select(Chunk.id).where(Chunk.tsv.op("@@")(match)))

    # Assert
    assert found.all() == []


async def test_cosine_distance_returns_the_nearest_chunk(db_session):
    # Arrange
    dim = EMBEDDING_DIMS["e5"]
    near = [1.0] + [0.0] * (dim - 1)
    far = [0.0, 1.0] + [0.0] * (dim - 2)
    db_session.add(make_chunk("law1:5:0", emb_e5=near))
    db_session.add(make_chunk("law1:6:0", article="6", seq=1, emb_e5=far))
    await db_session.commit()

    # Act
    query = [0.99, 0.01] + [0.0] * (dim - 2)
    nearest = await db_session.scalars(
        select(Chunk.id).order_by(Chunk.emb_e5.cosine_distance(query)).limit(1)
    )

    # Assert
    assert nearest.one() == "law1:5:0"
