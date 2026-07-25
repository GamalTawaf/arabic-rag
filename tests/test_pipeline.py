"""Integration tests for the corpus -> chunks pipeline (needs the pgvector DB)."""

import pytest
from sqlalchemy import func, select

from app.models.chunks import Chunk
from ingestion.fetch import CorpusDoc
from ingestion.normalize import normalize_for_index
from ingestion.pipeline import ingest_documents

DOC_ID = "test-law-1-2020"

# Preamble + two articles, with diacritics and Arabic-Indic digits so the
# normalization and id-numbering paths are both exercised.
TWO_ARTICLE_TEXT = """قانونٌ تجريبي رقم ١ لسنة ٢٠٢٠
المادة (1)
يُعمَل بأحكام هذا القانون اعتباراً من تاريخ نشره.
المادة (٢)
تسري أحكامُ هذا القانون على جميع العُمّال الخاضعين له.
"""


def make_doc(text: str = TWO_ARTICLE_TEXT, doc_id: str = DOC_ID) -> CorpusDoc:
    return CorpusDoc(
        doc_id=doc_id,
        title="قانون تجريبي",
        source_url=None,
        license="public-domain",
        text=text,
    )


class FakeEmbedder:
    """Deterministic vectors, zero ML dependencies."""

    def __init__(self, model_key: str = "e5", dim: int = 1024) -> None:
        self.model_key = model_key
        self.dim = dim
        self.batch_sizes: list[int] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.batch_sizes.append(len(texts))
        return [[float(len(text) % 10)] * self.dim for text in texts]


async def fetch_rows(session) -> list[Chunk]:
    result = await session.execute(select(Chunk).order_by(Chunk.id))
    return list(result.scalars())


async def test_ingests_two_article_document_with_stable_ids(db_session):
    # Arrange
    docs = [make_doc()]

    # Act
    stats = await ingest_documents(docs, db_session)

    # Assert
    rows = await fetch_rows(db_session)
    assert stats.documents == 1
    assert stats.chunks_written == 3
    assert stats.chunks_skipped == 0
    assert [row.id for row in rows] == [
        f"{DOC_ID}:1:0",
        f"{DOC_ID}:2:0",
        f"{DOC_ID}:p:0",
    ]
    assert [row.article for row in rows] == ["1", "2", None]
    assert all(row.doc_id == DOC_ID and row.seq == 0 for row in rows)


async def test_stores_original_text_and_normalized_index_form(db_session):
    # Arrange / Act
    await ingest_documents([make_doc()], db_session)

    # Assert
    article_one = await db_session.get(Chunk, f"{DOC_ID}:1:0")
    assert "يُعمَل" in article_one.text  # diacritics preserved for display
    assert article_one.text_normalized == normalize_for_index(article_one.text)
    assert "ُ" not in article_one.text_normalized
    assert article_one.emb_e5 is None  # phase-2 backfill fills the vectors


async def test_reingesting_same_document_is_idempotent(db_session):
    # Arrange
    await ingest_documents([make_doc()], db_session)
    first = await fetch_rows(db_session)

    # Act
    stats = await ingest_documents([make_doc()], db_session)

    # Assert
    second = await fetch_rows(db_session)
    assert stats.chunks_written == 3
    assert [row.id for row in second] == [row.id for row in first]
    assert await db_session.scalar(select(func.count()).select_from(Chunk)) == 3


async def test_edited_source_text_updates_the_row_instead_of_duplicating(db_session):
    # Arrange
    await ingest_documents([make_doc()], db_session)
    edited = TWO_ARTICLE_TEXT.replace("جميع العُمّال", "جميع العاملين")

    # Act
    await ingest_documents([make_doc(edited)], db_session)

    # Assert
    assert await db_session.scalar(select(func.count()).select_from(Chunk)) == 3
    article_two = await db_session.get(Chunk, f"{DOC_ID}:2:0")
    await db_session.refresh(article_two)
    assert "العاملين" in article_two.text
    assert "العُمّال" not in article_two.text
    assert article_two.text_normalized == normalize_for_index(article_two.text)


async def test_repeated_article_number_gets_a_distinct_id(db_session):
    # Arrange - "مادة 1 - إصدار" then the law's own "المادة 1", as in Law 14/2004
    text = (
        "المادة 1 - إصدار\nيُعمل بأحكام القانون المرفق.\n"
        "المادة 1\nفي تطبيق أحكام هذا القانون تكون للكلمات المعاني المبينة قرينها.\n"
    )

    # Act
    stats = await ingest_documents([make_doc(text)], db_session)

    # Assert
    rows = await fetch_rows(db_session)
    assert stats.chunks_written == 2
    assert [row.id for row in rows] == [f"{DOC_ID}:1:0", f"{DOC_ID}:1:1"]
    assert [row.seq for row in rows] == [0, 1]
    assert all(row.article == "1" for row in rows)


async def test_embedder_populates_its_own_vector_column(db_session):
    # Arrange
    embedder = FakeEmbedder(model_key="e5")

    # Act
    stats = await ingest_documents([make_doc()], db_session, embedder=embedder)

    # Assert
    rows = await fetch_rows(db_session)
    assert stats.chunks_written == 3
    assert embedder.batch_sizes == [3]  # one batch, under the 64 cap
    assert all(len(row.emb_e5) == 1024 for row in rows)
    assert all(row.emb_bge is None and row.emb_openai is None for row in rows)


async def test_embedder_returning_wrong_dimension_raises(db_session):
    # Arrange
    embedder = FakeEmbedder(model_key="e5", dim=8)

    # Act / Assert
    with pytest.raises(ValueError, match="dim 8"):
        await ingest_documents([make_doc()], db_session, embedder=embedder)
    assert await db_session.scalar(select(func.count()).select_from(Chunk)) == 0


async def test_unknown_embedder_model_key_raises(db_session):
    # Arrange
    embedder = FakeEmbedder(model_key="nope")

    # Act / Assert
    with pytest.raises(ValueError, match="unknown embedder model_key"):
        await ingest_documents([make_doc()], db_session, embedder=embedder)


async def test_empty_document_list_returns_zeroed_stats(db_session):
    # Act
    stats = await ingest_documents([], db_session)

    # Assert
    assert (stats.documents, stats.chunks_written, stats.chunks_skipped) == (0, 0, 0)
    assert await db_session.scalar(select(func.count()).select_from(Chunk)) == 0
