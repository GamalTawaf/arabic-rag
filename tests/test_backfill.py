"""Integration tests for the embedding backfill (needs the pgvector DB).

No real model is ever loaded: `ingestion.backfill.get_embedder` is monkeypatched
to a deterministic fake, which is also how the tests assert that the backfill
uses `embed_passages` and not `embed_queries`.
"""

import pytest
from sqlalchemy import select

from app.models.chunks import Chunk
from ingestion.backfill import backfill_embeddings

DIM = 1024


class FakeEmbedder:
    """Deterministic vectors, zero ML dependencies. Records how it was called."""

    def __init__(self, model_key: str = "e5", dim: int = DIM, fail_after: int | None = None):
        self.model_key = model_key
        self.dim = dim
        self.fail_after = fail_after  # raise once this many texts have been embedded
        self.passage_batches: list[list[str]] = []
        self.query_batches: list[list[str]] = []
        self.embedded = 0

    async def embed_passages(self, texts: list[str]) -> list[list[float]]:
        if self.fail_after is not None and self.embedded >= self.fail_after:
            raise RuntimeError("simulated interrupt")
        self.passage_batches.append(list(texts))
        self.embedded += len(texts)
        return [[float(len(text) % 7 + 1)] * self.dim for text in texts]

    async def embed_queries(self, texts: list[str]) -> list[list[float]]:
        self.query_batches.append(list(texts))
        return await self.embed_passages(texts)


@pytest.fixture()
def fake_embedder(monkeypatch):
    """Install one fake embedder; the test mutates it before calling the backfill."""
    embedder = FakeEmbedder()
    monkeypatch.setattr("ingestion.backfill.get_embedder", lambda key: embedder)
    return embedder


async def seed(session, count: int = 5) -> list[str]:
    """Insert `count` chunks with NULL vector columns. Returns their ids in order."""
    chunks = [
        Chunk(
            id=f"doc-a:{index}:0",
            doc_id="doc-a",
            article=str(index),
            seq=0,
            text=f"المادة {index} نصٌ تجريبي" + "ـ" * index,
            text_normalized=f"المادة {index} نص تجريبي",
        )
        for index in range(count)
    ]
    session.add_all(chunks)
    await session.commit()
    return [chunk.id for chunk in chunks]


async def vectors(session) -> dict[str, list[float] | None]:
    rows = (await session.execute(select(Chunk.id, Chunk.emb_e5))).all()
    return {chunk_id: vector for chunk_id, vector in rows}


async def test_fills_only_null_columns_by_default(db_session, fake_embedder):
    # Arrange - one row already carries a vector from an earlier run
    ids = await seed(db_session, 5)
    sentinel = [0.5] * DIM
    await db_session.execute(
        Chunk.__table__.update().where(Chunk.id == ids[0]).values(emb_e5=sentinel)
    )
    await db_session.commit()

    # Act
    stats = await backfill_embeddings(db_session, "e5", batch_size=2)

    # Assert
    stored = await vectors(db_session)
    assert (stats.chunks_embedded, stats.chunks_skipped) == (4, 1)
    assert pytest.approx(list(stored[ids[0]])) == sentinel  # untouched
    assert all(stored[chunk_id] is not None for chunk_id in ids)
    assert fake_embedder.embedded == 4


async def test_only_missing_false_re_embeds_everything(db_session, fake_embedder):
    # Arrange
    ids = await seed(db_session, 3)
    sentinel = [0.5] * DIM
    await db_session.execute(
        Chunk.__table__.update().where(Chunk.id == ids[0]).values(emb_e5=sentinel)
    )
    await db_session.commit()

    # Act
    stats = await backfill_embeddings(db_session, "e5", batch_size=2, only_missing=False)

    # Assert
    stored = await vectors(db_session)
    assert (stats.chunks_embedded, stats.chunks_skipped) == (3, 0)
    assert list(stored[ids[0]]) != sentinel  # overwritten
    assert fake_embedder.embedded == 3


async def test_embeds_original_text_via_embed_passages(db_session, fake_embedder):
    # Arrange
    await seed(db_session, 3)
    originals = {row.text for row in (await db_session.execute(select(Chunk))).scalars()}

    # Act
    await backfill_embeddings(db_session, "e5", batch_size=10)

    # Assert - passage path only, and the raw text (tatweel included), not the index form
    assert fake_embedder.query_batches == []
    assert set(fake_embedder.passage_batches[0]) == originals


async def test_batches_are_committed_so_an_interrupt_is_resumable(db_session, fake_embedder):
    # Arrange - blow up once two chunks (one batch) are embedded
    await seed(db_session, 5)
    fake_embedder.fail_after = 2

    # Act
    with pytest.raises(RuntimeError, match="simulated interrupt"):
        await backfill_embeddings(db_session, "e5", batch_size=2)

    # Assert - the first batch survived the failure
    partial = await vectors(db_session)
    assert sum(vector is not None for vector in partial.values()) == 2

    # Act again - a rerun only picks up what is still NULL
    fake_embedder.fail_after = None
    stats = await backfill_embeddings(db_session, "e5", batch_size=2)

    # Assert
    stored = await vectors(db_session)
    assert (stats.chunks_embedded, stats.chunks_skipped) == (3, 2)
    assert all(vector is not None for vector in stored.values())


async def test_second_run_embeds_nothing(db_session, fake_embedder):
    # Arrange
    await seed(db_session, 4)
    await backfill_embeddings(db_session, "e5", batch_size=2)

    # Act
    stats = await backfill_embeddings(db_session, "e5", batch_size=2)

    # Assert
    assert (stats.chunks_embedded, stats.chunks_skipped) == (0, 4)
    assert fake_embedder.embedded == 4  # not re-embedded


async def test_stats_report_the_model_and_a_positive_duration(db_session, fake_embedder):
    # Arrange
    await seed(db_session, 3)

    # Act
    stats = await backfill_embeddings(db_session, "e5", batch_size=1)

    # Assert
    assert stats.model_key == "e5"
    assert stats.chunks_embedded + stats.chunks_skipped == 3
    assert stats.seconds > 0
    assert [len(batch) for batch in fake_embedder.passage_batches] == [1, 1, 1]


async def test_unknown_model_key_raises_before_loading_a_model(db_session, monkeypatch):
    # Arrange - any attempt to build an embedder is a failure
    def explode(key):  # pragma: no cover - must not run
        raise AssertionError("model must not be loaded for an unknown key")

    monkeypatch.setattr("ingestion.backfill.get_embedder", explode)
    await seed(db_session, 2)

    # Act / Assert
    with pytest.raises(ValueError, match="unknown embedding model_key 'nope'"):
        await backfill_embeddings(db_session, "nope")


async def test_non_positive_batch_size_raises(db_session, fake_embedder):
    # Act / Assert
    with pytest.raises(ValueError, match="batch_size must be >= 1"):
        await backfill_embeddings(db_session, "e5", batch_size=0)


async def test_wrong_dimension_from_the_embedder_raises(db_session, fake_embedder):
    # Arrange
    await seed(db_session, 2)
    fake_embedder.dim = 8

    # Act / Assert
    with pytest.raises(ValueError, match="returned dim 8, expected 1024"):
        await backfill_embeddings(db_session, "e5", batch_size=2)
    assert all(vector is None for vector in (await vectors(db_session)).values())


async def test_empty_table_returns_zeroed_stats(db_session, fake_embedder):
    # Act
    stats = await backfill_embeddings(db_session, "e5")

    # Assert
    assert (stats.chunks_embedded, stats.chunks_skipped) == (0, 0)
    assert fake_embedder.passage_batches == []
