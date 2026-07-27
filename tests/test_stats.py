"""GET /stats — real numbers out of the real database.

No fixtures fake the counts: rows are inserted and the endpoint has to report
them. That is the only version of this test worth having, because /stats exists
precisely to answer "is this instance actually loaded".
"""

from __future__ import annotations

from app.config import settings
from app.deps import SERVICE_MODEL_KEY
from app.observability.cost import SpendTracker, get_spend_tracker
from app.retrieval.cache import lookup as cache_lookup
from app.retrieval.cache import store as cache_store
from tests.test_service import NOTICE_TEXT, make_chunk, unit_vector


async def seed_mixed_corpus(session) -> None:
    """Three chunks in two documents; two of them embedded with bge, one not."""
    embedded = [
        make_chunk("law:49:0", NOTICE_TEXT, article="49", axis=0),
        make_chunk("law:79:0", "نص المادة التاسعة والسبعين", article="79", axis=1),
    ]
    bare = make_chunk("decree:3:0", "نص القرار الوزاري", article="3")
    bare.emb_bge = None
    session.add_all([*embedded, bare])
    await session.commit()


async def test_stats_reports_the_real_corpus_and_embedding_coverage(client, db_session):
    # Arrange
    await seed_mixed_corpus(db_session)

    # Act
    body = (await client.get("/stats")).json()

    # Assert
    assert body["corpus"]["chunks"] == 3
    assert body["corpus"]["documents"] == 2
    assert body["corpus"]["embeddings"]["bge"] == {"chunks": 2, "coverage": 0.6667}
    # A model nothing was backfilled with reads zero, not missing
    assert body["corpus"]["embeddings"]["openai"] == {"chunks": 0, "coverage": 0.0}


async def test_stats_reports_cache_size_and_hit_ratio(client, db_session):
    # Arrange — one stored answer, served once
    pipeline = "hybrid+rerank|r20|c5|rr1|fake:fake-1"
    await cache_store(
        db_session, "سؤال", unit_vector(0), "bge", pipeline, "جواب", ["law:49:0"]
    )
    assert (
        await cache_lookup(db_session, "سؤال", unit_vector(0), "bge", pipeline)
        is not None
    )

    # Act
    body = (await client.get("/stats")).json()

    # Assert — one entry (one generation) and one hit -> 1 of 2 lookups served
    assert body["cache"]["entries"] == 1
    assert body["cache"]["hits"] == 1
    assert body["cache"]["hit_ratio"] == 0.5
    assert body["cache"]["threshold"] == settings.semantic_cache_threshold


async def test_stats_is_empty_but_valid_on_a_cold_instance(client, db_session):
    # Act — nothing seeded at all
    body = (await client.get("/stats")).json()

    # Assert
    assert body["corpus"]["chunks"] == 0
    assert body["corpus"]["embeddings"]["bge"]["coverage"] == 0.0
    assert body["cache"] == {
        "entries": 0,
        "hits": 0,
        "hit_ratio": 0.0,
        "enabled": settings.semantic_cache_enabled,
        "threshold": settings.semantic_cache_threshold,
    }


async def test_stats_reports_todays_spend_against_the_cap(client, db_session):
    # Arrange
    from app.main import app

    tracker = SpendTracker(5.0)
    tracker.record(1.25)
    app.dependency_overrides[get_spend_tracker] = lambda: tracker

    # Act
    body = (await client.get("/stats")).json()
    app.dependency_overrides.pop(get_spend_tracker, None)

    # Assert
    assert body["spend"]["usd"] == 1.25
    assert body["spend"]["calls"] == 1
    assert body["spend"]["cap_usd"] == 5.0
    assert body["spend"]["remaining_usd"] == 3.75


async def test_stats_separates_configured_providers_from_usable_ones(client, db_session):
    # Act
    body = (await client.get("/stats")).json()

    # Assert — no API keys in this environment, so every configured provider is
    # missing its key. That difference is the reason both lists are printed.
    assert body["providers"]["configured"] == ["anthropic", "gemini"]
    assert body["providers"]["available"] == []
    assert body["providers"]["missing_keys"] == ["anthropic", "gemini"]


async def test_stats_names_the_retrieval_stack_it_would_use(client, db_session):
    # Act
    body = (await client.get("/stats")).json()

    # Assert
    assert body["retrieval"]["model_key"] == SERVICE_MODEL_KEY
    assert body["retrieval"]["rerank_min_score"] == settings.rerank_min_score
    assert body["retrieval"]["top_k_context"] == settings.top_k_context
