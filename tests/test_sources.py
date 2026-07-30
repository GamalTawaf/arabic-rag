"""Source links: doc_id -> the page on Al Meezan the text was taken from.

The corpus manifest already records a `source_url` per law (it is what
`ingestion fetch` re-downloads from). This module is the read path that puts it
on a citation, so a reader can check the article against the portal instead of
trusting the excerpt.
"""

from __future__ import annotations

import json

import pytest

from app.lib import sources


@pytest.fixture(autouse=True)
def _clear_cache():
    """Both lookups are cached per process; each test gets cold ones."""
    sources.source_url.cache_clear()
    sources._manifest_urls.cache_clear()
    yield
    sources.source_url.cache_clear()
    sources._manifest_urls.cache_clear()


def test_returns_the_manifest_url_for_a_corpus_document():
    # Arrange / Act
    url = sources.source_url("qatar-labour-law-14-2004")

    # Assert
    assert url is not None
    assert url.startswith("https://www.almeezan.qa/")
    assert "LawID=3961" in url  # the labour law, not another one


def test_every_committed_document_has_a_link():
    # Arrange: a citation with no source is a citation a reader cannot verify
    manifest = json.loads(
        (sources.CORPUS_DIR / "manifest.json").read_text(encoding="utf-8")
    )

    # Act / Assert
    for document in manifest["documents"]:
        assert sources.source_url(document["doc_id"]), document["doc_id"]


def test_an_unknown_document_has_no_link():
    # Arrange / Act: anything ingested over POST /ingest is not in the manifest
    assert sources.source_url("something-a-user-posted") is None


def test_a_missing_manifest_is_not_an_error():
    # Arrange: the manifest ships in the image, but a link is decoration — losing
    # it must degrade the citation, not fail the request
    original = sources.CORPUS_DIR
    sources.CORPUS_DIR = original / "does-not-exist"
    sources.source_url.cache_clear()
    sources._manifest_urls.cache_clear()

    # Act / Assert
    try:
        assert sources.source_url("qatar-labour-law-14-2004") is None
    finally:
        sources.CORPUS_DIR = original
        sources._manifest_urls.cache_clear()


def test_the_manifest_is_read_once_per_process():
    # Act
    sources.source_url("qatar-labour-law-14-2004")
    sources.source_url("qatar-labour-law-14-2004")

    # Assert: /ask builds up to five citations per request; five file reads per
    # request for a value that changes when the image changes is waste
    assert sources._manifest_urls.cache_info().misses == 1
