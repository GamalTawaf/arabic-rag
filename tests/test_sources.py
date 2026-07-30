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
    sources._manifest_sections.cache_clear()
    yield
    sources.source_url.cache_clear()
    sources._manifest_urls.cache_clear()
    sources._manifest_sections.cache_clear()


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
    sources._manifest_sections.cache_clear()

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


# ---------------------------------------------------------- chapter anchors


def test_an_article_links_to_the_chapter_that_contains_it():
    # Arrange / Act: المادة 103 sits in الفصل العاشر, "السلامة والصحة المهنية (99-107)"
    url = sources.source_url("qatar-labour-law-14-2004", "103")

    # Assert
    assert url is not None
    assert url.endswith("#Section_12653")


def test_the_same_chapter_serves_every_article_in_its_range():
    # Arrange / Act
    boundaries = [sources.source_url("qatar-labour-law-14-2004", n) for n in (99, 103, 107)]

    # Assert: the range is inclusive at both ends
    assert boundaries == [boundaries[0]] * 3
    assert sources.source_url("qatar-labour-law-14-2004", 98) != boundaries[0]
    assert sources.source_url("qatar-labour-law-14-2004", 108) != boundaries[0]


def test_an_article_in_two_overlapping_chapters_takes_the_one_it_is_printed_in():
    # Arrange: 115 matches both "الفصل الحادي عشر (108-115)" and the inserted
    # "الفصل الحادي عشر مكرراً (115-115)", which is a different article
    url = sources.source_url("qatar-labour-law-14-2004", 115)

    # Assert: page order wins, so plain المادة 115 lands in its own chapter
    assert url is not None and url.endswith("#Section_12655")


def test_the_issuance_preamble_never_shadows_the_laws_own_articles():
    # Arrange / Act: "مواد الإصدار (1-4)" precedes "الفصل الأول (1-10)" in the ToC
    url = sources.source_url("qatar-labour-law-14-2004", 1)

    # Assert: المادة 1 is the law's article 1, not the first issuance article
    assert url is not None and url.endswith("#Section_12635")


def test_a_law_whose_page_has_no_contents_falls_back_to_the_plain_url():
    # Arrange: the domestic-workers law ships no table of contents to scrape
    plain = sources.source_url("qatar-domestic-workers-law-15-2017")
    anchored = sources.source_url("qatar-domestic-workers-law-15-2017", "5")

    # Assert
    assert anchored == plain
    assert "#" not in anchored


def test_an_article_outside_every_chapter_falls_back_to_the_plain_url():
    # Arrange / Act: law 3961's chapters stop at 148
    assert sources.source_url("qatar-labour-law-14-2004", 900) == sources.source_url(
        "qatar-labour-law-14-2004"
    )


def test_arabic_indic_digits_resolve_to_the_same_chapter():
    # Arrange / Act: int() reads Unicode decimal digits, so ١٠٣ is article 103
    assert sources.source_url("qatar-labour-law-14-2004", "١٠٣") == sources.source_url(
        "qatar-labour-law-14-2004", "103"
    )


@pytest.mark.parametrize("article", [None, "", "  ", "مكرر", "115 مكرر"])
def test_an_unparsable_article_degrades_to_the_law_page(article):
    # Arrange / Act: never guess an anchor from something that is not a number
    url = sources.source_url("qatar-labour-law-14-2004", article)

    # Assert
    assert url == sources.source_url("qatar-labour-law-14-2004")


def test_an_unknown_document_has_no_link_even_with_an_article():
    assert sources.source_url("something-a-user-posted", "3") is None
