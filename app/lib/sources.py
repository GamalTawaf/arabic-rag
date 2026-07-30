"""Where a citation's text came from: doc_id -> the law's page on Al Meezan.

A citation carries an excerpt and a score, which asks the reader to trust that the
excerpt is really what the article says. The manifest under ``data/corpus/``
already records the portal URL each law was fetched from — the same field
``ingestion fetch`` re-downloads from — so putting it on the citation turns
"trust the excerpt" into "check it yourself".

Read from the manifest rather than stored on the chunk: it is one value per
document, it changes only when the corpus is re-fetched, and a column would mean a
migration plus 233 copies of the same string.

**Chapter-level, not article-level.** Al Meezan's ``LawView.aspx?LawID=…`` opens
the whole law, and the page's only anchors are chapters: 34 ``Section_`` ids for
law 3961's 145 articles, every one a فصل. So a citation to المادة 103 links to
``#Section_12653`` — "السلامة والصحة المهنية (99-107)", nine articles rather than
145. There is no finer anchor to link to; this is the portal's ceiling, not a
shortcut. ``scripts/scrape_sections.py`` records the ranges in the manifest.

Two laws get no anchor and fall back to the plain URL: anything ingested over
``POST /ingest`` (not in the manifest at all), and the domestic-workers law, whose
page ships no table of contents.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from ingestion.fetch import DEFAULT_CORPUS_DIR, load_manifest

# Module-level so a test can point it somewhere else. Not a setting: the corpus
# directory is part of the image, not part of the deployment.
CORPUS_DIR: Path = DEFAULT_CORPUS_DIR


@lru_cache(maxsize=1)
def _manifest_urls() -> dict[str, str]:
    """doc_id -> source_url for every manifest entry that has one.

    Never raises. A missing or malformed manifest means no links, and no links is
    a worse citation, not a failed request — /ask must not 500 because a decoration
    is unavailable.
    """
    try:
        documents = load_manifest(CORPUS_DIR)
    except (FileNotFoundError, ValueError):
        return {}
    return {
        str(document["doc_id"]): str(document["source_url"])
        for document in documents
        if document.get("doc_id") and document.get("source_url")
    }


@lru_cache(maxsize=1)
def _manifest_sections() -> dict[str, tuple[tuple[int, int, str], ...]]:
    """doc_id -> (first_article, last_article, anchor) in page order.

    Preamble chapters ("مواد الإصدار") are dropped here rather than skipped at
    lookup time: their articles 1-4 are a separate numbering that would otherwise
    shadow the law's own articles 1-4 and send المادة 1 to the wrong text.

    Never raises, for the same reason as :func:`_manifest_urls` — a missing
    anchor costs a less precise link, not a failed request.
    """
    try:
        documents = load_manifest(CORPUS_DIR)
    except (FileNotFoundError, ValueError):
        return {}
    sections: dict[str, tuple[tuple[int, int, str], ...]] = {}
    for document in documents:
        doc_id = str(document.get("doc_id") or "")
        if not doc_id:
            continue
        try:
            ranges = tuple(
                (int(s["first_article"]), int(s["last_article"]), str(s["anchor"]))
                for s in document.get("sections") or ()
                if not s.get("preamble")
            )
        except (KeyError, TypeError, ValueError):
            continue  # a malformed entry costs that law its anchors, nothing else
        if ranges:
            sections[doc_id] = ranges
    return sections


@lru_cache(maxsize=1024)
def source_url(doc_id: str, article: str | int | None = None) -> str | None:
    """The page this document's text was taken from, or None if there isn't one.

    With ``article``, appends the anchor of the chapter containing it, so the
    reader lands on the chapter instead of the top of the law.

    None for anything that arrived over ``POST /ingest`` or Pub/Sub: a caller's
    document has no entry in the committed manifest, and inventing a link for it
    would be worse than showing none.
    """
    url = _manifest_urls().get(doc_id)
    if url is None or article is None:
        return url
    anchor = _anchor_for(doc_id, article)
    return f"{url}#{anchor}" if anchor else url


def _anchor_for(doc_id: str, article: str | int) -> str | None:
    """The anchor of the first chapter whose range contains ``article``.

    **First in page order, not narrowest.** Law 3961 has both "الفصل الحادي عشر
    (108-115)" and the later-inserted "الفصل الحادي عشر مكرراً (115-115)", so
    article 115 matches two chapters. Narrowest-wins would send plain المادة 115
    to the مكرر chapter, which is a different article; page order sends it to the
    chapter it is actually printed in.

    # ponytail: plain int(), which already reads Unicode decimal digits — ١٠٣ and
    # 103 resolve alike, so no digit-folding pass is needed. The committed corpus
    # stores ASCII (verified: all 228 chunks that carry an article number).
    # Anything int() rejects — "", "115 مكرر" — returns None and the link degrades
    # to the law page rather than guessing at a chapter.
    """
    try:
        number = int(str(article).strip())
    except (TypeError, ValueError):
        return None
    for first, last, anchor in _manifest_sections().get(doc_id, ()):
        if first <= number <= last:
            return anchor
    return None
