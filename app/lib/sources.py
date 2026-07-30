"""Where a citation's text came from: doc_id -> the law's page on Al Meezan.

A citation carries an excerpt and a score, which asks the reader to trust that the
excerpt is really what the article says. The manifest under ``data/corpus/``
already records the portal URL each law was fetched from — the same field
``ingestion fetch`` re-downloads from — so putting it on the citation turns
"trust the excerpt" into "check it yourself".

Read from the manifest rather than stored on the chunk: it is one value per
document, it changes only when the corpus is re-fetched, and a column would mean a
migration plus 233 copies of the same string.

**Law-level, not article-level.** Al Meezan's ``LawView.aspx?LawID=…`` opens the
whole law; per-article deep links exist in the portal but their ids are not in the
manifest and are not derivable from an article number. So the link opens the law
and the reader finds the article — honest, and better than no link. Scraping the
per-article ids is the upgrade path if that ever matters.
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


@lru_cache(maxsize=256)
def source_url(doc_id: str) -> str | None:
    """The page this document's text was taken from, or None if there isn't one.

    None for anything that arrived over ``POST /ingest`` or Pub/Sub: a caller's
    document has no entry in the committed manifest, and inventing a link for it
    would be worse than showing none.
    """
    return _manifest_urls().get(doc_id)
