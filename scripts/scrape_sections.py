"""One-time scrape: record each law's chapter anchors in the corpus manifest.

Al Meezan's ``LawView.aspx`` page carries a table of contents whose entries are
``<a href='#Section_NNNNN'>الفصل الرابع (38-57)</a>`` — an anchor plus the range
of articles that chapter covers. That range is what turns a citation's article
number into a deep link, so a reader lands on the chapter instead of the top of a
145-article page.

**Chapter granularity is the ceiling, not a shortcut.** The page has no
per-article anchor: 34 ``Section_`` ids for law 3961's 145 articles, every one a
فصل. Landing on the right chapter is the best the portal allows.

Run after ``ingestion fetch``, or whenever the portal's pagination changes:

    python -m scripts.scrape_sections --dry-run   # print what it found
    python -m scripts.scrape_sections             # rewrite manifest.json

Committed rather than resolved at request time: it is one HTTP call per law
against a third-party portal, the answer changes only when Al Meezan restructures
a law, and /ask must not depend on almeezan.qa being reachable.

# ponytail: regex over the ToC, no HTML parser. The anchors are machine-generated
# and uniform; if the portal ever ships hand-edited markup, that is the day to add
# a real parser, not before.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path

import httpx

# The portal serves an incomplete certificate chain; ingestion already carries the
# missing intermediate and the polite User-Agent. Same host, same fix — importing
# it beats discovering the same SSL error twice.
from ingestion.fetch import _ssl_context

MANIFEST = Path("data/corpus/manifest.json")
HEADERS = {"User-Agent": "arabic-rag-corpus-fetcher/0.1 (+public legal texts)"}

# <a href='#Section_12641'>الفصل الرابع (38-57)</a> — the label may contain
# nested tags, so strip them rather than trying to match around them.
_TOC_ENTRY = re.compile(r"href='#(Section_\d+)'>(.*?)</a>", re.DOTALL)
_RANGE = re.compile(r"\((\d+)\s*-\s*(\d+)\)")
_TAG = re.compile(r"<[^>]+>")

# The issuance preamble is numbered 1-4 in its own sequence and would otherwise
# shadow the law's own articles 1-4, sending المادة 1 to the wrong text.
_PREAMBLE = "مواد الإصدار"

TIMEOUT_S = 60


def parse_sections(page: str) -> list[dict]:
    """Chapter anchors with their article ranges, in page order, deduplicated.

    Each chapter appears twice in the ToC — once as "الفصل الرابع (38-57)" and
    once as its title, "علاقة العمل الفردية (38-57)". Both carry the same range
    and both scroll to the same place; the first occurrence wins.
    """
    sections: dict[str, dict] = {}
    for anchor, raw_label in _TOC_ENTRY.findall(page):
        label = _TAG.sub("", html.unescape(raw_label)).strip()
        label = re.sub(r"\s+", " ", label)
        span = _RANGE.search(label)
        if not span or anchor in sections:
            continue
        sections[anchor] = {
            "anchor": anchor,
            "label": label,
            "first_article": int(span.group(1)),
            "last_article": int(span.group(2)),
            "preamble": label.startswith(_PREAMBLE),
        }
    return list(sections.values())


def scrape(url: str) -> list[dict]:
    response = httpx.get(
        url,
        timeout=TIMEOUT_S,
        follow_redirects=True,
        headers=HEADERS,
        verify=_ssl_context(),
    )
    response.raise_for_status()
    return parse_sections(response.text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m scripts.scrape_sections")
    parser.add_argument("--dry-run", action="store_true", help="print, do not write")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args(argv)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    for document in manifest["documents"]:
        url = document.get("source_url")
        if not url:
            print(f"{document['doc_id']}: no source_url, skipped", file=sys.stderr)
            continue
        try:
            sections = scrape(url)
        except httpx.HTTPError as exc:
            print(f"{document['doc_id']}: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        print(f"\n{document['doc_id']}: {len(sections)} sections")
        for section in sections:
            flag = " (preamble)" if section["preamble"] else ""
            print(
                f"  {section['anchor']}  "
                f"{section['first_article']:>3}-{section['last_article']:<3}  "
                f"{section['label']}{flag}"
            )
        document["sections"] = sections

    if args.dry_run:
        print("\ndry run, manifest not written", file=sys.stderr)
        return 0

    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nwrote {args.manifest}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
