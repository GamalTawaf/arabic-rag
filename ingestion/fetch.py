"""Corpus acquisition: download public Qatari legal texts, or load the committed copy.

Two halves that mirror each other:

* ``fetch_corpus`` re-downloads every manifest entry that has a ``source_url``
  and rewrites ``<doc_id>.txt``. Needs network.
* ``load_corpus`` reads the committed ``.txt`` files. Offline, and the only path
  the rest of the pipeline (chunking, embedding, evals) depends on.

The CLI lives in ``ingestion/__main__.py`` (``python -m ingestion fetch``); this
module stays import-only so there is exactly one entrypoint to keep working.
"""

from __future__ import annotations

import json
import ssl
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

import certifi
import httpx

DEFAULT_CORPUS_DIR = Path("data/corpus")
MANIFEST_NAME = "manifest.json"

# trade-off: almeezan.qa serves its leaf certificate without the DigiCert intermediate,
# so Python (unlike curl/browsers, which chase the AIA extension) cannot build a chain.
# Pinning the public intermediate below keeps full verification on. Valid to 2031-03-29;
# when it expires, replace it from the CA Issuers URI in the server certificate.
_MISSING_INTERMEDIATE_CA = """-----BEGIN CERTIFICATE-----
MIIEyDCCA7CgAwIBAgIQDPW9BitWAvR6uFAsI8zwZjANBgkqhkiG9w0BAQsFADBh
MQswCQYDVQQGEwJVUzEVMBMGA1UEChMMRGlnaUNlcnQgSW5jMRkwFwYDVQQLExB3
d3cuZGlnaWNlcnQuY29tMSAwHgYDVQQDExdEaWdpQ2VydCBHbG9iYWwgUm9vdCBH
MjAeFw0yMTAzMzAwMDAwMDBaFw0zMTAzMjkyMzU5NTlaMFkxCzAJBgNVBAYTAlVT
MRUwEwYDVQQKEwxEaWdpQ2VydCBJbmMxMzAxBgNVBAMTKkRpZ2lDZXJ0IEdsb2Jh
bCBHMiBUTFMgUlNBIFNIQTI1NiAyMDIwIENBMTCCASIwDQYJKoZIhvcNAQEBBQAD
ggEPADCCAQoCggEBAMz3EGJPprtjb+2QUlbFbSd7ehJWivH0+dbn4Y+9lavyYEEV
cNsSAPonCrVXOFt9slGTcZUOakGUWzUb+nv6u8W+JDD+Vu/E832X4xT1FE3LpxDy
FuqrIvAxIhFhaZAmunjZlx/jfWardUSVc8is/+9dCopZQ+GssjoP80j812s3wWPc
3kbW20X+fSP9kOhRBx5Ro1/tSUZUfyyIxfQTnJcVPAPooTncaQwywa8WV0yUR0J8
osicfebUTVSvQpmowQTCd5zWSOTOEeAqgJnwQ3DPP3Zr0UxJqyRewg2C/Uaoq2yT
zGJSQnWS+Jr6Xl6ysGHlHx+5fwmY6D36g39HaaECAwEAAaOCAYIwggF+MBIGA1Ud
EwEB/wQIMAYBAf8CAQAwHQYDVR0OBBYEFHSFgMBmx9833s+9KTeqAx2+7c0XMB8G
A1UdIwQYMBaAFE4iVCAYlebjbuYP+vq5Eu0GF485MA4GA1UdDwEB/wQEAwIBhjAd
BgNVHSUEFjAUBggrBgEFBQcDAQYIKwYBBQUHAwIwdgYIKwYBBQUHAQEEajBoMCQG
CCsGAQUFBzABhhhodHRwOi8vb2NzcC5kaWdpY2VydC5jb20wQAYIKwYBBQUHMAKG
NGh0dHA6Ly9jYWNlcnRzLmRpZ2ljZXJ0LmNvbS9EaWdpQ2VydEdsb2JhbFJvb3RH
Mi5jcnQwQgYDVR0fBDswOTA3oDWgM4YxaHR0cDovL2NybDMuZGlnaWNlcnQuY29t
L0RpZ2lDZXJ0R2xvYmFsUm9vdEcyLmNybDA9BgNVHSAENjA0MAsGCWCGSAGG/WwC
ATAHBgVngQwBATAIBgZngQwBAgEwCAYGZ4EMAQICMAgGBmeBDAECAzANBgkqhkiG
9w0BAQsFAAOCAQEAkPFwyyiXaZd8dP3A+iZ7U6utzWX9upwGnIrXWkOH7U1MVl+t
wcW1BSAuWdH/SvWgKtiwla3JLko716f2b4gp/DA/JIS7w7d7kwcsr4drdjPtAFVS
slme5LnQ89/nD/7d+MS5EHKBCQRfz5eeLjJ1js+aWNJXMX43AYGyZm0pGrFmCW3R
bpD0ufovARTFXFZkAdl9h6g4U5+LXUZtXMYnhIHUfoyMo5tS58aI7Dd8KvvwVVo4
chDYABPPTHPbqjc1qCmBaZx2vN4Ye5DUys/vZwP9BFohFrH/6j/f3IL16/RZkiMN
JCqVJUzKoZHm1Lesh3Sz8W2jmdv51b2EQJ8HmA==
-----END CERTIFICATE-----"""

# Tags whose boundaries are real line breaks in the rendered page. Everything
# else (span, a, b, ...) is inline and must not split a sentence.
_BLOCK_TAGS = frozenset(
    {
        "br", "hr", "p", "div", "li", "ul", "ol", "tr", "td", "th",
        "table", "section", "article", "h1", "h2", "h3", "h4", "h5", "h6",
    }
)
_SKIP_TAGS = frozenset({"script", "style", "noscript", "head"})

# trade-off: a 30-line HTMLParser instead of beautifulsoup4/trafilatura. Ceiling is
# hand-written legal-portal HTML, which is all we fetch. If we ever ingest arbitrary
# web pages, swap this for trafilatura and delete the marker slicing below.


class _TextExtractor(HTMLParser):
    """Collect visible text, one logical block per line."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        elif tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._parts.append(data)

    @property
    def text(self) -> str:
        return "".join(self._parts)


def strip_html(html: str) -> str:
    """HTML -> plain text: no tags, no blank-line runs, whitespace collapsed per line."""
    extractor = _TextExtractor()
    extractor.feed(html)
    lines = (" ".join(line.split()) for line in extractor.text.split("\n"))
    return "\n".join(line for line in lines if line)


def slice_between(text: str, start: str | None, end: str | None) -> str:
    """Drop site chrome by keeping only what lies between two literal markers.

    Markers are matched on the stripped text and kept in the output (the start
    marker is usually the first line of the preamble). A missing marker is an
    error, not a silent full-page fallback.
    """
    begin = 0
    if start:
        begin = text.find(start)
        if begin < 0:
            raise ValueError(f"start marker not found in fetched text: {start!r}")
    stop = len(text)
    if end:
        stop = text.find(end, begin)
        if stop < 0:
            raise ValueError(f"end marker not found in fetched text: {end!r}")
    return text[begin:stop].strip()


def _ssl_context() -> ssl.SSLContext:
    """Standard verification (certifi roots) plus the intermediate almeezan.qa omits."""
    context = ssl.create_default_context(cafile=certifi.where())
    context.load_verify_locations(cadata=_MISSING_INTERMEDIATE_CA)
    return context


@dataclass(frozen=True)
class CorpusDoc:
    doc_id: str
    title: str
    source_url: str | None
    license: str
    text: str


def load_manifest(src: Path = DEFAULT_CORPUS_DIR) -> list[dict]:
    """Read manifest.json. Raises if it is missing or not a list of documents."""
    path = Path(src) / MANIFEST_NAME
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"corpus manifest not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"corpus manifest is not valid JSON: {path}: {exc}") from exc

    docs = raw.get("documents") if isinstance(raw, dict) else raw
    if not isinstance(docs, list) or not docs:
        raise ValueError(f"corpus manifest has no documents: {path}")
    for entry in docs:
        missing = {"doc_id", "title", "license"} - set(entry)
        if missing:
            raise ValueError(f"manifest entry {entry.get('doc_id', '?')} missing {sorted(missing)}")
    return docs


def fetch_corpus(dest: Path = DEFAULT_CORPUS_DIR, timeout: float = 30.0) -> list[Path]:
    """Download every manifest entry that has a source_url; write <doc_id>.txt.

    Returns the paths written. Any HTTP or extraction failure raises with the
    offending doc_id — a partially refreshed corpus is worse than a loud failure.
    """
    dest = Path(dest)
    written: list[Path] = []
    headers = {"User-Agent": "arabic-rag-corpus-fetcher/0.1 (+public legal texts)"}

    with httpx.Client(
        timeout=timeout, follow_redirects=True, headers=headers, verify=_ssl_context()
    ) as client:
        for entry in load_manifest(dest):
            url = entry.get("source_url")
            if not url:
                continue
            doc_id = entry["doc_id"]
            try:
                response = client.get(url)
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise RuntimeError(
                    f"{doc_id}: {url} returned HTTP {exc.response.status_code}"
                ) from exc
            except httpx.HTTPError as exc:
                raise RuntimeError(f"{doc_id}: could not fetch {url}: {exc}") from exc

            try:
                text = slice_between(
                    strip_html(response.text),
                    entry.get("start_marker"),
                    entry.get("end_marker"),
                )
            except ValueError as exc:
                raise RuntimeError(f"{doc_id}: page layout changed at {url}: {exc}") from exc
            if not text:
                raise RuntimeError(f"{doc_id}: extracted no text from {url}")

            path = dest / f"{doc_id}.txt"
            path.write_text(text + "\n", encoding="utf-8")
            written.append(path)
    return written


def load_corpus(src: Path = DEFAULT_CORPUS_DIR) -> list[CorpusDoc]:
    """Read the committed corpus. Offline, no network, no optional deps."""
    src = Path(src)
    docs: list[CorpusDoc] = []
    for entry in load_manifest(src):
        path = src / f"{entry['doc_id']}.txt"
        try:
            text = path.read_text(encoding="utf-8").strip()
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"{entry['doc_id']}: manifest lists a document with no text file at {path}"
            ) from exc
        if not text:
            raise ValueError(f"{entry['doc_id']}: text file is empty: {path}")
        docs.append(
            CorpusDoc(
                doc_id=entry["doc_id"],
                title=entry["title"],
                source_url=entry.get("source_url"),
                license=entry["license"],
                text=text,
            )
        )
    return docs
