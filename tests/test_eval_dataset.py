"""Integrity guards for the merged eval dataset (``evals/data/eval_pairs.jsonl``).

These tests exist because a bad label is silent: a ``source_chunk_id`` that does
not exist, or a pair that quietly vanishes in a merge, scores 0.0 recall forever
and drags every benchmark number and CI gate down with it. Cheap assertions here
are the only thing standing between a typo and a poisoned conclusion.

**Why no ``db_session`` fixture here.** ``tests/conftest.py`` TRUNCATEs every
table in ``Base.metadata`` (including ``chunks``) at the start of each
``db_session`` test, to give DB tests a clean slate. Asking that fixture whether
our 306 cited chunk ids exist would therefore query an *empty* table and fail for
a reason that has nothing to do with the dataset. So these tests assert the chunk
id FORMAT only -- the shape ``ingestion.chunk`` mints -- and existence-checking is
left to the eval harness, which runs against the real ingested corpus. A format
guard still catches the realistic failure (a hand-edited or hallucinated id),
while staying runnable with no database at all.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pytest

from evals.schema import DIALECT_TAGS, load_pairs

DATASET_PATH = Path(__file__).parent.parent / "evals" / "data" / "eval_pairs.jsonl"

# Floors, not targets. The dataset is ~283 pairs; these trip only on real erosion
# (a merge that drops a file, a subset silently emptied), not on normal editing.
MIN_PAIRS = 200
MIN_GULF = 40
MIN_UNANSWERABLE = 10

# "<source_doc>:<article>:<seq>" as minted by ingestion.chunk. Doc ids are
# lowercase-hyphenated with digits and contain no colon, article is a number,
# seq is the 0-based index of the chunk within that article.
CHUNK_ID_RE = re.compile(r"^[a-z0-9-]+:\d+:\d+$")


@pytest.fixture(scope="module")
def pairs():
    """Load once per module. ``load_pairs`` validates every line as a side effect."""
    if not DATASET_PATH.exists():
        pytest.fail(f"eval dataset missing at {DATASET_PATH}; run the merge step")
    return load_pairs(DATASET_PATH)


def test_dataset_loads_and_every_pair_validates(pairs):
    # load_pairs runs validate_pair on every line and raises ValueError on the
    # first bad one, so reaching here at all means the whole file is well formed.
    assert pairs, "dataset loaded but is empty"


def test_no_duplicate_ids(pairs):
    duplicates = {pid: n for pid, n in Counter(p.id for p in pairs).items() if n > 1}

    assert duplicates == {}, f"duplicate pair ids: {duplicates}"


def test_dataset_meets_size_floors(pairs):
    gulf = [p for p in pairs if p.dialect_tag == "gulf"]
    unanswerable = [p for p in pairs if not p.source_chunk_ids]

    assert len(pairs) >= MIN_PAIRS, f"only {len(pairs)} pairs, expected >= {MIN_PAIRS}"
    assert len(gulf) >= MIN_GULF, f"only {len(gulf)} gulf pairs, expected >= {MIN_GULF}"
    assert len(unanswerable) >= MIN_UNANSWERABLE, (
        f"only {len(unanswerable)} unanswerable pairs, expected >= {MIN_UNANSWERABLE}"
    )


def test_dialect_tags_are_known(pairs):
    unknown = {p.dialect_tag for p in pairs} - set(DIALECT_TAGS)

    assert unknown == set(), f"unknown dialect_tag(s): {unknown}"


def test_answerable_flag_matches_chunk_ids(pairs):
    # The two states must never blur: an "answerable" pair with no chunks scores
    # 0.0 recall forever, and an "unanswerable" pair with chunks corrupts the
    # refusal metric. `answerable` is derived from source_chunk_ids, so this
    # asserts the dataset only ever expresses one of the two coherent shapes.
    empty_but_answerable = [p.id for p in pairs if p.answerable and not p.source_chunk_ids]
    nonempty_but_unanswerable = [p.id for p in pairs if not p.answerable and p.source_chunk_ids]

    assert empty_but_answerable == []
    assert nonempty_but_unanswerable == []


def test_answerable_pairs_cite_at_least_one_chunk(pairs):
    answerable = [p for p in pairs if p.answerable]
    without_chunks = [p.id for p in answerable if len(p.source_chunk_ids) < 1]

    assert answerable, "dataset has no answerable pairs at all"
    assert without_chunks == [], f"answerable pairs citing nothing: {without_chunks}"


def test_chunk_ids_match_ingestion_id_format(pairs):
    malformed = [
        (p.id, chunk_id)
        for p in pairs
        for chunk_id in p.source_chunk_ids
        if not CHUNK_ID_RE.match(chunk_id)
    ]

    assert malformed == [], f"chunk ids not matching '<doc>:<article>:<seq>': {malformed}"


def test_chunk_ids_are_prefixed_by_their_source_doc(pairs):
    # validate_pair already enforces this; re-asserted here so that if anyone ever
    # relaxes the schema, the dataset guard still fails loudly.
    mismatched = [
        (p.id, chunk_id)
        for p in pairs
        for chunk_id in p.source_chunk_ids
        if not chunk_id.startswith(f"{p.source_doc}:")
    ]

    assert mismatched == [], f"chunk ids not belonging to their source_doc: {mismatched}"


def test_no_duplicate_chunk_ids_within_a_pair(pairs):
    # A repeated id inflates that pair's citation count and would double-count in
    # recall denominators.
    repeated = [
        (p.id, p.source_chunk_ids)
        for p in pairs
        if len(set(p.source_chunk_ids)) != len(p.source_chunk_ids)
    ]

    assert repeated == [], f"pairs citing the same chunk twice: {repeated}"


def test_gulf_pairs_match_the_msa_pair_they_derive_from():
    """A gulf pair is a dialect rephrasing, so its ground truth must be identical.

    Reads the raw JSON rather than EvalPair because `derived_from` is an extra key
    the schema deliberately ignores.
    """
    import json

    raw = [
        json.loads(line)
        for line in DATASET_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_id = {r["id"]: r for r in raw}

    dangling_parents = []
    mismatches = []
    for row in raw:
        parent_id = row.get("derived_from")
        if not parent_id:
            continue
        parent = by_id.get(parent_id)
        if parent is None:
            dangling_parents.append((row["id"], parent_id))
            continue
        if sorted(row["source_chunk_ids"]) != sorted(parent["source_chunk_ids"]):
            mismatches.append((row["id"], parent_id))

    assert dangling_parents == [], f"derived_from pointing at missing pairs: {dangling_parents}"
    assert mismatches == [], f"gulf pairs disagreeing with their MSA parent: {mismatches}"


def test_questions_and_answers_are_not_placeholders(pairs):
    # Guards against a half-finished annotation pass landing in the merged file.
    too_short = [p.id for p in pairs if len(p.question.strip()) < 10 or len(p.answer.strip()) < 10]

    assert too_short == [], f"pairs with suspiciously short question/answer: {too_short}"
