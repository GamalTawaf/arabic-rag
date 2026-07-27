"""Eval dataset schema: labelled Arabic Q/A pairs stored as JSONL.

One JSON object per line, e.g.

    {"id": "q001", "question": "...", "dialect_tag": "msa", "answer": "...",
     "source_doc": "labour-law-2004", "source_chunk_ids": ["labour-law-2004:12:0"]}

An empty ``source_chunk_ids`` marks an *unanswerable* pair: there is no correct
chunk to retrieve, so retrieval metrics are undefined and the pair is scored on
refusal behaviour instead (see ``evals.metrics``).

Every non-empty ``source_chunk_ids`` entry must start with ``<source_doc>:`` —
that is the id shape ``ingestion.chunk`` mints, and it is the only automatic link
between the corpus and hand-written ground truth.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DIALECT_TAGS = ("msa", "gulf")

_REQUIRED_FIELDS = (
    "id",
    "question",
    "dialect_tag",
    "answer",
    "source_doc",
    "source_chunk_ids",
)


@dataclass(frozen=True)
class EvalPair:
    id: str
    question: str
    dialect_tag: str  # "msa" | "gulf"
    answer: str
    source_doc: str
    source_chunk_ids: list[str]  # empty list == unanswerable

    @property
    def answerable(self) -> bool:
        return bool(self.source_chunk_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "dialect_tag": self.dialect_tag,
            "answer": self.answer,
            "source_doc": self.source_doc,
            "source_chunk_ids": list(self.source_chunk_ids),
        }


def validate_pair(raw: dict) -> EvalPair:
    """Validate one raw JSON object and return an EvalPair.

    Trust boundary: the dataset is hand-edited, so every field is checked and a
    ValueError naming the offending id/field is raised on anything unexpected.
    """
    # Every failure here is bad *data*, not a bad call signature, so callers only
    # ever have to catch ValueError (noqa: TRY004 for the isinstance checks).
    if not isinstance(raw, dict):
        raise ValueError(f"eval pair must be a JSON object, got {type(raw).__name__}")  # noqa: TRY004

    pair_id = raw.get("id")
    if not isinstance(pair_id, str) or not pair_id.strip():
        raise ValueError(f"field 'id' must be a non-empty string, got {pair_id!r}")

    missing = [field for field in _REQUIRED_FIELDS if field not in raw]
    if missing:
        raise ValueError(f"pair {pair_id!r}: missing field(s) {', '.join(sorted(missing))}")

    # trade-off: extra keys are ignored rather than rejected so annotators can keep
    # notes in the file. Upgrade path: switch to a pydantic model with extra="forbid".
    for field in ("question", "answer", "source_doc"):
        value = raw[field]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"pair {pair_id!r}: field {field!r} must be a non-blank string, got {value!r}")

    dialect_tag = raw["dialect_tag"]
    if dialect_tag not in DIALECT_TAGS:
        raise ValueError(
            f"pair {pair_id!r}: field 'dialect_tag' must be one of {DIALECT_TAGS}, got {dialect_tag!r}"
        )

    chunk_ids = raw["source_chunk_ids"]
    if not isinstance(chunk_ids, list):
        raise ValueError(  # noqa: TRY004
            f"pair {pair_id!r}: field 'source_chunk_ids' must be a list, got {type(chunk_ids).__name__}"
        )
    for chunk_id in chunk_ids:
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError(
                f"pair {pair_id!r}: field 'source_chunk_ids' must contain non-blank strings, got {chunk_id!r}"
            )

    # Seam guard: ingestion mints ids as "<doc_id>:<article>:<seq>", so a ground-truth
    # id that is not prefixed by its source_doc can never be retrieved — it would just
    # score 0 recall forever instead of failing loudly here.
    source_doc = raw["source_doc"]
    for chunk_id in chunk_ids:
        if not chunk_id.startswith(f"{source_doc}:"):
            raise ValueError(
                f"pair {pair_id!r}: source_chunk_id {chunk_id!r} does not belong to "
                f"source_doc {source_doc!r} (expected the id to start with {source_doc + ':'!r})"
            )

    return EvalPair(
        id=pair_id,
        question=raw["question"],
        dialect_tag=dialect_tag,
        answer=raw["answer"],
        source_doc=raw["source_doc"],
        source_chunk_ids=list(chunk_ids),
    )


def load_pairs(path: Path) -> list[EvalPair]:
    """Read a JSONL eval file. Raises ValueError on any invalid or duplicate pair."""
    pairs: list[EvalPair] = []
    seen: set[str] = set()

    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON ({exc.msg})") from exc
            try:
                pair = validate_pair(raw)
            except ValueError as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
            if pair.id in seen:
                raise ValueError(f"{path}:{line_no}: duplicate pair id {pair.id!r}")
            seen.add(pair.id)
            pairs.append(pair)

    return pairs


def save_pairs(pairs: Iterable[EvalPair], path: Path) -> None:
    """Write pairs as JSONL (UTF-8, Arabic kept readable). Rejects duplicate ids."""
    seen: set[str] = set()
    lines: list[str] = []
    for pair in pairs:
        if pair.id in seen:
            raise ValueError(f"duplicate pair id {pair.id!r}")
        seen.add(pair.id)
        lines.append(json.dumps(pair.to_dict(), ensure_ascii=False, sort_keys=True))

    Path(path).write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
