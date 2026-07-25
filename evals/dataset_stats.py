"""Print the composition of the merged eval dataset.

    python -m evals.dataset_stats [path/to/eval_pairs.jsonl]

Stdlib + ``evals.schema`` only, on purpose: this runs in CI next to the retrieval
gate, and a stats script that needs a database or a plotting library is a stats
script that stops getting run.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

from evals.schema import EvalPair, load_pairs

DEFAULT_PATH = Path(__file__).parent / "data" / "eval_pairs.jsonl"

# Size of the frozen corpus snapshot the labels were written against. Coverage is
# meaningless without a denominator, and this module deliberately does not open a
# database connection to find one. Re-derive after any re-ingest with:
#     select count(*) from chunks;
CORPUS_CHUNK_COUNT = 233


def _histogram(counts: Counter[int]) -> list[tuple[int, int]]:
    return sorted(counts.items())


def format_stats(pairs: list[EvalPair], corpus_size: int = CORPUS_CHUNK_COUNT) -> str:
    """Render the composition report. Pure function so tests can assert on it."""
    if not pairs:
        return "empty dataset\n"

    lines: list[str] = []
    add = lines.append

    answerable = [p for p in pairs if p.answerable]
    unanswerable = [p for p in pairs if not p.answerable]

    add(f"total pairs: {len(pairs)}")

    add("\nby dialect_tag:")
    for tag, n in sorted(Counter(p.dialect_tag for p in pairs).items()):
        add(f"  {tag:<6} {n:>4}  ({n / len(pairs):.1%})")

    add("\nanswerable vs unanswerable:")
    add(f"  answerable    {len(answerable):>4}  ({len(answerable) / len(pairs):.1%})")
    add(f"  unanswerable  {len(unanswerable):>4}  ({len(unanswerable) / len(pairs):.1%})")

    add("\npairs per source_doc:")
    for doc, n in Counter(p.source_doc for p in pairs).most_common():
        add(f"  {n:>4}  {doc}")

    add("\nchunks cited per pair:")
    for n_chunks, n_pairs in _histogram(Counter(len(p.source_chunk_ids) for p in pairs)):
        label = "0 (unanswerable)" if n_chunks == 0 else str(n_chunks)
        add(f"  {label:<18} {n_pairs:>4} pairs")

    mean_q = sum(len(p.question) for p in pairs) / len(pairs)
    mean_q_words = sum(len(p.question.split()) for p in pairs) / len(pairs)
    add("\nquestion length:")
    add(f"  mean {mean_q:.1f} chars / {mean_q_words:.1f} words")
    add(f"  min  {min(len(p.question) for p in pairs)} chars")
    add(f"  max  {max(len(p.question) for p in pairs)} chars")

    cited = {cid for p in pairs for cid in p.source_chunk_ids}
    add("\ncorpus coverage:")
    add(f"  distinct chunks cited: {len(cited)} / {corpus_size} ({len(cited) / corpus_size:.1%})")
    add(f"  total citations:       {sum(len(p.source_chunk_ids) for p in pairs)}")

    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> int:
    path = Path(argv[1]) if len(argv) > 1 else DEFAULT_PATH
    try:
        pairs = load_pairs(path)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(format_stats(pairs), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
