# Eval dataset — `eval_pairs.jsonl`

283 labelled Arabic Q/A pairs over 5 real Qatari legal documents (233 ingested chunks).
This file is the ground truth behind every retrieval number, the CI regression gate, and
the published writeup. A wrong `source_chunk_id` does not throw — it silently scores 0.0
recall forever — so the integrity guards in `tests/test_eval_dataset.py` matter more than
the pair count does.

## Composition

| | |
|---|---|
| Total pairs | 283 |
| MSA / Gulf | 229 / 54 |
| Answerable / unanswerable | 268 / 15 |
| Distinct chunks cited | 186 of 233 (79.8%) |
| Chunks per pair | 230×1, 38×2, 15×0 (unanswerable) |
| Mean question length | 74.4 chars / 13.6 words |

Per document: `qatar-labour-law-14-2004` 239, `qatar-domestic-workers-law-15-2017` 19,
`qatar-labour-dispute-committees-decision-6-2018` 10, `qatar-heat-stress-decision-17-2021` 9,
`qatar-minimum-wage-law-17-2020` 6.

The distribution follows the corpus: the labour law is by far the largest document. It is
*not* balanced across documents, so a per-document recall breakdown is more honest than a
single headline number — the four smaller documents contribute 44 pairs between them and a
regression confined to one of them can hide inside the aggregate.

## Format

One JSON object per line, UTF-8, `ensure_ascii=False`, sorted keys, sorted by `id`:

```json
{"answer": "...", "derived_from": "msa-1-002", "dialect_tag": "gulf", "id": "gulf-001",
 "question": "...", "source_chunk_ids": ["qatar-labour-law-14-2004:100:0"],
 "source_doc": "qatar-labour-law-14-2004"}
```

`evals/schema.py::validate_pair` is the contract. Every chunk id must start with
`<source_doc>:` — that prefix rule is the only automatic link between the corpus and
hand-written ground truth. `source_chunk_ids: []` marks an **unanswerable** pair: there is
no correct chunk, so retrieval metrics are undefined and the pair scores refusal behaviour
instead.

`derived_from` is an optional extra key (the schema deliberately ignores unknown keys). It
links a Gulf pair to the MSA pair it rephrases, and is what lets the benchmark run a *paired*
MSA-vs-Gulf comparison instead of comparing two unrelated question sets.

## The dialect subset

54 Gulf-dialect questions against the same MSA documents, 50 of which are rephrasings of a
specific MSA pair with **identical** `source_chunk_ids` (the other 4 are unanswerable Gulf
questions). Because ground truth is held constant, any recall gap between the MSA pair and
its Gulf twin is attributable to the dialect of the question and nothing else. That is the
measurement the dialect→MSA query-rewrite step in `app/planning/` has to justify itself
against.

## How it was built

1. **Drafted** by LLM directly from the real ingested chunk text — never from memory of
   Qatari law — by 8 annotator passes working on disjoint slices of the corpus (6 MSA
   slices, 1 Gulf, 1 unanswerable), written to `pairs_*.jsonl`.
2. **Merged** into `eval_pairs.jsonl` sorted by `id`, each line re-validated through
   `validate_pair`.
3. **Integrity-audited** against the live database (see below). Nothing was dropped —
   every audit came back clean.

The `pairs_*.jsonl` files are kept as the provenance record of who wrote what. Regenerating
the merged file from them is deterministic.

## Audit results

Run against the live 233-chunk corpus at merge time:

| Check | Result |
|---|---|
| Dangling chunk ids (cited but not in `chunks`) | **0** of 306 citations |
| Duplicate pair ids | **0** |
| Near-duplicate questions (token Jaccard ≥ 0.8) | **0** |
| Question↔chunk-sentence leakage (Jaccard ≥ 0.6) | **0** |
| Gulf pairs disagreeing with their `derived_from` parent | **0** |

Two caveats on those zeros, because a clean audit is only as good as its threshold:

- **The stated thresholds were never approached.** Max observed question-question similarity
  is 0.50 and max question↔sentence overlap is 0.42, so the 0.8/0.6 gates could not have
  fired. Word-level Jaccard is a weak signal in Arabic — clitics (`و`, `ال`, `ب`, `ل`) attach
  to the stem, so two phrasings of the same idea share fewer surface tokens than they would in
  English. The checks were re-run at 0.65 and 0.45 respectively and still found nothing, and a
  stricter question-token-containment lens put the worst case at 0.71 with a 0.28 mean.
- **The duplicate risk never materialised structurally.** The 6 MSA annotators cited
  **zero chunks in common** — they were cleanly partitioned by article range. The 25 cases of
  two pairs sharing a chunk set are all *within* one annotator's file and ask about genuinely
  different aspects of the same article (art. 82 is cited 4×: eligibility, pay scaling,
  termination, forfeiture). Those are intentional, not duplicates.

An additional check not in the original plan: **answer grounding**, measuring how much of each
answer's content actually appears in its cited chunk (median 0.81, min 0.41). The 25
worst-scoring citations were read against the chunk text by hand and all were correct — the
low scores are Arabic morphology (`تعيين` vs `يعين`), not mislabels. Chunk-id *existence* is
not chunk-id *correctness*, and this is the check that would catch a plausible-but-wrong
citation.

## Limitations — read before quoting any number from this

- **Single annotator, no inter-annotator agreement.** Every pair was written and reviewed by
  the same pipeline. There is no second opinion and therefore no κ statistic. Treat the
  labels as one careful person's judgement, not as adjudicated ground truth.
- **LLM-assisted drafting.** Questions and answers were LLM-drafted from real chunk text.
  Grounding in the cited chunk was verified programmatically and spot-checked by hand, but
  not every one of the 283 answers has been independently verified against the statute by a
  lawyer. These are retrieval labels, **not legal advice**.
- **Single-chunk bias.** 230 of 268 answerable pairs cite exactly one chunk, so the dataset
  mostly measures single-hop retrieval. Multi-hop performance rests on just 38 pairs — too
  few to carry a confident claim.
- **Unanswerable set is small** (15 pairs, 5.3%). Enough to detect a model that never
  refuses; not enough to put a tight confidence interval on the refusal rate.
- **Corpus coverage is 79.8%.** 47 chunks are cited by no pair — mostly definitional and
  preamble articles. Retrieval quality on those is unmeasured.
- **Gulf dialect is one writer's register.** Qatari/Khaleeji as written by one person. It
  does not cover the full spread of Gulf variation, and a real user population would be
  noisier (typos, code-switching, Arabizi) than this set is.
- **Frozen against a corpus snapshot.** The labels are only valid for the 233-chunk ingest
  they were written against. Re-chunking the corpus (different token cap, different article
  splitting) silently invalidates every `source_chunk_id`. Re-run the audit after any
  ingestion change.

## Commands

```bash
# Composition report
python -m evals.dataset_stats                      # defaults to evals/data/eval_pairs.jsonl
python -m evals.dataset_stats path/to/other.jsonl

# Integrity guards (no database required — runs in ~0.01s)
pytest tests/test_eval_dataset.py -q

# Validate the file the way every downstream consumer does
python -c "from evals.schema import load_pairs; print(len(load_pairs('evals/data/eval_pairs.jsonl')))"
```

### Regenerating the merged file

`eval_pairs.jsonl` is a deterministic merge of the `pairs_*.jsonl` files — sort by `id`,
validate every line, write with `ensure_ascii=False, sort_keys=True`:

```bash
python - <<'PY'
import json
from pathlib import Path
from evals.schema import validate_pair

data = Path("evals/data")
rows, seen = [], set()
for f in sorted(data.glob("pairs_*.jsonl")):
    for ln, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        obj = json.loads(line)
        validate_pair(obj)
        if obj["id"] in seen:
            raise SystemExit(f"duplicate id {obj['id']!r} at {f.name}:{ln}")
        seen.add(obj["id"])
        rows.append(obj)
rows.sort(key=lambda r: r["id"])
(data / "eval_pairs.jsonl").write_text(
    "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows),
    encoding="utf-8",
)
print(f"wrote {len(rows)} pairs")
PY
```

The `chunks` table must be populated to re-run the dangling-id and grounding audits; the
merge itself needs no database.

### Verifying chunk ids still exist after a re-ingest

Existence checking is deliberately **not** a pytest test: `tests/conftest.py`'s `db_session`
fixture TRUNCATEs `chunks`, so a test using it would query an empty table. The unit tests
assert id *format* only; existence is checked here, against the real corpus:

```bash
python -c "
import subprocess
from evals.schema import load_pairs
cited = {c for p in load_pairs('evals/data/eval_pairs.jsonl') for c in p.source_chunk_ids}
out = subprocess.run(['docker','exec','arabic-rag-pg','psql','-U','rag_user','-d','rag_db',
                      '-t','-A','-c','select id from chunks'], capture_output=True, text=True)
live = {l.strip() for l in out.stdout.splitlines() if l.strip()}
print('dangling:', sorted(cited - live) or 'NONE')
print('coverage:', len(cited & live), '/', len(live))
"
```
