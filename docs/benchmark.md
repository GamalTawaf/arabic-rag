# Retrieval benchmark — Arabic legal RAG

**What this measures:** how well four retrieval configurations find the right article of Qatari
labour legislation, for the same question asked in Modern Standard Arabic and in Gulf dialect.

Every number on this page is read out of [`benchmark/results/results.json`](../benchmark/results/results.json).
The tables are generated from that file, not typed by hand.

## Setup

| | |
|---|---|
| Corpus | 5 Qatari labour-law documents (Law 14/2004, Law 15/2017 domestic workers, Law 17/2020 minimum wage, Decision 17/2021 heat stress, Decision 6/2018 dispute committees), chunked on `المادة` boundaries into 233 chunks |
| Index | one Postgres table: original text, Arabic-normalized text, a generated `tsvector` (`simple` config), one pgvector column per embedding model, HNSW cosine indexes |
| Dataset | 283 labelled question/chunk-id pairs — 268 answerable, 15 unanswerable; 229 MSA, 54 Gulf dialect. Ground truth is chunk ids, so scoring is deterministic and needs no LLM |
| Models measured | `intfloat/multilingual-e5-large` (`e5`) and `BAAI/bge-m3` (`bge`), both local on Apple Silicon MPS |
| Reranker | `BAAI/bge-reranker-v2-m3` cross-encoder, local |
| Configs | `dense` (pgvector cosine), `lexical` (Postgres FTS), `hybrid` (both, fused with RRF), `hybrid+rerank` (fusion then cross-encoder) |
| Metrics | recall@3, recall@10, hit@3, hit@10, MRR, mean/p95 per-query retrieval latency |

All configs retrieve `top_k=20`. `hybrid+rerank` reranks those 20 down to 10, deliberately *not*
down to the service's production 5 — recall@10 over a 5-item list would measure the truncation
rather than the reranker. Production top-5 behaviour is what the recall@3 column tracks.

The 15 unanswerable questions have no correct chunk, so recall is undefined for them; they are
scored separately, on the score each config assigns to its best wrong answer.

### Reproducing it

```bash
docker compose up -d db
alembic upgrade head
python -m ingestion ingest                  # 5 documents -> 233 chunks
python -m ingestion backfill --model e5     # ~10s on MPS
python -m ingestion backfill --model bge    # ~10s on MPS

DATABASE_URL=postgresql+asyncpg://rag_user:rag_pass@localhost:5433/rag_db \
  python -m benchmark.run                   # writes benchmark/results/results.json

python -m benchmark.run --tables            # re-renders every table below
python -m benchmark.run --headline e5:dense # re-renders the highlight table below
```

The whole matrix is 28 cells — 2 models × 4 configs × 4 splits, minus the lexical rows, which are
measured once and shared because full-text search never touches an embedding column. It took
**1280 s (21 min)** on an M-series laptop; the two reranked configurations are ~95% of that.

## The headline: what the dialect costs

**Ask the same question in Gulf dialect instead of Modern Standard Arabic and retrieval gets worse
— but how much worse depends entirely on the embedding model, and the damage lands on *ranking*
rather than on *coverage*.**

With `multilingual-e5-large`, holding the target chunks fixed (the "MSA matched" control), moving
the question from MSA to Gulf dialect costs:

| e5, dense retrieval | MSA matched (n=68) | Gulf (n=50) | change |
|---|---|---|---|
| recall@10 | 0.949 | 0.850 | **-9.9 pts** |
| recall@3 | 0.882 | 0.750 | **-13.2 pts** |
| mrr | 0.815 | 0.617 | **-19.7 pts** |

The right article is usually still *somewhere* in the top 10; it is much less often at the top.
That is the practical shape of the problem — a RAG service that feeds the top 3 chunks to a
generator loses far more than the recall@10 column suggests.

With `BGE-m3` the same comparison is nearly flat: −0.4 points of recall@10, −3.3 recall@3, −5.9 MRR.
Over the 50 Gulf questions, e5's dense leg misses the gold chunk 6 times and bge misses it twice —
and bge's two failures are a strict subset of e5's six. **The entire model-level difference on
Gulf-dialect questions is four questions.** That is a real difference in the same direction across
every metric, and it is also far too small a sample to claim a general property of the two models.

Reranking recovers most of the loss without removing it: the cross-encoder lifts e5's Gulf recall@3
from 0.750 to 0.860 and bge's from 0.790 to 0.910, but the MSA-matched/Gulf gap for e5 stays at
−9.3 points of recall@10. **Reranking makes the retrieved set better; it does not make the dialect
gap go away.**

### How confident should you be in that?

Not very, on the size of the effect. The Gulf subset is **50 answerable questions**, so a recall
figure of ~0.9 carries a 95% Wilson interval of roughly **±8 to ±10 points** — for instance e5's
Gulf recall@10 of 0.850 is [0.726, 0.924], which overlaps the MSA-matched interval of
[0.868, 0.981]. Taken on its own, the headline gap is **not statistically separable from noise**,
and no significance test was run.

What makes it worth believing anyway is *consistency*: the gap points the same way in all four e5
configurations (−9.9, −9.6, −9.3 points of recall@10 for dense, hybrid, hybrid+rerank, and −7.8 for
lexical), and it is much larger on the ranking metrics than the k=10 metric, which is the signature
you would expect from a real register mismatch rather than from sampling. One cell does invert:
e5's `hybrid` recall@3 is 6.6 points *better* on Gulf than on matched MSA — but hybrid's recall@3
has been wrecked by fusion (0.50 against dense's 0.86, see below), so that column is measuring
fusion damage, not dialect. Directionally: believe
it. Numerically: treat "roughly ten points of recall@10 and roughly twenty of MRR, for e5" as an
order of magnitude, not a measurement.

## Full results

<!-- generated by `python -m benchmark.run --tables`; do not edit by hand -->

Run 2026-07-25T04:39:06+00:00 · commit `6b70b6d` · device `mps` · 233 chunks / 5 documents · 283 pairs (268 answerable, 229 MSA / 54 Gulf) · matrix wall clock 1280.5s

### The dialect penalty

| model | config | MSA all | MSA matched | Gulf | Gulf − MSA matched | relative |
|---|---|---|---|---|---|---|
| bge | dense | 0.950 (n=218) | 0.934 (n=68) | 0.930 (n=50) | -0.004 | -0% |
| bge | lexical* | 0.346 (n=218) | 0.478 (n=68) | 0.400 (n=50) | -0.078 | -16% |
| bge | hybrid | 0.920 (n=218) | 0.919 (n=68) | 0.870 (n=50) | -0.049 | -5% |
| bge | hybrid+rerank | 0.970 (n=218) | 0.978 (n=68) | 0.940 (n=50) | -0.038 | -4% |
| e5 | dense | 0.945 (n=218) | 0.949 (n=68) | 0.850 (n=50) | -0.099 | -10% |
| e5 | lexical* | 0.346 (n=218) | 0.478 (n=68) | 0.400 (n=50) | -0.078 | -16% |
| e5 | hybrid | 0.933 (n=218) | 0.956 (n=68) | 0.860 (n=50) | -0.096 | -10% |
| e5 | hybrid+rerank | 0.966 (n=218) | 0.993 (n=68) | 0.900 (n=50) | -0.093 | -9% |

**MSA matched** is the control: the MSA pairs whose gold chunks are also targeted by a Gulf question, so the retrieval target is held fixed and only the register of the question changes. **MSA all** is every MSA pair and is *not* a like-for-like comparison. The last two columns are Gulf minus MSA-matched, in absolute recall@10 points and as a share of the matched score; negative means Gulf-dialect questions retrieve worse.

### Full matrix

**bge — BAAI/bge-m3**

| config | split | n | recall@3 | recall@10 | hit@3 | hit@10 | MRR | mean ms | p95 ms |
|---|---|---|---|---|---|---|---|---|---|
| dense | all | 268 | 0.864 | 0.946 | 0.899 | 0.963 | 0.829 | 2.6 | 3.9 |
| dense | msa | 218 | 0.881 | 0.950 | 0.917 | 0.963 | 0.848 | 2.5 | 3.4 |
| dense | gulf | 50 | 0.790 | 0.930 | 0.820 | 0.960 | 0.744 | 2.0 | 3.7 |
| dense | msa_matched | 68 | 0.824 | 0.934 | 0.853 | 0.941 | 0.803 | 1.4 | 2.1 |
| lexical* | all | 268 | 0.188 | 0.356 | 0.202 | 0.373 | 0.181 | 2.5 | 4.1 |
| lexical* | msa | 218 | 0.195 | 0.346 | 0.211 | 0.367 | 0.186 | 2.4 | 3.5 |
| lexical* | gulf | 50 | 0.160 | 0.400 | 0.160 | 0.400 | 0.158 | 1.9 | 3.0 |
| lexical* | msa_matched | 68 | 0.265 | 0.478 | 0.279 | 0.485 | 0.234 | 2.5 | 3.8 |
| hybrid | all | 268 | 0.489 | 0.910 | 0.526 | 0.937 | 0.480 | 4.0 | 5.6 |
| hybrid | msa | 218 | 0.491 | 0.920 | 0.532 | 0.945 | 0.484 | 4.4 | 6.0 |
| hybrid | gulf | 50 | 0.480 | 0.870 | 0.500 | 0.900 | 0.466 | 4.3 | 6.5 |
| hybrid | msa_matched | 68 | 0.537 | 0.919 | 0.559 | 0.941 | 0.528 | 4.8 | 6.9 |
| hybrid+rerank | all | 268 | 0.922 | 0.965 | 0.952 | 0.981 | 0.883 | 973.5 | 1031.3 |
| hybrid+rerank | msa | 218 | 0.924 | 0.970 | 0.954 | 0.982 | 0.903 | 978.2 | 1031.8 |
| hybrid+rerank | gulf | 50 | 0.910 | 0.940 | 0.940 | 0.980 | 0.794 | 987.6 | 1050.5 |
| hybrid+rerank | msa_matched | 68 | 0.912 | 0.978 | 0.941 | 0.985 | 0.901 | 978.0 | 1021.5 |

**e5 — intfloat/multilingual-e5-large**

| config | split | n | recall@3 | recall@10 | hit@3 | hit@10 | MRR | mean ms | p95 ms |
|---|---|---|---|---|---|---|---|---|---|
| dense | all | 268 | 0.855 | 0.927 | 0.899 | 0.948 | 0.818 | 2.6 | 3.6 |
| dense | msa | 218 | 0.878 | 0.945 | 0.927 | 0.963 | 0.865 | 1.4 | 2.2 |
| dense | gulf | 50 | 0.750 | 0.850 | 0.780 | 0.880 | 0.617 | 2.4 | 4.7 |
| dense | msa_matched | 68 | 0.882 | 0.949 | 0.912 | 0.956 | 0.815 | 2.5 | 3.9 |
| lexical* | all | 268 | 0.188 | 0.356 | 0.202 | 0.373 | 0.181 | 2.5 | 4.1 |
| lexical* | msa | 218 | 0.195 | 0.346 | 0.211 | 0.367 | 0.186 | 2.4 | 3.5 |
| lexical* | gulf | 50 | 0.160 | 0.400 | 0.160 | 0.400 | 0.158 | 1.9 | 3.0 |
| lexical* | msa_matched | 68 | 0.265 | 0.478 | 0.279 | 0.485 | 0.234 | 2.5 | 3.8 |
| hybrid | all | 268 | 0.502 | 0.920 | 0.537 | 0.944 | 0.474 | 3.9 | 5.3 |
| hybrid | msa | 218 | 0.477 | 0.933 | 0.514 | 0.959 | 0.470 | 4.1 | 5.8 |
| hybrid | gulf | 50 | 0.610 | 0.860 | 0.640 | 0.880 | 0.489 | 4.5 | 7.1 |
| hybrid | msa_matched | 68 | 0.544 | 0.956 | 0.559 | 0.971 | 0.524 | 5.0 | 7.1 |
| hybrid+rerank | all | 268 | 0.910 | 0.953 | 0.944 | 0.970 | 0.881 | 985.0 | 1030.9 |
| hybrid+rerank | msa | 218 | 0.922 | 0.966 | 0.954 | 0.982 | 0.903 | 978.8 | 1035.8 |
| hybrid+rerank | gulf | 50 | 0.860 | 0.900 | 0.900 | 0.920 | 0.785 | 982.2 | 1036.3 |
| hybrid+rerank | msa_matched | 68 | 0.926 | 0.993 | 0.956 | 1.000 | 0.915 | 986.1 | 1044.0 |

`*` marks the lexical rows: full-text search uses no embedding model, so the same single measurement is repeated under each model for reading convenience. It is one run, not two.

### Unanswerable questions (refusal signal)

| model | config | unanswerable pairs | returned hits | mean top score | max top score |
|---|---|---|---|---|---|
| bge | dense | 15 | 15 | 0.6081 | 0.6759 |
| bge | lexical* | 15 | 15 | 2.1067 | 3.5000 |
| bge | hybrid | 15 | 15 | 0.0317 | 0.0328 |
| bge | hybrid+rerank | 15 | 15 | 0.3208 | 0.9547 |
| e5 | dense | 15 | 15 | 0.8471 | 0.8677 |
| e5 | lexical* | 15 | 15 | 2.1067 | 3.5000 |
| e5 | hybrid | 15 | 15 | 0.0318 | 0.0328 |
| e5 | hybrid+rerank | 15 | 15 | 0.3208 | 0.9547 |

Retrieval cannot refuse — it always returns its top-k — so what is measured is the score each config puts on its best *wrong* answer. A usable 'not in corpus' threshold has to separate this distribution from the answerable one.

### Not measured

| model | why it was not measured |
|---|---|
| cohere | no API key |
| openai | no API key |

## Which configuration wins

`hybrid+rerank` on `bge-m3` is the best retriever measured: recall@10 0.965, recall@3 0.922,
MRR 0.883 over all 268 answerable pairs, and it is also the strongest on Gulf dialect (recall@10
0.940). It costs **~975 ms per query** against ~2.6 ms for dense alone — roughly **375× the
latency** for **+1.9 points of recall@10** and **+5.8 points of recall@3** over plain dense bge.

Whether that trade is worth making is a product question, not a retrieval question. For a p95
budget of 3.5 s end to end with generation still to pay for, a ~1 s reranker is affordable but not
free, and the honest summary is:

- **Best quality:** `bge` + `hybrid+rerank`.
- **Best quality per millisecond, by a wide margin:** `bge` + `dense`. It reaches 0.946 recall@10
  and 0.864 recall@3 in under 3 ms.
- **e5 vs bge:** bge is ahead on almost every cell, but by margins (1.9 points of recall@10 overall,
  four questions on the Gulf subset) that this dataset cannot resolve. Anyone reading this as
  "bge-m3 beats multilingual-e5-large" is over-reading it. What the data supports is "they are
  close, and bge was never worse."

### Where lexical wins, and where it does not

Lexical search does not win anywhere here. Postgres FTS reaches **recall@10 0.356** against 0.927+
for either dense model — it is beaten by a factor of two and a half. The cause is stated plainly in
the code: the `tsvector` uses the `simple` configuration, which has **no Arabic stemmer and no
stopword list**, so matching is exact-token-after-normalization. Arabic morphology — clitics,
broken plurals, `الأجور` vs `الأجر` — is entirely unhandled, and a natural-language question shares
few surface tokens with the article that answers it.

Its one relative strength is visible in the matched control: on the chunks that Gulf questions
target, lexical scores 0.478 rather than its 0.346 overall — those articles happen to contain
distinctive, rare terms. That is a property of the subset, not a dialect effect (see below).

### Hybrid fusion is *worse* than dense alone here — and that is not a bug

The most counter-intuitive number in the table: fusing dense with lexical **drops recall@3 from
0.864 to 0.489** for bge, and from 0.855 to 0.502 for e5, while barely moving recall@10.

This is unweighted Reciprocal Rank Fusion doing exactly what it is defined to do with one strong
list and one weak one. RRF scores a document `1/(60 + rank)` per list and sums. A chunk that the
dense leg ranks **1st** and lexical never returns scores `1/61 = 0.0164`. A chunk that dense ranks
**3rd** and lexical ranks **4th** scores `1/63 + 1/64 = 0.0315` — nearly double. Appearing in both
lists beats being right in one, and with a lexical leg whose own recall@3 is 0.19, "appearing in
both lists" is mostly noise agreeing with itself.

Traced on a single query (`gulf-001`, gold chunk `qatar-labour-law-14-2004:100:0`):

```
dense   top-5: [100:0, 71:0, 59:0, 120:0, 145:0]     <- gold at rank 1
lexical top-5: [62:0, 3:1, heat-stress:4:0, 39:0, 1:2]  <- gold absent
hybrid  top-5: [59:0, 62:0, 100:0, 39:0, 70:0]       <- gold demoted to rank 3
scores:        [0.03102, 0.02955, 0.02921, ...]
```

So hybrid fusion is only worth its slot in this stack because the reranker sits behind it and
repairs the top of the list. Fusion widens the candidate pool cheaply; it does not order it. If the
reranker were removed, the right production configuration would be **dense-only**, not hybrid —
and the benchmark is the reason we know that rather than assuming hybrid is free.

## Anomalies investigated before publishing

Three results looked wrong enough to stop and check. All three survived; none was a plumbing bug.

1. **Gulf scored *higher* than MSA on lexical search** (0.400 vs 0.346) — the exact "surprising
   finding that is probably a bug" pattern. It is neither: it is a population artifact. The 50 Gulf
   questions target only 47 distinct gold chunk sets, and the full MSA set targets far more. Adding
   the **MSA-matched control** — the MSA pairs aiming at the *same* gold chunks — reverses it:
   0.478 MSA vs 0.400 Gulf. The MSA-vs-Gulf comparison was never like-for-like, and every headline
   number on this page now uses the matched control instead. This is why the `msa_matched` split
   exists in `results.json`.
2. **bge showing essentially no dialect penalty** while e5 showed ten points. Checked by
   re-running both dense legs over the Gulf pairs and diffing the failures: e5 misses 6 of 50,
   bge misses 2, bge's failures ⊂ e5's. Both vector columns are populated (233/233 each), hold
   unit-norm vectors, and are genuinely different (cosine between the two models' vectors for the
   same chunk ≈ 0.46). The difference is real and it is four questions wide.
3. **Hybrid underperforming dense.** Traced to the RRF tie arithmetic above on a specific query.
   Expected behaviour of unweighted RRF with an unbalanced pair of legs, not a fusion bug.

One more check worth stating: all **186 distinct gold chunk ids referenced by the dataset resolve
to rows in the corpus** (0 missing), so no part of the recall ceiling is caused by dangling ground
truth.

### Why the absolute numbers are high

Recall@10 of ~0.95 on a 233-chunk corpus means the correct article is in the top 4% of the index.
That is a genuinely easy retrieval problem: the corpus is small, the chunks are article-sized and
topically distinct, and the MSA questions were LLM-drafted *from the chunks themselves*, which
gives them more lexical and semantic overlap with the gold chunk than a real user's question would
have. **These numbers should not be read as what this system would score on a real query log.**
The Gulf-dialect subset, hand-written against the same targets, is the closest thing here to an
out-of-distribution test — and it is the subset that scores worst.

## Limitations

These are the reasons not to over-read the tables above. They are listed because they are real,
not as a formality.

- **One annotator.** Every pair was written and reviewed by one person (the author). There is no
  second annotator and therefore **no inter-annotator agreement figure**. Where "the right chunk"
  is genuinely arguable — a question whose answer spans two articles — the label reflects one
  judgement, and nothing here measures how often that judgement is contestable.
- **LLM-assisted drafting.** MSA candidate questions were drafted by an LLM from corpus chunks and
  then manually reviewed and edited. The review was real, but the *distribution* of questions is
  still shaped by what a model finds natural to ask about a legal article, which is not the
  distribution of what a worker or an HR manager would actually ask.
- **The corpus is one document with satellites.** 239 of 283 pairs — and 167 of 233 chunks — come
  from Labour Law 14/2004. The aggregate numbers are, to a first approximation, that document's
  numbers. The per-document breakdown is in `results.json` (`by_doc`) precisely so a regression on
  a 6-pair document cannot hide inside a good average.
- **47 of 233 chunks are never cited** by any pair. Those chunks are in the index and can be
  retrieved as false positives, but nothing in this dataset rewards retrieving them correctly, so
  recall here is measured over 80% of the corpus, not all of it.
- **The Gulf subset is small: 50 answerable pairs.** A recall@10 measured on 50 items has a 95%
  confidence interval roughly ±10 points wide. Gaps smaller than that are not distinguishable from
  sampling noise, and no significance test was run.
- **Gulf questions are rephrasings, not field data.** They were hand-written by one Gulf-Arabic
  speaker as dialect versions of existing MSA questions. Real dialect queries are shorter, more
  elliptical, more typo-prone, and frequently code-switch with English. The dialect penalty here is
  therefore a *lower bound* on what production traffic would produce.
- **Two of four planned models were measured.** `text-embedding-3-large` and Cohere `embed-v4.0`
  are implemented and unit-tested against mocked transports, but no API keys were available, so
  they were not run. They appear in `results.json` as
  `{"status": "not_run", "reason": "no API key"}` rather than being omitted.
- **Latency is a laptop number, not a production p95.** Measured on Apple Silicon against Postgres
  in a local Docker container, with the query embedding batched across the whole dataset and
  reported separately (`latency.embed_batch_ms`) rather than smeared into the per-query figure. A
  production p95 has to include per-query embedding, network round-trips, connection-pool waits and
  a cold HNSW cache. Treat these as *relative* costs between configs, which is what they are useful
  for, and nothing more.
- **No cost column.** The design calls for $/1K queries; with both measured models running locally
  the honest value is "electricity", and the two models that would have a real price were not run.

## What would move these numbers

In rough order of expected value:

1. **Dialect→MSA query rewriting** before retrieval — the planned next ablation, and the direct
   response to the headline finding.
2. **An Arabic-aware text search configuration.** The lexical leg uses Postgres' `simple` config:
   no stemmer, no stopword list, exact token matching after normalization. Arabic morphology
   (clitics, broken plurals, `الأجور`/`الأجر`) is entirely unhandled, which is most of the gap
   between the lexical and dense columns.
3. **Weighted fusion.** Unweighted RRF lets a lexical leg with recall@3 of 0.19 demote correct
   dense hits (see above). A per-leg weight, or simply dropping the lexical leg until it is worth
   fusing, is a one-line change the benchmark can then re-score.
4. **More Gulf pairs.** 50 is enough to see a large effect and not enough to size a small one.
5. **The two API models**, to check whether a much larger multilingual embedding model narrows the
   dialect gap or just raises both numbers.
