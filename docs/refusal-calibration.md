# Calibrating the refusal threshold

**Result: there is no good threshold.** Swept over all 283 labelled eval pairs,
the cross-encoder's top score does not separate answerable questions from
unanswerable ones well enough to gate on — refusal precision peaks at **0.171**,
and the shipped 0.15 was refusing **39 answerable questions to catch 7
unanswerable ones**, with the correct article already sitting in the context
window of all 39. It also refused **54% of Gulf-dialect questions against 5.5% of
MSA ones**. `rerank_min_score` is therefore now **0.0** — the score gate is off —
and refusal needs a different signal. The measurement and the reasoning are
below; the [upgrade paths](#what-to-do-instead) are the point of writing it up
rather than quietly deleting the feature.

Reproduce:

```bash
PYTHONPATH=. python -m evals.refusal --sweep --compare 0.15   # instant, cached scores
PYTHONPATH=. python -m evals.refusal --sweep --audit 0.15     # ~40 s, the audit table below
PYTHONPATH=. python -m evals.refusal --sweep --rescore        # ~5 min, re-runs the models
```

## The defect

`settings.rerank_min_score` decides whether `/ask` answers or refuses: below it,
the service returns "not in corpus" in the user's own register and never calls a
provider. The value was 0.15, chosen by hand. The 30-question latency replay
(`python -m benchmark.replay --n 30`, seed 0) refused **5 of 30** answerable
questions with it:

| pair | dialect | top score | question |
|---|---|---|---|
| `gulf-026` | gulf | 0.0041 | صار لي سنه ونص بالشركة وابغى اقدم استقالتي، شكثر لازم انبههم قبل ما امشي؟ |
| `msa-6-001` | msa | 0.0256 | كم ساعة يجوز لرب الأسرة أن يشغّل السائق أو الطباخ في اليوم الواحد؟ |
| `gulf-025` | gulf | 0.0285 | بعد ما اطلع من الشركة شكثر يقدرون يمنعوني اشتغل عند شركة تنافسهم؟ |
| `gulf-019` | gulf | 0.0324 | انا وافد، شنو اللي لازم يكون عندي عشان يطلعون لي رخصة شغل؟ |
| `msa-4-032` | msa | 0.0934 | أمضيت ثلاث سنوات في العمل وقدمت استقالتي، ما مدة الإشعار المطلوبة وهل يستمر صرف راتبي خلالها؟ |

Three of the six Gulf questions in that sample were refused. Every one of these
is a plain labour-law question the corpus answers.

Those five rows are also this document's first sanity check: they were produced
by `RagService` inside the replay, and they fall out again, question for
question, from the independently-written scoring path in `evals/refusal.py`. The
number swept below is the number the service compares.

## Method

`evals/refusal.py` runs the **real** retrieval path — `plan → embed → search per
planned query → RRF → cross-encoder` — over all 283 pairs once, recording the top
rerank score per question, and caches those scores in `evals/refusal_scores.json`
so every sweep afterwards is instant and needs no model. Thresholds are then
swept offline, 0.00 to 0.90 in 0.02 steps.

The rerank query is `plan.rewritten or plan.original` and the comparison is
`hits[0].score < threshold`, both exactly as `RagService._is_refusal` does it —
including the Gulf→MSA rewrite, so the dialect numbers below are what the
*planned* pipeline achieves, not what the raw dialect question would.

Configuration: `hybrid+rerank`, `bge-m3` embeddings, `BAAI/bge-reranker-v2-m3`
scored through a sigmoid to 0–1 (see `app/retrieval/rerank.py`), rules planner,
`top_k_retrieve=20`, `top_k_context=5`, 233 chunks, Apple Silicon / MPS.
Dataset: 268 answerable pairs (218 MSA, 50 Gulf) and 15 written to be
unanswerable (11 MSA, 4 Gulf).

## The score distributions

| population | n | min | p25 | median | p75 | max |
|---|---|---|---|---|---|---|
| answerable | 268 | 0.0005 | 0.3622 | 0.8458 | 0.9724 | 0.9997 |
| unanswerable | 15 | 0.0156 | 0.0622 | 0.2355 | 0.5610 | 0.9547 |
| — MSA answerable | 218 | 0.0039 | 0.5985 | **0.9289** | 0.9806 | 0.9997 |
| — Gulf answerable | 50 | 0.0005 | 0.0285 | **0.1168** | 0.4344 | 0.9320 |
| — MSA unanswerable | 11 | 0.0156 | 0.1084 | 0.3601 | 0.7106 | 0.9547 |
| — Gulf unanswerable | 4 | 0.0178 | 0.0486 | 0.0622 | 0.0622 | 0.5610 |

All 15 unanswerable scores, sorted: 0.0156, 0.0178, 0.0259, 0.0486, 0.0622,
0.0683, 0.1084, 0.2355, 0.3601, 0.3916, 0.4788, 0.5610, 0.7106, 0.8390, 0.9547.
Eight of them outscore the median Gulf answerable question (0.1168). The two populations
are interleaved, not stacked.

Read the two bolded medians together: the median **answerable Gulf** question
scores 0.117 and the median **answerable MSA** question scores 0.929. That is an
eight-fold gap between two sets of questions with *identical ground truth* —
the Gulf pairs are hand-written rephrasings of MSA ones. The score is reacting
to register, not to whether the corpus contains the answer.

## The curve

Refuse when `top_score < threshold`. "Unanswerable refused" is the gate working;
"answerable refused" is the gate destroying a good answer.

| threshold | unanswerable refused | answerable refused | refusal recall | refusal precision | F1 | false-refusal rate |
|---|---|---|---|---|---|---|
| 0.00 | 0/15 | 0/268 | 0.000 | 0.000 | 0.000 | 0.000 |
| 0.02 | 2/15 | 10/268 | 0.133 | 0.167 | 0.148 | 0.037 |
| 0.04 | 3/15 | 22/268 | 0.200 | 0.120 | 0.150 | 0.082 |
| 0.06 | 4/15 | 27/268 | 0.267 | 0.129 | 0.174 | 0.101 |
| 0.08 | 6/15 | 29/268 | 0.400 | **0.171** | 0.240 | 0.108 |
| 0.12 | 7/15 | 36/268 | 0.467 | 0.163 | 0.241 | 0.134 |
| 0.16 | 7/15 | 40/268 | 0.467 | 0.149 | 0.226 | 0.149 |
| 0.20 | 7/15 | 46/268 | 0.467 | 0.132 | 0.206 | 0.172 |
| 0.24 | 8/15 | 51/268 | 0.533 | 0.136 | 0.216 | 0.190 |
| 0.32 | 8/15 | 61/268 | 0.533 | 0.116 | 0.190 | 0.228 |
| 0.40 | 10/15 | 75/268 | 0.667 | 0.118 | 0.200 | 0.280 |
| 0.48 | 11/15 | 84/268 | 0.733 | 0.116 | 0.200 | 0.313 |
| 0.56 | 11/15 | 98/268 | 0.733 | 0.101 | 0.177 | 0.366 |
| 0.64 | 12/15 | 109/268 | 0.800 | 0.099 | 0.176 | 0.407 |
| 0.72 | 13/15 | 118/268 | 0.867 | 0.099 | 0.178 | 0.440 |
| 0.80 | 13/15 | 126/268 | 0.867 | 0.094 | 0.169 | 0.470 |
| 0.88 | 14/15 | 148/268 | 0.933 | 0.086 | 0.158 | 0.552 |
| 0.90 | 14/15 | 152/268 | 0.933 | 0.084 | 0.155 | 0.567 |

(Every 0.02 step from 0.00 to 0.90 is swept; rows are thinned above 0.24 for
reading. `--every 1` prints all 46.)

There is no knee. **Refusal precision never exceeds 0.171**, anywhere: at the
best point in the whole sweep, five out of six refusals are wrong. Buying recall
costs answerable questions almost linearly — going from catching 7 of 15 to
catching 12 of 15 costs another 66 good answers (36 refused at 0.12, 102 at 0.60).

Threshold-free, the same conclusion:

| population | AUC (answerable vs unanswerable) | n |
|---|---|---|
| all | **0.7649** | 268 / 15 |
| MSA only | 0.8178 | 218 / 11 |
| Gulf only | **0.5300** | 50 / 4 |

AUC is the probability that a random unanswerable question scores below a random
answerable one — 0.5 is a coin flip. Within MSA the score carries real signal
(0.82). Within Gulf it carries none (0.53, and at n=4 unanswerable that number is
barely more than a gesture — but it cannot be *rescued* by more data either, given
what the same score does next).

The same score separates **MSA-answerable from Gulf-answerable at AUC 0.8886**.
It is a better dialect classifier than it is a relevance classifier. That is the
whole finding in one number.

## Refusal is not "retrieval failed"

The obvious defence of a confidence gate is that the questions it refuses are the
ones retrieval got wrong, so no answer was lost. That is checkable, and it is
false here. For every answerable question scoring below 0.20, whether the gold
chunk was in the top-5 context the generator would have received:

| threshold | answerable refused | gold chunk in top-5 context | gold chunk at rank 1 |
|---|---|---|---|
| 0.04 | 22 | **22 (100%)** | 15 |
| 0.08 | 29 | **29 (100%)** | 21 |
| 0.15 | 39 | **39 (100%)** | 30 |
| 0.20 | 46 | 44 (96%) | 33 |

At the shipped threshold, **all 39 refused answerable questions had the correct
article already retrieved and ranked into the context** (27 Gulf, 12 MSA), 30 of
them at position one. The pipeline found the answer and then threw it away.
Whatever cost ratio you assume between the two failure modes, that is not a
safety feature.

`--audit 0.15` reproduces that row; it re-retrieves only the pairs the threshold
refuses (~40 s) and compares each context against the pair's
`source_chunk_ids`.

## The dialect bias

Answerable pairs only — a refused unanswerable question is the gate working.

| threshold | Gulf refused | Gulf rate | MSA refused | MSA rate | disparity |
|---|---|---|---|---|---|
| 0.02 | 9/50 | 18.0% | 1/218 | 0.5% | 39x |
| 0.04 | 17/50 | 34.0% | 5/218 | 2.3% | 15x |
| 0.08 | 22/50 | 44.0% | 7/218 | 3.2% | 14x |
| **0.15 (shipped)** | **27/50** | **54.0%** | **12/218** | **5.5%** | **10x** |
| 0.20 | 29/50 | 58.0% | 17/218 | 7.8% | 7x |
| 0.30 | 31/50 | 62.0% | 26/218 | 11.9% | 5x |
| 0.50 | 40/50 | 80.0% | 47/218 | 21.6% | 4x |

The bias is not marginal and it does not go away by tuning: at every threshold
that catches anything, a Gulf speaker is 4–39x likelier to be refused than an MSA
speaker asking **the same question with the same correct answer**. More than half
of Gulf questions were being refused in production configuration. The disparity
shrinks at high thresholds only because the gate starts failing everyone.

This was predicted, at n=20, in `app/retrieval/rerank.py`'s score-semantics note
and in the README's "known" list. This is that prediction measured at n=268, and
it is worse than the note assumed.

## The decision

**`rerank_min_score`: 0.15 → 0.0.** One line in `app/config.py`. A sigmoid score
is never negative, so 0.0 disables the score comparison; `_is_refusal` still
refuses when retrieval returns nothing at all, which remains the only automatic
refusal path.

The trade, stated explicitly because the asymmetry is real and runs the *other*
way. For a legal-information service:

- Wrongly **refusing** a question the corpus can answer is a **mild** failure. The
  user is told to look elsewhere and is no worse off than before asking.
- Wrongly **answering** a question the corpus does not cover is a **serious**
  failure. The user is handed confident text about their own employment rights
  that no article supports.

So the honest default is to bias *toward* refusing, and `evals/refusal.recommend`
does exactly that: it maximises refusal recall — not F1, which would weigh the
two failures identically — subject to two named constraints, a 10% cap on the
false-refusal rate (`MAX_FALSE_REFUSAL_RATE`) and a 50% floor under refusal
precision (`MIN_REFUSAL_PRECISION`, i.e. a refusal must be right more often than
wrong).

**Nothing in the sweep clears both**, so the policy returns 0.0 and says why.
Turning the gate off is not a preference for answering; it is the finding that
this particular score cannot buy the safety it was supposed to buy. It was
refusing 39 questions whose answers it had already retrieved, in order to block 7
of 15 uncovered ones, while failing dialect speakers ten times more often than
MSA speakers.

One honest counter-argument, stated because it is the strongest one against this
decision. Precision depends on how much of the population is unanswerable, and
this dataset's 15/283 (5.3%) is an authoring decision, not traffic — real users
ask off-topic questions far more often. The prevalence at which each threshold's
refusals would be right half the time is `fr / (recall + fr)`:

| threshold | break-even unanswerable share of traffic |
|---|---|
| 0.08 | 21.3% |
| 0.14 | 23.3% |
| 0.20 | 26.9% |
| 0.40 | 29.6% |

So if roughly a quarter of real traffic were uncovered, a threshold around 0.08
would clear the precision floor and become defensible. That is unknowable without
a query log, and it does not rescue the other two findings: the refused questions
still have their gold chunk in context, and the gate still fails Gulf speakers
10x more often. Prevalence changes the precision arithmetic; it does not make a
dialect detector into a relevance detector. **Revisit this the day there is a
query log**, with the break-even column above as the test.

## What to do instead

Ranked by measured promise, not by novelty.

1. **Let the model abstain.** The system prompt already instructs it (rule 4 in
   `app/generation/budget.py`): say "لا تتضمن المواد المتاحة إجابة عن هذا السؤال."
   when the attached articles do not answer the question. That decision reads the
   actual Arabic text of the retrieved articles instead of a scalar, and it is
   register-blind in a way the cross-encoder demonstrably is not. It costs one
   LLM call — which the score gate was avoiding, so this is a real cost, not a
   free lunch. **Unmeasured: there is no API key in this environment.** Measuring
   it is the first thing to do the day there is one, against these same 15
   unanswerable pairs.
2. **A margin feature, not an absolute score.** The top-1 score is dialect-
   sensitive; the *gap* between top-1 and top-5 may not be, since both legs shift
   together under register. Cheap to test — it needs no new model, only the full
   reranked list instead of `hits[0]`, and `score_pairs` would extend to record
   it. Worth an hour before anything heavier.
3. **A dialect-aware threshold.** `plan.register` is already known at the gate,
   and the per-dialect distributions above are what a two-threshold gate would be
   calibrated from. This is a patch on the symptom rather than the cause, and 4
   unanswerable Gulf pairs is far too few to fit a Gulf threshold — the eval set
   would need extending first.
4. **A calibrated classifier.** Fit a small logistic model on features that are
   free at request time (top-1, top-5 margin, mean of top-k, hit count, register,
   query length) against this same label. Its output would be an actual
   probability, which is what a threshold assumes it has. Needs more unanswerable
   pairs than 15 to be worth fitting, and it is the heaviest option — do 1 and 2
   first.

## Limitations

- **15 unanswerable pairs, 4 of them Gulf.** Every refusal-recall figure moves in
  steps of 6.7 points, and the Gulf-only AUC of 0.53 rests on 4 questions. The
  direction of every finding here is well-supported by the 268 answerable pairs;
  the precision numbers are not precise.
- **The unanswerable pairs are authored, not observed.** They are deliberately
  near-miss questions — other Gulf states' labour laws, tax, banking, traffic
  fines — so they are plausibly *harder* than typical off-topic traffic, which
  would flatter the gate rather than this conclusion.
- **233 chunks.** Top-5 is 2% of the index. On a larger corpus the reranker's
  absolute scores would sit in a different place, and this calibration would have
  to be re-run — `--rescore` is one command.
- **One model pair.** bge-m3 + bge-reranker-v2-m3. The score semantics are not
  comparable across rerankers, so the *number* 0.0 is not portable even if the
  method is.
- **Generation has never run.** Upgrade path 1 — the one this document recommends
  most — is therefore an argument, not a measurement.
