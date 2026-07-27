"""Cross-encoder reranking: the last retrieval stage before generation.

Fusion gives us top-20 candidates ordered by *lexical/dense agreement*; a
cross-encoder reads the query and each candidate together and orders them by
actual relevance. It is also where the "not in corpus" decision is made, so the
score it emits has to mean something.

**Score semantics.** ``BAAI/bge-reranker-v2-m3`` emits an unbounded relevance
logit — measured on this corpus it runs about -11 (plainly unrelated) to +6
(clearly the answer). A raw logit is useless as a threshold, so every score here
is passed through a sigmoid and reported as a **0-1 value**: >0.5 means the model
leans "relevant", and ``settings.rerank_min_score`` is compared against this
transformed value, not the logit. The mapping is monotonic, so ordering is
unaffected — only the threshold becomes interpretable.

Honest caveat: 0-1 is a *bounded* score, not a calibrated probability, and it is
not comparable across models. Measured over 20 eval pairs (gold chunk vs 5 random
corpus chunks, all 20 ranked correctly): distractors sit at 0.00004-0.008, while
the gold chunk spans 0.0003-0.999 — median 0.92 for MSA questions but median 0.01
for Gulf-dialect ones, because the cross-encoder is far less certain about
dialect. Ranking is unaffected; the *absolute* value is not dialect-neutral.

Consequence for the "not in corpus" floor: a single global
``settings.rerank_min_score`` refuses roughly half of the answerable Gulf pairs
while excluding every distractor by a wide margin.

**That prediction was then measured at n=283 and it is worse than this note
assumed, so the floor is now off** — ``rerank_min_score = 0.0``. The old 0.15
refused 39 answerable questions (54% of Gulf pairs, 5.5% of MSA) to catch 7 of 15
unanswerable ones, and all 39 had the gold article already in the context window.
No threshold in a 0.00–0.90 sweep clears both a 10% false-refusal cap and a 50%
refusal-precision floor. The finding is that this score is not a relevance
detector at all; the ranked list it produces is still excellent. Sweep, dialect
table and the three ranked alternatives (model abstention, a top1-vs-top5 margin
feature, a dialect-aware threshold): ``docs/refusal-calibration.md``, reproducible
with ``python -m evals.refusal --sweep``.

``torch`` and ``sentence_transformers`` are imported *inside* ``_build_model``
on purpose: importing them at module scope costs seconds and hundreds of MB in
the API process and in every test run that never reranks anything.
"""

from __future__ import annotations

import asyncio
import dataclasses
import math
import threading
from collections.abc import Sequence
from functools import cache
from typing import Protocol

from app.retrieval.search import Hit

CROSS_ENCODER_MODEL = "BAAI/bge-reranker-v2-m3"
MAX_SEQUENCE_LENGTH = 512  # query + chunk; longer pairs are truncated by the tokenizer
RERANK_SOURCE = "rerank"


class Reranker(Protocol):
    """Reorders fused candidates. `rerank` never mutates the hits it is given."""

    name: str

    async def rerank(self, query: str, hits: Sequence[Hit], top_k: int = 5) -> list[Hit]: ...


def _sigmoid(logit: float) -> float:
    """Numerically stable logistic — `math.exp` overflows past ~709."""
    if logit >= 0:
        return 1.0 / (1.0 + math.exp(-logit))
    odds = math.exp(logit)
    return odds / (1.0 + odds)


def _check_top_k(top_k: int) -> None:
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")


class CrossEncoderReranker:
    """BAAI/bge-reranker-v2-m3. Weights load on first use and stay on the instance."""

    name = "bge"

    def __init__(
        self,
        model_name: str = CROSS_ENCODER_MODEL,
        max_length: int = MAX_SEQUENCE_LENGTH,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self._model = None
        self._load_lock = threading.Lock()

    def _build_model(self):
        import torch
        from sentence_transformers import CrossEncoder

        # activation_fn=Identity: sentence-transformers would otherwise apply this
        # model's configured Sigmoid itself, and squashing twice collapses the
        # scale (a clear non-match lands on 0.5000 instead of 0.00002). We own the
        # single sigmoid in `_score` so the number a threshold sees is defined here.
        return CrossEncoder(
            self.model_name,
            device=_device(),
            max_length=self.max_length,
            activation_fn=torch.nn.Identity(),
        )

    def _load(self):
        # Locked: `_score` runs in worker threads, so two concurrent cold-start
        # requests would otherwise both see `_model is None` and build two copies
        # of a ~2 GB model. Held for the whole build — it is paid once per process.
        with self._load_lock:
            if self._model is None:
                self._model = self._build_model()
            return self._model

    def _score(self, query: str, texts: Sequence[str]) -> list[float]:
        """Blocking: loads weights on first call and runs the forward pass."""
        logits = self._load().predict([(query, text) for text in texts])
        return [_sigmoid(float(logit)) for logit in logits]

    async def rerank(self, query: str, hits: Sequence[Hit], top_k: int = 5) -> list[Hit]:
        """Top-`top_k` hits by cross-encoder relevance, scores replaced by 0-1 values.

        Ties keep their incoming order (the sort is stable). The model call is a
        few hundred ms of blocking CPU/MPS work, so it runs in a worker thread —
        one /ask must not stall every other in-flight request.
        """
        _check_top_k(top_k)
        if not hits:
            return []

        scores = await asyncio.to_thread(self._score, query, [hit.text for hit in hits])
        ranked = sorted(zip(hits, scores), key=lambda pair: pair[1], reverse=True)
        return [
            dataclasses.replace(hit, score=score, source=RERANK_SOURCE)
            for hit, score in ranked[:top_k]
        ]


class NoopReranker:
    """Ablation baseline: keeps fusion's order, so the benchmark can isolate rerank gain."""

    name = "noop"

    async def rerank(self, query: str, hits: Sequence[Hit], top_k: int = 5) -> list[Hit]:
        _check_top_k(top_k)
        return list(hits[:top_k])


def _device() -> str:
    # trade-off: Apple Silicon dev box + CPU-only CI is the whole deployment surface
    # today, so cuda is deliberately not probed. Add it here when a GPU box exists.
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


_RERANKERS = {"bge": CrossEncoderReranker, "noop": NoopReranker}


@cache
def get_reranker(name: str = "bge") -> Reranker:
    """Cached per name — reranker instances hold ~2GB of weights; build one, reuse it."""
    try:
        return _RERANKERS[name]()
    except KeyError:
        raise ValueError(
            f"unknown reranker {name!r}; expected one of {sorted(_RERANKERS)}"
        ) from None
