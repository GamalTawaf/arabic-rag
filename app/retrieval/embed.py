"""Embedding providers — one class per benchmarked model, one interface.

Every embedder returns **L2-normalized** vectors, so cosine distance and inner
product rank identically and pgvector's `<=>` on the HNSW cosine indexes agrees
with a plain dot product. sentence-transformers does it via
`normalize_embeddings=True`; API results are normalized here by hand.

There are two embed methods rather than one because the models are asymmetric:
multilingual-e5 was trained with a "query: " / "passage: " prefix and loses
measurable recall without it, and Cohere embed-v4 wants
`input_type=search_query` vs `search_document`. BGE-m3 and OpenAI take neither,
so the asymmetry has to live per-model instead of in the caller.

trade-off: torch / sentence_transformers are imported inside the methods that need
them. The FastAPI service and the test suite must stay importable without a
2 GB ML stack loaded. Ceiling: the first local query pays the model load.
"""

from __future__ import annotations

import asyncio
import math
import threading
from typing import Any, Protocol

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

from app.config import settings
from app.constants import (
    API_BATCH,
    API_TIMEOUT_S,
    EMBEDDING_DIMS,
    LOCAL_MODELS,
    MAX_TRIES,
    PREFIXES,
    ST_BATCH,
)


class Embedder(Protocol):
    """Anything that turns Arabic text into unit vectors for one model."""

    model_key: str  # a key of app.models.chunks.EMBEDDING_COLUMNS
    dim: int

    async def embed_passages(self, texts: list[str]) -> list[list[float]]: ...

    async def embed_queries(self, texts: list[str]) -> list[list[float]]: ...


class _IngestCompat:
    """`ingestion.pipeline.Embedder` calls `embed()`; it only ever embeds corpus text.

    trade-off: an alias instead of touching pipeline.py, whose protocol predates the
    query/passage split. Ceiling: a caller could embed a *query* through it and
    silently get the passage prefix. Upgrade path: widen pipeline's protocol to
    `embed_passages` and delete this.
    """

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return await self.embed_passages(texts)  # type: ignore[attr-defined]


class LocalEmbedder(_IngestCompat):
    """sentence-transformers model, loaded lazily and cached on the instance."""

    def __init__(self, model_key: str) -> None:
        if model_key not in LOCAL_MODELS:
            raise ValueError(
                f"unknown local model_key {model_key!r}; "
                f"expected one of {sorted(LOCAL_MODELS)}"
            )
        self.model_key = model_key
        self.model_name = LOCAL_MODELS[model_key]
        self.dim = EMBEDDING_DIMS[model_key]
        self.query_prefix, self.passage_prefix = PREFIXES[model_key]
        self._model: Any = None
        self._load_lock = threading.Lock()

    def load(self) -> Any:
        """Load the model once (first call pays ~10s + weights) and cache it.

        Locked: two concurrent cold-start requests would otherwise both see
        `_model is None` and build two copies of a multi-GB model. The lock is
        held for the whole load because it is paid once per process.
        """
        with self._load_lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name, device=_device())
            return self._model

    async def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return await self._encode(texts, self.passage_prefix)

    async def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return await self._encode(texts, self.query_prefix)

    def _load_and_encode(self, texts: list[str], prefix: str) -> Any:
        """Blocking: loads weights on first call and runs the forward pass."""
        return self.load().encode(
            [prefix + text for text in texts],
            batch_size=ST_BATCH,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    async def _encode(self, texts: list[str], prefix: str) -> list[list[float]]:
        if not texts:
            return []
        # to_thread: both the ~10s first-call weight load and encode() are
        # blocking, and the service embeds queries on the request path.
        vectors = await asyncio.to_thread(self._load_and_encode, texts, prefix)
        return _check_dims([[float(x) for x in row] for row in vectors], self.model_key)


class OpenAIEmbedder(_IngestCompat):
    """text-embedding-3-large over the REST API. No prefixes; symmetric model."""

    URL = "https://api.openai.com/v1/embeddings"
    MODEL_NAME = "text-embedding-3-large"

    def __init__(
        self, api_key: str | None = None, transport: httpx.AsyncBaseTransport | None = None
    ) -> None:
        self.model_key = "openai"
        self.dim = EMBEDDING_DIMS["openai"]
        self._api_key = api_key if api_key is not None else settings.openai_api_key
        self._transport = transport  # test seam; None means a real network client
        if not self._api_key:
            raise ValueError(
                "OpenAIEmbedder needs an API key: set OPENAI_API_KEY in the "
                "environment or .env (see available_embedders())"
            )

    async def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return await self._embed(texts)

    async def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return await self._embed(texts)

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        headers = {"Authorization": f"Bearer {self._api_key}"}
        vectors: list[list[float]] = []
        async with _client(self._transport) as client:
            for batch in _batches(texts):
                payload = {"model": self.MODEL_NAME, "input": batch}
                body = await _post_json(client, self.URL, headers, payload)
                items = sorted(body["data"], key=lambda item: item["index"])
                vectors.extend(_l2_normalize(item["embedding"]) for item in items)
        return _check_dims(vectors, self.model_key)


class CohereEmbedder(_IngestCompat):
    """Cohere embed-v4 over the REST API. Asymmetry is the `input_type` field."""

    URL = "https://api.cohere.com/v2/embed"
    MODEL_NAME = "embed-v4.0"

    def __init__(
        self, api_key: str | None = None, transport: httpx.AsyncBaseTransport | None = None
    ) -> None:
        self.model_key = "cohere"
        self.dim = EMBEDDING_DIMS["cohere"]
        self._api_key = api_key if api_key is not None else settings.cohere_api_key
        self._transport = transport  # test seam; None means a real network client
        if not self._api_key:
            raise ValueError(
                "CohereEmbedder needs an API key: set COHERE_API_KEY in the "
                "environment or .env (see available_embedders())"
            )

    async def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return await self._embed(texts, "search_document")

    async def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return await self._embed(texts, "search_query")

    async def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        if not texts:
            return []
        headers = {"Authorization": f"Bearer {self._api_key}"}
        vectors: list[list[float]] = []
        async with _client(self._transport) as client:
            for batch in _batches(texts):
                payload = {
                    "model": self.MODEL_NAME,
                    "texts": batch,
                    "input_type": input_type,
                    "embedding_types": ["float"],
                    "output_dimension": self.dim,
                }
                body = await _post_json(client, self.URL, headers, payload)
                vectors.extend(
                    _l2_normalize(vector) for vector in body["embeddings"]["float"]
                )
        return _check_dims(vectors, self.model_key)


def get_embedder(model_key: str) -> Embedder:
    """Build the embedder for one benchmarked model. Raises ValueError if unknown."""
    if model_key in LOCAL_MODELS:
        return LocalEmbedder(model_key)
    if model_key == "openai":
        return OpenAIEmbedder()
    if model_key == "cohere":
        return CohereEmbedder()
    raise ValueError(
        f"unknown embedding model_key {model_key!r}; "
        f"expected one of {sorted(EMBEDDING_DIMS)}"
    )


def available_embedders() -> list[str]:
    """Model keys usable right now: local always, API models only with a key set."""
    keys = list(LOCAL_MODELS)
    if settings.openai_api_key:
        keys.append("openai")
    if settings.cohere_api_key:
        keys.append("cohere")
    return keys


def _device() -> str:
    import torch

    # Apple Silicon first, CPU otherwise. No cuda on this machine or on Cloud Run.
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _batches(texts: list[str]) -> list[list[str]]:
    return [texts[start : start + API_BATCH] for start in range(0, len(texts), API_BATCH)]


def _client(transport: httpx.AsyncBaseTransport | None) -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=API_TIMEOUT_S, transport=transport)


def _is_retryable(exc: BaseException) -> bool:
    """429 and 5xx are worth another try; 4xx (bad key, bad request) never are."""
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status == 429 or status >= 500
    return isinstance(exc, httpx.TransportError)  # timeouts, connection resets


@retry(
    reraise=True,
    stop=stop_after_attempt(MAX_TRIES),
    wait=wait_random_exponential(multiplier=0.5, max=8),
    retry=retry_if_exception(_is_retryable),
)
async def _post_json(
    client: httpx.AsyncClient, url: str, headers: dict[str, str], payload: dict[str, Any]
) -> dict[str, Any]:
    response = await client.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0.0:
        return [float(x) for x in vector]  # degenerate; nothing to scale
    return [float(x) / norm for x in vector]


def _check_dims(vectors: list[list[float]], model_key: str) -> list[list[float]]:
    expected = EMBEDDING_DIMS[model_key]
    for vector in vectors:
        if len(vector) != expected:
            raise ValueError(
                f"embedder {model_key!r} returned dim {len(vector)}, "
                f"expected dim {expected}"
            )
    return vectors
