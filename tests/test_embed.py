"""Embedder unit tests. No network, no API keys, no model weights loaded."""

import httpx
import pytest

from app.retrieval import embed as embed_module
from app.retrieval.embed import (
    CohereEmbedder,
    LocalEmbedder,
    OpenAIEmbedder,
    available_embedders,
    get_embedder,
)

QUESTION = "ما هي مدة الإجازة السنوية للعامل؟"
PASSAGE = "للعامل الحق في إجازة سنوية مدفوعة الأجر لا تقل عن ثلاثة أسابيع."


class FakeModel:
    """Stands in for a SentenceTransformer; records what encode() was given."""

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim
        self.texts: list[str] = []
        self.kwargs: dict = {}

    def encode(self, texts, **kwargs):
        self.texts = list(texts)
        self.kwargs = kwargs
        return [[1.0] + [0.0] * (self.dim - 1) for _ in texts]


def local(model_key: str, dim: int = 1024) -> tuple[LocalEmbedder, FakeModel]:
    """A LocalEmbedder wired to a fake model, so no weights are ever loaded."""
    embedder = LocalEmbedder(model_key)
    model = FakeModel(dim)
    embedder._model = model
    return embedder, model


def mock_transport(handler) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


@pytest.fixture(autouse=True)
def no_api_keys(monkeypatch):
    """Default every test to the real local situation: no keys configured."""
    monkeypatch.setattr(embed_module.settings, "openai_api_key", "")
    monkeypatch.setattr(embed_module.settings, "cohere_api_key", "")


# --- get_embedder / available_embedders -------------------------------------


@pytest.mark.parametrize(
    ("model_key", "model_name"),
    [("e5", "intfloat/multilingual-e5-large"), ("bge", "BAAI/bge-m3")],
)
def test_get_embedder_returns_local_model_with_right_dim(model_key, model_name):
    # Act
    embedder = get_embedder(model_key)

    # Assert
    assert isinstance(embedder, LocalEmbedder)
    assert (embedder.model_key, embedder.dim, embedder.model_name) == (
        model_key,
        1024,
        model_name,
    )


def test_get_embedder_returns_openai_class_and_dim_when_key_present(monkeypatch):
    # Arrange
    monkeypatch.setattr(embed_module.settings, "openai_api_key", "sk-test")

    # Act
    embedder = get_embedder("openai")

    # Assert
    assert isinstance(embedder, OpenAIEmbedder)
    assert (embedder.model_key, embedder.dim) == ("openai", 3072)


def test_get_embedder_returns_cohere_class_and_dim_when_key_present(monkeypatch):
    # Arrange
    monkeypatch.setattr(embed_module.settings, "cohere_api_key", "co-test")

    # Act
    embedder = get_embedder("cohere")

    # Assert
    assert isinstance(embedder, CohereEmbedder)
    assert (embedder.model_key, embedder.dim) == ("cohere", 1536)


def test_get_embedder_with_unknown_key_raises():
    # Act / Assert
    with pytest.raises(ValueError, match="unknown embedding model_key 'voyage'"):
        get_embedder("voyage")


def test_available_embedders_excludes_api_models_when_keys_are_absent():
    # Act / Assert
    assert available_embedders() == ["e5", "bge"]


def test_available_embedders_includes_api_models_when_keys_are_set(monkeypatch):
    # Arrange
    monkeypatch.setattr(embed_module.settings, "openai_api_key", "sk-test")
    monkeypatch.setattr(embed_module.settings, "cohere_api_key", "co-test")

    # Act / Assert
    assert available_embedders() == ["e5", "bge", "openai", "cohere"]


# --- asymmetric prefixes ----------------------------------------------------


async def test_e5_prefixes_queries_with_query_and_passages_with_passage():
    # Arrange
    embedder, model = local("e5")

    # Act
    await embedder.embed_queries([QUESTION])
    query_texts = model.texts
    await embedder.embed_passages([PASSAGE])

    # Assert
    assert query_texts == [f"query: {QUESTION}"]
    assert model.texts == [f"passage: {PASSAGE}"]


async def test_bge_sends_text_unprefixed_in_both_directions():
    # Arrange
    embedder, model = local("bge")

    # Act
    await embedder.embed_queries([QUESTION])
    query_texts = model.texts
    await embedder.embed_passages([PASSAGE])

    # Assert
    assert query_texts == [QUESTION]
    assert model.texts == [PASSAGE]


async def test_local_encode_asks_for_normalized_vectors_in_batches_of_16():
    # Arrange
    embedder, model = local("e5")

    # Act
    await embedder.embed_passages([PASSAGE])

    # Assert
    assert model.kwargs["normalize_embeddings"] is True
    assert model.kwargs["batch_size"] == 16


async def test_local_embed_alias_uses_the_passage_prefix():
    # Arrange — ingestion.pipeline's protocol calls embed(), and it embeds corpus text
    embedder, model = local("e5")

    # Act
    vectors = await embedder.embed([PASSAGE])

    # Assert
    assert model.texts == [f"passage: {PASSAGE}"]
    assert len(vectors) == 1


async def test_local_embed_of_empty_list_never_loads_the_model():
    # Arrange
    embedder = LocalEmbedder("e5")

    # Act / Assert — a real load() here would download/read weights
    assert await embedder.embed_passages([]) == []
    assert embedder._model is None


def test_local_embedder_rejects_a_non_local_model_key():
    # Act / Assert
    with pytest.raises(ValueError, match="unknown local model_key 'openai'"):
        LocalEmbedder("openai")


async def test_local_dim_mismatch_raises_naming_both_dims():
    # Arrange
    embedder, _ = local("e5", dim=8)

    # Act / Assert
    with pytest.raises(ValueError, match=r"returned dim 8, expected dim 1024"):
        await embedder.embed_passages([PASSAGE])


# --- API embedders ----------------------------------------------------------


def test_openai_embedder_without_a_key_raises_a_clear_error():
    # Act / Assert
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIEmbedder()


def test_cohere_embedder_without_a_key_raises_a_clear_error():
    # Act / Assert
    with pytest.raises(ValueError, match="COHERE_API_KEY"):
        CohereEmbedder()


def test_get_embedder_for_api_model_without_a_key_raises():
    # Act / Assert
    with pytest.raises(ValueError, match="COHERE_API_KEY"):
        get_embedder("cohere")


async def test_openai_l2_normalizes_the_returned_vectors():
    # Arrange — an unnormalized vector of norm 3.0
    raw = [3.0] + [0.0] * 3071
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": raw}]})

    embedder = OpenAIEmbedder(api_key="sk-test", transport=mock_transport(handler))

    # Act
    vectors = await embedder.embed_queries([QUESTION])

    # Assert
    assert pytest.approx(sum(x * x for x in vectors[0]) ** 0.5) == 1.0
    assert vectors[0][0] == pytest.approx(1.0)
    assert requests[0].headers["authorization"] == "Bearer sk-test"


async def test_openai_reorders_embeddings_by_index():
    # Arrange — the API may return items out of order
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [0.0, 2.0] + [0.0] * 3070},
                    {"index": 0, "embedding": [2.0, 0.0] + [0.0] * 3070},
                ]
            },
        )

    embedder = OpenAIEmbedder(api_key="sk-test", transport=mock_transport(handler))

    # Act
    vectors = await embedder.embed_passages([PASSAGE, QUESTION])

    # Assert
    assert vectors[0][0] == pytest.approx(1.0)
    assert vectors[1][1] == pytest.approx(1.0)


async def test_cohere_sends_search_query_and_search_document_input_types():
    # Arrange
    payloads: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        payloads.append(json.loads(request.content))
        return httpx.Response(
            200, json={"embeddings": {"float": [[1.0] + [0.0] * 1535]}}
        )

    embedder = CohereEmbedder(api_key="co-test", transport=mock_transport(handler))

    # Act
    await embedder.embed_queries([QUESTION])
    await embedder.embed_passages([PASSAGE])

    # Assert
    assert [payload["input_type"] for payload in payloads] == [
        "search_query",
        "search_document",
    ]
    assert payloads[0]["model"] == "embed-v4.0"
    assert payloads[0]["output_dimension"] == 1536


async def test_cohere_dim_mismatch_raises_naming_both_dims():
    # Arrange
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"embeddings": {"float": [[1.0, 0.0, 0.0]]}})

    embedder = CohereEmbedder(api_key="co-test", transport=mock_transport(handler))

    # Act / Assert
    with pytest.raises(ValueError, match=r"returned dim 3, expected dim 1536"):
        await embedder.embed_queries([QUESTION])


async def test_api_embedder_retries_a_429_and_then_succeeds():
    # Arrange
    attempts = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        if len(attempts) == 1:
            return httpx.Response(429, json={"error": "rate limited"})
        return httpx.Response(
            200, json={"data": [{"index": 0, "embedding": [1.0] + [0.0] * 3071}]}
        )

    embedder = OpenAIEmbedder(api_key="sk-test", transport=mock_transport(handler))

    # Act
    vectors = await embedder.embed_queries([QUESTION])

    # Assert
    assert len(attempts) == 2
    assert len(vectors[0]) == 3072


async def test_api_embedder_does_not_retry_an_auth_failure():
    # Arrange
    attempts = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        return httpx.Response(401, json={"error": "invalid api key"})

    embedder = OpenAIEmbedder(api_key="sk-bad", transport=mock_transport(handler))

    # Act / Assert
    with pytest.raises(httpx.HTTPStatusError):
        await embedder.embed_queries([QUESTION])
    assert len(attempts) == 1


async def test_api_embedders_skip_the_network_for_an_empty_batch():
    # Arrange
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("should not call the API for zero texts")

    openai = OpenAIEmbedder(api_key="sk-test", transport=mock_transport(handler))
    cohere = CohereEmbedder(api_key="co-test", transport=mock_transport(handler))

    # Act / Assert
    assert await openai.embed_passages([]) == []
    assert await cohere.embed_queries([]) == []
