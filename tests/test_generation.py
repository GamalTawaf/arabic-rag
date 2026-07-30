"""Unit tests for generation providers and failover.

No network and no API keys: both SDK clients are injected as fakes through the
`client=` seam, and the failover tests use hand-written `Provider` doubles so the
policy is exercised without either SDK in the picture. The one place real SDK
objects appear is error translation, where the whole point is that the mapping
matches the vendors' actual exception classes.

Retry backoff is zeroed by an autouse fixture - the tests assert *how many*
attempts happen, never how long they take.
"""

from __future__ import annotations

import asyncio
import json

import anthropic
import httpx
import pytest
from google.genai import errors as genai_errors
from opentelemetry.sdk.trace import TracerProvider

from app.config import settings
from app.generation import failover as failover_module
from app.generation.base import (
    Completion,
    ErrorKind,
    ProviderError,
    Usage,
    usage_cost_usd,
)
from app.generation.failover import AllProvidersFailed, FailoverProvider
from app.generation.providers import (
    ANTHROPIC_PRICES,
    CHARS_PER_TOKEN,
    GEMINI_PRICES,
    AnthropicProvider,
    GeminiProvider,
    HuggingFaceProvider,
    available_providers,
    get_provider,
)

HAIKU = "claude-haiku-4-5"
FLASH = "gemini-2.5-flash"
HF_MODEL = "Qwen/Qwen2.5-72B-Instruct"
HF_BASE = "https://router.huggingface.co/v1"
MESSAGES = [{"role": "user", "content": "ما هي مدة الإجازة السنوية؟"}]
SYSTEM = "أجب بالعربية الفصحى."


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    """Keep retry semantics, drop the wall-clock cost of jittered backoff."""
    monkeypatch.setattr(failover_module, "RETRY_MULTIPLIER_S", 0.0)
    monkeypatch.setattr(failover_module, "RETRY_MAX_WAIT_S", 0.0)


# --- Anthropic SDK doubles ---------------------------------------------------


class _TextBlock:
    def __init__(self, text: str):
        self.type = "text"
        self.text = text


class _SdkUsage:
    def __init__(self, input_tokens: int, output_tokens: int):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _SdkMessage:
    def __init__(self, text: str, input_tokens: int, output_tokens: int):
        self.content = [_TextBlock(text)]
        self.usage = _SdkUsage(input_tokens, output_tokens)


class _FakeAnthropicStream:
    def __init__(self, chunks, message, error=None):
        self._chunks = chunks
        self._message = message
        self._error = error

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    @property
    def text_stream(self):
        async def generate():
            for chunk in self._chunks:
                yield chunk
            if self._error is not None:
                raise self._error

        return generate()

    async def get_final_message(self):
        return self._message


class FakeAnthropicClient:
    """Stands in for AsyncAnthropic: `.messages.create` / `.messages.stream`."""

    def __init__(self, message=None, chunks=(), error=None, stream_error=None):
        self._message = message
        self._chunks = list(chunks)
        self._error = error
        self._stream_error = stream_error
        self.calls: list[dict] = []
        self.messages = self

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return self._message

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return _FakeAnthropicStream(self._chunks, self._message, self._stream_error)


# --- Gemini SDK doubles ------------------------------------------------------


class _GeminiUsage:
    def __init__(self, prompt: int | None, candidates: int | None):
        self.prompt_token_count = prompt
        self.candidates_token_count = candidates


class _GeminiChunk:
    def __init__(self, text: str | None, usage_metadata=None):
        self.text = text
        self.usage_metadata = usage_metadata


class _GeminiResponse:
    def __init__(self, text: str, usage_metadata=None):
        self.text = text
        self.usage_metadata = usage_metadata


class _FakeGeminiModels:
    def __init__(self, response=None, chunks=(), error=None):
        self._response = response
        self._chunks = list(chunks)
        self._error = error
        self.calls: list[dict] = []

    async def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return self._response

    async def generate_content_stream(self, **kwargs):
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        chunks = self._chunks

        async def generate():
            for chunk in chunks:
                yield chunk

        return generate()


class FakeGeminiClient:
    """Stands in for genai.Client: `.aio.models.generate_content[_stream]`."""

    def __init__(self, response=None, chunks=(), error=None):
        self.models = _FakeGeminiModels(response, chunks, error)
        self.aio = self


# --- Provider doubles for the failover tests ---------------------------------


class FakeProvider:
    """Scripted `Provider`. Each entry is raised if it is an exception, else returned."""

    def __init__(
        self,
        name: str,
        *,
        script=None,
        chunk_script=None,
        model: str = "fake-model",
        price: tuple[float, float] = (1.0, 2.0),
        delay_s: float = 0.0,
    ):
        self.name = name
        self.model = model
        self._price = price
        self._script = list(script or [])
        self._chunk_script = list(chunk_script or [])
        self.delay_s = delay_s
        self.complete_calls = 0
        self.stream_calls = 0
        self.closed_streams = 0

    def price(self) -> tuple[float, float]:
        return self._price

    def _completion(self, text: str) -> Completion:
        return Completion(
            text=text,
            usage=Usage(input_tokens=10, output_tokens=5, cost_usd=0.0),
            provider=self.name,
            model=self.model,
        )

    async def complete(self, system, messages, max_tokens=1024) -> Completion:
        self.complete_calls += 1
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        step = self._script.pop(0) if self._script else "ok"
        if isinstance(step, BaseException):
            raise step
        return self._completion(step)

    async def stream(self, system, messages, max_tokens=1024, *, on_usage=None):
        self.stream_calls += 1
        try:
            if self.delay_s:
                await asyncio.sleep(self.delay_s)
            steps = self._chunk_script.pop(0) if self._chunk_script else ["ok"]
            for step in steps:
                if isinstance(step, BaseException):
                    raise step
                yield step
        except GeneratorExit:  # only ever raised by an explicit aclose()
            self.closed_streams += 1
            raise
        if on_usage is not None:
            on_usage(Usage(input_tokens=10, output_tokens=5, cost_usd=0.25))


def _error(name: str, kind: ErrorKind) -> ProviderError:
    return ProviderError(name, kind, f"synthetic {kind.value}")


def _http_error(cls, status: int):
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return cls("boom", response=httpx.Response(status, request=request), body=None)


# --- Anthropic provider ------------------------------------------------------


async def test_anthropic_complete_returns_text_usage_and_attribution():
    # Arrange
    client = FakeAnthropicClient(message=_SdkMessage("إجازة سنوية", 1000, 200))
    provider = AnthropicProvider(model=HAIKU, api_key="k", client=client)

    # Act
    completion = await provider.complete(SYSTEM, MESSAGES, max_tokens=256)

    # Assert
    assert completion.text == "إجازة سنوية"
    assert completion.provider == "anthropic"
    assert completion.model == HAIKU
    assert completion.usage.input_tokens == 1000
    assert completion.usage.output_tokens == 200
    assert client.calls[0]["model"] == HAIKU
    assert client.calls[0]["max_tokens"] == 256
    assert client.calls[0]["system"] == SYSTEM


async def test_anthropic_complete_prices_real_token_counts():
    # Arrange: 1M input + 1M output at haiku's $1 / $5 per million.
    client = FakeAnthropicClient(message=_SdkMessage("x", 1_000_000, 1_000_000))
    provider = AnthropicProvider(model=HAIKU, api_key="k", client=client)

    # Act
    completion = await provider.complete(SYSTEM, MESSAGES)

    # Assert
    assert completion.usage.cost_usd == pytest.approx(6.0)


async def test_anthropic_stream_yields_chunks_and_reports_measured_usage():
    # Arrange
    client = FakeAnthropicClient(
        message=_SdkMessage("ignored", 300, 40), chunks=["مرحبا ", "بك"]
    )
    provider = AnthropicProvider(model=HAIKU, api_key="k", client=client)
    seen: list[Usage] = []

    # Act
    chunks = [c async for c in provider.stream(SYSTEM, MESSAGES, on_usage=seen.append)]

    # Assert
    assert chunks == ["مرحبا ", "بك"]
    assert seen[0].input_tokens == 300
    assert seen[0].output_tokens == 40
    assert seen[0].cost_usd == pytest.approx((300 * 1.0 + 40 * 5.0) / 1_000_000)


@pytest.mark.parametrize(
    ("exc", "kind"),
    [
        (_http_error(anthropic.RateLimitError, 429), ErrorKind.RATE_LIMIT),
        (_http_error(anthropic.InternalServerError, 500), ErrorKind.SERVER),
        (_http_error(anthropic.BadRequestError, 400), ErrorKind.FATAL),
        (_http_error(anthropic.AuthenticationError, 401), ErrorKind.FATAL),
        (_http_error(anthropic.PermissionDeniedError, 403), ErrorKind.FATAL),
    ],
)
async def test_anthropic_translates_status_errors_to_kinds(exc, kind):
    # Arrange
    provider = AnthropicProvider(
        model=HAIKU, api_key="k", client=FakeAnthropicClient(error=exc)
    )

    # Act / Assert
    with pytest.raises(ProviderError) as caught:
        await provider.complete(SYSTEM, MESSAGES)
    assert caught.value.kind is kind
    assert caught.value.provider == "anthropic"


async def test_anthropic_translates_timeout_and_connection_errors():
    # Arrange
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    timeout_provider = AnthropicProvider(
        model=HAIKU,
        api_key="k",
        client=FakeAnthropicClient(error=anthropic.APITimeoutError(request=request)),
    )
    connection_provider = AnthropicProvider(
        model=HAIKU,
        api_key="k",
        client=FakeAnthropicClient(
            error=anthropic.APIConnectionError(message="reset", request=request)
        ),
    )

    # Act / Assert
    with pytest.raises(ProviderError) as timed_out:
        await timeout_provider.complete(SYSTEM, MESSAGES)
    assert timed_out.value.kind is ErrorKind.TIMEOUT

    with pytest.raises(ProviderError) as refused:
        await connection_provider.complete(SYSTEM, MESSAGES)
    assert refused.value.kind is ErrorKind.CONNECTION


async def test_anthropic_translates_unknown_exception_as_fatal():
    # Arrange: an SDK-shaped bug must surface, not silently burn the next provider.
    provider = AnthropicProvider(
        model=HAIKU, api_key="k", client=FakeAnthropicClient(error=TypeError("oops"))
    )

    # Act / Assert
    with pytest.raises(ProviderError) as caught:
        await provider.complete(SYSTEM, MESSAGES)
    assert caught.value.kind is ErrorKind.FATAL
    assert caught.value.should_fail_over is False


async def test_anthropic_translates_an_error_raised_mid_stream():
    # Arrange: the failure lands after the SDK stream has already opened.
    client = FakeAnthropicClient(
        message=_SdkMessage("", 1, 1),
        chunks=["مر"],
        stream_error=_http_error(anthropic.InternalServerError, 503),
    )
    provider = AnthropicProvider(model=HAIKU, api_key="k", client=client)

    # Act / Assert
    with pytest.raises(ProviderError) as caught:
        [c async for c in provider.stream(SYSTEM, MESSAGES)]
    assert caught.value.kind is ErrorKind.SERVER


async def test_translation_does_not_double_wrap_a_provider_error():
    # Arrange: an already-classified error must keep its kind, not become FATAL.
    already = ProviderError("anthropic", ErrorKind.RATE_LIMIT, "pre-classified")
    client = FakeAnthropicClient(
        message=_SdkMessage("", 1, 1), chunks=[], stream_error=already
    )
    provider = AnthropicProvider(model=HAIKU, api_key="k", client=client)

    # Act / Assert
    with pytest.raises(ProviderError) as caught:
        [c async for c in provider.stream(SYSTEM, MESSAGES)]
    assert caught.value is already


def test_anthropic_builds_and_caches_a_real_sdk_client():
    # Arrange: constructing the SDK client touches no network.
    provider = AnthropicProvider(model=HAIKU, api_key="not-a-real-key")

    # Act
    client = provider._build_client()

    # Assert
    assert type(client).__name__ == "AsyncAnthropic"
    assert provider._build_client() is client  # built once, reused
    assert provider.price() == ANTHROPIC_PRICES[HAIKU]


def test_gemini_builds_and_caches_a_real_sdk_client():
    # Arrange
    provider = GeminiProvider(model=FLASH, api_key="not-a-real-key")

    # Act
    client = provider._build_client()

    # Assert
    assert type(client).__name__ == "Client"
    assert provider._build_client() is client
    assert provider.price() == GEMINI_PRICES[FLASH]


def test_anthropic_requires_an_api_key():
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        AnthropicProvider(model=HAIKU, api_key="")


def test_provider_rejects_a_model_with_no_published_price():
    with pytest.raises(ValueError, match="no published price"):
        AnthropicProvider(model="claude-imaginary-9", api_key="k")


# --- Gemini provider ---------------------------------------------------------


async def test_gemini_complete_maps_roles_and_reports_usage():
    # Arrange
    client = FakeGeminiClient(
        response=_GeminiResponse("جواب", _GeminiUsage(500, 100)),
    )
    provider = GeminiProvider(model=FLASH, api_key="k", client=client)
    history = [
        {"role": "user", "content": "س"},
        {"role": "assistant", "content": "ج"},
    ]

    # Act
    completion = await provider.complete(SYSTEM, history, max_tokens=128)

    # Assert
    assert completion.text == "جواب"
    assert completion.provider == "gemini"
    assert completion.usage.cost_usd == pytest.approx(
        (500 * 0.30 + 100 * 2.50) / 1_000_000
    )
    call = client.models.calls[0]
    assert [content["role"] for content in call["contents"]] == ["user", "model"]
    assert call["config"]["system_instruction"] == SYSTEM
    assert call["config"]["max_output_tokens"] == 128


async def test_gemini_stream_uses_the_last_usage_metadata_it_sees():
    # Arrange: google-genai reports cumulative counts on the final chunk.
    client = FakeGeminiClient(
        chunks=[
            _GeminiChunk("مر", _GeminiUsage(20, 1)),
            _GeminiChunk("حبا", _GeminiUsage(20, 7)),
        ]
    )
    provider = GeminiProvider(model=FLASH, api_key="k", client=client)
    seen: list[Usage] = []

    # Act
    chunks = [c async for c in provider.stream(SYSTEM, MESSAGES, on_usage=seen.append)]

    # Assert
    assert chunks == ["مر", "حبا"]
    assert (seen[0].input_tokens, seen[0].output_tokens) == (20, 7)


async def test_gemini_stream_without_usage_metadata_estimates_rather_than_zero():
    # Arrange: 40 characters, no metadata anywhere in the stream.
    text = "a" * 40
    client = FakeGeminiClient(chunks=[_GeminiChunk(text, None)])
    provider = GeminiProvider(model=FLASH, api_key="k", client=client)
    seen: list[Usage] = []

    # Act
    [c async for c in provider.stream(SYSTEM, MESSAGES, on_usage=seen.append)]

    # Assert
    assert seen[0].output_tokens == 40 // CHARS_PER_TOKEN  # derived, not hardcoded
    assert seen[0].cost_usd > 0  # never a silent zero - the spend cap must move


@pytest.mark.parametrize(
    ("exc", "kind"),
    [
        (genai_errors.ClientError(429, {"error": {"message": "quota"}}), ErrorKind.RATE_LIMIT),
        (genai_errors.ServerError(503, {"error": {"message": "down"}}), ErrorKind.SERVER),
        (genai_errors.ClientError(400, {"error": {"message": "bad"}}), ErrorKind.FATAL),
        (genai_errors.ClientError(403, {"error": {"message": "denied"}}), ErrorKind.FATAL),
        # `code` is Optional in google-genai: an error body with no status leaves
        # it None. It must classify as retriable, not crash the taxonomy.
        (genai_errors.APIError(None, {"error": {"message": "no status"}}), ErrorKind.SERVER),
    ],
)
async def test_gemini_translates_api_errors_to_kinds(exc, kind):
    # Arrange
    provider = GeminiProvider(model=FLASH, api_key="k", client=FakeGeminiClient(error=exc))

    # Act / Assert
    with pytest.raises(ProviderError) as caught:
        await provider.complete(SYSTEM, MESSAGES)
    assert caught.value.kind is kind
    assert caught.value.provider == "gemini"


@pytest.mark.parametrize(
    ("exc", "kind"),
    [
        (httpx.ReadTimeout("slow"), ErrorKind.TIMEOUT),
        (httpx.ConnectError("refused"), ErrorKind.CONNECTION),
        (RuntimeError("bug"), ErrorKind.FATAL),
        # already classified: passed through, never re-wrapped as FATAL
        (ProviderError("gemini", ErrorKind.RATE_LIMIT, "pre-classified"), ErrorKind.RATE_LIMIT),
    ],
)
async def test_gemini_translates_transport_and_unknown_errors(exc, kind):
    # Arrange
    provider = GeminiProvider(model=FLASH, api_key="k", client=FakeGeminiClient(error=exc))

    # Act / Assert
    with pytest.raises(ProviderError) as caught:
        await provider.complete(SYSTEM, MESSAGES)
    assert caught.value.kind is kind


async def test_gemini_translates_an_error_raised_mid_stream():
    # Arrange
    provider = GeminiProvider(
        model=FLASH,
        api_key="k",
        client=FakeGeminiClient(
            error=genai_errors.ServerError(500, {"error": {"message": "down"}})
        ),
    )

    # Act / Assert
    with pytest.raises(ProviderError) as caught:
        [c async for c in provider.stream(SYSTEM, MESSAGES)]
    assert caught.value.kind is ErrorKind.SERVER


async def test_gemini_estimates_when_metadata_reports_zero_output_tokens():
    # Arrange: metadata present but output count missing - still not a free answer.
    text = "b" * 20
    client = FakeGeminiClient(response=_GeminiResponse(text, _GeminiUsage(99, 0)))
    provider = GeminiProvider(model=FLASH, api_key="k", client=client)

    # Act
    completion = await provider.complete(SYSTEM, MESSAGES)

    # Assert
    assert completion.usage.input_tokens == 99  # the count we did get is kept
    assert completion.usage.output_tokens == 20 // CHARS_PER_TOKEN
    assert completion.usage.cost_usd > 0


def test_gemini_requires_an_api_key():
    with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
        GeminiProvider(model=FLASH, api_key="")


# --- Registry ----------------------------------------------------------------


def test_get_provider_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="unknown generation provider"):
        get_provider("llama")


def test_get_provider_builds_the_known_providers(monkeypatch):
    # Arrange
    monkeypatch.setattr(settings, "anthropic_api_key", "a")
    monkeypatch.setattr(settings, "google_api_key", "g")

    # Act / Assert
    assert get_provider("anthropic").name == "anthropic"
    assert get_provider("gemini").name == "gemini"


def test_available_providers_is_empty_without_keys(monkeypatch):
    # Arrange: the state of this machine - no generation keys anywhere.
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    monkeypatch.setattr(settings, "google_api_key", "")

    # Act / Assert
    assert available_providers() == []


@pytest.mark.parametrize(
    ("anthropic_key", "google_key", "expected"),
    [
        ("a", "g", ["anthropic", "gemini"]),
        ("", "g", ["gemini"]),
        ("a", "", ["anthropic"]),
    ],
)
def test_available_providers_lists_only_configured_keys(
    monkeypatch, anthropic_key, google_key, expected
):
    # Arrange
    monkeypatch.setattr(settings, "anthropic_api_key", anthropic_key)
    monkeypatch.setattr(settings, "google_api_key", google_key)

    # Act / Assert
    assert available_providers() == expected


def test_price_tables_cover_the_configured_models():
    assert settings.anthropic_model in ANTHROPIC_PRICES
    assert settings.gemini_model in GEMINI_PRICES


# --- Cost arithmetic ---------------------------------------------------------


@pytest.mark.parametrize(
    ("input_tokens", "output_tokens", "price", "expected"),
    [
        (0, 0, (1.0, 5.0), 0.0),
        (1_000_000, 0, (1.0, 5.0), 1.0),
        (0, 1_000_000, (1.0, 5.0), 5.0),
        (1_500, 500, (1.0, 5.0), 0.004),
        (10_000, 2_000, (0.30, 2.50), 0.008),
    ],
)
def test_usage_cost_usd(input_tokens, output_tokens, price, expected):
    assert usage_cost_usd(input_tokens, output_tokens, price) == pytest.approx(expected)


# --- Failover: complete ------------------------------------------------------


async def test_failover_uses_the_primary_when_it_answers():
    # Arrange
    primary = FakeProvider("primary", script=["first"])
    secondary = FakeProvider("secondary")
    provider = FailoverProvider([primary, secondary], timeout_s=1.0)

    # Act
    completion = await provider.complete(SYSTEM, MESSAGES)

    # Assert
    assert completion.text == "first"
    assert completion.provider == "primary"
    assert secondary.complete_calls == 0


async def test_failover_retries_the_same_provider_on_429_then_moves_on():
    # Arrange: three 429s exhausts MAX_ATTEMPTS on the primary.
    primary = FakeProvider(
        "primary", script=[_error("primary", ErrorKind.RATE_LIMIT)] * 3
    )
    secondary = FakeProvider("secondary", script=["rescued"])
    provider = FailoverProvider([primary, secondary], timeout_s=1.0)

    # Act
    completion = await provider.complete(SYSTEM, MESSAGES)

    # Assert
    assert primary.complete_calls == 3  # retried in place before failing over
    assert completion.text == "rescued"


async def test_failover_stops_retrying_as_soon_as_the_rate_limit_clears():
    # Arrange
    primary = FakeProvider(
        "primary", script=[_error("primary", ErrorKind.RATE_LIMIT), "recovered"]
    )
    provider = FailoverProvider([primary], timeout_s=1.0)

    # Act
    completion = await provider.complete(SYSTEM, MESSAGES)

    # Assert
    assert primary.complete_calls == 2
    assert completion.text == "recovered"


async def test_failover_does_not_retry_a_timeout_it_just_moves_on():
    # Arrange: the primary hangs past the deadline.
    primary = FakeProvider("primary", delay_s=0.5)
    secondary = FakeProvider("secondary", script=["fast"])
    provider = FailoverProvider([primary, secondary], timeout_s=0.02)

    # Act
    completion = await provider.complete(SYSTEM, MESSAGES)

    # Assert
    assert primary.complete_calls == 1  # a second timeout would cost the same again
    assert completion.text == "fast"


async def test_failover_does_not_fail_over_on_a_bad_request():
    # Arrange
    primary = FakeProvider("primary", script=[_error("primary", ErrorKind.FATAL)])
    secondary = FakeProvider("secondary")
    provider = FailoverProvider([primary, secondary], timeout_s=1.0)

    # Act / Assert
    with pytest.raises(ProviderError) as caught:
        await provider.complete(SYSTEM, MESSAGES)
    assert caught.value.kind is ErrorKind.FATAL
    assert primary.complete_calls == 1
    assert secondary.complete_calls == 0  # our bug; the next provider cannot help


async def test_failover_raises_all_providers_failed_naming_every_reason():
    # Arrange
    primary = FakeProvider("primary", delay_s=0.5)
    secondary = FakeProvider(
        "secondary", script=[_error("secondary", ErrorKind.CONNECTION)]
    )
    provider = FailoverProvider([primary, secondary], timeout_s=0.02)

    # Act / Assert
    with pytest.raises(AllProvidersFailed) as caught:
        await provider.complete(SYSTEM, MESSAGES)
    message = str(caught.value)
    assert "primary" in message and "timeout" in message
    assert "secondary" in message and "connection" in message
    assert [name for name, _reason in caught.value.failures] == ["primary", "secondary"]


async def test_failover_records_a_span_event_per_handoff():
    # Arrange
    primary = FakeProvider("primary", script=[_error("primary", ErrorKind.TIMEOUT)])
    secondary = FakeProvider("secondary", script=["ok"])
    provider = FailoverProvider([primary, secondary], timeout_s=1.0)
    tracer = TracerProvider().get_tracer(__name__)

    # Act
    with tracer.start_as_current_span("ask") as span:
        await provider.complete(SYSTEM, MESSAGES)

    # Assert
    events = [event for event in span.events if event.name == "provider.failover"]
    assert len(events) == 1
    assert events[0].attributes["from"] == "primary"
    assert events[0].attributes["to"] == "secondary"
    assert "timeout" in events[0].attributes["reason"]


async def test_failover_emits_no_span_event_when_nothing_is_left_to_try():
    # Arrange
    only = FakeProvider("only", script=[_error("only", ErrorKind.CONNECTION)])
    provider = FailoverProvider([only], timeout_s=1.0)
    tracer = TracerProvider().get_tracer(__name__)

    # Act / Assert
    with tracer.start_as_current_span("ask") as span, pytest.raises(AllProvidersFailed):
        await provider.complete(SYSTEM, MESSAGES)
    assert [event for event in span.events if event.name == "provider.failover"] == []


def test_failover_requires_at_least_one_provider():
    with pytest.raises(ValueError, match="at least one provider"):
        FailoverProvider([])


def test_failover_exposes_the_primary_as_its_identity():
    # Arrange
    primary = FakeProvider("primary", model="m1", price=(1.0, 2.0))
    provider = FailoverProvider([primary, FakeProvider("secondary")], timeout_s=1.0)

    # Assert
    assert provider.model == "m1"
    assert provider.price() == (1.0, 2.0)
    assert provider.name == "failover:primary,secondary"


def test_from_settings_builds_the_configured_chain(monkeypatch):
    # Arrange
    monkeypatch.setattr(settings, "anthropic_api_key", "a")
    monkeypatch.setattr(settings, "google_api_key", "g")
    monkeypatch.setattr(settings, "providers", "anthropic, gemini")

    # Act
    provider = FailoverProvider.from_settings()

    # Assert
    assert provider.name == "failover:anthropic,gemini"


def test_from_settings_rejects_an_empty_provider_list(monkeypatch):
    monkeypatch.setattr(settings, "providers", "  ")
    with pytest.raises(ValueError, match="settings.providers is empty"):
        FailoverProvider.from_settings()


# --- Failover: streaming -----------------------------------------------------


async def test_stream_fails_over_before_the_first_token():
    # Arrange
    primary = FakeProvider(
        "primary", chunk_script=[[_error("primary", ErrorKind.TIMEOUT)]]
    )
    secondary = FakeProvider("secondary", chunk_script=[["مر", "حبا"]])
    provider = FailoverProvider([primary, secondary], timeout_s=1.0)

    # Act
    chunks = [c async for c in provider.stream(SYSTEM, MESSAGES)]

    # Assert
    assert chunks == ["مر", "حبا"]
    assert primary.stream_calls == 1


async def test_stream_does_not_fail_over_after_the_first_token():
    # Arrange: the token is already on the wire, so a swap cannot be transparent.
    primary = FakeProvider(
        "primary", chunk_script=[["مر", _error("primary", ErrorKind.SERVER)]]
    )
    secondary = FakeProvider("secondary", chunk_script=[["never"]])
    provider = FailoverProvider([primary, secondary], timeout_s=1.0)
    received: list[str] = []

    # Act / Assert
    with pytest.raises(ProviderError) as caught:
        async for chunk in provider.stream(SYSTEM, MESSAGES):
            received.append(chunk)
    assert received == ["مر"]  # partial output stays delivered
    assert caught.value.kind is ErrorKind.SERVER  # propagated, not wrapped
    assert secondary.stream_calls == 0


async def test_stream_retries_the_same_provider_before_the_first_token():
    # Arrange
    primary = FakeProvider(
        "primary",
        chunk_script=[
            [_error("primary", ErrorKind.SERVER)],
            [_error("primary", ErrorKind.SERVER)],
            ["late"],
        ],
    )
    provider = FailoverProvider([primary], timeout_s=1.0)

    # Act
    chunks = [c async for c in provider.stream(SYSTEM, MESSAGES)]

    # Assert
    assert primary.stream_calls == 3
    assert chunks == ["late"]


async def test_stream_closes_the_provider_when_the_consumer_disconnects():
    # Arrange: an SSE client that hangs up after the first token.
    primary = FakeProvider("primary", chunk_script=[["مر", "حبا", "بك"]])
    provider = FailoverProvider([primary], timeout_s=1.0)

    # Act
    stream = provider.stream(SYSTEM, MESSAGES)
    received = [await stream.__anext__()]
    await stream.aclose()

    # Assert: the upstream generator is closed, not left holding an open response.
    assert received == ["مر"]
    assert primary.closed_streams == 1


async def test_stream_surfaces_a_bad_request_without_failing_over():
    # Arrange
    primary = FakeProvider("primary", chunk_script=[[_error("primary", ErrorKind.FATAL)]])
    secondary = FakeProvider("secondary", chunk_script=[["never"]])
    provider = FailoverProvider([primary, secondary], timeout_s=1.0)

    # Act / Assert
    with pytest.raises(ProviderError) as caught:
        [c async for c in provider.stream(SYSTEM, MESSAGES)]
    assert caught.value.kind is ErrorKind.FATAL
    assert secondary.stream_calls == 0


async def test_stream_raises_all_providers_failed_when_none_produce_a_token():
    # Arrange
    primary = FakeProvider("primary", chunk_script=[[_error("primary", ErrorKind.TIMEOUT)]])
    secondary = FakeProvider(
        "secondary", chunk_script=[[_error("secondary", ErrorKind.CONNECTION)]]
    )
    provider = FailoverProvider([primary, secondary], timeout_s=1.0)

    # Act / Assert
    with pytest.raises(AllProvidersFailed) as caught:
        [c async for c in provider.stream(SYSTEM, MESSAGES)]
    assert [name for name, _reason in caught.value.failures] == ["primary", "secondary"]


async def test_stream_of_an_empty_response_is_not_treated_as_a_failure():
    # Arrange: a provider that returns no tokens at all is a (bad) answer, not
    # an error - failing over would double the spend for the same empty result.
    primary = FakeProvider("primary", chunk_script=[[]])
    secondary = FakeProvider("secondary", chunk_script=[["never"]])
    provider = FailoverProvider([primary, secondary], timeout_s=1.0)

    # Act
    chunks = [c async for c in provider.stream(SYSTEM, MESSAGES)]

    # Assert
    assert chunks == []
    assert secondary.stream_calls == 0


async def test_stream_forwards_the_usage_callback_to_the_serving_provider():
    # Arrange
    primary = FakeProvider("primary", chunk_script=[[_error("primary", ErrorKind.TIMEOUT)]])
    secondary = FakeProvider("secondary", chunk_script=[["ok"]])
    provider = FailoverProvider([primary, secondary], timeout_s=1.0)
    seen: list[Usage] = []

    # Act
    [c async for c in provider.stream(SYSTEM, MESSAGES, on_usage=seen.append)]

    # Assert
    assert seen == [Usage(input_tokens=10, output_tokens=5, cost_usd=0.25)]


# --- Hugging Face router ------------------------------------------------------
#
# No SDK doubles here: this adapter speaks the OpenAI-compatible HTTP shape
# directly, so the seam is an `httpx.MockTransport` and the tests assert on the
# wire format the router actually returns.


def _hf(handler, **kwargs) -> HuggingFaceProvider:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url=HF_BASE)
    return HuggingFaceProvider(model=HF_MODEL, api_key="k", client=client, **kwargs)


def _hf_body(text: str, prompt: int, completion: int) -> dict:
    return {
        "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion},
    }


def _sse(*payloads: dict) -> bytes:
    lines = [f"data: {json.dumps(p)}\n\n" for p in payloads]
    return ("".join(lines) + "data: [DONE]\n\n").encode()


def _delta(text: str) -> dict:
    return {"choices": [{"delta": {"content": text}}]}


async def test_hf_complete_returns_text_usage_and_attribution():
    # Arrange
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_hf_body("إجازة سنوية", 1000, 200))

    provider = _hf(handler)

    # Act
    completion = await provider.complete(SYSTEM, MESSAGES, max_tokens=64)

    # Assert
    assert completion.text == "إجازة سنوية"
    assert completion.provider == "huggingface"
    assert completion.model == HF_MODEL
    assert completion.usage.input_tokens == 1000
    assert completion.usage.output_tokens == 200
    # The system prompt becomes a system message, not a prefix on the user turn.
    assert seen["body"]["messages"][0] == {"role": "system", "content": SYSTEM}
    assert seen["body"]["messages"][1] == MESSAGES[0]
    assert seen["body"]["max_tokens"] == 64
    assert seen["auth"] == "Bearer k"
    assert seen["url"].endswith("/chat/completions")


async def test_hf_complete_prices_with_the_configured_rate():
    # Arrange: the rate is settings, not a table - so assert it is actually read.
    provider = _hf(
        lambda r: httpx.Response(200, json=_hf_body("x", 1_000_000, 1_000_000)),
        price=(0.25, 1.50),
    )

    # Act
    completion = await provider.complete(SYSTEM, MESSAGES)

    # Assert
    assert completion.usage.cost_usd == pytest.approx(1.75)


async def test_hf_stream_yields_deltas_and_reports_usage_from_the_final_chunk():
    # Arrange: include_usage puts the totals on a trailing chunk with no choices.
    body = _sse(
        _delta("مدة "),
        _delta("الإجازة"),
        {"choices": [], "usage": {"prompt_tokens": 800, "completion_tokens": 12}},
    )
    provider = _hf(lambda r: httpx.Response(200, content=body))
    seen: list[Usage] = []

    # Act
    chunks = [c async for c in provider.stream(SYSTEM, MESSAGES, on_usage=seen.append)]

    # Assert
    assert chunks == ["مدة ", "الإجازة"]
    assert seen[0].input_tokens == 800
    assert seen[0].output_tokens == 12
    assert seen[0].provider == "huggingface"


async def test_hf_stream_asks_the_router_for_usage():
    # Arrange: without stream_options the router omits usage and every streamed
    # answer would be billed off a character estimate.
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, content=_sse(_delta("x")))

    provider = _hf(handler)

    # Act
    [c async for c in provider.stream(SYSTEM, MESSAGES)]

    # Assert
    assert seen["body"]["stream"] is True
    assert seen["body"]["stream_options"] == {"include_usage": True}


async def test_hf_stream_without_usage_estimates_rather_than_zero():
    # Arrange
    text = "كلمات" * 20
    provider = _hf(lambda r: httpx.Response(200, content=_sse(_delta(text))))
    seen: list[Usage] = []

    # Act
    [c async for c in provider.stream(SYSTEM, MESSAGES, on_usage=seen.append)]

    # Assert
    assert seen[0].output_tokens == len(text) // CHARS_PER_TOKEN
    assert seen[0].cost_usd > 0


async def test_hf_ignores_keepalive_and_unparsable_sse_lines():
    # Arrange: the router interleaves ": keep-alive" comments; one bad frame must
    # not abort a stream that is otherwise fine.
    body = (
        b": keep-alive\n\n"
        + b"data: not-json\n\n"
        + f"data: {json.dumps(_delta('نعم'))}\n\n".encode()
        + b"data: [DONE]\n\n"
    )
    provider = _hf(lambda r: httpx.Response(200, content=body))

    # Act
    chunks = [c async for c in provider.stream(SYSTEM, MESSAGES)]

    # Assert
    assert chunks == ["نعم"]


@pytest.mark.parametrize(
    "status,kind",
    [
        (429, ErrorKind.RATE_LIMIT),
        (500, ErrorKind.SERVER),
        (503, ErrorKind.SERVER),
        (401, ErrorKind.FATAL),
        (400, ErrorKind.FATAL),
    ],
)
async def test_hf_translates_status_codes_to_kinds(status, kind):
    # Arrange
    provider = _hf(lambda r: httpx.Response(status, text="nope"))

    # Act / Assert
    with pytest.raises(ProviderError) as excinfo:
        await provider.complete(SYSTEM, MESSAGES)
    assert excinfo.value.kind is kind


async def test_hf_translates_status_codes_on_the_streaming_path_too():
    # Arrange: a 429 arriving before the first byte of the body must still be a
    # classified ProviderError, not an unhandled httpx error mid-generator.
    provider = _hf(lambda r: httpx.Response(429, text="slow down"))

    # Act / Assert
    with pytest.raises(ProviderError) as excinfo:
        [c async for c in provider.stream(SYSTEM, MESSAGES)]
    assert excinfo.value.kind is ErrorKind.RATE_LIMIT


@pytest.mark.parametrize(
    "exc,kind",
    [
        (httpx.ReadTimeout("slow"), ErrorKind.TIMEOUT),
        (httpx.ConnectError("down"), ErrorKind.CONNECTION),
    ],
)
async def test_hf_translates_transport_errors(exc, kind):
    # Arrange
    def handler(request: httpx.Request) -> httpx.Response:
        raise exc

    provider = _hf(handler)

    # Act / Assert
    with pytest.raises(ProviderError) as excinfo:
        await provider.complete(SYSTEM, MESSAGES)
    assert excinfo.value.kind is kind


async def test_hf_treats_a_malformed_200_as_a_server_error():
    # Arrange: a 200 whose body is not the promised shape should fail over, not
    # surface as a 500 while a healthy provider sits unused.
    provider = _hf(lambda r: httpx.Response(200, json={"choices": []}))

    # Act / Assert
    with pytest.raises(ProviderError) as excinfo:
        await provider.complete(SYSTEM, MESSAGES)
    assert excinfo.value.kind is ErrorKind.SERVER


def test_hf_requires_an_api_key():
    # Act / Assert
    with pytest.raises(ValueError, match="HF_API_KEY"):
        HuggingFaceProvider(api_key="")


def test_hf_is_reachable_through_the_provider_registry():
    # Arrange / Act
    provider = get_provider("huggingface", api_key="k")

    # Assert
    assert provider.name == "huggingface"


def test_available_providers_reports_huggingface_when_its_key_is_set(monkeypatch):
    # Arrange
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    monkeypatch.setattr(settings, "google_api_key", "")
    monkeypatch.setattr(settings, "hf_api_key", "hf_x")

    # Act / Assert
    assert available_providers() == ["huggingface"]
