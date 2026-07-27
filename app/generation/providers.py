"""Anthropic and Gemini adapters - two SDKs, one `Provider` interface.

Each adapter owns three things its SDK makes specific: the request shape, the
usage fields, and the exception hierarchy. Everything downstream
(`FailoverProvider`, `/ask`, cost accounting) sees only `Completion`, `Usage`,
and `ProviderError`.

**Pricing.** The tables below are published list prices in USD per one million
tokens, **checked 2026-07** against the vendors' public pricing pages
(platform.claude.com/docs/en/pricing and ai.google.dev/gemini-api/docs/pricing).
They are keyed by exact model id, including the dated Anthropic snapshot, so a
model swap fails loudly at construction instead of silently reporting last
year's cost. Neither table covers cache-read/cache-write discounts or Gemini's
long-context tier - see the trade-off note on `_lookup_price`.

**SDK imports live inside methods.** The service must stay importable, and the
test suite runnable, when only one vendor SDK is installed - and importing
`google.genai` drags in pydantic, httpx and google.auth for a process that may
only ever call Anthropic. Ceiling: the first call of each kind pays the import.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Any

from app.config import settings
from app.generation.base import (
    Completion,
    ErrorKind,
    Provider,
    ProviderError,
    Usage,
    usage_cost_usd,
)
from app.generation.budget import CHARS_PER_TOKEN as _CHARS_PER_TOKEN

# model id -> (USD / 1M input tokens, USD / 1M output tokens). Checked 2026-07.
ANTHROPIC_PRICES: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-haiku-4-5-20251001": (1.00, 5.00),
}

# Checked 2026-07. Gemini 2.5 Flash text rates; output includes thinking tokens.
GEMINI_PRICES: dict[str, tuple[float, float]] = {
    "gemini-2.5-flash": (0.30, 2.50),
}

#: Re-exported, not redefined. This module used to carry its own
#: ``CHARS_PER_TOKEN = 4`` ("the usual English ballpark") next to budget.py's
#: researched Arabic value of 3, so the two halves of the same cost calculation
#: disagreed by a third — and the one used for *billing* on the degraded path was
#: the one that admitted in its own comment to being wrong on Arabic. One
#: constant, defined where its reasoning lives.
CHARS_PER_TOKEN = _CHARS_PER_TOKEN


class AnthropicProvider:
    """claude-haiku-4-5 via the official async SDK.

    Usage is exact on both paths: non-streaming responses carry `usage`, and the
    streaming helper accumulates it into `get_final_message()`, so `on_usage`
    reports measured tokens rather than an estimate.
    """

    name = "anthropic"

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        client: Any = None,
    ) -> None:
        self.model = model or settings.anthropic_model
        self._price = _lookup_price(ANTHROPIC_PRICES, self.model, self.name)
        self._api_key = api_key if api_key is not None else settings.anthropic_api_key
        self._client = client  # test seam; None means build a real SDK client
        if self._client is None and not self._api_key:
            raise ValueError(
                "AnthropicProvider needs an API key: set ANTHROPIC_API_KEY in the "
                "environment or .env (see available_providers())"
            )

    def price(self) -> tuple[float, float]:
        return self._price

    def _build_client(self) -> Any:
        if self._client is None:
            from anthropic import AsyncAnthropic

            # max_retries=0: FailoverProvider owns retry policy. Leaving the SDK
            # default of 2 on would multiply into 3 x 3 attempts per provider and
            # silently blow the generation timeout budget.
            self._client = AsyncAnthropic(api_key=self._api_key, max_retries=0)
        return self._client

    async def complete(
        self, system: str, messages: list[dict], max_tokens: int = 1024
    ) -> Completion:
        client = self._build_client()
        try:
            message = await client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=messages,
            )
        except Exception as exc:
            raise self._translate(exc) from exc
        # Parsing is inside its own guard, not left bare after the try: the SDK
        # does not validate, so response shape is a transport failure like any
        # other and must reach the failover chain as a ProviderError.
        try:
            text = "".join(
                block.text
                for block in message.content
                if getattr(block, "type", "") == "text"
            )
            usage = self._usage(message.usage.input_tokens, message.usage.output_tokens)
        except (AttributeError, TypeError, ValueError) as exc:
            raise _malformed(self.name, exc) from exc
        return Completion(
            text=text,
            usage=usage,
            provider=self.name,
            model=self.model,
        )

    async def stream(
        self,
        system: str,
        messages: list[dict],
        max_tokens: int = 1024,
        *,
        on_usage: Callable[[Usage], None] | None = None,
    ) -> AsyncIterator[str]:
        client = self._build_client()
        try:
            async with client.messages.stream(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=messages,
            ) as events:
                async for chunk in events.text_stream:
                    yield chunk
                final = await events.get_final_message()
        except Exception as exc:
            raise self._translate(exc) from exc
        if on_usage is not None:
            try:
                usage = self._usage(
                    final.usage.input_tokens, final.usage.output_tokens
                )
            except (AttributeError, TypeError, ValueError) as exc:
                raise _malformed(self.name, exc) from exc
            on_usage(usage)

    def _usage(self, input_tokens: int, output_tokens: int) -> Usage:
        return Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=usage_cost_usd(input_tokens, output_tokens, self._price),
            provider=self.name,
            model=self.model,
        )

    def _translate(self, exc: Exception) -> ProviderError:
        import anthropic

        if isinstance(exc, ProviderError):
            return exc
        # APITimeoutError subclasses APIConnectionError - order matters.
        if isinstance(exc, anthropic.APITimeoutError):
            return ProviderError(self.name, ErrorKind.TIMEOUT, "request timed out")
        if isinstance(exc, anthropic.APIConnectionError):
            return ProviderError(self.name, ErrorKind.CONNECTION, str(exc))
        if isinstance(exc, anthropic.APIStatusError):
            return ProviderError(
                self.name, _kind_for_status(exc.status_code), f"HTTP {exc.status_code}"
            )
        return ProviderError(self.name, ErrorKind.FATAL, f"{type(exc).__name__}: {exc}")


class GeminiProvider:
    """gemini-2.5-flash via google-genai.

    Requests are built from plain dicts (`ContentDict` / `GenerateContentConfigDict`)
    rather than `google.genai.types` objects, so nothing in this module needs the
    SDK's type tree imported to construct a call.

    **Streamed usage is best-effort.** google-genai attaches `usage_metadata` to
    stream chunks and the final chunk normally carries the cumulative totals, but
    that is not contractual: a truncated or error-terminated stream can end with
    none. When that happens `on_usage` reports a character-count *estimate*
    flagged by the `CHARS_PER_TOKEN` trade-off note above, never a silent zero -
    an under-reported cost is a cost cap that never fires.
    """

    name = "gemini"

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        client: Any = None,
    ) -> None:
        self.model = model or settings.gemini_model
        self._price = _lookup_price(GEMINI_PRICES, self.model, self.name)
        self._api_key = api_key if api_key is not None else settings.google_api_key
        self._client = client  # test seam; None means build a real SDK client
        if self._client is None and not self._api_key:
            raise ValueError(
                "GeminiProvider needs an API key: set GOOGLE_API_KEY in the "
                "environment or .env (see available_providers())"
            )

    def price(self) -> tuple[float, float]:
        return self._price

    def _build_client(self) -> Any:
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self._api_key)
        return self._client

    async def complete(
        self, system: str, messages: list[dict], max_tokens: int = 1024
    ) -> Completion:
        client = self._build_client()
        try:
            response = await client.aio.models.generate_content(
                model=self.model,
                contents=_to_gemini_contents(messages),
                config=_gemini_config(system, max_tokens),
            )
            text = response.text or ""
        except Exception as exc:
            raise self._translate(exc) from exc
        try:
            usage = self._usage_from_metadata(response.usage_metadata, text)
        except (AttributeError, TypeError, ValueError) as exc:
            raise _malformed(self.name, exc) from exc
        return Completion(
            text=text,
            usage=usage,
            provider=self.name,
            model=self.model,
        )

    async def stream(
        self,
        system: str,
        messages: list[dict],
        max_tokens: int = 1024,
        *,
        on_usage: Callable[[Usage], None] | None = None,
    ) -> AsyncIterator[str]:
        client = self._build_client()
        chunks: list[str] = []
        metadata: Any = None
        try:
            stream = await client.aio.models.generate_content_stream(
                model=self.model,
                contents=_to_gemini_contents(messages),
                config=_gemini_config(system, max_tokens),
            )
            async for chunk in stream:
                if chunk.usage_metadata is not None:
                    metadata = chunk.usage_metadata  # last one wins: it is cumulative
                text = chunk.text
                if text:
                    chunks.append(text)
                    yield text
        except Exception as exc:
            raise self._translate(exc) from exc
        if on_usage is not None:
            on_usage(self._usage_from_metadata(metadata, "".join(chunks)))

    def _usage_from_metadata(self, metadata: Any, text: str) -> Usage:
        if metadata is None:
            return self._estimated_usage(text)
        input_tokens = metadata.prompt_token_count or 0
        output_tokens = metadata.candidates_token_count or 0
        if output_tokens == 0 and text:
            return self._estimated_usage(text, input_tokens=input_tokens)
        return self._usage(input_tokens, output_tokens)

    def _estimated_usage(self, text: str, input_tokens: int = 0) -> Usage:
        # trade-off: see CHARS_PER_TOKEN. An estimate is reported rather than a
        # zero so the spend cap still moves when usage metadata goes missing.
        return self._usage(input_tokens, len(text) // CHARS_PER_TOKEN)

    def _usage(self, input_tokens: int, output_tokens: int) -> Usage:
        return Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=usage_cost_usd(input_tokens, output_tokens, self._price),
            provider=self.name,
            model=self.model,
        )

    def _translate(self, exc: Exception) -> ProviderError:
        import httpx
        from google.genai import errors

        if isinstance(exc, ProviderError):
            return exc
        if isinstance(exc, errors.APIError):
            return ProviderError(
                self.name, _kind_for_status(exc.code), f"HTTP {exc.code}: {exc.message}"
            )
        if isinstance(exc, httpx.TimeoutException):
            return ProviderError(self.name, ErrorKind.TIMEOUT, "request timed out")
        if isinstance(exc, httpx.TransportError):
            return ProviderError(self.name, ErrorKind.CONNECTION, str(exc))
        return ProviderError(self.name, ErrorKind.FATAL, f"{type(exc).__name__}: {exc}")


_PROVIDERS: dict[str, type] = {"anthropic": AnthropicProvider, "gemini": GeminiProvider}


def get_provider(name: str) -> Provider:
    """Build one provider by name. Raises ValueError if unknown or keyless."""
    try:
        factory = _PROVIDERS[name]
    except KeyError:
        raise ValueError(
            f"unknown generation provider {name!r}; expected one of {sorted(_PROVIDERS)}"
        ) from None
    return factory()


def available_providers() -> list[str]:
    """Providers usable right now - i.e. the ones whose API key is actually set."""
    keys = []
    if settings.anthropic_api_key:
        keys.append("anthropic")
    if settings.google_api_key:
        keys.append("gemini")
    return keys


def _malformed(provider: str, exc: Exception) -> ProviderError:
    """A 200 whose body is not the shape the SDK promised.

    ``ErrorKind.SERVER``, deliberately, not FATAL: a response that fails to parse
    is an upstream problem, and FATAL is the one kind that does **not** fail over
    (``ProviderError.should_fail_over``). Both SDKs parse leniently — anthropic
    builds models with ``construct_type`` and no validation, so a 200 that omits
    ``usage`` yields ``message.usage is None`` rather than an SDK error. Left
    unguarded that surfaced as a bare AttributeError, escaped the taxonomy
    entirely, and returned a 500 while a healthy backup provider sat unused.
    """
    return ProviderError(
        provider, ErrorKind.SERVER, f"malformed response: {type(exc).__name__}: {exc}"
    )


def _kind_for_status(status: int | None) -> ErrorKind:
    """HTTP status -> failover policy. 429 and 5xx are worth another try; 4xx is ours.

    ``None`` is accepted because google-genai's ``APIError.code`` is optional —
    its own ``_get_code`` returns None when the error body carries no status. It
    maps to SERVER, i.e. retriable: an unclassifiable transport failure must fall
    through to the next provider rather than escape the taxonomy as a TypeError
    and bypass failover entirely.
    """
    if status is None:
        return ErrorKind.SERVER
    if status == 429:
        return ErrorKind.RATE_LIMIT
    if status >= 500:
        return ErrorKind.SERVER
    return ErrorKind.FATAL


def _lookup_price(
    table: dict[str, tuple[float, float]], model: str, provider: str
) -> tuple[float, float]:
    """Fail at construction, not at billing time, when a model has no price.

    trade-off: a flat per-model table, so cached input and Gemini's long-context
    tier are billed at the standard rate. Ceiling: cost is understated for cache
    hits and overstated past Gemini's 200k threshold. Upgrade path: a per-model
    rate object once the semantic cache reports cache-read tokens.
    """
    try:
        return table[model]
    except KeyError:
        raise ValueError(
            f"no published price for {provider} model {model!r}; "
            f"add it to the {provider} price table (known: {sorted(table)})"
        ) from None


def _to_gemini_contents(messages: list[dict]) -> list[dict]:
    """Anthropic-shaped messages -> Gemini `contents`. Gemini calls it "model"."""
    return [
        {
            "role": "model" if message["role"] == "assistant" else "user",
            "parts": [{"text": message["content"]}],
        }
        for message in messages
    ]


def _gemini_config(system: str, max_tokens: int) -> dict:
    return {
        "system_instruction": system or None,
        "max_output_tokens": max_tokens,
    }
