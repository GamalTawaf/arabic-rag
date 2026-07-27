"""Ordered provider failover - the reliability story for `/ask`.

`FailoverProvider` *is* a `Provider`: it takes an ordered list and satisfies the
same two methods, so callers never branch on "am I failing over". Policy, in one
place, driven entirely by `ProviderError.kind` (see `base.py` for the table):

* **429 / 5xx** - retried against the *same* provider first (jittered exponential
  backoff, ``MAX_ATTEMPTS`` tries), because a rate limit is usually a few hundred
  milliseconds of bad luck, not an outage. Only when those are exhausted do we
  move on.
* **timeout / connection error** - fail over immediately. Retrying a provider
  that just ate ``settings.generation_timeout_s`` spends the user's latency
  budget twice for the same answer.
* **400 / auth** - re-raised as-is, never failed over. Those are our bug and the
  next provider would reject the identical request; failing over would hide the
  real error behind "all providers failed".

Every hand-off emits an OpenTelemetry span **event** named ``provider.failover``
with ``from`` / ``to`` / ``reason`` attributes, so a degraded request is visible
in the trace of the request it degraded rather than in a log nobody greps.

**Streaming failover has a hard boundary.** Once a token has been handed to the
caller it is already on the wire to the browser: a second provider cannot
retract it, and would restart mid-sentence. So failover applies **only before
the first token**. The first-token attempt - connection, retries, timeout - is
guarded; everything after it propagates to the caller, mid-stream, unwrapped.
`test_stream_does_not_fail_over_after_first_token` pins this.

trade-off: the deadline is `asyncio.timeout` here rather than per-SDK timeout
settings, so both vendors behave identically and streams get the same guard.
Ceiling: a provider used *without* this wrapper keeps its SDK's default timeout
(10 minutes for Anthropic). Upgrade path: pass the timeout into the SDK clients
too once anything calls a provider bare in the request path.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from typing import Any, TypeVar

from opentelemetry import trace
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

from app.config import settings
from app.generation.base import Completion, ErrorKind, Provider, ProviderError, Usage

MAX_ATTEMPTS = 3  # per provider, including the first try
RETRY_MULTIPLIER_S = 0.5
RETRY_MAX_WAIT_S = 8.0
FAILOVER_EVENT = "provider.failover"

T = TypeVar("T")


class AllProvidersFailed(Exception):
    """Every configured provider failed. Names each one and why.

    One error, not a chain of swallowed ones: the caller gets a single message
    listing every provider tried and its reason, and `.failures` carries the
    same as structured pairs for span attributes and `/stats`.
    """

    def __init__(self, failures: Sequence[tuple[str, str]]) -> None:
        detail = "; ".join(f"{name} ({reason})" for name, reason in failures)
        super().__init__(f"all generation providers failed - {detail}")
        self.failures = tuple(failures)


class FailoverProvider:
    """Wraps an ordered list of providers. First one that answers, wins."""

    def __init__(
        self,
        providers: Sequence[Provider],
        timeout_s: float | None = None,
        max_attempts: int = MAX_ATTEMPTS,
    ) -> None:
        if not providers:
            raise ValueError("FailoverProvider needs at least one provider")
        self._providers = tuple(providers)
        self.timeout_s = (
            timeout_s if timeout_s is not None else settings.generation_timeout_s
        )
        self.max_attempts = max_attempts

    @classmethod
    def from_settings(cls) -> FailoverProvider:
        """Build from `settings.providers` - first entry primary, rest failover."""
        from app.generation.providers import get_provider

        names = [name.strip() for name in settings.providers.split(",") if name.strip()]
        if not names:
            raise ValueError("settings.providers is empty; expected e.g. 'anthropic,gemini'")
        return cls([get_provider(name) for name in names])

    @property
    def name(self) -> str:
        return "failover:" + ",".join(provider.name for provider in self._providers)

    @property
    def model(self) -> str:
        """The primary's model. `Completion.model` reports what actually ran."""
        return self._providers[0].model

    def price(self) -> tuple[float, float]:
        """The primary's rates. Per-call cost comes from `Completion.usage`."""
        return self._providers[0].price()

    async def complete(
        self, system: str, messages: list[dict], max_tokens: int = 1024
    ) -> Completion:
        failures: list[tuple[str, str]] = []
        for index, provider in enumerate(self._providers):
            try:
                return await self._with_retry(
                    provider,
                    lambda p=provider: p.complete(system, messages, max_tokens),
                )
            except ProviderError as error:
                if not error.should_fail_over:
                    raise
                failures.append((provider.name, error.reason))
                self._record_failover(index, error)
        raise AllProvidersFailed(failures)

    async def stream(
        self,
        system: str,
        messages: list[dict],
        max_tokens: int = 1024,
        *,
        on_usage: Callable[[Usage], None] | None = None,
    ) -> AsyncIterator[str]:
        """Yield tokens, failing over **only** until the first one is produced.

        After the first yield the caller has already seen output, so a later
        failure propagates untouched - see the module docstring.
        """
        failures: list[tuple[str, str]] = []
        for index, provider in enumerate(self._providers):
            try:
                chunks, first = await self._with_retry(
                    provider,
                    lambda p=provider: _open_stream(
                        p, system, messages, max_tokens, on_usage
                    ),
                )
            except ProviderError as error:
                if not error.should_fail_over:
                    raise
                failures.append((provider.name, error.reason))
                self._record_failover(index, error)
                continue
            # Committed: from here on, errors reach the caller mid-stream.
            try:
                if first is not None:
                    yield first
                async for chunk in chunks:
                    yield chunk
            finally:
                # A browser that closes an SSE connection throws GeneratorExit in
                # here; `async for` alone would leave the provider's generator -
                # and its open HTTP response - pinned until the GC ran it.
                await chunks.aclose()
            return
        raise AllProvidersFailed(failures)

    async def _with_retry(
        self, provider: Provider, call: Callable[[], Awaitable[T]]
    ) -> T:
        """Retry `call` on 429/5xx, each attempt under the generation deadline."""
        try:
            async for attempt in AsyncRetrying(
                reraise=True,
                stop=stop_after_attempt(self.max_attempts),
                wait=wait_random_exponential(
                    multiplier=RETRY_MULTIPLIER_S, max=RETRY_MAX_WAIT_S
                ),
                retry=retry_if_exception(_is_retryable),
            ):
                with attempt:
                    return await self._with_deadline(provider, call)
        except RetryError as error:  # pragma: no cover - reraise=True precludes it
            raise error.last_attempt.exception() from error
        raise AssertionError(  # pragma: no cover - AsyncRetrying returns or raises
            "unreachable: AsyncRetrying always returns or raises"
        )

    async def _with_deadline(
        self, provider: Provider, call: Callable[[], Awaitable[T]]
    ) -> T:
        try:
            async with asyncio.timeout(self.timeout_s):
                return await call()
        except TimeoutError as exc:
            raise ProviderError(
                provider.name,
                ErrorKind.TIMEOUT,
                f"no response within {self.timeout_s}s",
            ) from exc

    def _record_failover(self, index: int, error: ProviderError) -> None:
        """Span event per hand-off. No-op when nothing is next - that is a failure."""
        if index + 1 >= len(self._providers):
            return
        # The observability agent owns app/observability/; swap this for its
        # helper when it lands. get_current_span() needs no setup - it degrades
        # to a no-op span when tracing is not configured.
        trace.get_current_span().add_event(
            FAILOVER_EVENT,
            {
                "from": self._providers[index].name,
                "to": self._providers[index + 1].name,
                "reason": error.reason,
            },
        )


async def _open_stream(
    provider: Provider,
    system: str,
    messages: list[dict],
    max_tokens: int,
    on_usage: Callable[[Usage], None] | None,
) -> tuple[AsyncIterator[str], str | None]:
    """Start a stream and pull its first chunk, so failures land before any yield.

    Returns the live iterator plus that first chunk (None for an empty stream).
    No explicit close on the failure path: an async generator that raises out of
    `__anext__` has already terminated, and a deadline cancellation is delivered
    *into* the generator's own frame, which runs its cleanup there.
    """
    chunks: Any = provider.stream(system, messages, max_tokens, on_usage=on_usage)
    try:
        first = await chunks.__anext__()
    except StopAsyncIteration:
        return chunks, None
    return chunks, first


def _is_retryable(exc: BaseException) -> bool:
    return isinstance(exc, ProviderError) and exc.is_retryable
