"""The generation contract: one completion type, one error taxonomy.

Two providers (Anthropic, Gemini) with two different SDKs and two different
exception hierarchies have to be interchangeable behind ``FailoverProvider``.
That only works if failure is *classified* at the edge, by the adapter that
knows its own SDK, rather than sniffed later by string-matching a traceback.
So every provider translates whatever its SDK raised into a single
``ProviderError`` carrying an ``ErrorKind``, and the failover policy is a pure
function of that kind:

======================  =====================  ==========================
kind                    retry same provider?   fail over to the next one?
======================  =====================  ==========================
``RATE_LIMIT`` (429)    yes                    yes, once retries run out
``SERVER`` (5xx)        yes                    yes, once retries run out
``TIMEOUT``             no                     yes
``CONNECTION``          no                     yes
``FATAL`` (4xx, auth)   no                     **no** - surface it
======================  =====================  ==========================

``FATAL`` never fails over on purpose. A 400 or a 401 is our bug - a malformed
request or a missing key - and it will fail identically on the next provider.
Failing over would just burn a second provider's quota and bury the real error
behind a generic "all providers failed".

trade-off: one exception class with a `kind` field instead of a five-class
hierarchy. Ceiling: callers cannot `except RateLimitError` - they branch on
`.kind`. Upgrade path: subclass per kind if a caller ever needs that.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

TOKENS_PER_MILLION = 1_000_000


@dataclass(frozen=True)
class Usage:
    """Real token counts from the provider, priced with that provider's rates.

    ``provider``/``model`` name whoever *actually* produced the tokens. They live
    here rather than being read off the caller's provider handle because under
    failover that handle is the ``FailoverProvider`` wrapper: on the streaming
    path there is no ``Completion`` to read attribution from, so cost and traces
    were being filed under ``failover:anthropic,gemini`` and the *primary's*
    model even when the backup served the request. Carrying it on the immutable
    usage record keeps it correct without a mutable "last used" field that two
    concurrent streams would race over.
    """

    input_tokens: int
    output_tokens: int
    cost_usd: float
    provider: str | None = None
    model: str | None = None


@dataclass(frozen=True)
class Completion:
    """A finished generation. `provider`/`model` are recorded, not assumed.

    Under failover the answer may not come from the provider the caller asked
    for, so cost attribution and traces read these fields rather than config.
    """

    text: str
    usage: Usage
    provider: str
    model: str


class ErrorKind(str, Enum):
    RATE_LIMIT = "rate_limit"
    SERVER = "server"
    TIMEOUT = "timeout"
    CONNECTION = "connection"
    FATAL = "fatal"


#: Kinds worth another attempt against the *same* provider before failing over.
RETRYABLE_KINDS = frozenset({ErrorKind.RATE_LIMIT, ErrorKind.SERVER})


class ProviderError(Exception):
    """A provider call failed, normalized across SDKs.

    Raised by every provider adapter; `FailoverProvider` reads `.kind` to decide
    between retrying, failing over, and re-raising.
    """

    def __init__(self, provider: str, kind: ErrorKind, detail: str) -> None:
        super().__init__(f"{provider}: {kind.value}: {detail}")
        self.provider = provider
        self.kind = kind
        self.detail = detail

    @property
    def is_retryable(self) -> bool:
        """True when another attempt against the same provider might work."""
        return self.kind in RETRYABLE_KINDS

    @property
    def should_fail_over(self) -> bool:
        """False only for FATAL - the next provider would fail the same way."""
        return self.kind is not ErrorKind.FATAL

    @property
    def reason(self) -> str:
        """One-line cause, for span events and the AllProvidersFailed message."""
        return f"{self.kind.value}: {self.detail}"


class Provider(Protocol):
    """Anything that turns a prompt into text, with real usage attached.

    `on_usage` is not part of the two-method sketch in the design: `stream`
    yields plain strings, so a streamed response has nowhere to put its token
    counts. The callback is that hole plugged - it fires once, after the last
    chunk, so `/ask` can bill an SSE response as accurately as a JSON one.
    """

    name: str
    model: str

    async def complete(
        self, system: str, messages: list[dict], max_tokens: int = 1024
    ) -> Completion: ...

    def stream(
        self,
        system: str,
        messages: list[dict],
        max_tokens: int = 1024,
        *,
        on_usage: Callable[[Usage], None] | None = None,
    ) -> AsyncIterator[str]: ...

    def price(self) -> tuple[float, float]:
        """(USD per 1M input tokens, USD per 1M output tokens)."""
        ...


def usage_cost_usd(
    input_tokens: int, output_tokens: int, price: tuple[float, float]
) -> float:
    """Cost of one call from real token counts and per-million list prices."""
    input_per_million, output_per_million = price
    return (
        input_tokens * input_per_million + output_tokens * output_per_million
    ) / TOKENS_PER_MILLION
