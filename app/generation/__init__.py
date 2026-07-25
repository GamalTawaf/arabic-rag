"""Generation: provider-agnostic LLM calls with ordered failover.

`base` defines the contract (`Completion`, `Usage`, `ProviderError`),
`providers` adapts the two vendor SDKs, and `failover` composes them.
"""

from app.generation.base import (
    Completion,
    ErrorKind,
    Provider,
    ProviderError,
    Usage,
    usage_cost_usd,
)
from app.generation.failover import AllProvidersFailed, FailoverProvider
from app.generation.providers import (
    ANTHROPIC_PRICES,
    GEMINI_PRICES,
    AnthropicProvider,
    GeminiProvider,
    available_providers,
    get_provider,
)

__all__ = [
    "ANTHROPIC_PRICES",
    "GEMINI_PRICES",
    "AllProvidersFailed",
    "AnthropicProvider",
    "Completion",
    "ErrorKind",
    "FailoverProvider",
    "GeminiProvider",
    "Provider",
    "ProviderError",
    "Usage",
    "available_providers",
    "get_provider",
    "usage_cost_usd",
]
