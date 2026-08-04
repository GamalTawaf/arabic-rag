"""Generation: provider-agnostic LLM calls with ordered failover.

`base` defines the contract (`Completion`, `Usage`, `ProviderError`),
`providers` adapts the two vendor SDKs, and `failover` composes them.
"""

from app.constants import ANTHROPIC_PRICES, GEMINI_PRICES
from app.data import Completion, Usage
from app.generation.base import (
    ErrorKind,
    Provider,
    ProviderError,
    usage_cost_usd,
)
from app.generation.failover import AllProvidersFailed, FailoverProvider
from app.generation.providers import (
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
