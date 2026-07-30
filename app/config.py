from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    service_name: str = "arabic-rag"
    env: str = "development"
    database_url: str = "postgresql+asyncpg://rag_user:rag_pass@localhost:5433/rag_db"

    # Retrieval
    embedding_model: str = "text-embedding-3-large"
    top_k_retrieve: int = 20
    top_k_context: int = 5
    rerank_enabled: bool = True
    # Below this -> "not in corpus"; 0.0 disables the score gate (sigmoid scores are
    # never negative) while an empty result set still refuses. Calibrated over all
    # 283 eval pairs: refusal precision peaks at 0.171, and the old 0.15 refused 39
    # answerable questions — every one with the gold article already in context — to
    # catch 7 of 15 unanswerable. See docs/refusal-calibration.md.
    rerank_min_score: float = 0.0

    # Generation (provider-agnostic; first entry is primary, rest are failover)
    providers: str = "anthropic,gemini"
    anthropic_model: str = "claude-haiku-4-5-20251001"
    gemini_model: str = "gemini-2.5-flash"
    # Hugging Face Inference Providers: an OpenAI-compatible router in front of
    # third-party hosts. Prices are per-model *and* per-routed-host and change
    # without a published table to check them against, so unlike Anthropic and
    # Gemini the rate is configuration, not a constant in the code. The default
    # is an order-of-magnitude estimate for a ~70B open model — set it to your
    # account's real rate, because the daily spend cap does this arithmetic.
    hf_model: str = "Qwen/Qwen2.5-72B-Instruct"
    hf_base_url: str = "https://router.huggingface.co/v1"
    hf_price_input_usd_per_million: float = 0.60
    hf_price_output_usd_per_million: float = 0.60
    generation_timeout_s: float = 20.0
    max_context_tokens: int = 6000

    # Cost controls
    daily_spend_cap_usd: float = 5.0
    semantic_cache_enabled: bool = True
    semantic_cache_threshold: float = 0.95

    # Access control. Both default to the open/demo setting so a fresh clone runs
    # with no configuration; both are honestly limited — see the README's
    # "Security posture" note before pointing anything real at this.
    # Empty -> /ingest is unauthenticated. Set it and the route requires a
    # matching `x-api-key` header.
    ingest_api_key: str = ""
    # Requests per client IP per minute on /ask; 0 disables the limiter.
    ask_rate_limit_per_minute: int = 60

    anthropic_api_key: str = ""
    google_api_key: str = ""
    hf_api_key: str = ""
    openai_api_key: str = ""
    cohere_api_key: str = ""


settings = Settings()
