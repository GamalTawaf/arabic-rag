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
    rerank_min_score: float = 0.15  # below this -> "not in corpus"

    # Generation (provider-agnostic; first entry is primary, rest are failover)
    providers: str = "anthropic,gemini"
    anthropic_model: str = "claude-haiku-4-5-20251001"
    gemini_model: str = "gemini-2.5-flash"
    generation_timeout_s: float = 20.0
    max_context_tokens: int = 6000

    # Cost controls
    daily_spend_cap_usd: float = 5.0
    semantic_cache_enabled: bool = True
    semantic_cache_threshold: float = 0.95

    anthropic_api_key: str = ""
    google_api_key: str = ""
    openai_api_key: str = ""
    cohere_api_key: str = ""


settings = Settings()
