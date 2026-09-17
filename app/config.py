"""Central configuration, loaded from environment variables / .env."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM
    llm_provider: str = "anthropic"
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-3-5-haiku-20241022"

    # Vector store
    vector_store: str = "faiss"
    vector_store_path: str = "./storage/index"

    # Embeddings — "tfidf" (default, lightweight) or "sentence_transformer" (optional, see README)
    embedding_provider: str = "tfidf"

    # Retrieval / grounding
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 4
    min_similarity: float = 0.15

    # Simple demo auth
    api_key: str = "change-me-local-dev-key"


settings = Settings()
