from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # NVIDIA NIM API
    nvidia_api_key: str

    # ChromaDB Cloud
    chroma_host: str = "api.trychroma.com"
    chroma_api_key: str
    chroma_tenant: str
    chroma_database: str
    chroma_collection_name: str = "pma_documents"

    # PostgreSQL
    postgres_host: str
    postgres_port: int
    postgres_user: str
    postgres_password: str
    postgres_db: str
    postgres_uri: str = ""

    @model_validator(mode="after")
    def _build_postgres_uri(self) -> "Settings":
        if not self.postgres_uri:
            self.postgres_uri = (
                f"postgresql://{self.postgres_user}:{self.postgres_password}"
                f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
            )
        return self

    # Model IDs
    llm_model: str = "nvidia/nemotron-mini-4b-instruct"
    embedding_model: str = "nvidia/nv-embedqa-e5-v5"
    reranker_model: str = "nvidia/llama-nemotron-rerank-1b-v2"

    # Chunking
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Retrieval
    vector_search_k: int = 20
    bm25_search_k: int = 20
    ensemble_weights: list[float] = [0.5, 0.5]
    reranker_top_n: int = 10

    # Confidence thresholds (calibrated for sigmoid-normalized logit scores)
    # nvidia/llama-nemotron-rerank-1b-v2 returns logits; after sigmoid:
    #   logit -0.85 -> prob ~0.30 (decent match)
    #   logit  0.0  -> prob  0.50 (neutral)
    #   logit  1.0  -> prob ~0.73 (strong match)
    confidence_high_threshold: float = 0.35
    confidence_high_min_docs: int = 2
    confidence_high_doc_threshold: float = 0.15
    confidence_medium_threshold: float = 0.15
    confidence_medium_min_docs: int = 1
    confidence_medium_doc_threshold: float = 0.10

    # Connection pool
    postgres_pool_max_size: int = 10

    # API
    api_rate_limit: str = "10/minute"
    api_cors_origins: list[str] = ["*"]


settings = Settings()
