from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Insurance Claims Copilot"
    app_env: str = "development"
    app_host: str = "127.0.0.1"
    app_port: int = 8000

    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    qdrant_url: str = "local:data/qdrant"
    qdrant_api_key: str = ""
    qdrant_collection: str = "insurance_claims"
    qdrant_cache_collection: str = "semantic_answer_cache"

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384

    sqlite_path: str = "data/copilot.db"
    document_dir: str = "data"
    upload_dir: str = "data/uploads"
    bm25_index_path: str = "data/bm25_index.json"
    auto_ingest_pdfs_on_startup: bool = True

    semantic_cache_threshold: float = 0.88
    self_rag_max_loops: int = 2
    retrieval_top_k: int = 8
    rerank_top_k: int = 5
    low_latency_mode: bool = True
    enable_query_rewrite: bool = True
    max_sources_to_llm: int = 5
    max_evidence_chars_per_source: int = 900

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    def ensure_directories(self) -> None:
        Path(self.sqlite_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.document_dir).mkdir(parents=True, exist_ok=True)
        Path(self.upload_dir).mkdir(parents=True, exist_ok=True)
        Path(self.bm25_index_path).parent.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
