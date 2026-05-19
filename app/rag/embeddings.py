from functools import lru_cache

from app.core.config import settings


class EmbeddingModel:
    def __init__(self) -> None:
        from langchain_huggingface import HuggingFaceEmbeddings

        self.model = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            encode_kwargs={"normalize_embeddings": True},
        )

    def embed_query(self, text: str) -> list[float]:
        return self.model.embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self.model.embed_documents(texts)


@lru_cache
def get_embedding_model() -> EmbeddingModel:
    return EmbeddingModel()
