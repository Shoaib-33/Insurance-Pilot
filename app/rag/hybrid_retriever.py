from typing import Any

from app.core.config import settings
from app.rag.bm25 import BM25Index
from app.rag.embeddings import get_embedding_model
from app.rag.qdrant_store import QdrantVectorStore
from app.rag.rrf import reciprocal_rank_fusion


class HybridRetriever:
    def __init__(self) -> None:
        self.embeddings = get_embedding_model()
        self.qdrant = QdrantVectorStore()
        self.bm25 = BM25Index.load_or_create()

    def retrieve(self, query: str, metadata_filter: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        vector = self.embeddings.embed_query(query)
        vector_hits = self.qdrant.search_chunks(
            vector=vector,
            top_k=settings.retrieval_top_k,
            metadata_filter=metadata_filter,
        )
        for hit in vector_hits:
            hit["metadata"] = {**hit.get("metadata", {}), "retriever": "qdrant"}
        bm25_hits = self.bm25.search(query, top_k=settings.retrieval_top_k)
        return reciprocal_rank_fusion(
            [bm25_hits, vector_hits],
            top_k=max(settings.retrieval_top_k, settings.rerank_top_k),
        )
