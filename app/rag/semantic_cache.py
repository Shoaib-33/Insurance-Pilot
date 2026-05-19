import json
from typing import Any

from app.core.config import settings
from app.rag.embeddings import get_embedding_model
from app.rag.qdrant_store import QdrantVectorStore
from app.rag.text import new_id, normalize_text


ANSWER_CACHE_VERSION = "rag-grounded-v4"


class SemanticAnswerCache:
    def __init__(self) -> None:
        self.embeddings = get_embedding_model()
        self.store = QdrantVectorStore()

    def lookup(self, query: str) -> dict[str, Any] | None:
        vector = self.embeddings.embed_query(normalize_text(query))
        hits = self.store.search_cache(vector, top_k=1)
        if not hits:
            return None
        best = hits[0]
        if best["score"] < settings.semantic_cache_threshold:
            return None
        payload = best["payload"]
        if payload.get("cache_version") != ANSWER_CACHE_VERSION:
            return None
        return {
            "answer": payload.get("answer", ""),
            "confidence": float(payload.get("confidence", 0.0)),
            "sources": json.loads(payload.get("sources_json", "[]")),
            "score": best["score"],
        }

    def save(self, query: str, answer: str, confidence: float, sources: list[dict[str, Any]]) -> None:
        if confidence < 0.75:
            return
        normalized = normalize_text(query)
        vector = self.embeddings.embed_query(normalized)
        cache_id = new_id("cache")
        self.store.upsert_cache_answer(
            cache_id=cache_id,
            vector=vector,
            payload={
                "query": query,
                "normalized_query": normalized,
                "answer": answer,
                "confidence": confidence,
                "cache_version": ANSWER_CACHE_VERSION,
                "sources_json": json.dumps(sources, ensure_ascii=True),
            },
        )
