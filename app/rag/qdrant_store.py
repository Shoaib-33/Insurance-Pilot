import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams

from app.core.config import settings

logger = logging.getLogger(__name__)


@lru_cache
def get_qdrant_client() -> QdrantClient:
    if settings.qdrant_url.startswith("local:"):
        path = settings.qdrant_url.removeprefix("local:")
        if path == ":memory:":
            return QdrantClient(":memory:")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        return QdrantClient(path=path)
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
        timeout=10,
    )


class QdrantVectorStore:
    def __init__(self) -> None:
        self.client = get_qdrant_client()
    def ensure_collections(self) -> None:
        for name in [settings.qdrant_collection, settings.qdrant_cache_collection]:
            existing = [c.name for c in self.client.get_collections().collections]
            if name not in existing:
                self.client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(size=settings.embedding_dim, distance=Distance.COSINE),
                )
                logger.info("Created Qdrant collection %s", name)

    def upsert_chunks(self, points: list[dict[str, Any]]) -> None:
        if not points:
            return
        self.client.upsert(
            collection_name=settings.qdrant_collection,
            points=[
                PointStruct(id=p["id"], vector=p["vector"], payload=p["payload"])
                for p in points
            ],
        )

    def search_chunks(
        self,
        vector: list[float],
        top_k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        query_filter = self._build_filter(metadata_filter)
        hits = self.client.query_points(
            collection_name=settings.qdrant_collection,
            query=vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        ).points
        return [
            {
                "id": str(hit.id),
                "score": float(hit.score),
                "text": hit.payload.get("text", ""),
                "source_name": hit.payload.get("source_name", "unknown"),
                "metadata": hit.payload,
            }
            for hit in hits
        ]

    def upsert_cache_answer(self, cache_id: str, vector: list[float], payload: dict[str, Any]) -> None:
        self.client.upsert(
            collection_name=settings.qdrant_cache_collection,
            points=[PointStruct(id=cache_id, vector=vector, payload=payload)],
        )

    def search_cache(self, vector: list[float], top_k: int = 1) -> list[dict[str, Any]]:
        hits = self.client.query_points(
            collection_name=settings.qdrant_cache_collection,
            query=vector,
            limit=top_k,
            with_payload=True,
        ).points
        return [
            {
                "id": str(hit.id),
                "score": float(hit.score),
                "payload": hit.payload or {},
            }
            for hit in hits
        ]

    def _build_filter(self, metadata_filter: dict[str, Any] | None) -> Filter | None:
        if not metadata_filter:
            return None
        conditions = [
            FieldCondition(key=key, match=MatchValue(value=value))
            for key, value in metadata_filter.items()
            if value is not None and value != ""
        ]
        if not conditions:
            return None
        return Filter(must=conditions)
