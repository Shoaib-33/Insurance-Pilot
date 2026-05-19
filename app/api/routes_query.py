import json
import time
from functools import lru_cache
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.db.sqlite import db
from app.rag.graph import ClaimsRAGGraph

router = APIRouter(tags=["query"])


@lru_cache
def get_rag_graph() -> ClaimsRAGGraph:
    return ClaimsRAGGraph()


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    user_id: str = "default_user"
    metadata_filter: dict[str, Any] | None = None


@router.post("/query")
def query_copilot(payload: QueryRequest) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        result = get_rag_graph().run(payload.query, payload.metadata_filter, user_id=payload.user_id)
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        raw_sources = result.get("reranked_sources") or result.get("sources", [])
        sources = [
            {
                "source_name": source.get("source_name", "unknown"),
                "text": source.get("text", ""),
                "score": source.get("score", source.get("rrf_score", 0.0)),
                "page": source.get("metadata", {}).get("page"),
            }
            for source in raw_sources
        ]
        return {
            "request_id": result["request_id"],
            "answer": result.get("answer", ""),
            "confidence": result.get("confidence", 0.0),
            "latency_ms": latency_ms,
            "sources": sources,
            "from_cache": result.get("cache_hit", False),
            "retrieval_used": bool(result.get("should_retrieve", True)),
            "self_rag": {
                **result.get("self_rag", {}),
                "iterations": result.get("iteration", 0),
            },
            "memory_used": result.get("memory_context", "") != "No prior memory for this user.",
            "trace": result.get("trace", []),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/traces/{request_id}")
def get_trace(request_id: str) -> dict[str, Any]:
    with db() as conn:
        row = conn.execute("SELECT * FROM traces WHERE request_id = ?", (request_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Trace not found")
    return {
        "request_id": row["request_id"],
        "query": row["query"],
        "trace": json.loads(row["trace_json"]),
        "created_at": row["created_at"],
    }
