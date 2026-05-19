import json
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.db.sqlite import db
from app.rag.graph import ClaimsRAGGraph
from app.rag.text import new_id

router = APIRouter(tags=["review"])


class ApproveRequest(BaseModel):
    request_id: str
    original_answer: str
    approved_answer: str
    reviewer: str = "human_adjuster"


class RegenerateRequest(BaseModel):
    query: str = Field(min_length=1)
    metadata_filter: dict[str, Any] | None = None


@router.post("/review/approve")
def approve(payload: ApproveRequest) -> dict[str, Any]:
    review_id = new_id("review")
    with db() as conn:
        conn.execute(
            """
            INSERT INTO reviews(review_id, request_id, original_answer, approved_answer, reviewer, status)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                review_id,
                payload.request_id,
                payload.original_answer,
                payload.approved_answer,
                payload.reviewer,
                "approved",
            ),
        )
    return {"status": "approved", "review_id": review_id}


@router.post("/review/regenerate")
def regenerate(payload: RegenerateRequest) -> dict[str, Any]:
    result = ClaimsRAGGraph().run(payload.query, payload.metadata_filter)
    return {
        "request_id": result["request_id"],
        "answer": result.get("answer", ""),
        "confidence": result.get("confidence", 0.0),
        "sources": result.get("reranked_sources") or result.get("sources", []),
        "self_rag": result.get("self_rag", {}),
    }


@router.get("/claims/{claim_id}")
def get_claim(claim_id: str) -> dict[str, Any]:
    return {
        "claim_id": claim_id,
        "customer": "Sample Customer",
        "status": "Needs Review",
        "loss_type": "Property",
        "summary": "No live claims system is connected yet. This placeholder is ready for plan lookup integration.",
    }


@router.get("/reviews")
def list_reviews() -> dict[str, Any]:
    with db() as conn:
        rows = conn.execute("SELECT * FROM reviews ORDER BY created_at DESC LIMIT 25").fetchall()
    return {
        "reviews": [
            {
                "review_id": row["review_id"],
                "request_id": row["request_id"],
                "approved_answer": row["approved_answer"],
                "reviewer": row["reviewer"],
                "status": row["status"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]
    }
