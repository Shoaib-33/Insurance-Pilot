from typing import Any, Literal, TypedDict


class Source(TypedDict, total=False):
    id: str
    text: str
    source_name: str
    score: float
    metadata: dict[str, Any]


class Critique(TypedDict, total=False):
    passed: bool
    retrieve: bool
    isrel: bool
    issup: bool
    isuse: bool
    confidence: float
    relevance_score: float
    faithfulness_score: float
    evidence_score: float
    needs_rewrite: bool
    rewrite_query: str | None
    issues: list[str]


class RagState(TypedDict, total=False):
    request_id: str
    query: str
    sanitized_query: str
    retrieval_query: str
    normalized_query: str
    intent: str
    should_retrieve: bool
    risk_level: Literal["low", "medium", "high"]
    cache_hit: bool
    use_cache: bool
    answer: str
    confidence: float
    user_id: str
    memory_context: str
    metadata_filter: dict[str, Any]
    sources: list[Source]
    reranked_sources: list[Source]
    self_rag: Critique
    iteration: int
    trace: list[dict[str, Any]]
