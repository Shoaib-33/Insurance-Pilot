from collections import defaultdict
from typing import Any


def reciprocal_rank_fusion(
    result_sets: list[list[dict[str, Any]]],
    k: int = 60,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    scores: dict[str, float] = defaultdict(float)
    docs: dict[str, dict[str, Any]] = {}

    for results in result_sets:
        for rank, doc in enumerate(results, start=1):
            doc_id = doc["id"]
            scores[doc_id] += 1.0 / (k + rank)
            docs[doc_id] = {**doc, "rrf_score": scores[doc_id]}

    ranked = sorted(docs.values(), key=lambda item: scores[item["id"]], reverse=True)
    return ranked[:top_k]
