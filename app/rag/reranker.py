import logging
import re
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)


@lru_cache
def get_flashrank_ranker():
    from flashrank import Ranker

    return Ranker(max_length=256)


class FlashRankReranker:
    def __init__(self) -> None:
        self._ranker = None

    @property
    def ranker(self):
        if self._ranker is None:
            self._ranker = get_flashrank_ranker()
        return self._ranker

    def rerank(self, query: str, docs: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        if not docs:
            return []
        try:
            from flashrank import RerankRequest

            passages = [
                {
                    "id": doc["id"],
                    "text": doc["text"],
                    "meta": {**doc.get("metadata", {}), "source_name": doc.get("source_name")},
                }
                for doc in docs
            ]
            result = self.ranker.rerank(RerankRequest(query=query, passages=passages))
            by_id = {doc["id"]: doc for doc in docs}
            reranked = []
            seen = set()
            for item in result:
                doc = by_id[str(item["id"])]
                fingerprint = self._fingerprint(doc.get("text", ""))
                if fingerprint in seen:
                    continue
                seen.add(fingerprint)
                reranked.append(
                    {
                        **doc,
                        "score": float(item.get("score", doc.get("score", 0.0))),
                        "metadata": {**doc.get("metadata", {}), "reranker": "flashrank"},
                    }
                )
                if len(reranked) >= top_k:
                    break
            return reranked
        except Exception as exc:  # pragma: no cover - fallback for missing model cache
            logger.warning("FlashRank fallback used: %s", exc)
            return self._dedupe(
                sorted(docs, key=lambda d: d.get("score", d.get("rrf_score", 0.0)), reverse=True),
                top_k=top_k,
            )

    def _dedupe(self, docs: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        selected = []
        seen = set()
        for doc in docs:
            fingerprint = self._fingerprint(doc.get("text", ""))
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            selected.append(doc)
            if len(selected) >= top_k:
                break
        return selected

    def _fingerprint(self, text: str) -> str:
        normalized = re.sub(r"\s+", " ", text.lower()).strip()
        return normalized[:500]
