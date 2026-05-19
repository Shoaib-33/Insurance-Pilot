import json
from pathlib import Path
from typing import Any

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from app.core.config import settings
from app.rag.text import tokenize


class BM25Index:
    def __init__(self, docs: list[dict[str, Any]]) -> None:
        self.docs = docs
        documents = [
            Document(
                page_content=doc["text"],
                metadata={
                    **doc.get("metadata", {}),
                    "id": doc["id"],
                    "source_name": doc.get("source_name", "unknown"),
                },
            )
            for doc in docs
        ]
        self.retriever = BM25Retriever.from_documents(
            documents,
            preprocess_func=tokenize,
        ) if documents else None

    @classmethod
    def load_or_create(cls) -> "BM25Index":
        path = Path(settings.bm25_index_path)
        if not path.exists():
            return cls([])
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return cls(payload.get("docs", []))
        except (OSError, json.JSONDecodeError):
            return cls([])

    def save(self) -> None:
        path = Path(settings.bm25_index_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"docs": self.docs}, ensure_ascii=True), encoding="utf-8")

    def rebuild(self, docs: list[dict[str, Any]] | None = None) -> None:
        fresh = BM25Index(docs or self.docs)
        self.docs = fresh.docs
        self.retriever = fresh.retriever
        self.save()

    def search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        if not self.retriever or not self.docs:
            return []
        self.retriever.k = top_k
        results = self.retriever.invoke(query)
        hits = []
        for rank, doc in enumerate(results, start=1):
            metadata = dict(doc.metadata)
            hits.append(
                {
                    "id": str(metadata.get("id", f"bm25-{rank}")),
                    "text": doc.page_content,
                    "source_name": str(metadata.get("source_name", "unknown")),
                    "score": 1.0 / rank,
                    "metadata": {**metadata, "retriever": "langchain_bm25"},
                }
            )
        return hits
