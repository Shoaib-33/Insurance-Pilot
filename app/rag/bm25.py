import json
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from app.core.config import settings
from app.db.sqlite import db
from app.rag.text import tokenize


class BM25Index:
    def __init__(self, docs: list[dict[str, Any]]) -> None:
        self.docs = docs
        self.tokens = [tokenize(d["text"]) for d in docs]
        self.index = BM25Okapi(self.tokens) if self.tokens else None

    @classmethod
    def from_db(cls) -> "BM25Index":
        with db() as conn:
            rows = conn.execute(
                """
                SELECT c.chunk_id, c.text, c.metadata_json, d.source_name
                FROM chunks c
                JOIN documents d ON d.doc_id = c.doc_id
                """
            ).fetchall()
        docs = [
            {
                "id": row["chunk_id"],
                "text": row["text"],
                "source_name": row["source_name"],
                "metadata": json.loads(row["metadata_json"]),
            }
            for row in rows
        ]
        return cls(docs)

    @classmethod
    def load_or_create(cls) -> "BM25Index":
        path = Path(settings.bm25_index_path)
        if not path.exists():
            return cls.from_db()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return cls(payload.get("docs", []))
        except (OSError, json.JSONDecodeError):
            return cls.from_db()

    def save(self) -> None:
        path = Path(settings.bm25_index_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"docs": self.docs}, ensure_ascii=True), encoding="utf-8")

    def rebuild(self) -> None:
        fresh = self.from_db()
        self.docs = fresh.docs
        self.tokens = fresh.tokens
        self.index = fresh.index
        self.save()

    def search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        if not self.index or not self.docs:
            return []
        scores = self.index.get_scores(tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:top_k]
        return [
            {
                **self.docs[idx],
                "score": float(score),
                "metadata": {**self.docs[idx].get("metadata", {}), "retriever": "bm25"},
            }
            for idx, score in ranked
            if score > 0
        ]
