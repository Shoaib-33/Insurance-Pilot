import json
from dataclasses import dataclass

from app.db.sqlite import db
from app.rag.text import hamming_distance, normalize_text, sha256_text, simhash


@dataclass
class DocumentCacheDecision:
    should_embed: bool
    status: str
    matched_doc_id: str | None = None
    reason: str = ""


class DocumentCache:
    near_duplicate_hamming_threshold = 4

    def inspect(self, raw_text: str) -> DocumentCacheDecision:
        normalized = normalize_text(raw_text)
        file_hash = sha256_text(raw_text)
        normalized_hash = sha256_text(normalized)
        signature = simhash(normalized)

        with db() as conn:
            exact = conn.execute(
                "SELECT doc_id FROM documents WHERE file_hash = ? OR normalized_hash = ?",
                (file_hash, normalized_hash),
            ).fetchone()
            if exact:
                return DocumentCacheDecision(
                    should_embed=False,
                    status="skipped_exact_duplicate",
                    matched_doc_id=exact["doc_id"],
                    reason="Document hash already exists.",
                )

            rows = conn.execute("SELECT doc_id, simhash FROM documents").fetchall()
            for row in rows:
                distance = hamming_distance(signature, int(row["simhash"]))
                if distance <= self.near_duplicate_hamming_threshold:
                    return DocumentCacheDecision(
                        should_embed=False,
                        status="skipped_near_duplicate",
                        matched_doc_id=row["doc_id"],
                        reason=f"Near-duplicate SimHash distance {distance}.",
                    )

        return DocumentCacheDecision(should_embed=True, status="new_document")

    def chunk_exists(self, text_hash: str) -> bool:
        with db() as conn:
            row = conn.execute("SELECT chunk_id FROM chunks WHERE text_hash = ?", (text_hash,)).fetchone()
            return row is not None

    def save_document(
        self,
        doc_id: str,
        source_name: str,
        raw_text: str,
        status: str,
    ) -> None:
        normalized = normalize_text(raw_text)
        with db() as conn:
            conn.execute(
                """
                INSERT INTO documents(doc_id, source_name, file_hash, normalized_hash, simhash, status)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    doc_id,
                    source_name,
                    sha256_text(raw_text),
                    sha256_text(normalized),
                    str(simhash(normalized)),
                    status,
                ),
            )

    def save_chunk(
        self,
        chunk_id: str,
        doc_id: str,
        chunk_index: int,
        text: str,
        text_hash: str,
        metadata: dict,
        embedded: bool,
    ) -> None:
        with db() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO chunks(chunk_id, doc_id, chunk_index, text, text_hash, metadata_json, embedded_at)
                VALUES (?, ?, ?, ?, ?, ?, CASE WHEN ? THEN CURRENT_TIMESTAMP ELSE NULL END)
                """,
                (chunk_id, doc_id, chunk_index, text, text_hash, json.dumps(metadata), int(embedded)),
            )
