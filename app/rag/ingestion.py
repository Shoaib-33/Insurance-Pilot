import logging
from pathlib import Path
from typing import Any

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings
from app.rag.bm25 import BM25Index
from app.rag.document_cache import DocumentCache
from app.rag.embeddings import get_embedding_model
from app.rag.qdrant_store import QdrantVectorStore
from app.rag.text import new_id, normalize_text, sha256_text

logger = logging.getLogger(__name__)


class DocumentIngestionService:
    def __init__(self) -> None:
        self.cache = DocumentCache()
        self._embeddings = None
        self.qdrant = QdrantVectorStore()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=900,
            chunk_overlap=140,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    @property
    def embeddings(self):
        if self._embeddings is None:
            self._embeddings = get_embedding_model()
        return self._embeddings

    def ingest_text(self, text: str, source_name: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        docs = [Document(page_content=text, metadata=metadata or {})]
        return self.ingest_documents(docs, source_name=source_name)

    def ingest_pdf(self, pdf_path: Path) -> dict[str, Any]:
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()
        metadata = {
            "document_type": "pdf",
            "source_path": str(pdf_path),
            "source_name": pdf_path.name,
        }
        enriched = [
            Document(page_content=doc.page_content, metadata={**metadata, **doc.metadata})
            for doc in docs
        ]
        return self.ingest_documents(enriched, source_name=pdf_path.name)

    def ingest_pdf_directory(self, directory: str | Path | None = None) -> dict[str, Any]:
        root = Path(directory or settings.document_dir)
        root.mkdir(parents=True, exist_ok=True)
        pdfs = sorted(
            path for path in root.rglob("*.pdf")
            if not any(part.startswith(".") for part in path.parts)
        )
        results = []
        for pdf in pdfs:
            try:
                results.append({"file": str(pdf), **self.ingest_pdf(pdf)})
            except Exception as exc:
                logger.exception("Failed to ingest PDF %s", pdf)
                results.append({"file": str(pdf), "status": "failed", "reason": str(exc)})
        return {
            "status": "scanned",
            "directory": str(root),
            "pdf_count": len(pdfs),
            "results": results,
        }

    def ingest_documents(self, docs: list[Document], source_name: str) -> dict[str, Any]:
        text = "\n\n".join(doc.page_content for doc in docs if doc.page_content.strip())
        if not text.strip():
            return {
                "status": "skipped_empty_document",
                "source_name": source_name,
                "embedded_chunks": 0,
                "skipped_chunks": 0,
            }
        decision = self.cache.inspect(text)
        if not decision.should_embed:
            return {
                "status": decision.status,
                "matched_doc_id": decision.matched_doc_id,
                "reason": decision.reason,
                "embedded_chunks": 0,
                "skipped_chunks": 0,
            }

        doc_id = new_id("doc")
        self.cache.save_document(doc_id, source_name, text, "embedded")

        split_docs = self.splitter.split_documents(docs)
        chunk_records = []
        new_chunk_texts = []
        bm25_docs = []
        skipped_chunks = 0

        for index, split_doc in enumerate(split_docs):
            chunk = split_doc.page_content.strip()
            if not chunk:
                continue
            text_hash = sha256_text(normalize_text(chunk))
            chunk_id = new_id("chunk")
            chunk_metadata = {
                **split_doc.metadata,
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "chunk_index": index,
                "source_name": source_name,
                "text_hash": text_hash,
            }
            bm25_docs.append(
                {
                    "id": chunk_id,
                    "text": chunk,
                    "source_name": source_name,
                    "metadata": chunk_metadata,
                }
            )
            if self.cache.chunk_exists(text_hash):
                skipped_chunks += 1
                self.cache.save_chunk(chunk_id, doc_id, index, chunk, text_hash, chunk_metadata, embedded=False)
                continue
            chunk_records.append((chunk_id, index, chunk, text_hash, chunk_metadata))
            new_chunk_texts.append(chunk)

        vectors = self.embeddings.embed_documents(new_chunk_texts)
        points = []
        for (chunk_id, index, chunk, text_hash, chunk_metadata), vector in zip(chunk_records, vectors):
            self.cache.save_chunk(chunk_id, doc_id, index, chunk, text_hash, chunk_metadata, embedded=True)
            points.append(
                {
                    "id": chunk_id,
                    "vector": vector,
                    "payload": {**chunk_metadata, "text": chunk},
                }
            )

        self.qdrant.upsert_chunks(points)
        self._merge_bm25_docs(bm25_docs)

        return {
            "status": "embedded",
            "doc_id": doc_id,
            "embedded_chunks": len(points),
            "skipped_chunks": skipped_chunks,
        }

    def _merge_bm25_docs(self, docs: list[dict[str, Any]]) -> None:
        current = BM25Index.load_or_create()
        by_hash = {
            str(doc.get("metadata", {}).get("text_hash") or doc.get("id")): doc
            for doc in current.docs
        }
        for doc in docs:
            key = str(doc.get("metadata", {}).get("text_hash") or doc.get("id"))
            by_hash[key] = doc
        BM25Index(list(by_hash.values())).save()
