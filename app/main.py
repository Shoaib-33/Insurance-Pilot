from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.api.routes_health import router as health_router
from app.api.routes_ingest import router as ingest_router
from app.api.routes_query import router as query_router
from app.api.routes_review import router as review_router
from app.core.config import settings
from app.core.logging import configure_logging
from app.db.sqlite import init_db
from app.rag.ingestion import DocumentIngestionService
from app.rag.qdrant_store import QdrantVectorStore


configure_logging()

app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router, prefix="/api")
app.include_router(query_router, prefix="/api")
app.include_router(ingest_router, prefix="/api")
app.include_router(review_router, prefix="/api")

frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
assets_dir = frontend_dir / "assets"
app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")


@app.on_event("startup")
def startup() -> None:
    settings.ensure_directories()
    init_db()
    QdrantVectorStore().ensure_collections()
    if settings.auto_ingest_pdfs_on_startup:
        DocumentIngestionService().ingest_pdf_directory(settings.document_dir)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(frontend_dir / "index.html")
