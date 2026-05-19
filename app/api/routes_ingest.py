from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.rag.ingestion import DocumentIngestionService

router = APIRouter(tags=["documents"])


class IngestTextRequest(BaseModel):
    text: str = Field(min_length=1)
    source_name: str = "manual_input"
    metadata: dict[str, Any] | None = None


@router.post("/documents/ingest")
def ingest_text(payload: IngestTextRequest) -> dict[str, Any]:
    return DocumentIngestionService().ingest_text(
        text=payload.text,
        source_name=payload.source_name,
        metadata=payload.metadata,
    )


@router.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    policy_type: str = Form(default=""),
    jurisdiction: str = Form(default=""),
) -> dict[str, Any]:
    raw = await file.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="Only UTF-8 text files are supported in this scaffold.") from exc

    metadata = {
        "policy_type": policy_type,
        "jurisdiction": jurisdiction,
    }
    return DocumentIngestionService().ingest_text(
        text=text,
        source_name=file.filename or "uploaded_document.txt",
        metadata=metadata,
    )
