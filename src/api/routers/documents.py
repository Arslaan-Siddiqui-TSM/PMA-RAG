import os
import tempfile

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from slowapi import Limiter
from slowapi.util import get_remote_address

from src.api.dependencies import AppComponents, get_components
from src.api.schemas import (
    VALID_DOC_TYPES,
    DeleteDocumentResponse,
    DocTypesResponse,
    DocumentDetailOut,
    DocumentOut,
    UploadResponse,
    UploadResult,
)
from src.ingestion.pipeline import ingest_document

limiter = Limiter(key_func=get_remote_address)

router = APIRouter(tags=["documents"])


@router.post("/documents/upload", response_model=UploadResponse)
@limiter.limit("10/minute")
async def upload_documents(
    request: Request,
    files: list[UploadFile] = File(...),
    doc_type: str = Form(...),
    components: AppComponents = Depends(get_components),
):
    if doc_type not in VALID_DOC_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid doc_type '{doc_type}'. Must be one of: {VALID_DOC_TYPES}",
        )

    results: list[UploadResult] = []
    total_chunks = 0

    for upload_file in files:
        suffix = os.path.splitext(upload_file.filename or "file")[1]
        if suffix.lower() not in {".pdf", ".docx", ".md"}:
            results.append(UploadResult(
                file_name=upload_file.filename or "unknown",
                doc_type=doc_type,
                chunk_count=0,
                status="error",
                detail=f"Unsupported file type '{suffix}'. Supported: .pdf, .docx, .md",
            ))
            continue

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await upload_file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            chunks = await ingest_document(
                file_path=tmp_path,
                doc_type=doc_type,
                metadata_store=components.metadata_store,
                vectorstore_manager=components.vectorstore_manager,
                bm25_index=components.bm25_index,
                original_name=upload_file.filename,
            )
            if chunks > 0:
                total_chunks += chunks
                results.append(UploadResult(
                    file_name=upload_file.filename or "unknown",
                    doc_type=doc_type,
                    chunk_count=chunks,
                    status="ingested",
                ))
            else:
                results.append(UploadResult(
                    file_name=upload_file.filename or "unknown",
                    doc_type=doc_type,
                    chunk_count=0,
                    status="skipped",
                    detail="Already ingested (duplicate file hash)",
                ))
        except Exception as e:
            results.append(UploadResult(
                file_name=upload_file.filename or "unknown",
                doc_type=doc_type,
                chunk_count=0,
                status="error",
                detail=str(e),
            ))
        finally:
            os.unlink(tmp_path)

    return UploadResponse(results=results, total_chunks=total_chunks)


@router.get("/documents", response_model=list[DocumentOut])
@limiter.limit("10/minute")
async def list_documents(
    request: Request,
    components: AppComponents = Depends(get_components),
):
    rows = await components.metadata_store.list_documents()
    return [DocumentOut(**row) for row in rows]


@router.get("/documents/{doc_id}", response_model=DocumentDetailOut)
@limiter.limit("10/minute")
async def get_document(
    request: Request,
    doc_id: int,
    components: AppComponents = Depends(get_components),
):
    row = await components.metadata_store.get_document(doc_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentDetailOut(**row)


@router.delete("/documents/{doc_id}", response_model=DeleteDocumentResponse)
@limiter.limit("10/minute")
async def delete_document(
    request: Request,
    doc_id: int,
    components: AppComponents = Depends(get_components),
):
    row = await components.metadata_store.get_document(doc_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Document not found")

    file_name = row["file_name"]

    components.vectorstore_manager.delete_by_source_file(file_name)
    components.bm25_index.delete_by_source_file(file_name)
    await components.metadata_store.delete_document(doc_id)

    return DeleteDocumentResponse(id=doc_id, file_name=file_name)


@router.get("/doc-types", response_model=DocTypesResponse)
@limiter.limit("10/minute")
async def get_doc_types(
    request: Request,
    components: AppComponents = Depends(get_components),
):
    doc_types = await components.metadata_store.get_all_doc_types()
    return DocTypesResponse(doc_types=doc_types)
