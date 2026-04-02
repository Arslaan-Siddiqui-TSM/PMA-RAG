import os
import tempfile
from uuid import UUID

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from slowapi import Limiter
from slowapi.util import get_remote_address

from src.api.dependencies import AppComponents, get_components, require_active_project
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
    project_id: UUID = Form(...),
    components: AppComponents = Depends(get_components),
):
    pid = str(project_id)
    project = await require_active_project(pid, components)

    if doc_type not in VALID_DOC_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid doc_type '{doc_type}'. Must be one of: {VALID_DOC_TYPES}",
        )

    collection_name = project["collection_name"]
    results: list[UploadResult] = []
    total_chunks = 0

    for upload_file in files:
        suffix = os.path.splitext(upload_file.filename or "file")[1]
        if suffix.lower() not in {".pdf", ".docx", ".md"}:
            results.append(
                UploadResult(
                    file_name=upload_file.filename or "unknown",
                    doc_type=doc_type,
                    chunk_count=0,
                    status="error",
                    detail=f"Unsupported file type '{suffix}'. Supported: .pdf, .docx, .md",
                )
            )
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
                project_id=pid,
                collection_name=collection_name,
                original_name=upload_file.filename,
            )
            if chunks > 0:
                total_chunks += chunks
                results.append(
                    UploadResult(
                        file_name=upload_file.filename or "unknown",
                        doc_type=doc_type,
                        chunk_count=chunks,
                        status="ingested",
                    )
                )
            else:
                results.append(
                    UploadResult(
                        file_name=upload_file.filename or "unknown",
                        doc_type=doc_type,
                        chunk_count=0,
                        status="skipped",
                        detail="Already ingested (duplicate file hash)",
                    )
                )
        except Exception as e:
            results.append(
                UploadResult(
                    file_name=upload_file.filename or "unknown",
                    doc_type=doc_type,
                    chunk_count=0,
                    status="error",
                    detail=str(e),
                )
            )
        finally:
            os.unlink(tmp_path)

    return UploadResponse(results=results, total_chunks=total_chunks)


@router.get("/documents", response_model=list[DocumentOut])
@limiter.limit("10/minute")
async def list_documents(
    request: Request,
    project_id: UUID = Query(...),
    doc_type: str | None = Query(default=None),
    components: AppComponents = Depends(get_components),
):
    pid = str(project_id)
    await require_active_project(pid, components)
    rows = await components.metadata_store.list_documents(pid, doc_type_filter=doc_type)
    return [DocumentOut(**row) for row in rows]


@router.get("/documents/{doc_id}", response_model=DocumentDetailOut)
@limiter.limit("10/minute")
async def get_document(
    request: Request,
    doc_id: int,
    project_id: UUID = Query(...),
    components: AppComponents = Depends(get_components),
):
    pid = str(project_id)
    await require_active_project(pid, components)
    row = await components.metadata_store.get_document(pid, doc_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentDetailOut(**row)


@router.delete("/documents/{doc_id}", response_model=DeleteDocumentResponse)
@limiter.limit("10/minute")
async def delete_document(
    request: Request,
    doc_id: int,
    project_id: UUID = Query(...),
    components: AppComponents = Depends(get_components),
):
    pid = str(project_id)
    project = await require_active_project(pid, components)

    row = await components.metadata_store.get_document(pid, doc_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Document not found")

    file_name = row["file_name"]
    collection_name = project["collection_name"]

    chunk_ids = await components.metadata_store.list_chunk_ids_for_document(pid, doc_id)
    if chunk_ids and collection_name:
        components.vectorstore_manager.delete_by_ids(collection_name, chunk_ids)

    await components.bm25_index.delete_by_source_file(pid, file_name)
    await components.metadata_store.delete_document(pid, doc_id)

    return DeleteDocumentResponse(id=doc_id, file_name=file_name)


@router.get("/doc-types", response_model=DocTypesResponse)
@limiter.limit("10/minute")
async def get_doc_types(
    request: Request,
    project_id: UUID = Query(...),
    components: AppComponents = Depends(get_components),
):
    pid = str(project_id)
    await require_active_project(pid, components)
    doc_types = await components.metadata_store.get_all_doc_types(pid)
    return DocTypesResponse(doc_types=doc_types)
