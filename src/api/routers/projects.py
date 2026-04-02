from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from slowapi import Limiter
from slowapi.util import get_remote_address

from src.api.dependencies import AppComponents, get_components
from src.api.schemas import ProjectCreateRequest, ProjectListResponse, ProjectOut

limiter = Limiter(key_func=get_remote_address)

router = APIRouter(tags=["projects"])


@router.post("/projects", response_model=ProjectOut, status_code=201)
@limiter.limit("10/minute")
async def create_project(
    request: Request,
    body: ProjectCreateRequest,
    components: AppComponents = Depends(get_components),
):
    try:
        row = await components.metadata_store.create_project(
            name=body.name, description=body.description
        )
    except Exception as exc:
        if "idx_projects_unique_active_name" in str(exc):
            raise HTTPException(status_code=409, detail="Project name already exists")
        raise
    return ProjectOut(**row)


@router.get("/projects", response_model=ProjectListResponse)
@limiter.limit("10/minute")
async def list_projects(
    request: Request,
    components: AppComponents = Depends(get_components),
):
    rows = await components.metadata_store.list_active_projects()
    return ProjectListResponse(projects=[ProjectOut(**r) for r in rows])


@router.delete("/projects/{project_id}", status_code=204)
@limiter.limit("10/minute")
async def delete_project(
    request: Request,
    project_id: UUID,
    components: AppComponents = Depends(get_components),
):
    pid = str(project_id)

    existing = await components.metadata_store.get_project(pid, include_deleted=True)
    if existing is None:
        raise HTTPException(status_code=404, detail="Project not found")
    if existing.get("deleted_at") is not None:
        raise HTTPException(status_code=410, detail="Project already deleted")

    deleted = await components.metadata_store.soft_delete_project(pid)
    if deleted is None:
        raise HTTPException(status_code=410, detail="Project already deleted")

    collection_name = deleted.get("collection_name")

    thread_ids = await components.metadata_store.delete_threads_for_project(pid)
    for tid in thread_ids:
        await _delete_thread_artifacts(tid, components)

    await components.metadata_store.hard_delete_project_documents(pid)

    if collection_name:
        try:
            components.vectorstore_manager.delete_collection(collection_name)
        except Exception:
            await components.metadata_store.mark_vector_cleanup_pending(pid, True)

    return Response(status_code=204)


async def _delete_thread_artifacts(thread_id: str, components: AppComponents) -> None:
    from src.db.postgres import get_pool

    pool = await get_pool()
    async with pool.connection() as conn:
        await conn.execute(
            "DELETE FROM checkpoint_writes WHERE thread_id = %s", (thread_id,)
        )
        await conn.execute(
            "DELETE FROM checkpoint_blobs WHERE thread_id = %s", (thread_id,)
        )
        await conn.execute("DELETE FROM checkpoints WHERE thread_id = %s", (thread_id,))
    await components.chat_store.delete_thread_data(thread_id)
