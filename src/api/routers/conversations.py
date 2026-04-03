import uuid
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response

from src.api.dependencies import (
    AppComponents,
    get_components,
    limiter,
    require_active_project,
)
from src.api.schemas import (
    ConversationCreateRequest,
    ConversationListResponse,
    ConversationOut,
)
from src.db.postgres import delete_thread_checkpoints

router = APIRouter(tags=["conversations"])


@router.post("/conversations", response_model=ConversationOut, status_code=201)
@limiter.limit("10/minute")
async def create_conversation(
    request: Request,
    body: ConversationCreateRequest,
    components: AppComponents = Depends(get_components),
):
    pid = str(body.project_id)
    await require_active_project(pid, components)

    thread_id = str(uuid.uuid4())
    await components.metadata_store.create_thread(thread_id, pid)
    return ConversationOut(thread_id=thread_id)


@router.get("/conversations", response_model=ConversationListResponse)
@limiter.limit("10/minute")
async def list_conversations(
    request: Request,
    project_id: UUID = Query(...),
    components: AppComponents = Depends(get_components),
):
    pid = str(project_id)
    await require_active_project(pid, components)

    thread_ids = await components.metadata_store.list_threads(pid)
    conversations = [ConversationOut(thread_id=tid) for tid in thread_ids]
    return ConversationListResponse(conversations=conversations)


@router.delete("/conversations/{thread_id}", status_code=204)
@limiter.limit("10/minute")
async def delete_conversation(
    request: Request,
    thread_id: str,
    project_id: UUID = Query(...),
    components: AppComponents = Depends(get_components),
):
    pid = str(project_id)
    await require_active_project(pid, components)

    try:
        uuid.UUID(thread_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="thread_id must be a valid UUID")

    bound_project = await components.metadata_store.get_thread_project_id(thread_id)
    if bound_project is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    if bound_project != pid:
        raise HTTPException(status_code=404, detail="Conversation not found")

    await delete_thread_checkpoints(thread_id)
    await components.chat_store.delete_thread_data(thread_id)
    await components.metadata_store.delete_thread(thread_id, pid)

    return Response(status_code=204)
