import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from src.api.dependencies import AppComponents, get_components
from src.api.schemas import (
    ConversationDeleteResponse,
    ConversationListResponse,
    ConversationOut,
)
from src.db.postgres import get_pool

limiter = Limiter(key_func=get_remote_address)

router = APIRouter(tags=["conversations"])


@router.post("/conversations", response_model=ConversationOut, status_code=201)
@limiter.limit("10/minute")
async def create_conversation(
    request: Request,
):
    thread_id = str(uuid.uuid4())
    return ConversationOut(thread_id=thread_id)


@router.get("/conversations", response_model=ConversationListResponse)
@limiter.limit("10/minute")
async def list_conversations(
    request: Request,
    components: AppComponents = Depends(get_components),
):
    pool = await get_pool()
    async with pool.connection() as conn:
        result = await conn.execute(
            """
            SELECT DISTINCT thread_id
            FROM checkpoints
            ORDER BY thread_id
            """
        )
        rows = await result.fetchall()

    conversations = [ConversationOut(thread_id=row["thread_id"]) for row in rows]
    return ConversationListResponse(conversations=conversations)


@router.delete(
    "/conversations/{thread_id}",
    response_model=ConversationDeleteResponse,
)
@limiter.limit("10/minute")
async def delete_conversation(
    request: Request,
    thread_id: str,
    components: AppComponents = Depends(get_components),
):
    pool = await get_pool()
    async with pool.connection() as conn:
        result = await conn.execute(
            "SELECT 1 FROM checkpoints WHERE thread_id = %s LIMIT 1",
            (thread_id,),
        )
        if await result.fetchone() is None:
            raise HTTPException(status_code=404, detail="Conversation not found")

        await conn.execute(
            "DELETE FROM checkpoint_writes WHERE thread_id = %s",
            (thread_id,),
        )
        await conn.execute(
            "DELETE FROM checkpoint_blobs WHERE thread_id = %s",
            (thread_id,),
        )
        await conn.execute(
            "DELETE FROM checkpoints WHERE thread_id = %s",
            (thread_id,),
        )

    return ConversationDeleteResponse(thread_id=thread_id)
