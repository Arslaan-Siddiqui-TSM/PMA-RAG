import json
import uuid
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Request
from langchain_core.documents import Document
from sse_starlette.sse import EventSourceResponse

from src.api.dependencies import (
    AppComponents,
    get_components,
    limiter,
    require_active_project,
)
from src.api.schemas import (
    ChatRequest,
    ChatResponse,
    Citation,
    FeedbackRequest,
    FeedbackResponse,
)
from src.graph.project_context import build_project_context
from src.graph.state import build_default_state

router = APIRouter(tags=["chat"])


async def _resolve_thread(
    body: ChatRequest,
    project: dict,
    components: AppComponents,
) -> str:
    """Return a valid thread_id bound to the project.

    - If thread_id is omitted, generate a new one and bind it.
    - If thread_id is supplied but unknown, generate a new one and bind it.
    - If thread_id is known, enforce project binding (409 on mismatch).
    """
    pid = str(body.project_id)

    if body.thread_id:
        try:
            uuid.UUID(body.thread_id)
        except ValueError:
            raise HTTPException(
                status_code=422, detail="thread_id must be a valid UUID"
            )

        bound_project = await components.metadata_store.get_thread_project_id(
            body.thread_id
        )
        if bound_project is not None:
            if bound_project != pid:
                raise HTTPException(
                    status_code=409,
                    detail="Thread is bound to a different project",
                )
            return body.thread_id

    thread_id = str(uuid.uuid4())
    await components.metadata_store.create_thread(thread_id, pid)
    return thread_id


async def _build_initial_state(
    question: str,
    project_id: str,
    collection_name: str,
    project_context: str,
    thread_id: str,
    components: AppComponents,
) -> dict:
    chat_history = await components.chat_store.get_history(thread_id)
    prior_docs = await components.chat_store.get_reranked_docs(thread_id)
    document_catalog = await components.metadata_store.list_documents(project_id)

    return build_default_state(
        question=question,
        project_id=project_id,
        collection_name=collection_name,
        project_context=project_context,
        chat_history=chat_history,
        reranked_documents=prior_docs,
        document_catalog=document_catalog,
    )


async def _persist_turn(
    thread_id: str,
    question: str,
    final_state: dict,
    components: AppComponents,
) -> None:
    generation = final_state.get("generation", "")
    await components.chat_store.append_messages(thread_id, question, generation)

    reranked_docs: list[Document] = list(final_state.get("reranked_documents") or [])
    await components.chat_store.save_reranked_docs(thread_id, reranked_docs)


def _make_run_config(
    thread_id: str,
    run_id: str,
    question: str,
    project_id: str,
    chat_history_length: int,
) -> dict:
    return {
        "configurable": {"thread_id": thread_id},
        "run_id": run_id,
        "metadata": {
            "user_question": question,
            "project_id": project_id,
            "chat_history_length": chat_history_length,
        },
        "tags": ["api"],
    }


@router.post("/chat", response_model=ChatResponse)
@limiter.limit("10/minute")
async def chat(
    request: Request,
    body: ChatRequest,
    components: AppComponents = Depends(get_components),
):
    pid = str(body.project_id)
    project = await require_active_project(pid, components)
    collection_name = project["collection_name"]

    thread_id = await _resolve_thread(body, project, components)
    run_id = str(uuid.uuid4())
    all_projects = await components.metadata_store.list_active_projects()
    project_context = build_project_context(
        active_project=project,
        all_projects=all_projects,
        max_projects=20,
    )

    initial_state = await _build_initial_state(
        body.question,
        pid,
        collection_name,
        project_context,
        thread_id,
        components,
    )

    config = _make_run_config(
        thread_id,
        run_id,
        body.question,
        pid,
        len(initial_state["chat_history"]),
    )

    final_state = await components.rag_graph.ainvoke(initial_state, config=config)

    await _persist_turn(thread_id, body.question, final_state, components)

    citations = [Citation(**c) for c in final_state.get("source_citations", [])]

    return ChatResponse(
        answer=final_state.get("generation", ""),
        confidence=final_state.get("confidence", ""),
        citations=citations,
        validation_passed=final_state.get("quality_passed", True),
        validation_reason=final_state.get("quality_reason", ""),
        thread_id=thread_id,
        run_id=run_id,
        search_documents=final_state.get("search_documents", True),
        response_style=final_state.get("response_style", "default"),
    )


@router.post("/chat/stream")
@limiter.limit("10/minute")
async def chat_stream(
    request: Request,
    body: ChatRequest,
    components: AppComponents = Depends(get_components),
):
    pid = str(body.project_id)
    project = await require_active_project(pid, components)
    collection_name = project["collection_name"]

    thread_id = await _resolve_thread(body, project, components)
    run_id = str(uuid.uuid4())
    all_projects = await components.metadata_store.list_active_projects()
    project_context = build_project_context(
        active_project=project,
        all_projects=all_projects,
        max_projects=20,
    )

    initial_state = await _build_initial_state(
        body.question,
        pid,
        collection_name,
        project_context,
        thread_id,
        components,
    )

    config = _make_run_config(
        thread_id,
        run_id,
        body.question,
        pid,
        len(initial_state["chat_history"]),
    )

    async def event_generator() -> AsyncGenerator[dict, None]:
        yield {
            "event": "thread_id",
            "data": json.dumps({"thread_id": thread_id, "run_id": run_id}),
        }

        final_state: dict = {}

        async for event in components.rag_graph.astream_events(
            initial_state,
            config=config,
            version="v2",
        ):
            kind = event.get("event", "")

            if kind == "on_chain_end" and event.get("name") == "classify_intent":
                output = event.get("data", {}).get("output", {})
                yield {
                    "event": "intent",
                    "data": json.dumps(
                        {
                            "intent": output.get("intent", ""),
                            "search_documents": output.get("search_documents", True),
                            "response_style": output.get("response_style", "default"),
                        }
                    ),
                }

            elif kind == "on_chain_end" and event.get("name") == "reflect_on_retrieval":
                output = event.get("data", {}).get("output", {})
                sufficient = output.get("retrieval_sufficient", True)
                yield {
                    "event": "retrieval_status",
                    "data": json.dumps({
                        "retrieval_sufficient": sufficient,
                        "confidence": output.get("confidence", ""),
                    }),
                }

            elif kind == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    yield {
                        "event": "token",
                        "data": json.dumps({"token": chunk.content}),
                    }

            elif kind == "on_chain_end" and event.get("name") == "LangGraph":
                final_state = event.get("data", {}).get("output", {})

        await _persist_turn(thread_id, body.question, final_state, components)

        citations = final_state.get("source_citations", [])
        yield {
            "event": "done",
            "data": json.dumps(
                {
                    "answer": final_state.get("generation", ""),
                    "confidence": final_state.get("confidence", ""),
                    "citations": citations,
                    "validation_passed": final_state.get("quality_passed", True),
                    "validation_reason": final_state.get("quality_reason", ""),
                    "thread_id": thread_id,
                    "run_id": run_id,
                    "search_documents": final_state.get("search_documents", True),
                    "response_style": final_state.get("response_style", "default"),
                }
            ),
        }

    return EventSourceResponse(event_generator())


@router.post("/feedback", response_model=FeedbackResponse)
@limiter.limit("20/minute")
async def submit_feedback(
    request: Request,
    body: FeedbackRequest,
    components: AppComponents = Depends(get_components),
):
    feedback_id = await components.chat_store.save_feedback(
        thread_id=body.thread_id,
        run_id=body.run_id,
        score=body.score,
        comment=body.comment,
    )

    ls = components.langsmith_client
    if ls:
        ls.create_feedback(
            run_id=body.run_id,
            key="user_rating",
            score=body.score,
            comment=body.comment,
        )

    return FeedbackResponse(id=feedback_id)
