import json
import uuid
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends, Request
from langchain_core.documents import Document
from slowapi import Limiter
from slowapi.util import get_remote_address
from sse_starlette.sse import EventSourceResponse

from src.api.dependencies import AppComponents, get_components
from src.api.schemas import (
    ChatRequest,
    ChatResponse,
    Citation,
    FeedbackRequest,
    FeedbackResponse,
)

limiter = Limiter(key_func=get_remote_address)

router = APIRouter(tags=["chat"])


async def _build_initial_state(
    question: str,
    doc_type_filter: str | None,
    thread_id: str,
    components: AppComponents,
) -> dict:
    chat_history = await components.chat_store.get_history(thread_id)
    prior_docs = await components.chat_store.get_reranked_docs(thread_id)

    return {
        "question": question,
        "intent": "",
        "search_documents": True,
        "response_style": "default",
        "chat_history": chat_history,
        "reuse_prior_docs": False,
        "doc_type_filter": doc_type_filter,
        "documents": [],
        "reranked_documents": prior_docs,
        "relevance_scores": [],
        "confidence": "",
        "generation": "",
        "source_citations": [],
        "messages": [],
    }


async def _persist_turn(
    thread_id: str,
    question: str,
    final_state: dict,
    components: AppComponents,
) -> None:
    generation = final_state.get("generation", "")
    await components.chat_store.append_messages(thread_id, question, generation)

    reranked_docs: list[Document] = list(
        final_state.get("reranked_documents") or []
    )
    await components.chat_store.save_reranked_docs(thread_id, reranked_docs)


def _make_run_config(
    thread_id: str,
    run_id: str,
    question: str,
    doc_type_filter: str | None,
    chat_history_length: int,
) -> dict:
    return {
        "configurable": {"thread_id": thread_id},
        "run_id": run_id,
        "metadata": {
            "user_question": question,
            "doc_type_filter": doc_type_filter or "all",
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
    thread_id = body.thread_id or str(uuid.uuid4())
    run_id = str(uuid.uuid4())
    initial_state = await _build_initial_state(
        body.question, body.doc_type_filter, thread_id, components
    )

    config = _make_run_config(
        thread_id,
        run_id,
        body.question,
        body.doc_type_filter,
        len(initial_state["chat_history"]),
    )

    final_state = await components.rag_graph.ainvoke(initial_state, config=config)

    await _persist_turn(thread_id, body.question, final_state, components)

    citations = [
        Citation(**c) for c in final_state.get("source_citations", [])
    ]

    return ChatResponse(
        answer=final_state.get("generation", ""),
        confidence=final_state.get("confidence", ""),
        citations=citations,
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
    thread_id = body.thread_id or str(uuid.uuid4())
    run_id = str(uuid.uuid4())
    initial_state = await _build_initial_state(
        body.question, body.doc_type_filter, thread_id, components
    )

    config = _make_run_config(
        thread_id,
        run_id,
        body.question,
        body.doc_type_filter,
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

            elif kind == "on_chain_end" and event.get("name") == "check_relevance":
                output = event.get("data", {}).get("output", {})
                confidence = output.get("confidence", "")
                yield {
                    "event": "confidence",
                    "data": json.dumps({"confidence": confidence}),
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
            "data": json.dumps({
                "answer": final_state.get("generation", ""),
                "confidence": final_state.get("confidence", ""),
                "citations": citations,
                "thread_id": thread_id,
                "run_id": run_id,
                "search_documents": final_state.get("search_documents", True),
                "response_style": final_state.get("response_style", "default"),
            }),
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
