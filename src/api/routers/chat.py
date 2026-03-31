import json
import uuid
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from sse_starlette.sse import EventSourceResponse

from src.api.dependencies import AppComponents, get_components
from src.api.schemas import ChatRequest, ChatResponse, Citation

limiter = Limiter(key_func=get_remote_address)

router = APIRouter(tags=["chat"])


def _build_initial_state(question: str, doc_type_filter: str | None) -> dict:
    return {
        "question": question,
        "intent": "",
        "chat_history": [],
        "reuse_prior_docs": False,
        "doc_type_filter": doc_type_filter,
        "documents": [],
        "reranked_documents": [],
        "relevance_scores": [],
        "confidence": "",
        "generation": "",
        "source_citations": [],
        "messages": [],
    }


@router.post("/chat", response_model=ChatResponse)
@limiter.limit("10/minute")
async def chat(
    request: Request,
    body: ChatRequest,
    components: AppComponents = Depends(get_components),
):
    thread_id = body.thread_id or str(uuid.uuid4())
    initial_state = _build_initial_state(body.question, body.doc_type_filter)

    final_state = await components.rag_graph.ainvoke(
        initial_state,
        config={"configurable": {"thread_id": thread_id}},
    )

    citations = [
        Citation(**c) for c in final_state.get("source_citations", [])
    ]

    return ChatResponse(
        answer=final_state.get("generation", ""),
        confidence=final_state.get("confidence", ""),
        citations=citations,
        thread_id=thread_id,
    )


@router.post("/chat/stream")
@limiter.limit("10/minute")
async def chat_stream(
    request: Request,
    body: ChatRequest,
    components: AppComponents = Depends(get_components),
):
    thread_id = body.thread_id or str(uuid.uuid4())
    initial_state = _build_initial_state(body.question, body.doc_type_filter)

    async def event_generator() -> AsyncGenerator[dict, None]:
        yield {
            "event": "thread_id",
            "data": json.dumps({"thread_id": thread_id}),
        }

        final_state: dict = {}

        async for event in components.rag_graph.astream_events(
            initial_state,
            config={"configurable": {"thread_id": thread_id}},
            version="v2",
        ):
            kind = event.get("event", "")

            if kind == "on_chain_end" and event.get("name") == "classify_intent":
                output = event.get("data", {}).get("output", {})
                intent = output.get("intent", "")
                yield {
                    "event": "intent",
                    "data": json.dumps({"intent": intent}),
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

        citations = final_state.get("source_citations", [])
        yield {
            "event": "done",
            "data": json.dumps({
                "answer": final_state.get("generation", ""),
                "confidence": final_state.get("confidence", ""),
                "citations": citations,
                "thread_id": thread_id,
            }),
        }

    return EventSourceResponse(event_generator())
