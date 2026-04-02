import asyncio
import os
import sys
import tempfile
import uuid

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import chainlit as cl
from chainlit.input_widget import Select
from chainlit.types import Feedback
from engineio.payload import Payload
from langchain_core.messages import AIMessage, HumanMessage
from langsmith import Client as LangSmithClient

from config import settings
from src.db.metadata import MetadataStore, slugify_project_name
from src.db.postgres import close_pool, get_pool
from src.graph.builder import compile_graph
from src.graph.chainlit_nodes import (
    casual_response_node,
    check_relevance_node,
    classify_intent_node,
    decompose_query_node,
    generate_node,
    help_response_node,
    reformulate_query_node,
    rerank_node,
    retrieve_node,
    set_retrieval_components,
    validate_answer_node,
)
from src.ingestion.pipeline import ingest_document
from src.retrieval.bm25 import BM25Index
from src.retrieval.vectorstore import VectorStoreManager
from src.graph.project_context import build_project_context

CHAINLIT_NODE_MAP = {
    "classify_intent": classify_intent_node,
    "casual_response": casual_response_node,
    "help_response": help_response_node,
    "reformulate_query": reformulate_query_node,
    "decompose_query": decompose_query_node,
    "retrieve": retrieve_node,
    "rerank": rerank_node,
    "check_relevance": check_relevance_node,
    "generate": generate_node,
    "validate_answer": validate_answer_node,
}

Payload.max_decode_packets = 500

os.environ.setdefault("NVIDIA_API_KEY", settings.nvidia_api_key)
os.environ.setdefault("DATABASE_URL", settings.postgres_uri)

if settings.langsmith_api_key:
    os.environ.setdefault("LANGSMITH_API_KEY", settings.langsmith_api_key)
    if settings.langsmith_tracing:
        os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", settings.langsmith_project)
    os.environ.setdefault("LANGSMITH_ENDPOINT", settings.langsmith_endpoint)
    if settings.langsmith_workspace_id:
        os.environ.setdefault("LANGSMITH_WORKSPACE_ID", settings.langsmith_workspace_id)

_ls_client: LangSmithClient | None = None


def _get_langsmith_client() -> LangSmithClient | None:
    global _ls_client
    if not settings.langsmith_api_key:
        return None
    if _ls_client is None:
        _ls_client = LangSmithClient()
    return _ls_client


_vectorstore_manager: VectorStoreManager | None = None
_bm25_index: BM25Index | None = None
_metadata_store: MetadataStore | None = None


async def _get_components():
    global _vectorstore_manager, _bm25_index, _metadata_store

    if _vectorstore_manager is None:
        _vectorstore_manager = VectorStoreManager()
    if _metadata_store is None:
        pool = await get_pool()
        if _bm25_index is None:
            _bm25_index = BM25Index(pool)
        _metadata_store = MetadataStore(pool)
        await _metadata_store.setup()
    elif _bm25_index is None:
        pool = await get_pool()
        _bm25_index = BM25Index(pool)

    set_retrieval_components(_vectorstore_manager, _bm25_index)
    return _vectorstore_manager, _bm25_index, _metadata_store


async def _get_or_create_collection_name(
    metadata: MetadataStore, project: dict
) -> str:
    collection_name = str(project.get("collection_name") or "").strip()
    if collection_name:
        return collection_name
    project_id = str(project["id"])
    generated = f"{slugify_project_name(str(project['name']))}__{project_id}"
    await metadata.set_project_collection_name(project_id, generated)
    project["collection_name"] = generated
    return generated


@cl.on_chat_start
async def on_chat_start():
    vsm, bm25, metadata = await _get_components()

    thread_id = str(uuid.uuid4())
    cl.user_session.set("thread_id", thread_id)

    pool = await get_pool()
    rag_graph = await compile_graph(pool, node_map=CHAINLIT_NODE_MAP)
    cl.user_session.set("rag_graph", rag_graph)

    projects = await metadata.list_active_projects()
    project_names = [p["name"] for p in projects]

    if not project_names:
        await cl.Message(
            content=(
                "Welcome to the **PMA-RAG Chatbot**!\n\n"
                "No projects found. Please create a project via the API first, "
                "then refresh this page."
            )
        ).send()
        return

    project_map = {p["name"]: p for p in projects}
    cl.user_session.set("project_map", project_map)

    chat_settings = [
        Select(
            id="project_selector",
            label="Select Project",
            values=project_names,
            initial_value=project_names[0],
        ),
    ]
    await cl.ChatSettings(chat_settings).send()

    first_project = projects[0]
    first_collection_name = await _get_or_create_collection_name(metadata, first_project)
    cl.user_session.set("project_id", str(first_project["id"]))
    cl.user_session.set("collection_name", first_collection_name)

    await metadata.create_thread(thread_id, str(first_project["id"]))

    cl.user_session.set("chat_history", [])
    cl.user_session.set("prior_reranked_docs", [])

    await cl.Message(
        content=(
            f"Welcome to the **PMA-RAG Chatbot**! "
            f"Active project: **{first_project['name']}**\n\n"
            "- **Upload documents** using the button below "
            "(PDF, DOCX, or Markdown)\n"
            "- **Ask questions** about your uploaded documents\n"
            "- **Switch projects** using the settings panel\n\n"
            "Upload some documents to get started, or ask a question "
            "if documents are already loaded."
        )
    ).send()


@cl.on_settings_update
async def on_settings_update(new_settings: dict):
    project_name = new_settings.get("project_selector")
    if not project_name:
        return
    project_map = cl.user_session.get("project_map", {})
    project = project_map.get(project_name)
    if not project:
        return

    pid = str(project["id"])
    _, _, metadata = await _get_components()
    collection_name = await _get_or_create_collection_name(metadata, project)
    cl.user_session.set("project_id", pid)
    cl.user_session.set("collection_name", collection_name)

    thread_id = str(uuid.uuid4())
    cl.user_session.set("thread_id", thread_id)
    cl.user_session.set("chat_history", [])
    cl.user_session.set("prior_reranked_docs", [])

    await metadata.create_thread(thread_id, pid)

    await cl.Message(
        content=f"Switched to project **{project_name}**. New conversation started."
    ).send()


@cl.action_callback("upload_documents")
async def on_upload_action(action: cl.Action):
    await _handle_file_upload()


async def _handle_file_upload():
    project_id = cl.user_session.get("project_id")
    collection_name = cl.user_session.get("collection_name")
    if not project_id or not collection_name:
        await cl.Message(content="Please select a project first.").send()
        return

    files = await cl.AskFileMessage(
        content="Upload your documents (PDF, DOCX, or Markdown).",
        accept=[
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "text/markdown",
        ],
        max_size_mb=20,
        max_files=10,
    ).send()

    if not files:
        return

    vsm, bm25, metadata = await _get_components()

    processing_msg = cl.Message(content="Processing uploaded documents...")
    await processing_msg.send()

    total_chunks = 0
    results: list[str] = []

    for file in files:
        doc_type = await _ask_doc_type(file.name)

        tmp_path = file.path
        if not tmp_path:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=os.path.splitext(file.name)[1]
            ) as tmp:
                tmp.write(file.content)
                tmp_path = tmp.name

        try:
            chunks = await ingest_document(
                file_path=tmp_path,
                doc_type=doc_type,
                metadata_store=metadata,
                vectorstore_manager=vsm,
                bm25_index=bm25,
                project_id=project_id,
                collection_name=collection_name,
                original_name=file.name,
            )
            if chunks > 0:
                total_chunks += chunks
                results.append(f"  - **{file.name}** ({doc_type}): {chunks} chunks")
            else:
                results.append(f"  - **{file.name}**: already ingested (skipped)")
        except Exception as e:
            results.append(f"  - **{file.name}**: error - {e}")

    summary = "\n".join(results)
    await cl.Message(
        content=(
            f"Document ingestion complete! "
            f"**{total_chunks} total chunks** created.\n\n{summary}"
        )
    ).send()


async def _process_attached_files(elements: list) -> None:
    file_elements = [el for el in elements if hasattr(el, "path") and el.path]
    if not file_elements:
        return

    project_id = cl.user_session.get("project_id")
    collection_name = cl.user_session.get("collection_name")
    if not project_id or not collection_name:
        await cl.Message(content="Please select a project first.").send()
        return

    vsm, bm25, metadata = await _get_components()

    processing_msg = cl.Message(content="Processing uploaded documents...")
    await processing_msg.send()

    total_chunks = 0
    results: list[str] = []

    for el in file_elements:
        doc_type = await _ask_doc_type(el.name)

        try:
            chunks = await ingest_document(
                file_path=el.path,
                doc_type=doc_type,
                metadata_store=metadata,
                vectorstore_manager=vsm,
                bm25_index=bm25,
                project_id=project_id,
                collection_name=collection_name,
                original_name=el.name,
            )
            if chunks > 0:
                total_chunks += chunks
                results.append(f"  - **{el.name}** ({doc_type}): {chunks} chunks")
            else:
                results.append(f"  - **{el.name}**: already ingested (skipped)")
        except Exception as e:
            results.append(f"  - **{el.name}**: error - {e}")

    summary = "\n".join(results)
    await cl.Message(
        content=(
            f"Document ingestion complete! "
            f"**{total_chunks} total chunks** created.\n\n{summary}"
        )
    ).send()


async def _ask_doc_type(file_name: str) -> str:
    res = await cl.AskActionMessage(
        content=f"What type of document is **{file_name}**?",
        actions=[
            cl.Action(name="doc_type", payload={"value": "PRD"}, label="PRD"),
            cl.Action(name="doc_type", payload={"value": "BRD"}, label="BRD"),
            cl.Action(
                name="doc_type",
                payload={"value": "Technical Spec"},
                label="Technical Spec",
            ),
            cl.Action(
                name="doc_type", payload={"value": "Test Plan"}, label="Test Plan"
            ),
            cl.Action(name="doc_type", payload={"value": "Use Case"}, label="Use Case"),
            cl.Action(
                name="doc_type",
                payload={"value": "Functional Spec"},
                label="Functional Spec",
            ),
            cl.Action(
                name="doc_type",
                payload={"value": "Non-Functional Spec"},
                label="Non-Functional Spec",
            ),
            cl.Action(name="doc_type", payload={"value": "Other"}, label="Other"),
        ],
    ).send()

    if res and res.get("payload"):
        return res["payload"]["value"]
    return "Other"


@cl.on_message
async def on_message(message: cl.Message):
    if message.elements:
        await _process_attached_files(message.elements)
        if not message.content.strip():
            return

    content = message.content.strip()
    if not content:
        return

    if content.lower().startswith("/upload"):
        await _handle_file_upload()
        return

    rag_graph = cl.user_session.get("rag_graph")
    thread_id = cl.user_session.get("thread_id")
    project_id = cl.user_session.get("project_id")
    collection_name = cl.user_session.get("collection_name")

    if rag_graph is None or not project_id:
        await cl.Message(content="Session not initialized. Please refresh.").send()
        return

    chat_history = cl.user_session.get("chat_history", [])
    prior_docs = cl.user_session.get("prior_reranked_docs", [])
    _, _, metadata = await _get_components()
    projects = await metadata.list_active_projects()
    active_project = next(
        (project for project in projects if str(project["id"]) == str(project_id)),
        None,
    )
    project_context = build_project_context(
        active_project=active_project or {"name": "", "description": ""},
        all_projects=projects,
        max_projects=20,
    )

    run_id = str(uuid.uuid4())

    final_state = await rag_graph.ainvoke(
        {
            "project_id": project_id,
            "collection_name": collection_name,
            "project_context": project_context,
            "original_question": content,
            "reformulated_question": content,
            "question": content,
            "intent": "",
            "search_documents": True,
            "response_style": "default",
            "chat_history": chat_history,
            "reuse_prior_docs": False,
            "retrieval_filters": {},
            "sub_queries": [],
            "documents": [],
            "reranked_documents": prior_docs,
            "relevance_scores": [],
            "confidence": "",
            "generation": "",
            "source_citations": [],
            "validation_passed": True,
            "validation_reason": "",
            "validation_attempts": 0,
            "retry_with_strict_grounding": False,
            "force_retrieval_on_retry": False,
            "retrieval_log": {},
            "messages": [],
        },
        config={
            "configurable": {"thread_id": thread_id},
            "run_id": run_id,
            "metadata": {
                "user_question": content,
                "project_id": project_id,
                "chat_history_length": len(chat_history),
            },
            "tags": ["chainlit"],
        },
    )

    cl.user_session.set("last_run_id", run_id)

    generation = final_state.get("generation", "")
    confidence = final_state.get("confidence", "")
    citations = final_state.get("source_citations", [])

    cl.user_session.set(
        "prior_reranked_docs", list(final_state.get("reranked_documents") or [])
    )

    chat_history.append(HumanMessage(content=content))
    chat_history.append(AIMessage(content=generation))
    cl.user_session.set("chat_history", chat_history[-10:])

    parts: list[str] = [generation] if generation else ["(No response generated)"]

    conf_line = ""
    if confidence:
        emoji = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}.get(confidence, "⚪")
        conf_line = f"{emoji} <b>Confidence</b>: {confidence}"

    if citations:
        lines = []
        for i, c in enumerate(citations, 1):
            source = c.get("source_file", "Unknown")
            page = c.get("page", "")
            section = c.get("section", "")
            loc = f"p.{page}" if page else section if section else ""
            score = c.get("relevance_score", 0)
            entry = (
                f"{i}. <b>{source}</b>"
                + (f" ({loc})" if loc else "")
                + f" — {score:.0%}"
            )
            lines.append(entry)
        inner = f"<br>{'<br>'.join(lines)}"
        if conf_line:
            inner += f"<br><br>{conf_line}"
        parts.append(
            f"\n<details><summary>📄 <b>{len(citations)} source(s)</b></summary>"
            f"{inner}"
            f"</details>"
        )
    elif conf_line:
        parts.append(
            f"\n<details><summary>📊 <b>Retrieval</b></summary><br>{conf_line}</details>"
        )

    await cl.Message(content="\n".join(parts)).send()


@cl.on_feedback
async def on_feedback(feedback: Feedback):
    ls = _get_langsmith_client()
    run_id = cl.user_session.get("last_run_id")
    if ls and run_id:
        ls.create_feedback(
            run_id=run_id,
            key="user_rating",
            score=1.0 if feedback.value == 1 else 0.0,
            comment=feedback.comment or "",
        )


@cl.on_stop
async def on_stop():
    await close_pool()
