import asyncio
import os
import sys
import tempfile
import uuid

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import chainlit as cl
from chainlit.input_widget import Select
from engineio.payload import Payload
from langchain_core.messages import AIMessage, HumanMessage

from config import settings
from src.db.metadata import MetadataStore
from src.db.postgres import close_pool, get_pool
from src.graph.builder import compile_graph
from src.graph.nodes import set_retrieval_components
from src.ingestion.pipeline import ingest_document
from src.retrieval.bm25 import BM25Index
from src.retrieval.vectorstore import VectorStoreManager

Payload.max_decode_packets = 500

os.environ.setdefault("NVIDIA_API_KEY", settings.nvidia_api_key)

_vectorstore_manager: VectorStoreManager | None = None
_bm25_index: BM25Index | None = None
_metadata_store: MetadataStore | None = None


async def _get_components():
    global _vectorstore_manager, _bm25_index, _metadata_store

    if _vectorstore_manager is None:
        _vectorstore_manager = VectorStoreManager()
    if _bm25_index is None:
        _bm25_index = BM25Index()
    if _metadata_store is None:
        pool = await get_pool()
        _metadata_store = MetadataStore(pool)
        await _metadata_store.setup()

    set_retrieval_components(_vectorstore_manager, _bm25_index)
    return _vectorstore_manager, _bm25_index, _metadata_store


@cl.on_chat_start
async def on_chat_start():
    vsm, bm25, metadata = await _get_components()

    thread_id = str(uuid.uuid4())
    cl.user_session.set("thread_id", thread_id)

    pool = await get_pool()
    rag_graph = await compile_graph(pool)
    cl.user_session.set("rag_graph", rag_graph)

    doc_types = await metadata.get_all_doc_types()
    filter_options = ["All"] + doc_types

    chat_settings = [
        Select(
            id="doc_type_filter",
            label="Filter by Document Type",
            values=filter_options,
            initial_value="All",
        ),
    ]
    await cl.ChatSettings(chat_settings).send()
    cl.user_session.set("doc_type_filter", None)
    cl.user_session.set("chat_history", [])
    cl.user_session.set("prior_reranked_docs", [])

    await cl.Message(
        content=(
            "Welcome to the **PMA-RAG Chatbot**! I can answer questions "
            "about your project documents.\n\n"
            "- **Upload documents** using the button below "
            "(PDF, DOCX, or Markdown)\n"
            "- **Ask questions** about your uploaded documents\n"
            "- **Filter by document type** using the settings panel\n\n"
            "Upload some documents to get started, or ask a question "
            "if documents are already loaded."
        )
    ).send()


@cl.on_settings_update
async def on_settings_update(new_settings: dict):
    value = new_settings.get("doc_type_filter", "All")
    cl.user_session.set("doc_type_filter", None if value == "All" else value)


@cl.action_callback("upload_documents")
async def on_upload_action(action: cl.Action):
    await _handle_file_upload()


async def _handle_file_upload():
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
            )
            if chunks > 0:
                total_chunks += chunks
                results.append(f"  - **{file.name}** ({doc_type}): {chunks} chunks")
            else:
                results.append(f"  - **{file.name}**: already ingested (skipped)")
        except Exception as e:
            results.append(f"  - **{file.name}**: error - {e}")

    doc_types = await metadata.get_all_doc_types()
    filter_options = ["All"] + doc_types
    chat_settings = [
        Select(
            id="doc_type_filter",
            label="Filter by Document Type",
            values=filter_options,
            initial_value="All",
        ),
    ]
    await cl.ChatSettings(chat_settings).send()

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
            )
            if chunks > 0:
                total_chunks += chunks
                results.append(f"  - **{el.name}** ({doc_type}): {chunks} chunks")
            else:
                results.append(f"  - **{el.name}**: already ingested (skipped)")
        except Exception as e:
            results.append(f"  - **{el.name}**: error - {e}")

    doc_types = await metadata.get_all_doc_types()
    filter_options = ["All"] + doc_types
    chat_settings = [
        Select(
            id="doc_type_filter",
            label="Filter by Document Type",
            values=filter_options,
            initial_value="All",
        ),
    ]
    await cl.ChatSettings(chat_settings).send()

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
    doc_type_filter = cl.user_session.get("doc_type_filter")

    if rag_graph is None:
        await cl.Message(content="Session not initialized. Please refresh.").send()
        return

    chat_history = cl.user_session.get("chat_history", [])
    prior_docs = cl.user_session.get("prior_reranked_docs", [])

    final_state = await rag_graph.ainvoke(
        {
            "question": content,
            "intent": "",
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
        },
        config={"configurable": {"thread_id": thread_id}},
    )

    generation = final_state.get("generation", "")
    confidence = final_state.get("confidence", "")
    citations = final_state.get("source_citations", [])

    new_reranked = final_state.get("reranked_documents", [])
    if new_reranked:
        cl.user_session.set("prior_reranked_docs", new_reranked)

    chat_history.append(HumanMessage(content=content))
    chat_history.append(AIMessage(content=generation))
    cl.user_session.set("chat_history", chat_history[-10:])

    parts: list[str] = [generation] if generation else ["(No response generated)"]

    if confidence:
        parts.append(f"\n\n**Confidence**: {confidence}")

    if citations:
        parts.append("\n**Sources:**")
        parts.append("| # | Document | Location | Relevance |")
        parts.append("|---|----------|----------|-----------|")
        for i, c in enumerate(citations, 1):
            source = c.get("source_file", "Unknown")
            page = c.get("page", "")
            section = c.get("section", "")
            location = f"Page {page}" if page else section if section else "—"
            score = c.get("relevance_score", 0)
            parts.append(f"| {i} | {source} | {location} | {score:.2f} |")

    await cl.Message(content="\n".join(parts)).send()


@cl.on_stop
async def on_stop():
    await close_pool()
