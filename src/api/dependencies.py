import os
from dataclasses import dataclass

from langgraph.graph.state import CompiledStateGraph
from langsmith import Client as LangSmithClient

from config import settings
from src.db.chat_store import ChatStore
from src.db.metadata import MetadataStore
from src.db.postgres import get_pool
from src.graph.builder import compile_graph
from src.graph.nodes import set_retrieval_components
from src.retrieval.bm25 import BM25Index
from src.retrieval.vectorstore import VectorStoreManager

os.environ.setdefault("NVIDIA_API_KEY", settings.nvidia_api_key)

if settings.langsmith_api_key:
    os.environ.setdefault("LANGSMITH_API_KEY", settings.langsmith_api_key)
    if settings.langsmith_tracing:
        os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", settings.langsmith_project)
    os.environ.setdefault("LANGSMITH_ENDPOINT", settings.langsmith_endpoint)
    if settings.langsmith_workspace_id:
        os.environ.setdefault("LANGSMITH_WORKSPACE_ID", settings.langsmith_workspace_id)


@dataclass
class AppComponents:
    vectorstore_manager: VectorStoreManager
    bm25_index: BM25Index
    metadata_store: MetadataStore
    chat_store: ChatStore
    rag_graph: CompiledStateGraph
    langsmith_client: LangSmithClient | None


_components: AppComponents | None = None


def _init_langsmith_client() -> LangSmithClient | None:
    if not settings.langsmith_api_key:
        return None
    return LangSmithClient()


async def init_components() -> AppComponents:
    global _components
    if _components is not None:
        return _components

    pool = await get_pool()
    vsm = VectorStoreManager()
    bm25 = BM25Index(pool)
    metadata = MetadataStore(pool)
    await metadata.setup()

    chat_store = ChatStore(pool)
    await chat_store.setup()

    set_retrieval_components(vsm, bm25)

    rag_graph = await compile_graph(pool)

    _components = AppComponents(
        vectorstore_manager=vsm,
        bm25_index=bm25,
        metadata_store=metadata,
        chat_store=chat_store,
        rag_graph=rag_graph,
        langsmith_client=_init_langsmith_client(),
    )
    return _components


async def shutdown_components() -> None:
    global _components
    from src.db.postgres import close_pool

    _components = None
    await close_pool()


def get_components() -> AppComponents:
    if _components is None:
        raise RuntimeError("App components not initialized")
    return _components
