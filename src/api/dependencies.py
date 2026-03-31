import os
from dataclasses import dataclass

from langgraph.graph.state import CompiledStateGraph

from config import settings
from src.db.metadata import MetadataStore
from src.db.postgres import get_pool
from src.graph.builder import compile_graph
from src.graph.nodes import set_retrieval_components
from src.retrieval.bm25 import BM25Index
from src.retrieval.vectorstore import VectorStoreManager

os.environ.setdefault("NVIDIA_API_KEY", settings.nvidia_api_key)


@dataclass
class AppComponents:
    vectorstore_manager: VectorStoreManager
    bm25_index: BM25Index
    metadata_store: MetadataStore
    rag_graph: CompiledStateGraph


_components: AppComponents | None = None


async def init_components() -> AppComponents:
    global _components
    if _components is not None:
        return _components

    vsm = VectorStoreManager()
    bm25 = BM25Index()

    pool = await get_pool()
    metadata = MetadataStore(pool)
    await metadata.setup()

    set_retrieval_components(vsm, bm25)

    rag_graph = await compile_graph(pool)

    _components = AppComponents(
        vectorstore_manager=vsm,
        bm25_index=bm25,
        metadata_store=metadata,
        rag_graph=rag_graph,
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
