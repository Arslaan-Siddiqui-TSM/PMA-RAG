from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_core.retrievers import BaseRetriever

from config import settings
from src.retrieval.bm25 import BM25Index
from src.retrieval.vectorstore import VectorStoreManager


def build_ensemble_retriever(
    vectorstore_manager: VectorStoreManager,
    bm25_index: BM25Index,
    doc_type_filter: str | None = None,
) -> BaseRetriever:
    vector_retriever = vectorstore_manager.as_retriever(doc_type_filter=doc_type_filter)
    bm25_retriever = bm25_index.as_retriever(doc_type_filter=doc_type_filter)

    return EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=settings.ensemble_weights,
    )
