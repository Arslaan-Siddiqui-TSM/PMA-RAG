from langchain_classic.retrievers.contextual_compression import (
    ContextualCompressionRetriever,
)
from langchain_core.retrievers import BaseRetriever
from langchain_nvidia_ai_endpoints import NVIDIARerank

from config import settings


def build_reranking_retriever(
    base_retriever: BaseRetriever,
) -> ContextualCompressionRetriever:
    reranker = NVIDIARerank(
        model=settings.reranker_model,
        top_n=settings.reranker_top_n,
    )

    return ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=base_retriever,
    )
