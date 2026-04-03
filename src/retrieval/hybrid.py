from __future__ import annotations

import asyncio

from langchain_core.documents import Document

from config import settings
from src.retrieval.bm25 import BM25Index
from src.retrieval.vectorstore import VectorStoreManager


def reciprocal_rank_fusion(
    result_lists: list[list[Document]],
    *,
    k: int = 60,
) -> list[Document]:
    scores: dict[str, float] = {}
    docs_by_id: dict[str, Document] = {}

    for results in result_lists:
        for rank, doc in enumerate(results):
            doc_id = str(
                doc.metadata.get("chunk_id")
                or doc.metadata.get("id")
                or f"{doc.metadata.get('source_file', '')}:{rank}:{hash(doc.page_content)}"
            )
            docs_by_id[doc_id] = doc
            scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (k + rank + 1))

    ranked_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    fused_docs = [docs_by_id[doc_id] for doc_id in ranked_ids]
    for doc_id in ranked_ids:
        docs_by_id[doc_id].metadata["fusion_score"] = scores[doc_id]
    return fused_docs


async def hybrid_retrieve(
    vectorstore_manager: VectorStoreManager,
    bm25_index: BM25Index,
    query: str,
    *,
    project_id: str,
    collection_name: str,
    filters: dict[str, str] | None = None,
    vector_k: int | None = None,
    fts_k: int | None = None,
) -> list[Document]:
    vk = vector_k if vector_k is not None else settings.vector_search_k
    fk = fts_k if fts_k is not None else settings.fts_search_k
    merged_filters = dict(filters or {})
    vector_task = vectorstore_manager.similarity_search(
        collection_name,
        query,
        k=vk,
        filters=merged_filters or None,
    )
    fts_task = bm25_index.search(
        query,
        project_id=project_id,
        k=fk,
        doc_type_filter=merged_filters.get("doc_type"),
        source_file_filter=merged_filters.get("source_file"),
        section_filter=merged_filters.get("section_title"),
    )
    vector_docs, lexical_docs = await asyncio.gather(vector_task, fts_task)
    fused = reciprocal_rank_fusion([vector_docs, lexical_docs])
    top_k = max(vk, fk)
    return fused[:top_k]
