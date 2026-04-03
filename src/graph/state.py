from __future__ import annotations

from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage


class RAGState(TypedDict, total=False):
    # Core fields (always expected in the initial state)
    project_id: str
    collection_name: str
    project_context: str
    original_question: str
    reformulated_question: str
    question: str

    # Intent / triage
    intent: str
    search_documents: bool
    response_style: str

    # Conversation context
    chat_history: list[BaseMessage]
    reuse_prior_docs: bool
    retrieval_filters: dict[str, str]
    messages: list[BaseMessage]

    # Retrieval
    sub_queries: list[str]
    documents: list[Document]
    reranked_documents: list[Document]
    relevance_scores: list[float]

    # Generation
    confidence: str
    generation: str
    source_citations: list[dict]
    retrieval_log: dict

    # Document catalog (for planner corpus awareness)
    document_catalog: list[dict]

    # Planning fields
    query_type: str
    query_complexity: str
    retrieval_plan: dict
    dynamic_vector_k: int
    dynamic_fts_k: int
    dynamic_reranker_top_n: int
    min_relevance_threshold: float
    dynamic_max_context_chunks: int
    planned_filters: dict

    # Reflection fields
    retrieval_sufficient: bool
    retrieval_iterations: int
    missing_information: str
    prior_retrieved_chunk_ids: list[str]

    # Quality gate fields
    quality_passed: bool
    quality_diagnosis: str
    quality_reason: str
    quality_attempts: int


def build_default_state(
    *,
    question: str,
    project_id: str,
    collection_name: str,
    project_context: str,
    chat_history: list[BaseMessage] | None = None,
    reranked_documents: list[Document] | None = None,
    document_catalog: list[dict] | None = None,
) -> dict:
    """Build a default initial state dict for the RAG graph.

    Centralises the boilerplate so that the API, Chainlit, and eval runners
    all start from an identical baseline.
    """
    return {
        "project_id": project_id,
        "collection_name": collection_name,
        "project_context": project_context,
        "original_question": question,
        "reformulated_question": question,
        "question": question,
        "intent": "",
        "search_documents": True,
        "response_style": "default",
        "chat_history": chat_history or [],
        "reuse_prior_docs": False,
        "retrieval_filters": {},
        "sub_queries": [],
        "documents": [],
        "reranked_documents": reranked_documents or [],
        "relevance_scores": [],
        "confidence": "",
        "generation": "",
        "source_citations": [],
        "retrieval_log": {},
        "messages": [],
        "document_catalog": document_catalog or [],
        "query_type": "",
        "query_complexity": "",
        "retrieval_plan": {},
        "dynamic_vector_k": 0,
        "dynamic_fts_k": 0,
        "dynamic_reranker_top_n": 0,
        "min_relevance_threshold": 0.0,
        "dynamic_max_context_chunks": 0,
        "planned_filters": {},
        "retrieval_sufficient": True,
        "retrieval_iterations": 0,
        "missing_information": "",
        "prior_retrieved_chunk_ids": [],
        "quality_passed": True,
        "quality_diagnosis": "",
        "quality_reason": "",
        "quality_attempts": 0,
    }
