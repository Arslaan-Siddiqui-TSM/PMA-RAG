from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage


class RAGState(TypedDict):
    project_id: str
    collection_name: str
    project_context: str
    original_question: str
    reformulated_question: str
    question: str
    intent: str
    search_documents: bool
    response_style: str
    chat_history: list[BaseMessage]
    reuse_prior_docs: bool
    retrieval_filters: dict[str, str]
    sub_queries: list[str]
    documents: list[Document]
    reranked_documents: list[Document]
    relevance_scores: list[float]
    confidence: str
    generation: str
    source_citations: list[dict]
    validation_passed: bool
    validation_reason: str
    validation_attempts: int
    retry_with_strict_grounding: bool
    force_retrieval_on_retry: bool
    retrieval_log: dict
    messages: list[BaseMessage]
