from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage


class RAGState(TypedDict):
    question: str
    intent: str
    chat_history: list[BaseMessage]
    reuse_prior_docs: bool
    doc_type_filter: str | None
    documents: list[Document]
    reranked_documents: list[Document]
    relevance_scores: list[float]
    confidence: str
    generation: str
    source_citations: list[dict]
    messages: list[BaseMessage]
