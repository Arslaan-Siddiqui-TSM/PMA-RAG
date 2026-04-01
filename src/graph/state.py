from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage


class RAGState(TypedDict):
    question: str
    # intent: greeting | thanks_bye | help | needs_rag | chat_only
    intent: str
    # If True, run reformulation (when applicable) and document retrieval.
    search_documents: bool
    # "default" | "summary" — tone hint for unified generation.
    response_style: str
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
