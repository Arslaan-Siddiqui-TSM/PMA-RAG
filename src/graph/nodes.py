from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIARerank

from config import settings
from src.generation.confidence import compute_confidence, normalize_scores
from src.generation.prompts import (
    CASUAL_RESPONSES,
    HELP_RESPONSE,
    RAG_PROMPT,
    REFORMULATE_PROMPT,
)
from src.graph.intent import classify_intent
from src.graph.state import RAGState
from src.retrieval.bm25 import BM25Index
from src.retrieval.hybrid import build_ensemble_retriever
from src.retrieval.vectorstore import VectorStoreManager

_vectorstore_manager: VectorStoreManager | None = None
_bm25_index: BM25Index | None = None


def set_retrieval_components(
    vectorstore_manager: VectorStoreManager, bm25_index: BM25Index
) -> None:
    global _vectorstore_manager, _bm25_index
    _vectorstore_manager = vectorstore_manager
    _bm25_index = bm25_index


# ---------------------------------------------------------------------------
# Intent classification
# ---------------------------------------------------------------------------

async def classify_intent_node(state: RAGState) -> dict:
    question = state["question"]
    chat_history = state.get("chat_history", [])
    intent = await classify_intent(question, chat_history)
    return {"intent": intent}


# ---------------------------------------------------------------------------
# Casual / help responses (no retrieval)
# ---------------------------------------------------------------------------

async def casual_response_node(state: RAGState) -> dict:
    intent = state.get("intent", "greeting")
    response = CASUAL_RESPONSES.get(intent, CASUAL_RESPONSES["greeting"])
    return {
        "generation": response,
        "confidence": "",
        "source_citations": [],
    }


async def help_response_node(state: RAGState) -> dict:
    return {
        "generation": HELP_RESPONSE,
        "confidence": "",
        "source_citations": [],
    }


# ---------------------------------------------------------------------------
# Follow-up reformulation
# ---------------------------------------------------------------------------

async def reformulate_query_node(state: RAGState) -> dict:
    question = state["question"]
    chat_history: list[BaseMessage] = state.get("chat_history", [])
    prior_docs: list[Document] = state.get("reranked_documents", [])

    history_lines = []
    for msg in chat_history[-6:]:
        role = "User" if msg.type == "human" else "Assistant"
        history_lines.append(f"{role}: {msg.content[:300]}")
    history_text = "\n".join(history_lines) or "(no prior conversation)"

    llm = ChatNVIDIA(
        model=settings.llm_model,
        temperature=0.0,
        max_tokens=200,
    )
    prompt = REFORMULATE_PROMPT.format(
        chat_history=history_text,
        question=question,
    )
    response = await llm.ainvoke(prompt)
    raw = response.content.strip()

    reuse = False
    reformulated = question
    if raw.upper().startswith("REUSE:"):
        reuse = True
        reformulated = raw[6:].strip()
    elif raw.upper().startswith("RETRIEVE:"):
        reuse = False
        reformulated = raw[9:].strip()
    else:
        reformulated = raw

    if not reformulated:
        reformulated = question

    can_reuse = reuse and len(prior_docs) > 0

    return {
        "question": reformulated,
        "reuse_prior_docs": can_reuse,
    }


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

async def retrieve_node(state: RAGState) -> dict:
    question = state["question"]
    doc_type_filter = state.get("doc_type_filter")

    ensemble = build_ensemble_retriever(
        _vectorstore_manager, _bm25_index, doc_type_filter=doc_type_filter
    )
    documents = await ensemble.ainvoke(question)
    return {"documents": documents}


# ---------------------------------------------------------------------------
# Reranking
# ---------------------------------------------------------------------------

async def rerank_node(state: RAGState) -> dict:
    question = state["question"]
    documents = state["documents"]

    if not documents:
        return {
            "reranked_documents": [],
            "relevance_scores": [],
        }

    reranker = NVIDIARerank(
        model=settings.reranker_model,
        top_n=settings.reranker_top_n,
    )
    reranked = await reranker.acompress_documents(documents, query=question)
    reranked = list(reranked)

    raw_scores = [
        doc.metadata.get("relevance_score", 0.0) for doc in reranked
    ]
    normalized = normalize_scores(raw_scores)

    for doc, norm_score in zip(reranked, normalized):
        doc.metadata["relevance_score_raw"] = doc.metadata.get(
            "relevance_score", 0.0
        )
        doc.metadata["relevance_score"] = norm_score

    return {
        "reranked_documents": reranked,
        "relevance_scores": normalized,
    }


# ---------------------------------------------------------------------------
# Relevance check
# ---------------------------------------------------------------------------

async def check_relevance_node(state: RAGState) -> dict:
    scores = state.get("relevance_scores", [])
    confidence = compute_confidence(scores)
    return {"confidence": confidence}


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _format_context(documents: list[Document]) -> str:
    parts: list[str] = []
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("source_file", "Unknown")
        page = doc.metadata.get("page", "")
        section = doc.metadata.get("h1", "") or doc.metadata.get("h2", "")
        location = f"page {page}" if page else section if section else "N/A"
        parts.append(
            f"[{i}] Source: {source}, Location: {location}\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(parts)


def _build_citations(documents: list[Document]) -> list[dict]:
    citations: list[dict] = []
    for doc in documents:
        citations.append(
            {
                "source_file": doc.metadata.get("source_file", "Unknown"),
                "page": doc.metadata.get("page", ""),
                "section": (
                    doc.metadata.get("h1", "") or doc.metadata.get("h2", "")
                ),
                "doc_type": doc.metadata.get("doc_type", ""),
                "relevance_score": doc.metadata.get("relevance_score", 0.0),
            }
        )
    return citations


async def generate_node(state: RAGState) -> dict:
    question = state["question"]
    reranked_docs = state["reranked_documents"]
    context = _format_context(reranked_docs)

    chain = RAG_PROMPT | ChatNVIDIA(
        model=settings.llm_model,
        temperature=0.1,
        max_tokens=1024,
    )
    response = await chain.ainvoke(
        {"context": context, "question": question}
    )

    return {
        "generation": response.content,
        "source_citations": _build_citations(reranked_docs),
    }


# ---------------------------------------------------------------------------
# No answer fallback
# ---------------------------------------------------------------------------

async def no_answer_node(state: RAGState) -> dict:
    return {
        "generation": (
            "I don't have enough information in the documents "
            "to answer this question."
        ),
        "confidence": "Low",
        "source_citations": [],
    }
