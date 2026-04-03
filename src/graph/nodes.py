from __future__ import annotations

import asyncio
import re

import tiktoken
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIARerank
from langsmith import traceable

from config import settings
from src.generation.confidence import normalize_scores
from src.generation.prompts import (
    CASUAL_RESPONSES,
    HELP_RESPONSE,
    RAG_SUMMARY_PROMPT,
    REFORMULATE_PROMPT,
    RESPONSE_STYLE_HINTS,
    UNIFIED_PROMPT,
)
from src.graph.intent import run_intent_triage
from src.graph.state import RAGState
from src.graph.validation import run_quality_gate
from src.retrieval.bm25 import BM25Index
from src.retrieval.hybrid import hybrid_retrieve, reciprocal_rank_fusion
from src.retrieval.vectorstore import VectorStoreManager

_vectorstore_manager: VectorStoreManager | None = None
_bm25_index: BM25Index | None = None
_TOKENIZER = tiktoken.get_encoding("cl100k_base")
_CHAT_META_PATTERNS = re.compile(
    r"(what did i (just )?say|what was my last message|repeat (that|your last response)|"
    r"what did you say|in this chat|conversation so far|summarize our chat)",
    re.IGNORECASE,
)
_FACTUAL_QUERY_PATTERNS = re.compile(
    r"^(how many|what (are|is)|list|show|which|who|where|when|compare|count|give me)",
    re.IGNORECASE,
)
_AMBIGUOUS_FOLLOWUP_PATTERNS = re.compile(
    r"\b(this|that|it|they|them|their|those|these|same|before|above|latter|former|second one|first one)\b",
    re.IGNORECASE,
)
_EXPLAIN_INTENT = re.compile(
    r"^\s*(explain|describe|tell me about|give (me )?(an )?(overview|details?) of)\b",
    re.IGNORECASE,
)
_SUMMARY_INTENT = re.compile(
    r"^\s*(summarize|summary|briefly summarize|give me a summary|overview)\b",
    re.IGNORECASE,
)
_COMPARE_INTENT = re.compile(
    r"^\s*(compare|contrast|difference|differences|vs|versus)\b",
    re.IGNORECASE,
)
_LIST_INTENT = re.compile(r"^\s*(list|show|enumerate)\b", re.IGNORECASE)
_IDENTIFY_INTENT = re.compile(r"^\s*(which|who|what)\b", re.IGNORECASE)
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
    "about",
    "me",
    "my",
    "our",
    "your",
    "this",
    "that",
}

_REASONING_PATTERN = re.compile(
    r"<reasoning>.*?</reasoning>\s*", re.DOTALL | re.IGNORECASE
)


def set_retrieval_components(
    vectorstore_manager: VectorStoreManager, bm25_index: BM25Index
) -> None:
    global _vectorstore_manager, _bm25_index
    _vectorstore_manager = vectorstore_manager
    _bm25_index = bm25_index


def _chunk_id(doc: Document) -> str:
    """Stable unique key for deduplication across retrieval, reflection, and context selection."""
    cid = doc.metadata.get("chunk_id")
    if cid:
        return str(cid)
    return f"{doc.metadata.get('source_file', '')}:{doc.metadata.get('chunk_index', '')}:{hash(doc.page_content)}"


def _detect_query_intent(text: str) -> str:
    if _EXPLAIN_INTENT.search(text):
        return "explain"
    if _SUMMARY_INTENT.search(text):
        return "summary"
    if _COMPARE_INTENT.search(text):
        return "compare"
    if _LIST_INTENT.search(text):
        return "list"
    if _IDENTIFY_INTENT.search(text):
        return "identify"
    return "other"


def _content_tokens(text: str) -> set[str]:
    tokens = {t.lower() for t in re.findall(r"[A-Za-z0-9]+", text)}
    return {t for t in tokens if t not in _STOPWORDS and len(t) > 2}


def _is_standalone_query(question: str) -> bool:
    text = question.strip()
    if not text:
        return False
    if len(
        re.findall(r"[A-Za-z0-9]+", text)
    ) <= 4 and _AMBIGUOUS_FOLLOWUP_PATTERNS.search(text):
        return False
    return not bool(_AMBIGUOUS_FOLLOWUP_PATTERNS.search(text))


def _infer_referent_from_history(chat_history: list[BaseMessage]) -> str:
    recent_text = " ".join((msg.content or "") for msg in chat_history[-6:]).lower()
    if re.search(r"\bdocs?\b|\bdocuments?\b|\bsources?\b|\bchunks?\b", recent_text):
        return "documents"
    if re.search(r"\bprojects?\b", recent_text):
        return "projects"
    if re.search(r"\brequirements?\b", recent_text):
        return "requirements"
    if re.search(r"\bfeatures?\b", recent_text):
        return "features"
    return "items"


def _resolve_ambiguous_followup(question: str, chat_history: list[BaseMessage]) -> str:
    text = question.strip()
    lowered = text.lower()
    if not _AMBIGUOUS_FOLLOWUP_PATTERNS.search(lowered):
        return text

    referent = _infer_referent_from_history(chat_history)

    if re.match(r"(?i)^what\s+are\s+(they|these|those)\??$", text):
        return f"What are the {referent}?"

    if re.match(r"(?i)^explain\b", text):
        if referent == "documents":
            return "Explain all of the documents and what each one covers."
        return f"Explain all of the {referent}."

    if re.match(r"(?i)^summarize\b", text):
        if referent == "documents":
            return "Summarize all of the documents and what each one covers."
        return f"Summarize the {referent}."

    return f"Explain the {referent} mentioned earlier."


def _is_significantly_narrower(original: str, rewritten: str) -> bool:
    original_tokens = _content_tokens(original)
    rewritten_tokens = _content_tokens(rewritten)
    if not original_tokens or not rewritten_tokens:
        return False

    overlap = len(original_tokens & rewritten_tokens)
    coverage = overlap / max(1, len(original_tokens))
    length_ratio = len(rewritten_tokens) / max(1, len(original_tokens))
    if coverage < 0.55 or length_ratio < 0.6:
        return True

    original_intent = _detect_query_intent(original)
    rewritten_intent = _detect_query_intent(rewritten)
    if (
        original_intent in {"explain", "summary", "compare", "list"}
        and rewritten_intent == "identify"
    ):
        return True
    return False


# ---------------------------------------------------------------------------
# Intent classification
# ---------------------------------------------------------------------------


async def classify_intent_node(state: RAGState) -> dict:
    question = state["question"]
    chat_history = state.get("chat_history", [])
    return await run_intent_triage(question, chat_history)


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


@traceable(name="reformulate_query", run_type="chain")
async def reformulate_query_node(state: RAGState) -> dict:
    question = state["question"]
    original_question = state.get("original_question") or question
    chat_history: list[BaseMessage] = state.get("chat_history", [])
    prior_docs: list[Document] = state.get("reranked_documents", [])

    if _is_standalone_query(question):
        return {
            "original_question": original_question,
            "reformulated_question": question,
            "question": question,
            "reuse_prior_docs": False,
        }

    history_lines = []
    for msg in chat_history[-6:]:
        role = "User" if msg.type == "human" else "Assistant"
        history_lines.append(f"{role}: {msg.content[:300]}")
    history_text = "\n".join(history_lines) or "(no prior conversation)"

    llm = ChatNVIDIA(
        model=settings.llm_model,
        temperature=1,
        top_p=0.95,
        disable_streaming=True,
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

    if _AMBIGUOUS_FOLLOWUP_PATTERNS.search(question) and (
        reformulated.strip().lower() == question.strip().lower()
        or _AMBIGUOUS_FOLLOWUP_PATTERNS.search(reformulated)
    ):
        reformulated = _resolve_ambiguous_followup(question, chat_history)
        reuse = False

    if _is_significantly_narrower(question, reformulated):
        reformulated = question
        reuse = False

    can_reuse = reuse and len(prior_docs) > 0

    return {
        "original_question": original_question,
        "reformulated_question": reformulated,
        "question": reformulated,
        "reuse_prior_docs": can_reuse,
    }


# ---------------------------------------------------------------------------
# Retrieval (adaptive)
# ---------------------------------------------------------------------------


def _build_retrieval_filters(state: RAGState) -> dict[str, str]:
    base = dict(state.get("retrieval_filters") or {})
    planned = dict(state.get("planned_filters") or {})
    merged = {**base, **planned}
    return {k: v for k, v in merged.items() if v}


@traceable(name="retrieve", run_type="retriever")
async def retrieve_node(state: RAGState) -> dict:
    sub_queries = state.get("sub_queries") or [state["question"]]
    filters = _build_retrieval_filters(state)
    project_id = state["project_id"]
    collection_name = state["collection_name"]

    vector_k = int(state.get("dynamic_vector_k") or settings.vector_search_k)
    fts_k = int(state.get("dynamic_fts_k") or settings.fts_search_k)

    async def _run_one(query: str) -> list[Document]:
        return await hybrid_retrieve(
            _vectorstore_manager,
            _bm25_index,
            query,
            project_id=project_id,
            collection_name=collection_name,
            filters=filters,
            vector_k=vector_k,
            fts_k=fts_k,
        )

    docs_per_query = list(
        await asyncio.gather(*[_run_one(q) for q in sub_queries])
    )

    if len(docs_per_query) == 1:
        merged = docs_per_query[0]
    else:
        merged = reciprocal_rank_fusion(docs_per_query)
        merged = merged[: max(vector_k, fts_k)]

    seen_ids: set[str] = set()
    deduped: list[Document] = []
    for doc in merged:
        key = _chunk_id(doc)
        if key in seen_ids:
            continue
        seen_ids.add(key)
        deduped.append(doc)

    chunk_ids = [_chunk_id(doc) for doc in deduped]

    return {
        "documents": deduped,
        "prior_retrieved_chunk_ids": chunk_ids,
        "retrieval_filters": filters,
        "retrieval_log": {
            "queries": sub_queries,
            "retrieved_count": len(deduped),
            "vector_k": vector_k,
            "fts_k": fts_k,
        },
    }


# ---------------------------------------------------------------------------
# Reranking (adaptive)
# ---------------------------------------------------------------------------


@traceable(name="rerank", run_type="chain")
async def rerank_node(state: RAGState) -> dict:
    question = state["question"]
    documents = state["documents"]

    if not documents:
        return {
            "reranked_documents": [],
            "relevance_scores": [],
        }

    top_n = int(state.get("dynamic_reranker_top_n") or settings.reranker_top_n)

    reranker = NVIDIARerank(
        model=settings.reranker_model,
        top_n=top_n,
    )
    reranked = await reranker.acompress_documents(documents, query=question)
    reranked = list(reranked)

    raw_scores = [doc.metadata.get("relevance_score", 0.0) for doc in reranked]
    normalized = normalize_scores(raw_scores)

    for doc, norm_score in zip(reranked, normalized):
        doc.metadata["relevance_score_raw"] = doc.metadata.get("relevance_score", 0.0)
        doc.metadata["relevance_score"] = norm_score

    return {
        "reranked_documents": reranked,
        "relevance_scores": normalized,
    }


# ---------------------------------------------------------------------------
# Generation (with CoT and adaptive context)
# ---------------------------------------------------------------------------


def _token_length(text: str) -> int:
    return len(_TOKENIZER.encode(text))


def _select_context_documents(
    documents: list[Document],
    *,
    max_chunks: int | None = None,
    min_threshold: float | None = None,
) -> list[Document]:
    chunk_limit = max_chunks if max_chunks is not None else settings.max_context_chunks
    threshold = min_threshold if min_threshold is not None else 0.0

    deduped: list[Document] = []
    seen: set[str] = set()
    for doc in documents:
        key = _chunk_id(doc)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(doc)

    if not deduped:
        return []

    if threshold > 0:
        above = [
            d for d in deduped
            if float(d.metadata.get("relevance_score", 0.0)) >= threshold
        ]
        candidates = above if above else deduped
    else:
        candidates = deduped

    selected: list[Document] = []
    token_budget = 0
    for doc in candidates:
        if len(selected) >= chunk_limit:
            break
        cost = _token_length(doc.page_content)
        if token_budget + cost > settings.max_context_tokens:
            continue
        selected.append(doc)
        token_budget += cost
    return selected


def _format_context(documents: list[Document]) -> str:
    parts: list[str] = []
    for i, doc in enumerate(documents, 1):
        doc_type = doc.metadata.get("doc_type", "")
        source = doc.metadata.get("source_file", "Unknown")
        page = doc.metadata.get("page", "")
        section = (
            doc.metadata.get("section_title")
            or doc.metadata.get("h1", "")
            or doc.metadata.get("h2", "")
            or "N/A"
        )
        location = f"page {page}" if page else section if section else "N/A"
        parts.append(
            f"[{i}] [Doc: {doc_type or source} | Section: {section}] "
            f"Source: {source}, Location: {location}\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(parts)


def _format_chat_transcript(
    chat_history: list[BaseMessage], *, max_turns: int = 20
) -> str:
    lines: list[str] = []
    for msg in chat_history[-max_turns:]:
        role = "User" if msg.type == "human" else "Assistant"
        text = (msg.content or "").strip()
        if text:
            lines.append(f"{role}: {text}")
    if not lines:
        return "(no prior conversation in this session)"
    return "\n".join(lines)


def _build_citations(
    documents: list[Document], selected_refs: list[int] | None = None
) -> list[dict]:
    selected_indices = set(selected_refs or [])
    citations: list[dict] = []
    for idx, doc in enumerate(documents, 1):
        if selected_indices and idx not in selected_indices:
            continue
        citations.append(
            {
                "chunk_id": doc.metadata.get("chunk_id", ""),
                "source_file": doc.metadata.get("source_file", "Unknown"),
                "page": doc.metadata.get("page", ""),
                "section": (
                    doc.metadata.get("section_title")
                    or doc.metadata.get("h1", "")
                    or doc.metadata.get("h2", "")
                ),
                "doc_type": doc.metadata.get("doc_type", ""),
                "relevance_score": doc.metadata.get("relevance_score", 0.0),
            }
        )
    return citations


def _extract_inline_refs(answer: str) -> list[int]:
    refs = re.findall(r"\[(\d+)\]", answer)
    out: list[int] = []
    for ref in refs:
        try:
            out.append(int(ref))
        except ValueError:
            continue
    return sorted(set(out))


def _is_chat_meta_question(question: str) -> bool:
    return bool(_CHAT_META_PATTERNS.search(question.strip()))


def _should_block_no_search_answer(question: str) -> bool:
    question = question.strip()
    if _is_chat_meta_question(question):
        return False
    return bool(_FACTUAL_QUERY_PATTERNS.search(question))


def _strip_reasoning(text: str) -> str:
    return _REASONING_PATTERN.sub("", text).strip()


@traceable(name="generate", run_type="chain")
async def generate_node(state: RAGState) -> dict:
    retrieval_question = state["question"]
    user_question = state.get("original_question") or retrieval_question
    chat_history: list[BaseMessage] = state.get("chat_history", [])
    chat_transcript = _format_chat_transcript(chat_history)
    project_context = str(state.get("project_context") or "")

    max_chunks = state.get("dynamic_max_context_chunks") or None
    min_threshold = state.get("min_relevance_threshold") or None

    if state.get("search_documents", True):
        reranked_docs = _select_context_documents(
            list(state.get("reranked_documents") or []),
            max_chunks=max_chunks,
            min_threshold=min_threshold,
        )
    else:
        if _should_block_no_search_answer(user_question):
            return {
                "generation": "I don't have enough information in the documents to answer this question.",
                "source_citations": [],
                "confidence": "Low",
                "reranked_documents": [],
            }
        reranked_docs = []

    context = (
        _format_context(reranked_docs)
        if reranked_docs
        else (
            "(no document excerpts retrieved for this turn — answer from the "
            "chat transcript if the question is about the conversation, or say "
            "documents were not searched.)"
        )
    )

    style = state.get("response_style") or "default"
    if style not in RESPONSE_STYLE_HINTS:
        style = "default"
    style_hint = RESPONSE_STYLE_HINTS[style]

    quality_diagnosis = state.get("quality_diagnosis", "")
    if quality_diagnosis == "generation":
        style_hint += (
            " Strict retry mode: every claim must be grounded in context and cite "
            "source chunk references like [1], [2]."
        )

    temperature = 0.7 if state.get("search_documents", True) else 1
    llm = ChatNVIDIA(
        model=settings.llm_model,
        temperature=temperature,
        top_p=0.95,
        disable_streaming=True,
    )
    if style == "summary" and state.get("search_documents", True):
        chain = RAG_SUMMARY_PROMPT | llm
        response = await chain.ainvoke(
            {
                "context": context,
                "project_context": project_context,
                "question": user_question,
            }
        )
    else:
        chain = UNIFIED_PROMPT | llm
        response = await chain.ainvoke(
            {
                "chat_transcript": chat_transcript,
                "context": context,
                "project_context": project_context,
                "question": user_question,
                "response_style_hint": style_hint,
            }
        )

    answer = _strip_reasoning(str(response.content))
    refs = _extract_inline_refs(answer)

    return {
        "generation": answer,
        "source_citations": _build_citations(reranked_docs, refs),
        "reranked_documents": reranked_docs,
    }


# ---------------------------------------------------------------------------
# Quality gate (replaces validate_answer)
# ---------------------------------------------------------------------------


@traceable(name="quality_gate", run_type="chain")
async def quality_gate_node(state: RAGState) -> dict:
    attempts = int(state.get("quality_attempts", 0))
    original_question = state.get("original_question") or state.get("question", "")
    answer = state.get("generation", "")

    missing_info = state.get("missing_information", "")
    if missing_info in (
        "Max retrieval retries reached",
        "Re-retrieval returned same documents",
        "No documents retrieved after retries",
    ):
        return {
            "quality_passed": True,
            "quality_diagnosis": "exhausted_retrieval",
            "quality_reason": "Retrieval retries exhausted; accepting best answer.",
            "quality_attempts": attempts,
        }

    if not state.get("search_documents", True):
        transcript = _format_chat_transcript(state.get("chat_history", []))
        context = f"Conversation transcript:\n{transcript}"
    else:
        context_docs = _select_context_documents(
            list(state.get("reranked_documents") or []),
            max_chunks=state.get("dynamic_max_context_chunks"),
            min_threshold=state.get("min_relevance_threshold"),
        )
        if not context_docs:
            if attempts >= 2:
                return {
                    "quality_passed": True,
                    "quality_diagnosis": "missing_context",
                    "quality_reason": "No retrieved context available after retries.",
                    "quality_attempts": attempts,
                }
            return {
                "quality_passed": False,
                "quality_diagnosis": "missing_context",
                "quality_reason": "No retrieved context available; routing to retrieval.",
                "quality_attempts": attempts + 1,
                "search_documents": True,
                "reuse_prior_docs": False,
                "question": original_question,
                "reformulated_question": original_question,
                "documents": [],
                "reranked_documents": [],
                "source_citations": [],
                "confidence": "Low",
                "retrieval_iterations": 0,
            }
        context = _format_context(context_docs)

    try:
        result = await run_quality_gate(
            question=original_question,
            answer=answer,
            context=context,
        )
    except Exception:
        return {
            "quality_passed": True,
            "quality_diagnosis": "none",
            "quality_reason": "Quality gate LLM call failed; accepting answer.",
            "quality_attempts": attempts,
        }

    passed = (
        result["grounded"]
        and result["coverage"]
        and result["completeness"]
        and not result["hallucination"]
    )

    if passed:
        return {
            "quality_passed": True,
            "quality_diagnosis": "none",
            "quality_reason": result["reason"],
            "quality_attempts": attempts,
        }

    if attempts >= 2:
        return {
            "quality_passed": True,
            "quality_diagnosis": result["diagnosis"],
            "quality_reason": f"Quality gate failed after retries: {result['reason']}",
            "quality_attempts": attempts,
            "generation": answer,
        }

    diagnosis = result["diagnosis"]
    if diagnosis not in ("generation", "missing_context"):
        diagnosis = "generation"

    if not state.get("search_documents", True):
        diagnosis = "missing_context"

    base_result: dict = {
        "quality_passed": False,
        "quality_diagnosis": diagnosis,
        "quality_reason": result["reason"],
        "quality_attempts": attempts + 1,
    }

    if diagnosis == "missing_context":
        base_result.update(
            {
                "search_documents": True,
                "reuse_prior_docs": False,
                "question": original_question,
                "reformulated_question": original_question,
                "documents": [],
                "reranked_documents": [],
                "source_citations": [],
                "confidence": "Low",
                "retrieval_iterations": 0,
            }
        )

    return base_result
