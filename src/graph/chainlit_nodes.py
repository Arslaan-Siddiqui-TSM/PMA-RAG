import chainlit as cl

from config import settings
from src.graph import nodes as core_nodes
from src.graph.nodes import (
    _bm25_index,
    _build_citations,
    _format_context,
    set_retrieval_components,
)
from src.graph.state import RAGState

__all__ = [
    "set_retrieval_components",
    "classify_intent_node",
    "casual_response_node",
    "help_response_node",
    "reformulate_query_node",
    "retrieve_node",
    "rerank_node",
    "check_relevance_node",
    "generate_node",
    "no_answer_node",
]


# ---------------------------------------------------------------------------
# Intent classification
# ---------------------------------------------------------------------------

async def classify_intent_node(state: RAGState) -> dict:
    async with cl.Step(
        name="Classify Intent",
        type="tool",
        show_input=True,
    ) as step:
        step.input = f"**User message:** {state['question']}"
        result = await core_nodes.classify_intent_node(state)
        chat_history = state.get("chat_history", [])
        step.output = (
            f"**Detected intent: `{result['intent']}`**\n"
            f"**Chat history length:** {len(chat_history)} messages"
        )
    return result


# ---------------------------------------------------------------------------
# Casual / help responses (no retrieval)
# ---------------------------------------------------------------------------

async def casual_response_node(state: RAGState) -> dict:
    async with cl.Step(
        name="Casual Response",
        type="tool",
        show_input=True,
    ) as step:
        step.input = f"**Intent:** {state.get('intent', 'greeting')}"
        result = await core_nodes.casual_response_node(state)
        step.output = f"**Response:** {result['generation']}"
    return result


async def help_response_node(state: RAGState) -> dict:
    async with cl.Step(
        name="Help Response",
        type="tool",
        show_input=True,
    ) as step:
        step.input = "**Intent:** help"
        result = await core_nodes.help_response_node(state)
        step.output = f"**Response:**\n{result['generation']}"
    return result


# ---------------------------------------------------------------------------
# Follow-up reformulation
# ---------------------------------------------------------------------------

async def reformulate_query_node(state: RAGState) -> dict:
    question = state["question"]
    chat_history = state.get("chat_history", [])
    prior_docs = state.get("reranked_documents", [])

    async with cl.Step(
        name="Reformulate Follow-up",
        type="tool",
        show_input=True,
    ) as step:
        step.input = (
            f"**Original question:** {question}\n"
            f"**Chat history:** {len(chat_history)} messages\n"
            f"**Prior reranked docs available:** {len(prior_docs)}"
        )
        result = await core_nodes.reformulate_query_node(state)
        reformulated = result["question"]
        can_reuse = result["reuse_prior_docs"]
        step.output = (
            f"**Reformulated question:** {reformulated}\n"
            f"**Final routing:** "
            f"{'reuse prior docs → generate' if can_reuse else 'retrieve new docs'}"
        )
    return result


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

async def retrieve_node(state: RAGState) -> dict:
    question = state["question"]
    doc_type_filter = state.get("doc_type_filter")

    async with cl.Step(
        name="Retrieve Documents",
        type="retrieval",
        show_input=True,
    ) as step:
        step.input = (
            f"**Query:** {question}\n"
            f"**Filter:** {doc_type_filter or 'None (all documents)'}\n"
            f"**BM25 index size:** "
            f"{_bm25_index.document_count if _bm25_index else 0} chunks"
        )
        result = await core_nodes.retrieve_node(state)
        documents = result["documents"]

        if not documents:
            step.output = (
                "**No documents retrieved.** The vector store and BM25 index "
                "returned 0 results."
            )
        else:
            lines = [f"**Retrieved {len(documents)} documents:**\n"]
            for i, doc in enumerate(documents, 1):
                source = doc.metadata.get("source_file", "Unknown")
                page = doc.metadata.get("page", "")
                doc_type = doc.metadata.get("doc_type", "")
                loc = f", page {page}" if page else ""
                dtype = f" [{doc_type}]" if doc_type else ""
                lines.append(
                    f"---\n**[{i}] {source}{loc}{dtype}**\n"
                    f"```\n{doc.page_content}\n```\n"
                )
            step.output = "\n".join(lines)
    return result


# ---------------------------------------------------------------------------
# Reranking
# ---------------------------------------------------------------------------

async def rerank_node(state: RAGState) -> dict:
    question = state["question"]
    documents = state["documents"]

    async with cl.Step(
        name="Rerank Documents",
        type="rerank",
        show_input=True,
    ) as step:
        step.input = (
            f"**Query:** {question}\n"
            f"**Input documents:** {len(documents)}\n"
            f"**Reranker model:** {settings.reranker_model}\n"
            f"**Top N:** {settings.reranker_top_n}"
        )

        if not documents:
            step.output = (
                "**No documents to rerank.** Retrieval returned 0 results."
            )
            return {
                "reranked_documents": [],
                "relevance_scores": [],
            }

        result = await core_nodes.rerank_node(state)
        reranked = result["reranked_documents"]
        normalized = result["relevance_scores"]

        lines = [f"**Reranked to {len(reranked)} documents:**\n"]
        for i, doc in enumerate(reranked, 1):
            source = doc.metadata.get("source_file", "Unknown")
            page = doc.metadata.get("page", "")
            raw = doc.metadata.get("relevance_score_raw", 0.0)
            norm = doc.metadata.get("relevance_score", 0.0)
            loc = f", page {page}" if page else ""
            lines.append(
                f"---\n**[{i}] {source}{loc} — "
                f"raw logit: {raw:.4f} → probability: {norm:.4f}**\n"
                f"```\n{doc.page_content}\n```\n"
            )

        if normalized:
            lines.append(
                f"\n**Score summary (normalized):** "
                f"min={min(normalized):.4f}, "
                f"max={max(normalized):.4f}, "
                f"mean={sum(normalized)/len(normalized):.4f}"
            )
        step.output = "\n".join(lines)
    return result


# ---------------------------------------------------------------------------
# Relevance check
# ---------------------------------------------------------------------------

async def check_relevance_node(state: RAGState) -> dict:
    scores = state.get("relevance_scores", [])

    async with cl.Step(
        name="Check Relevance",
        type="tool",
        show_input=True,
    ) as step:
        step.input = (
            f"**Normalized scores (0-1):** {[round(s, 4) for s in scores]}\n"
            f"**Thresholds:**\n"
            f"  - High: top >= {settings.confidence_high_threshold} AND "
            f"{settings.confidence_high_min_docs}+ docs >= "
            f"{settings.confidence_high_doc_threshold}\n"
            f"  - Medium: top >= {settings.confidence_medium_threshold} OR "
            f"{settings.confidence_medium_min_docs}+ docs >= "
            f"{settings.confidence_medium_doc_threshold}\n"
            f"  - Low: everything else"
        )

        result = await core_nodes.check_relevance_node(state)
        confidence = result["confidence"]

        reranked_docs = state.get("reranked_documents", [])
        will_route = (
            "no_answer" if (not reranked_docs or not scores) else "generate"
        )

        top_display = f"{max(scores):.4f}" if scores else "N/A"
        step.output = (
            f"**Confidence: {confidence}**\n"
            f"**Top normalized score: {top_display}**\n"
            f"**Routing to: `{will_route}`**"
        )
    return result


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

async def generate_node(state: RAGState) -> dict:
    reranked_docs = state["reranked_documents"]
    context = _format_context(reranked_docs)

    async with cl.Step(
        name="Generate Answer",
        type="llm",
        show_input=True,
    ) as step:
        step.input = (
            f"**Model:** {settings.llm_model}\n"
            f"**Context chunks:** {len(reranked_docs)}\n"
            f"**Context length:** {len(context)} chars\n\n"
            f"**Full prompt context sent to LLM:**\n"
            f"```\n{context[:2000]}"
            f"{'...(truncated)' if len(context) > 2000 else ''}\n```"
        )
        result = await core_nodes.generate_node(state)
        step.output = f"**LLM Response:**\n{result['generation']}"
    return result


# ---------------------------------------------------------------------------
# No answer fallback
# ---------------------------------------------------------------------------

async def no_answer_node(state: RAGState) -> dict:
    scores = state.get("relevance_scores", [])

    async with cl.Step(
        name="No Answer (Insufficient Context)",
        type="tool",
        show_input=True,
    ) as step:
        step.input = (
            f"**Reason:** No relevant documents found after reranking\n"
            f"**Normalized scores:** {[round(s, 4) for s in scores]}"
        )
        step.output = (
            "Returning: *I don't have enough information in the documents "
            "to answer this question.*"
        )
    result = await core_nodes.no_answer_node(state)
    return result
