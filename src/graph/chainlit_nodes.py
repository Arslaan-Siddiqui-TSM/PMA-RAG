import chainlit as cl

from config import settings
from src.graph import nodes as core_nodes
from src.graph.nodes import (
    _format_context,
    set_retrieval_components,
)
from src.graph.planner import plan_retrieval_node as core_plan_retrieval_node
from src.graph.reflection import reflect_on_retrieval_node as core_reflect_node
from src.graph.state import RAGState

__all__ = [
    "set_retrieval_components",
    "classify_intent_node",
    "casual_response_node",
    "help_response_node",
    "reformulate_query_node",
    "plan_retrieval_node",
    "retrieve_node",
    "rerank_node",
    "reflect_on_retrieval_node",
    "generate_node",
    "quality_gate_node",
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
            f"**Intent / route: `{result['intent']}`**\n"
            f"**Search documents:** {result.get('search_documents', True)}\n"
            f"**Response style:** {result.get('response_style', 'default')}\n"
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
            f"{'reuse prior docs → generate' if can_reuse else 'plan retrieval → retrieve'}"
        )
    return result


# ---------------------------------------------------------------------------
# Retrieval planning (replaces decompose_query)
# ---------------------------------------------------------------------------


async def plan_retrieval_node(state: RAGState) -> dict:
    async with cl.Step(
        name="Plan Retrieval",
        type="tool",
        show_input=True,
    ) as step:
        step.input = (
            f"**Question:** {state['question']}\n"
            f"**Documents in catalog:** {len(state.get('document_catalog', []))}"
        )
        result = await core_plan_retrieval_node(state)
        sub_queries = result.get("sub_queries", [])
        step.output = (
            f"**Query type:** {result.get('query_type', 'unknown')}\n"
            f"**Complexity:** {result.get('query_complexity', 'unknown')}\n"
            f"**Sub-queries ({len(sub_queries)}):**\n"
            + "\n".join(f"- {q}" for q in sub_queries)
            + f"\n**Vector K:** {result.get('dynamic_vector_k')}\n"
            f"**FTS K:** {result.get('dynamic_fts_k')}\n"
            f"**Reranker top N:** {result.get('dynamic_reranker_top_n')}\n"
            f"**Relevance threshold:** {result.get('min_relevance_threshold')}\n"
            f"**Max context chunks:** {result.get('dynamic_max_context_chunks')}\n"
            f"**Filters:** {result.get('planned_filters', {})}"
        )
    return result


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


async def retrieve_node(state: RAGState) -> dict:
    question = state["question"]
    project_id = state.get("project_id", "")

    async with cl.Step(
        name="Retrieve Documents",
        type="retrieval",
        show_input=True,
    ) as step:
        step.input = (
            f"**Query:** {question}\n"
            f"**Project:** {project_id}\n"
            f"**Vector K:** {state.get('dynamic_vector_k', settings.vector_search_k)}\n"
            f"**FTS K:** {state.get('dynamic_fts_k', settings.fts_search_k)}"
        )
        result = await core_nodes.retrieve_node(state)
        documents = result["documents"]

        if not documents:
            step.output = (
                "**No documents retrieved.** The vector store and FTS "
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
        top_n = state.get("dynamic_reranker_top_n") or settings.reranker_top_n
        step.input = (
            f"**Query:** {question}\n"
            f"**Input documents:** {len(documents)}\n"
            f"**Reranker model:** {settings.reranker_model}\n"
            f"**Top N:** {top_n}"
        )

        result = await core_nodes.rerank_node(state)
        reranked = result["reranked_documents"]
        normalized = result["relevance_scores"]

        if not reranked:
            step.output = "**No documents to rerank.** Retrieval returned 0 results."
            return result

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
                f"mean={sum(normalized) / len(normalized):.4f}"
            )
        step.output = "\n".join(lines)
    return result


# ---------------------------------------------------------------------------
# Retrieval reflection (replaces check_relevance)
# ---------------------------------------------------------------------------


async def reflect_on_retrieval_node(state: RAGState) -> dict:
    reranked_docs = state.get("reranked_documents", [])
    iterations = state.get("retrieval_iterations", 0)

    async with cl.Step(
        name="Reflect on Retrieval",
        type="tool",
        show_input=True,
    ) as step:
        step.input = (
            f"**Question:** {state.get('original_question') or state['question']}\n"
            f"**Reranked documents:** {len(reranked_docs)}\n"
            f"**Retrieval iteration:** {iterations}\n"
            f"**Query type:** {state.get('query_type', 'unknown')}"
        )
        result = await core_reflect_node(state)
        sufficient = result.get("retrieval_sufficient", True)
        step.output = (
            f"**Sufficient:** {sufficient}\n"
            f"**Missing info:** {result.get('missing_information', 'none')}\n"
            f"**Routing to:** {'generate' if sufficient else 'retrieve (retry)'}"
        )
    return result


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


async def generate_node(state: RAGState) -> dict:
    reranked_docs = state.get("reranked_documents", [])
    if not state.get("search_documents", True):
        reranked_docs = []
    context = _format_context(reranked_docs) if reranked_docs else "(skipped retrieval)"
    chat_hist = state.get("chat_history", [])
    transcript_preview = "\n".join(
        f"{'U' if m.type == 'human' else 'A'}: {(m.content or '')[:200]}"
        for m in chat_hist[-6:]
    )

    async with cl.Step(
        name="Generate Answer",
        type="llm",
        show_input=True,
    ) as step:
        step.input = (
            f"**Model:** {settings.llm_model}\n"
            f"**Search documents:** {state.get('search_documents', True)}\n"
            f"**Response style:** {state.get('response_style', 'default')}\n"
            f"**Query type:** {state.get('query_type', 'unknown')}\n"
            f"**Context chunks:** {len(reranked_docs)}\n"
            f"**Context length:** {len(context)} chars\n\n"
            f"**Chat transcript (preview):**\n```\n{transcript_preview or '(empty)'}\n```\n\n"
            f"**Retrieved context (preview):**\n```\n{context[:2000]}"
            f"{'...(truncated)' if len(context) > 2000 else ''}\n```"
        )
        result = await core_nodes.generate_node(state)
        step.output = f"**LLM Response:**\n{result['generation']}"
    return result


# ---------------------------------------------------------------------------
# Quality gate (replaces validate_answer)
# ---------------------------------------------------------------------------


async def quality_gate_node(state: RAGState) -> dict:
    async with cl.Step(
        name="Quality Gate",
        type="tool",
        show_input=True,
    ) as step:
        step.input = (
            f"**Quality attempts:** {state.get('quality_attempts', 0)}\n"
            f"**Previous diagnosis:** {state.get('quality_diagnosis', 'none')}"
        )
        result = await core_nodes.quality_gate_node(state)
        step.output = (
            f"**Passed:** {result.get('quality_passed', True)}\n"
            f"**Diagnosis:** {result.get('quality_diagnosis', 'none')}\n"
            f"**Reason:** {result.get('quality_reason', '')}"
        )
    return result
