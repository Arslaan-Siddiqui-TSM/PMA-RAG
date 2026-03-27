import chainlit as cl
from langchain_core.documents import Document
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIARerank

from config import settings
from src.generation.confidence import compute_confidence, normalize_scores
from src.generation.prompts import RAG_PROMPT
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
            f"**BM25 index size:** {_bm25_index.document_count if _bm25_index else 0} chunks"
        )

        ensemble = build_ensemble_retriever(
            _vectorstore_manager, _bm25_index, doc_type_filter=doc_type_filter
        )
        documents = await ensemble.ainvoke(question)

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

    return {"documents": documents}


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

        lines = [f"**Reranked to {len(reranked)} documents:**\n"]
        for i, (doc, raw, norm) in enumerate(
            zip(reranked, raw_scores, normalized), 1
        ):
            source = doc.metadata.get("source_file", "Unknown")
            page = doc.metadata.get("page", "")
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
            lines.append(
                f"**Raw logit range:** "
                f"min={min(raw_scores):.4f}, max={max(raw_scores):.4f}"
            )
        step.output = "\n".join(lines)

    return {
        "reranked_documents": reranked,
        "relevance_scores": normalized,
    }


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

        confidence = compute_confidence(scores)

        reranked_docs = state.get("reranked_documents", [])
        will_route = "no_answer" if (not reranked_docs or not scores) else "generate"

        step.output = (
            f"**Confidence: {confidence}**\n"
            f"**Top normalized score: "
            f"{f'{max(scores):.4f}' if scores else 'N/A'}**\n"
            f"**Routing to: `{will_route}`**"
        )

    return {"confidence": confidence}


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
                "section": doc.metadata.get("h1", "") or doc.metadata.get("h2", ""),
                "doc_type": doc.metadata.get("doc_type", ""),
                "relevance_score": doc.metadata.get("relevance_score", 0.0),
            }
        )
    return citations


async def generate_node(state: RAGState) -> dict:
    question = state["question"]
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

        chain = RAG_PROMPT | ChatNVIDIA(
            model=settings.llm_model,
            temperature=0.1,
            max_tokens=1024,
        )
        response = await chain.ainvoke({"context": context, "question": question})

        step.output = f"**LLM Response:**\n{response.content}"

    return {
        "generation": response.content,
        "source_citations": _build_citations(reranked_docs),
    }


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

    return {
        "generation": (
            "I don't have enough information in the documents "
            "to answer this question."
        ),
        "confidence": "Low",
        "source_citations": [],
    }
