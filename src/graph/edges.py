from src.graph.state import RAGState


def route_after_relevance_check(state: RAGState) -> str:
    """Route to generate or no_answer.

    Only routes to no_answer when reranking returned zero documents.
    Otherwise always generates -- the LLM's system prompt instructs it
    to say "I don't know" if the context is truly insufficient, and the
    confidence score shown to the user communicates certainty level.
    """
    reranked_docs = state.get("reranked_documents", [])
    scores = state.get("relevance_scores", [])

    if not reranked_docs or not scores:
        return "no_answer"

    return "generate"
