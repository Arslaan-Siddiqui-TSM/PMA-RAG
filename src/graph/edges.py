from src.graph.state import RAGState


def route_after_intent(state: RAGState) -> str:
    """Route based on classified intent."""
    intent = state.get("intent", "doc_query")
    match intent:
        case "greeting" | "thanks_bye":
            return "casual_response"
        case "help":
            return "help_response"
        case "followup":
            return "reformulate_query"
        case _:
            return "retrieve"


def route_after_reformulate(state: RAGState) -> str:
    """After reformulating a follow-up, decide whether to reuse prior docs."""
    if state.get("reuse_prior_docs"):
        return "generate"
    return "retrieve"


def route_after_relevance_check(state: RAGState) -> str:
    """Route to generate or no_answer based on available documents."""
    reranked_docs = state.get("reranked_documents", [])
    scores = state.get("relevance_scores", [])

    if not reranked_docs or not scores:
        return "no_answer"

    return "generate"
