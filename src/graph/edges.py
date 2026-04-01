from src.graph.state import RAGState


def route_after_intent(state: RAGState) -> str:
    """Route after triage: casual/help, direct generate, or retrieval path."""
    intent = state.get("intent", "")
    match intent:
        case "greeting" | "thanks_bye":
            return "casual_response"
        case "help":
            return "help_response"
        case _:
            if not state.get("search_documents", True):
                return "generate"
            return "reformulate_query"


def route_after_reformulate(state: RAGState) -> str:
    """After reformulating a follow-up, decide whether to reuse prior docs."""
    if state.get("reuse_prior_docs"):
        return "generate"
    return "decompose_query"


def route_after_relevance_check(state: RAGState) -> str:
    """After reranking + confidence estimation, proceed to generation."""
    return "generate"


def route_after_validation(state: RAGState) -> str:
    """Retry generation, or re-enter retrieval, if validation fails."""
    if state.get("validation_passed", True):
        return "end"
    if state.get("force_retrieval_on_retry", False):
        return "reformulate_query"
    return "generate"
