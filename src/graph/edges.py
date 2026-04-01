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
    return "retrieve"


def route_after_relevance_check(state: RAGState) -> str:
    """Always generate; unified prompt handles empty or weak retrieval."""
    return "generate"
