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
    return "plan_retrieval"


def route_after_reflection(state: RAGState) -> str:
    """After evaluating retrieval quality, proceed to generate or re-retrieve."""
    if state.get("retrieval_sufficient", True):
        return "generate"
    return "retrieve"


def route_after_quality_gate(state: RAGState) -> str:
    """Route after the unified quality gate check."""
    if state.get("quality_passed", True):
        return "end"
    diagnosis = state.get("quality_diagnosis", "generation")
    if diagnosis == "missing_context":
        return "reformulate_query"
    return "generate"
