from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompt_templates"


def _load_prompt_text(filename: str) -> str:
    return (_PROMPTS_DIR / filename).read_text(encoding="utf-8").strip()


RAG_SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _load_prompt_text("rag_summary_system.txt")),
        ("human", _load_prompt_text("rag_summary_human.txt")),
    ]
)

TRIAGE_PROMPT = _load_prompt_text("triage.txt")

UNIFIED_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _load_prompt_text("unified_system.txt")),
        ("human", _load_prompt_text("unified_human.txt")),
    ]
)

RESPONSE_STYLE_HINTS = {
    "summary": (
        "The user asked for a summary or overview. If they explicitly asked for a "
        "brief recap or TL;DR, keep it tight; otherwise default to a comprehensive "
        "summary. Start with a short executive summary, then provide supporting "
        "detail in the format that best fits the material. Synthesize only what the "
        "available evidence supports."
    ),
    "default": (
        "Default to a detailed answer unless the user's request is clearly simple "
        "or explicitly asks for brevity. Choose the format that best fits the "
        "question. If the user mixes summary with deeper analysis, match the most "
        "detailed part of the request and remain comprehensive. For document "
        "questions, ground answers in retrieved context when present."
    ),
}

REFORMULATE_PROMPT = _load_prompt_text("reformulate.txt")

PLAN_RETRIEVAL_PROMPT = _load_prompt_text("plan_retrieval.txt")

REFLECT_RETRIEVAL_PROMPT = _load_prompt_text("reflect_retrieval.txt")

QUALITY_GATE_PROMPT = _load_prompt_text("quality_gate.txt")

CASUAL_RESPONSES = {
    "greeting": _load_prompt_text("casual_greeting.txt"),
    "thanks_bye": _load_prompt_text("casual_thanks_bye.txt"),
}

HELP_RESPONSE = _load_prompt_text("help_response.txt")

ENRICHMENT_PROMPT = _load_prompt_text("enrichment.txt")
