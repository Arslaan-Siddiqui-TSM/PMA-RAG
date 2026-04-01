from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompt_templates"


def _load_prompt_text(filename: str) -> str:
    return (_PROMPTS_DIR / filename).read_text(encoding="utf-8").strip()


RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _load_prompt_text("rag_system.txt")),
        ("human", _load_prompt_text("rag_human.txt")),
    ]
)

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
        "The user asked for a summary or overview. Prefer tight sections or bullets; "
        "lead with what the material is about; synthesize only what the retrieved "
        "context supports."
    ),
    "default": (
        "Answer in clear, scannable prose unless bullets clearly help. "
        "For document questions, ground answers in retrieved context when present."
    ),
}

REFORMULATE_PROMPT = _load_prompt_text("reformulate.txt")

CASUAL_RESPONSES = {
    "greeting": _load_prompt_text("casual_greeting.txt"),
    "thanks_bye": _load_prompt_text("casual_thanks_bye.txt"),
}

HELP_RESPONSE = _load_prompt_text("help_response.txt")
