import re

from langchain_core.messages import BaseMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from config import settings
from src.generation.prompts import INTENT_CLASSIFY_PROMPT

GREETING_PATTERNS = re.compile(
    r"^(hi|hello|hey|howdy|good\s*(morning|afternoon|evening)|"
    r"greetings|what'?s\s*up|yo|sup|hiya|namaste|salam|"
    r"i'?m\s+\w+|my\s+name\s+is)\b",
    re.IGNORECASE,
)

THANKS_BYE_PATTERNS = re.compile(
    r"^(thanks?|thank\s*you|thx|ty|cheers|"
    r"bye|goodbye|good\s*bye|see\s*you|that'?s\s*all|"
    r"no\s*more\s*questions?|i'?m\s*done|"
    r"great|awesome|perfect|nice|cool|ok\s*thanks?)\s*[.!]?$",
    re.IGNORECASE,
)

HELP_PATTERNS = re.compile(
    r"^(help|what\s+can\s+you\s+do|"
    r"what\s+are\s+your\s+capabilities|"
    r"how\s+do\s+(i|you)\s+use|"
    r"how\s+does\s+this\s+work|"
    r"what\s+commands|"
    r"show\s+me\s+help|"
    r"/help)\s*\??$",
    re.IGNORECASE,
)

VALID_INTENTS = {"doc_query", "followup", "comparison", "summary"}


def classify_by_heuristics(question: str) -> str | None:
    """Fast-path: detect obvious intents without an LLM call."""
    text = question.strip()

    if GREETING_PATTERNS.search(text):
        return "greeting"

    if THANKS_BYE_PATTERNS.match(text):
        return "thanks_bye"

    if HELP_PATTERNS.match(text):
        return "help"

    return None


async def classify_by_llm(
    question: str,
    chat_history: list[BaseMessage],
) -> str:
    """Use the NVIDIA LLM with few-shot examples to classify intent."""
    history_text = ""
    if chat_history:
        recent = chat_history[-6:]
        lines = []
        for msg in recent:
            role = "User" if msg.type == "human" else "Assistant"
            lines.append(f"{role}: {msg.content[:200]}")
        history_text = "\n".join(lines)

    llm = ChatNVIDIA(
        model=settings.llm_model,
        temperature=0.0,
        max_tokens=20,
    )
    prompt = INTENT_CLASSIFY_PROMPT.format(
        chat_history=history_text or "(no prior conversation)",
        question=question,
    )
    response = await llm.ainvoke(prompt)
    raw = response.content.strip().lower()

    for intent in VALID_INTENTS:
        if intent in raw:
            return intent

    return "doc_query"


async def classify_intent(
    question: str,
    chat_history: list[BaseMessage],
) -> str:
    """Hybrid intent classification: heuristics first, then LLM fallback."""
    heuristic_result = classify_by_heuristics(question)
    if heuristic_result is not None:
        return heuristic_result

    return await classify_by_llm(question, chat_history)
