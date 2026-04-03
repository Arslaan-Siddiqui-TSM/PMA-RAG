import re

from langchain_core.messages import BaseMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langsmith import traceable

from config import settings
from src.generation.prompts import TRIAGE_PROMPT

GREETING_PATTERNS = re.compile(
    r"^(hi(\s+there)?|hello(\s+there)?|hey(\s+there)?|howdy|"
    r"good\s*(morning|afternoon|evening)|"
    r"greetings|what'?s\s*up|yo|sup|hiya|namaste|salam|"
    r"i'?m\s+\w+|my\s+name\s+is\s+\w+)\s*[.!]?$",
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

MAX_HEURISTIC_LENGTH = 60


def classify_by_heuristics(question: str) -> str | None:
    """Fast-path: detect obvious intents without an LLM call."""
    text = question.strip()

    if len(text) > MAX_HEURISTIC_LENGTH:
        return None

    if GREETING_PATTERNS.match(text):
        return "greeting"

    if THANKS_BYE_PATTERNS.match(text):
        return "thanks_bye"

    if HELP_PATTERNS.match(text):
        return "help"

    return None


def _parse_triage_response(raw: str) -> tuple[bool, str]:
    """Parse SEARCH: yes/no and STYLE: default/summary from model output."""
    text = raw.strip()
    search = True
    if re.search(r"(?i)SEARCH:\s*NO\b", text):
        search = False
    elif re.search(r"(?i)SEARCH:\s*YES\b", text):
        search = True
    style = "summary" if re.search(r"(?i)STYLE:\s*SUMMARY\b", text) else "default"
    return search, style


@traceable(name="triage_by_llm", run_type="chain")
async def triage_by_llm(
    question: str,
    chat_history: list[BaseMessage],
) -> tuple[bool, str]:
    """LLM decides whether to search documents and summary vs default style."""
    history_text = ""
    if chat_history:
        recent = chat_history[-6:]
        lines = []
        for msg in recent:
            role = "User" if msg.type == "human" else "Assistant"
            lines.append(f"{role}: {msg.content[:200]}")
        history_text = "\n".join(lines)

    llm = ChatNVIDIA(
        model=settings.classifier_model,
        temperature=0.1,
        max_tokens=40,
        top_p=1,
        disable_streaming=True,
        model_kwargs={
            "chat_template_kwargs": {"enable_thinking": True, "reasoning_budget": 1024},
        },
    )
    prompt = TRIAGE_PROMPT.format(
        chat_history=history_text or "(no prior conversation)",
        question=question,
    )
    response = await llm.ainvoke(prompt)
    return _parse_triage_response(str(response.content))


async def run_intent_triage(
    question: str,
    chat_history: list[BaseMessage],
) -> dict:
    """Full triage: heuristics (casual/help) or LLM search/style decision."""
    heuristic = classify_by_heuristics(question)
    if heuristic is not None:
        return {
            "intent": heuristic,
            "search_documents": False,
            "response_style": "default",
        }

    search, style = await triage_by_llm(question, chat_history)
    return {
        "intent": "needs_rag" if search else "chat_only",
        "search_documents": search,
        "response_style": style,
    }
