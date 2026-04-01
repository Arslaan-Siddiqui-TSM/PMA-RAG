from __future__ import annotations

import re
import json

from langchain_nvidia_ai_endpoints import ChatNVIDIA

from config import settings

_VALIDATION_PROMPT = """\
You are a strict RAG validator. Evaluate an answer against provided context.

Return exactly three lines:
SUPPORTED: yes|no
COVERAGE: yes|no
REASON: <short reason>

Question:
{question}

Answer:
{answer}

Context:
{context}
"""


def parse_validation(raw: str) -> tuple[bool, bool, str]:
    text = raw.strip()

    # Try JSON first for robust parsing.
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            support_raw = str(
                parsed.get("supported", parsed.get("grounded", parsed.get("faithful", "no")))
            ).lower()
            coverage_raw = str(parsed.get("coverage", parsed.get("complete", "no"))).lower()
            supported = support_raw in {"yes", "true", "1", "supported", "pass"}
            coverage = coverage_raw in {"yes", "true", "1", "complete", "pass"}
            reason = str(parsed.get("reason", "")).strip() or "Validation JSON parsed"
            return supported, coverage, reason
    except Exception:
        pass

    supported = bool(
        re.search(r"(?i)(SUPPORTED|GROUNDED|FAITHFUL)\s*:\s*(YES|TRUE|PASS)\b", text)
    )
    coverage = bool(
        re.search(r"(?i)(COVERAGE|COMPLETE)\s*:\s*(YES|TRUE|PASS)\b", text)
    )
    reason_match = re.search(r"(?i)REASON:\s*(.+)", text)
    reason = reason_match.group(1).strip() if reason_match else text[:180] or "Validation parser fallback"
    return supported, coverage, reason


async def validate_answer(
    *,
    question: str,
    answer: str,
    context: str,
) -> tuple[bool, bool, str]:
    llm = ChatNVIDIA(
        model=settings.classifier_model,
        temperature=0.01,
        max_tokens=120,
        disable_streaming=True,
    )
    response = await llm.ainvoke(
        _VALIDATION_PROMPT.format(
            question=question,
            answer=answer,
            context=context[:6000],
        )
    )
    return parse_validation(str(response.content))

