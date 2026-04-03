from __future__ import annotations

import json
import re
import logging

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langsmith import traceable

from config import settings
from src.generation.prompts import QUALITY_GATE_PROMPT

logger = logging.getLogger(__name__)


def parse_quality_gate(raw: str) -> dict:
    """Parse the structured output from the quality gate LLM."""
    text = raw.strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            def _yn(val: object) -> bool:
                return str(val).lower() in {"yes", "true", "1", "pass"}

            grounded = _yn(parsed.get("grounded", parsed.get("GROUNDED", "no")))
            coverage = _yn(parsed.get("coverage", parsed.get("COVERAGE", "no")))
            completeness = _yn(parsed.get("completeness", parsed.get("COMPLETENESS", "no")))
            hallucination = _yn(parsed.get("hallucination", parsed.get("HALLUCINATION", "no")))
            diagnosis = str(
                parsed.get("diagnosis", parsed.get("DIAGNOSIS", "none"))
            ).lower()
            reason = str(parsed.get("reason", parsed.get("REASON", ""))).strip()
            return {
                "grounded": grounded,
                "coverage": coverage,
                "completeness": completeness,
                "hallucination": hallucination,
                "diagnosis": diagnosis,
                "reason": reason or "Quality gate JSON parsed",
            }
    except Exception:
        pass

    grounded = bool(re.search(r"(?i)GROUNDED\s*:\s*YES\b", text))
    coverage = bool(re.search(r"(?i)COVERAGE\s*:\s*YES\b", text))
    completeness = bool(re.search(r"(?i)COMPLETENESS\s*:\s*YES\b", text))
    hallucination = bool(re.search(r"(?i)HALLUCINATION\s*:\s*YES\b", text))

    diagnosis = "none"
    diag_match = re.search(
        r"(?i)DIAGNOSIS\s*:\s*(generation|missing_context|none)\b", text
    )
    if diag_match:
        diagnosis = diag_match.group(1).lower()

    reason_match = re.search(r"(?i)REASON\s*:\s*(.+)", text)
    reason = (
        reason_match.group(1).strip()
        if reason_match
        else text[:200] or "Quality gate parser fallback"
    )

    return {
        "grounded": grounded,
        "coverage": coverage,
        "completeness": completeness,
        "hallucination": hallucination,
        "diagnosis": diagnosis,
        "reason": reason,
    }


@traceable(name="quality_gate", run_type="chain")
async def run_quality_gate(
    *,
    question: str,
    answer: str,
    context: str,
) -> dict:
    llm = ChatNVIDIA(
        model=settings.quality_gate_model,
        temperature=0.3,
        top_p=0.95,
        disable_streaming=True,
        model_kwargs={
            "chat_template_kwargs": {
                "enable_thinking": True,
                "reasoning_budget": 1024,
                "reasoning_effort": "low",
            },
        },
    )

    prompt = QUALITY_GATE_PROMPT.format(
        question=question,
        answer=answer,
        context=context[:6000],
    )

    response = await llm.ainvoke(prompt)
    return parse_quality_gate(str(response.content))
