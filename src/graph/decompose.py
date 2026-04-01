from __future__ import annotations

import re

from langchain_nvidia_ai_endpoints import ChatNVIDIA

from config import settings

_COMPARE_PATTERN = re.compile(
    r"\b(compare|difference|differences|vs|versus|contrast)\b",
    re.IGNORECASE,
)


def should_decompose(query: str) -> bool:
    return bool(_COMPARE_PATTERN.search(query))


def _heuristic_decompose(query: str) -> list[str]:
    normalized = query.strip()
    lowered = normalized.lower()
    if " vs " in lowered:
        parts = re.split(r"(?i)\s+vs\s+", normalized, maxsplit=1)
        if len(parts) == 2 and parts[0].strip() and parts[1].strip():
            return [parts[0].strip(), parts[1].strip()]
    if " versus " in lowered:
        parts = re.split(r"(?i)\s+versus\s+", normalized, maxsplit=1)
        if len(parts) == 2 and parts[0].strip() and parts[1].strip():
            return [parts[0].strip(), parts[1].strip()]
    return []


async def decompose_query(query: str) -> list[str]:
    heuristic = _heuristic_decompose(query)
    if heuristic:
        return heuristic

    llm = ChatNVIDIA(
        model=settings.classifier_model,
        temperature=0.01,
        max_tokens=200,
        disable_streaming=True,
    )
    prompt = (
        "Split the following query into 2-4 retrieval sub-queries only when needed "
        "for comparisons or multi-part requests. "
        "Return one sub-query per line. If decomposition is not needed, return the original query.\n\n"
        f"Query: {query}"
    )
    response = await llm.ainvoke(prompt)
    lines = [line.strip("- ").strip() for line in str(response.content).splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return [query]
    if len(lines) == 1:
        return [lines[0]]
    return lines[:4]

