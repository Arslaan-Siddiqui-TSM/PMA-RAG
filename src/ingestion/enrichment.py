"""LLM-based metadata enrichment for document chunks.

For each chunk, generates a short summary, keywords, and hypothetical questions
that the chunk could answer. These are stored in chunk metadata and improve
retrieval matching significantly.
"""

from __future__ import annotations

import json
import logging

from langchain_core.documents import Document
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from config import settings

logger = logging.getLogger(__name__)

_ENRICHMENT_PROMPT = """\
You are a metadata extraction assistant. Given a text chunk from a project \
management document, produce a JSON object with exactly these keys:

- "summary": a 1-2 sentence summary of the chunk
- "keywords": a list of 3-8 key terms or phrases
- "questions": a list of 2-3 questions this chunk could answer

Respond ONLY with valid JSON, no markdown fences, no extra text.

Text chunk:
---
{chunk_text}
---"""

_BATCH_SIZE = 5


def _parse_enrichment(raw: str) -> dict:
    """Best-effort parse of the LLM JSON response."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text[:-3].strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {}

    result: dict = {}
    if isinstance(data.get("summary"), str):
        result["summary"] = data["summary"]
    if isinstance(data.get("keywords"), list):
        result["keywords"] = [str(k) for k in data["keywords"]]
    if isinstance(data.get("questions"), list):
        result["questions"] = [str(q) for q in data["questions"]]
    return result


async def enrich_chunks(chunks: list[Document]) -> list[Document]:
    """Enrich each chunk with LLM-generated summary, keywords, and questions.

    Processes in batches to manage API throughput. Failures on individual
    chunks are logged and skipped — the chunk keeps its original metadata.
    """
    if not chunks:
        return chunks

    llm = ChatNVIDIA(
        model=settings.llm_model,
        temperature=0.7,
        max_tokens=400,
        disable_streaming=True,
    )

    for i in range(0, len(chunks), _BATCH_SIZE):
        batch = chunks[i : i + _BATCH_SIZE]
        for chunk in batch:
            prompt = _ENRICHMENT_PROMPT.format(
                chunk_text=chunk.page_content[:2000],
            )
            try:
                response = await llm.ainvoke(prompt)
                enrichment = _parse_enrichment(response.content)
                if enrichment:
                    chunk.metadata.update(enrichment)
            except Exception:
                logger.warning(
                    "Enrichment failed for chunk %d, skipping",
                    chunk.metadata.get("chunk_index", -1),
                    exc_info=True,
                )

    return chunks
