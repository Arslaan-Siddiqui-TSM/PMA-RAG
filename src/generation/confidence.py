import math

from config import settings


def logit_to_probability(logit: float) -> float:
    """Convert a logit (log-odds) score to a 0-1 probability via sigmoid."""
    return 1.0 / (1.0 + math.exp(-logit))


def normalize_scores(raw_scores: list[float]) -> list[float]:
    """Normalize reranker scores to 0-1 probabilities.

    The NVIDIA reranker (llama-nemotron-rerank) returns raw logits which can
    be negative. A logit of 0 maps to 0.5 probability; negative logits map
    to < 0.5; positive logits map to > 0.5.

    Examples:
        -0.85 -> ~0.30  (decent match)
        -2.56 -> ~0.07  (weak match)
        -4.55 -> ~0.01  (poor match)
         0.0  -> 0.50
         2.0  -> 0.88
    """
    return [logit_to_probability(s) for s in raw_scores]


def compute_confidence(relevance_scores: list[float]) -> str:
    """Compute confidence from normalized (0-1) relevance scores."""
    if not relevance_scores:
        return "Low"

    top_score = max(relevance_scores)
    docs_above_high = sum(
        1 for s in relevance_scores if s >= settings.confidence_high_doc_threshold
    )
    docs_above_medium = sum(
        1 for s in relevance_scores if s >= settings.confidence_medium_doc_threshold
    )

    if (
        top_score >= settings.confidence_high_threshold
        and docs_above_high >= settings.confidence_high_min_docs
    ):
        return "High"

    if (
        top_score >= settings.confidence_medium_threshold
        or docs_above_medium >= settings.confidence_medium_min_docs
    ):
        return "Medium"

    return "Low"
