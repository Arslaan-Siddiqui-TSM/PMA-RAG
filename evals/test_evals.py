"""Pytest wrappers for PMA-RAG evaluations.

Usage:
    # Run fast heuristic tests only (no LLM, no LangSmith)
    pytest evals/test_evals.py -m "not eval"

    # Run full LangSmith-backed evaluations (requires API key + datasets uploaded)
    pytest evals/test_evals.py -m eval

    # Run everything
    pytest evals/test_evals.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.graph.intent import classify_by_heuristics

DATASETS_DIR = Path(__file__).resolve().parent / "datasets"

# ---------------------------------------------------------------------------
# Heuristic-only tests (fast, no LLM, no LangSmith)
# ---------------------------------------------------------------------------


def _load_intent_dataset() -> list[dict]:
    path = DATASETS_DIR / "intent_classification.json"
    with open(path) as f:
        return json.load(f)


class TestHeuristicClassification:
    """Test the regex heuristic classifier against labeled examples."""

    @pytest.fixture(autouse=True)
    def _load_examples(self):
        self.examples = _load_intent_dataset()

    def test_pure_greetings_detected(self):
        """Pure greeting messages should be caught by heuristics."""
        greeting_examples = [
            ex for ex in self.examples
            if ex["outputs"]["intent"] == "greeting"
            and ex.get("metadata", {}).get("heuristic_expected", False)
        ]
        assert len(greeting_examples) > 0, "No greeting test cases found"

        for ex in greeting_examples:
            result = classify_by_heuristics(ex["inputs"]["question"])
            assert result == "greeting", (
                f"Expected 'greeting' for: {ex['inputs']['question']!r}, got {result!r}"
            )

    def test_pure_thanks_bye_detected(self):
        """Pure thanks/bye messages should be caught by heuristics."""
        thanks_examples = [
            ex for ex in self.examples
            if ex["outputs"]["intent"] == "thanks_bye"
            and ex.get("metadata", {}).get("heuristic_expected", False)
        ]
        assert len(thanks_examples) > 0, "No thanks_bye test cases found"

        for ex in thanks_examples:
            result = classify_by_heuristics(ex["inputs"]["question"])
            assert result == "thanks_bye", (
                f"Expected 'thanks_bye' for: {ex['inputs']['question']!r}, got {result!r}"
            )

    def test_pure_help_detected(self):
        """Pure help messages should be caught by heuristics."""
        help_examples = [
            ex for ex in self.examples
            if ex["outputs"]["intent"] == "help"
            and ex.get("metadata", {}).get("heuristic_expected", False)
        ]
        assert len(help_examples) > 0, "No help test cases found"

        for ex in help_examples:
            result = classify_by_heuristics(ex["inputs"]["question"])
            assert result == "help", (
                f"Expected 'help' for: {ex['inputs']['question']!r}, got {result!r}"
            )

    def test_embedded_greetings_not_caught(self):
        """Messages with greetings embedded in real questions should NOT be
        classified as greeting by heuristics (they should fall through to LLM).
        """
        embedded_examples = [
            ex for ex in self.examples
            if ex["outputs"]["intent"] != "greeting"
            and ex["inputs"]["question"].lower().startswith(("hi", "hey", "hello"))
        ]

        for ex in embedded_examples:
            result = classify_by_heuristics(ex["inputs"]["question"])
            assert result != "greeting", (
                f"Should NOT be 'greeting' for: {ex['inputs']['question']!r}, "
                f"expected {ex['outputs']['intent']!r}"
            )

    def test_doc_queries_not_caught_by_heuristics(self):
        """Document queries should not match any heuristic pattern."""
        doc_examples = [
            ex for ex in self.examples
            if ex["outputs"]["intent"] == "doc_query"
            and not ex["inputs"]["question"].lower().startswith(("hi", "hey", "hello"))
        ]

        for ex in doc_examples:
            result = classify_by_heuristics(ex["inputs"]["question"])
            assert result is None, (
                f"Doc query should not match heuristics: {ex['inputs']['question']!r}, "
                f"got {result!r}"
            )


# ---------------------------------------------------------------------------
# LangSmith-backed evaluations (require API key + uploaded datasets)
# ---------------------------------------------------------------------------


@pytest.mark.eval
def test_intent_accuracy_above_threshold():
    """Triage search_documents accuracy should be at least 85%."""
    from evals.run_evals import run_intent_eval

    results = run_intent_eval()

    total = 0
    correct = 0
    for result in results:
        for ev in result.get("evaluation_results", {}).get("results", []):
            if ev.key == "search_documents_accuracy":
                total += 1
                correct += ev.score
                break

    assert total > 0, "No evaluation results returned"
    accuracy = correct / total
    assert accuracy >= 0.85, (
        f"search_documents accuracy {accuracy:.1%} is below 85% threshold "
        f"({correct}/{total} correct)"
    )


@pytest.mark.eval
def test_heuristic_accuracy_above_threshold():
    """Heuristic-only classification should correctly handle all heuristic cases."""
    from evals.run_evals import run_heuristic_eval

    results = run_heuristic_eval()

    heuristic_total = 0
    heuristic_correct = 0
    for result in results:
        example = getattr(result, "example", None)
        example_meta = {}
        if example is not None:
            md = getattr(example, "metadata", None)
            if isinstance(md, dict):
                example_meta = md
        if not example_meta.get("heuristic_expected", False):
            continue

        for ev in result.get("evaluation_results", {}).get("results", []):
            if ev.key == "intent_accuracy":
                heuristic_total += 1
                heuristic_correct += ev.score
                break

    if heuristic_total == 0:
        pytest.skip("No heuristic-expected examples found in dataset")

    accuracy = heuristic_correct / heuristic_total
    assert accuracy >= 0.95, (
        f"Heuristic accuracy {accuracy:.1%} is below 95% threshold "
        f"({heuristic_correct}/{heuristic_total} correct)"
    )
