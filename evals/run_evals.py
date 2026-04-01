"""Offline evaluation runner for PMA-RAG.

Runs triage (search vs no-search) and/or full RAG pipeline evaluations against
LangSmith datasets and reports results as LangSmith experiments.

Usage:
    # Run triage LLM evaluation only
    python -m evals.run_evals --intent

    # Run full RAG pipeline eval only
    python -m evals.run_evals --rag

    # Run both
    python -m evals.run_evals --intent --rag

    # Upload local datasets to LangSmith first, then run
    python -m evals.run_evals --upload --intent --rag
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.messages import AIMessage, HumanMessage
from langsmith import Client, evaluate

from src.graph.intent import classify_by_heuristics, run_intent_triage

load_dotenv()

DATASETS_DIR = Path(__file__).resolve().parent / "datasets"

# For confusion matrix over retrieval decision
SEARCH_MATRIX_KEYS = [False, True]


# ---------------------------------------------------------------------------
# Targets — functions that LangSmith evaluate() calls per example
# ---------------------------------------------------------------------------


def triage_target(inputs: dict) -> dict:
    """Full triage (heuristics + LLM) for one example."""
    question = inputs["question"]
    raw_history = inputs.get("chat_history", [])

    chat_history = []
    for msg in raw_history:
        if msg.get("type") == "human":
            chat_history.append(HumanMessage(content=msg["content"]))
        else:
            chat_history.append(AIMessage(content=msg["content"]))

    return asyncio.run(run_intent_triage(question, chat_history))


def heuristic_only_target(inputs: dict) -> dict:
    """Heuristic path only (no LLM); intent label for casual/help rows."""
    question = inputs["question"]
    result = classify_by_heuristics(question)
    return {"intent": result or "llm_required"}


# ---------------------------------------------------------------------------
# Evaluators
# ---------------------------------------------------------------------------


def search_documents_accuracy(run, example) -> dict:
    """Match on whether document retrieval should run."""
    predicted = run.outputs.get("search_documents")
    expected = example.outputs.get("search_documents")
    if expected is None:
        return {"key": "search_documents_accuracy", "score": 1}
    match = bool(predicted) == bool(expected)
    return {"key": "search_documents_accuracy", "score": int(match)}


def response_style_accuracy(run, example) -> dict:
    """Match summary vs default style hint."""
    predicted = run.outputs.get("response_style") or "default"
    expected = example.outputs.get("response_style") or "default"
    return {
        "key": "response_style_accuracy",
        "score": int(predicted == expected),
    }


def search_match_detail(run, example) -> dict:
    """For confusion matrix over search_documents."""
    predicted = run.outputs.get("search_documents")
    expected = example.outputs.get("search_documents")
    if expected is None:
        expected = "missing"
    return {
        "key": "search_match_detail",
        "score": int(bool(predicted) == bool(expected))
        if expected != "missing"
        else 1,
        "comment": f"predicted={predicted!r}, expected={expected!r}",
    }


def intent_accuracy(run, example) -> dict:
    """Exact-match intent (heuristic eval only)."""
    predicted = run.outputs.get("intent", "")
    expected = example.outputs.get("intent", "")
    return {"key": "intent_accuracy", "score": int(predicted == expected)}


def intent_category_match(run, example) -> dict:
    predicted = run.outputs.get("intent", "unknown")
    expected = example.outputs.get("intent", "unknown")
    return {
        "key": "intent_match_detail",
        "score": int(predicted == expected),
        "comment": f"predicted={predicted}, expected={expected}",
    }


# ---------------------------------------------------------------------------
# Confusion matrix helper (retrieval decision)
# ---------------------------------------------------------------------------


def print_search_confusion_matrix(results) -> None:
    """Print confusion matrix for search_documents (expected rows, predicted cols)."""
    matrix: dict[bool, Counter] = {True: Counter(), False: Counter()}

    for result in results:
        comment = ""
        for ev in result.get("evaluation_results", {}).get("results", []):
            if ev.key == "search_match_detail":
                comment = ev.comment or ""
                break

        if "expected=" not in comment:
            continue
        try:
            pred_part = comment.split(", ")[0]
            exp_part = comment.split(", ")[1]
            predicted = pred_part.split("=", 1)[1].strip() == "True"
            expected_raw = exp_part.split("=", 1)[1].strip()
            if expected_raw == "missing":
                continue
            expected = expected_raw == "True"
        except (IndexError, ValueError):
            continue

        matrix[expected][predicted] += 1

    print("\nConfusion matrix for search_documents (rows=expected, cols=predicted):")
    print(f"{'':>12}", end="")
    for p in SEARCH_MATRIX_KEYS:
        print(f"{str(p):>12}", end="")
    print()

    for expected in SEARCH_MATRIX_KEYS:
        row = matrix[expected]
        print(f"{str(expected):>12}", end="")
        for predicted in SEARCH_MATRIX_KEYS:
            count = row.get(predicted, 0)
            cell = str(count) if count > 0 else "."
            print(f"{cell:>12}", end="")
        print()


# ---------------------------------------------------------------------------
# Dataset upload
# ---------------------------------------------------------------------------


def upload_datasets(client: Client) -> None:
    """Upload local JSON datasets to LangSmith."""
    dataset_map = {
        "intent-classification-v1": DATASETS_DIR / "intent_classification.json",
        "rag-end-to-end-v1": DATASETS_DIR / "rag_end_to_end.json",
    }

    for name, path in dataset_map.items():
        if not path.exists():
            print(f"  SKIP  {name} — file not found: {path}")
            continue

        existing = list(client.list_datasets(dataset_name=name))
        if existing:
            print(f"  EXISTS {name}")
            continue

        with open(path) as f:
            examples = json.load(f)

        dataset = client.create_dataset(dataset_name=name)
        for ex in examples:
            client.create_example(
                inputs=ex["inputs"],
                outputs=ex.get("outputs", {}),
                metadata=ex.get("metadata", {}),
                dataset_id=dataset.id,
            )
        print(f"  CREATED {name} — {len(examples)} examples")


# ---------------------------------------------------------------------------
# Eval runners
# ---------------------------------------------------------------------------


def run_intent_eval(client: Client | None = None) -> object:
    """Run triage LLM evaluation (search_documents + response_style)."""
    results = evaluate(
        triage_target,
        data="intent-classification-v1",
        evaluators=[
            search_documents_accuracy,
            response_style_accuracy,
            search_match_detail,
        ],
        experiment_prefix="intent-triage",
        client=client,
    )
    return results


def run_heuristic_eval(client: Client | None = None) -> object:
    """Run heuristic-only evaluation (no LLM calls)."""
    results = evaluate(
        heuristic_only_target,
        data="intent-classification-v1",
        evaluators=[intent_accuracy, intent_category_match],
        experiment_prefix="intent-heuristic",
        client=client,
    )
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="PMA-RAG Evaluation Runner")
    parser.add_argument(
        "--upload", action="store_true", help="Upload datasets to LangSmith first"
    )
    parser.add_argument(
        "--intent", action="store_true", help="Run triage LLM evaluation"
    )
    parser.add_argument(
        "--heuristic", action="store_true", help="Run heuristic-only eval"
    )
    parser.add_argument(
        "--rag", action="store_true", help="Run full RAG pipeline eval (placeholder)"
    )
    args = parser.parse_args()

    if not any([args.upload, args.intent, args.heuristic, args.rag]):
        parser.print_help()
        return

    client = Client()

    if args.upload:
        print("Uploading datasets...")
        upload_datasets(client)
        print()

    if args.intent:
        print("Running triage evaluation...")
        run_intent_eval(client)
        print("\nTriage eval complete. View results in LangSmith dashboard.")

    if args.heuristic:
        print("Running heuristic-only evaluation...")
        run_heuristic_eval(client)
        print("\nHeuristic eval complete. View results in LangSmith dashboard.")

    if args.rag:
        print("Full RAG pipeline evaluation is not yet implemented.")
        print("It requires a running vector store and BM25 index with ingested documents.")
        print("This will be added once the pipeline supports headless execution.")


if __name__ == "__main__":
    sys.exit(main() or 0)
