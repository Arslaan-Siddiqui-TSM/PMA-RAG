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
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langsmith import Client, evaluate

from src.api.dependencies import init_components, shutdown_components
from src.graph.intent import classify_by_heuristics, run_intent_triage
from src.graph.project_context import build_project_context
from src.graph.state import build_default_state

load_dotenv()

DATASETS_DIR = Path(__file__).resolve().parent / "datasets"
LOGS_DIR = Path(__file__).resolve().parent / "logs"


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
        "score": int(bool(predicted) == bool(expected)) if expected != "missing" else 1,
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


def _build_rag_state(
    question: str,
    project_id: str,
    collection_name: str,
    project_context: str,
) -> dict:
    return build_default_state(
        question=question,
        project_id=project_id,
        collection_name=collection_name,
        project_context=project_context,
    )


def _precision_at_k(relevance: list[int], k: int) -> float:
    if k <= 0:
        return 0.0
    top = relevance[:k]
    if not top:
        return 0.0
    return sum(top) / len(top)


def _recall_at_k(relevance: list[int], relevant_total: int, k: int) -> float:
    if relevant_total <= 0:
        return 1.0
    top = relevance[:k]
    return min(sum(top) / relevant_total, 1.0)


def _mrr(relevance: list[int]) -> float:
    for idx, rel in enumerate(relevance, start=1):
        if rel:
            return 1.0 / idx
    return 0.0


def _ndcg_at_k(relevance: list[int], k: int) -> float:
    rel_k = relevance[:k]
    if not rel_k:
        return 0.0
    dcg = 0.0
    for i, rel in enumerate(rel_k, start=1):
        dcg += rel / math.log2(i + 1)
    ideal = sorted(rel_k, reverse=True)
    idcg = 0.0
    for i, rel in enumerate(ideal, start=1):
        idcg += rel / math.log2(i + 1)
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def _fact_coverage(answer: str, expected_key_facts: list[str]) -> float:
    if not expected_key_facts:
        return 1.0
    answer_l = answer.lower()
    matches = sum(1 for fact in expected_key_facts if fact.lower() in answer_l)
    return matches / len(expected_key_facts)


async def _run_rag_eval_async() -> dict:
    dataset_path = DATASETS_DIR / "rag_end_to_end.json"
    with open(dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)

    components = await init_components()
    logs: list[dict] = []

    recall_scores: list[float] = []
    precision_scores: list[float] = []
    mrr_scores: list[float] = []
    ndcg_scores: list[float] = []
    faithfulness_scores: list[float] = []
    fact_coverage_scores: list[float] = []

    project_id = os.environ.get("PMA_EVAL_PROJECT_ID", "")
    collection_name = os.environ.get("PMA_EVAL_COLLECTION_NAME", "")
    if not project_id or not collection_name:
        project = await components.metadata_store.list_active_projects()
        if not project:
            print("ERROR: No active projects. Create one before running evals.")
            await shutdown_components()
            return {"summary": {}, "runs": [], "log_path": ""}
        project_id = str(project[0]["id"])
        collection_name = project[0]["collection_name"]

    all_projects = await components.metadata_store.list_active_projects()
    active_project = next(
        (project for project in all_projects if str(project["id"]) == project_id),
        None,
    )
    project_context = build_project_context(
        active_project=active_project or {"name": "", "description": ""},
        all_projects=all_projects,
        max_projects=20,
    )

    for ex in dataset:
        question = ex["inputs"]["question"]
        expected_doc_types = ex["outputs"].get("expected_doc_types", [])
        expected_key_facts = ex["outputs"].get("expected_key_facts", [])

        state = _build_rag_state(
            question,
            project_id,
            collection_name,
            project_context,
        )
        run_id = str(uuid4())
        final_state = await components.rag_graph.ainvoke(
            state,
            config={
                "configurable": {"thread_id": f"eval-{run_id}"},
                "run_id": run_id,
                "tags": ["eval", "rag"],
            },
        )

        citations = final_state.get("source_citations", [])
        retrieved_doc_types = [c.get("doc_type", "") for c in citations]
        retrieved_chunks = [c.get("chunk_id", "") for c in citations]
        answer = final_state.get("generation", "")
        validation_passed = bool(final_state.get("quality_passed", True))

        if expected_doc_types:
            expected_set = set(expected_doc_types)
            relevance = [1 if dt in expected_set else 0 for dt in retrieved_doc_types]
            recall = _recall_at_k(
                relevance, relevant_total=len(expected_set), k=len(relevance) or 1
            )
            precision = _precision_at_k(relevance, len(relevance) or 1)
            mrr_value = _mrr(relevance)
            ndcg = _ndcg_at_k(relevance, len(relevance) or 1)
        else:
            recall = 1.0
            precision = 1.0
            mrr_value = 1.0
            ndcg = 1.0

        faithfulness = 1.0 if validation_passed else 0.0
        fact_coverage = _fact_coverage(answer, expected_key_facts)

        recall_scores.append(recall)
        precision_scores.append(precision)
        mrr_scores.append(mrr_value)
        ndcg_scores.append(ndcg)
        faithfulness_scores.append(faithfulness)
        fact_coverage_scores.append(fact_coverage)

        logs.append(
            {
                "query": question,
                "retrieved_chunks": retrieved_chunks,
                "retrieved_doc_types": retrieved_doc_types,
                "final_answer": answer,
                "confidence": final_state.get("confidence", ""),
                "validation_passed": validation_passed,
                "scores": {
                    "recall_at_k": recall,
                    "precision_at_k": precision,
                    "mrr": mrr_value,
                    "ndcg": ndcg,
                    "faithfulness": faithfulness,
                    "fact_coverage": fact_coverage,
                },
            }
        )

    await shutdown_components()

    summary = {
        "count": len(dataset),
        "recall_at_k": sum(recall_scores) / len(recall_scores)
        if recall_scores
        else 0.0,
        "precision_at_k": sum(precision_scores) / len(precision_scores)
        if precision_scores
        else 0.0,
        "mrr": sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0,
        "ndcg": sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0,
        "faithfulness": sum(faithfulness_scores) / len(faithfulness_scores)
        if faithfulness_scores
        else 0.0,
        "fact_coverage": sum(fact_coverage_scores) / len(fact_coverage_scores)
        if fact_coverage_scores
        else 0.0,
    }

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / (
        "rag_eval_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".json"
    )
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "runs": logs}, f, indent=2)

    print("\nRAG Eval Summary:")
    print(json.dumps(summary, indent=2))
    print(f"\nDetailed logs: {log_path}")
    return {"summary": summary, "runs": logs, "log_path": str(log_path)}


def run_rag_eval() -> dict:
    """Run end-to-end RAG evaluation with retrieval + answer metrics."""
    return asyncio.run(_run_rag_eval_async())


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
        "--rag", action="store_true", help="Run full RAG pipeline eval"
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
        print("Running full RAG pipeline evaluation...")
        run_rag_eval()


if __name__ == "__main__":
    sys.exit(main() or 0)
