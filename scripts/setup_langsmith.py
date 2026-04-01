"""One-time setup: create LangSmith project and upload evaluation datasets.

Usage:
    python -m scripts.setup_langsmith
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from dotenv import load_dotenv

from langsmith import Client

load_dotenv()

DATASETS_DIR = Path(__file__).resolve().parent.parent / "evals" / "datasets"

DATASET_FILES = {
    "intent-classification-v1": DATASETS_DIR / "intent_classification.json",
    "rag-end-to-end-v1": DATASETS_DIR / "rag_end_to_end.json",
}


def _upload_dataset(client: Client, name: str, path: Path) -> None:
    if not path.exists():
        print(f"  SKIP  {name} — file not found: {path}")
        return

    existing = list(client.list_datasets(dataset_name=name))
    if existing:
        print(f"  EXISTS {name} (id={existing[0].id})")
        return

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
    print(f"  CREATED {name} — {len(examples)} examples (id={dataset.id})")


def main() -> None:
    client = Client()

    print("LangSmith setup")
    print("=" * 50)

    info = client.info
    print(f"\nConnected to LangSmith (server version: {info.version})")

    print("\nUploading datasets:")
    for name, path in DATASET_FILES.items():
        _upload_dataset(client, name, path)

    print(
        "\nIf you use Chainlit feedback (thumbs), apply the data-layer schema once:\n"
        "  python -m scripts.init_chainlit_db"
    )
    print("\nOnline evaluators should be configured in the LangSmith UI:")
    print("  1. Go to your project → Settings → Online Evaluators")
    print("  2. Add 'Intent Consistency' — LLM-as-judge evaluator:")
    print("     Prompt: 'Given the user question: {input.question} and the")
    print("     classified intent: {output.intent}, is the classification")
    print("     correct? Score 1 if correct, 0 if wrong.'")
    print("  3. Add 'Answer Faithfulness' — LLM-as-judge evaluator:")
    print("     Prompt: 'Is the answer grounded in the provided context?")
    print("     Score 1 if faithful, 0 if hallucinated.'")
    print("  4. Add 'Answer Helpfulness' — LLM-as-judge evaluator:")
    print("     Prompt: 'Is the answer helpful and complete for the user")
    print("     question? Score 0-1.'")
    print("\nDone.")


if __name__ == "__main__":
    sys.exit(main() or 0)
