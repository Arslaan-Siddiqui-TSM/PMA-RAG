"""Destructive reset: wipe all projects, documents, chunks, threads,
chat store tables, LangGraph checkpoints, and all Chroma collections.

Usage:
    python -m scripts.reset_all
"""

from __future__ import annotations

import asyncio
import sys

from src.db.postgres import close_pool, get_pool
from src.retrieval.vectorstore import VectorStoreManager


async def _reset() -> None:
    pool = await get_pool()

    tables_to_truncate = [
        "threads",
        "chunks",
        "documents",
        "projects",
        "chat_messages",
        "thread_reranked_docs",
        "api_feedback",
        "checkpoint_writes",
        "checkpoint_blobs",
        "checkpoints",
    ]

    async with pool.connection() as conn:
        for table in tables_to_truncate:
            try:
                await conn.execute(f"DELETE FROM {table}")
                print(f"  Cleared {table}")
            except Exception as exc:
                print(f"  Skip {table}: {exc}")

    print("\nClearing Chroma collections...")
    vsm = VectorStoreManager()
    for name in vsm.list_collections():
        try:
            vsm.delete_collection(name)
            print(f"  Deleted collection {name}")
        except Exception as exc:
            print(f"  Failed to delete {name}: {exc}")

    await close_pool()
    print("\nReset complete.")


def main() -> None:
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(_reset())


if __name__ == "__main__":
    main()
