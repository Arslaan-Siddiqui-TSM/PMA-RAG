"""Server-side persistence for chat history and reranked documents per thread."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from psycopg_pool import AsyncConnectionPool

CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS chat_messages (
    id SERIAL PRIMARY KEY,
    thread_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chat_messages_thread
    ON chat_messages (thread_id, created_at);

CREATE TABLE IF NOT EXISTS thread_reranked_docs (
    id SERIAL PRIMARY KEY,
    thread_id TEXT NOT NULL UNIQUE,
    docs_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_thread_reranked_docs_thread
    ON thread_reranked_docs (thread_id);

CREATE TABLE IF NOT EXISTS api_feedback (
    id SERIAL PRIMARY KEY,
    thread_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    score DOUBLE PRECISION NOT NULL,
    comment TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_api_feedback_thread
    ON api_feedback (thread_id);
CREATE INDEX IF NOT EXISTS idx_api_feedback_run
    ON api_feedback (run_id);
"""

MAX_HISTORY_MESSAGES = 10


class ChatStore:
    def __init__(self, pool: AsyncConnectionPool) -> None:
        self._pool = pool

    async def setup(self) -> None:
        async with self._pool.connection() as conn:
            await conn.execute(CREATE_TABLES_SQL)

    # ------------------------------------------------------------------
    # Chat history
    # ------------------------------------------------------------------

    async def append_messages(
        self,
        thread_id: str,
        human_content: str,
        ai_content: str,
    ) -> None:
        now = datetime.now(timezone.utc)
        async with self._pool.connection() as conn:
            await conn.execute(
                """
                INSERT INTO chat_messages (thread_id, role, content, created_at)
                VALUES (%s, 'human', %s, %s), (%s, 'ai', %s, %s)
                """,
                (thread_id, human_content, now, thread_id, ai_content, now),
            )

    async def get_history(
        self, thread_id: str, *, limit: int = MAX_HISTORY_MESSAGES
    ) -> list[BaseMessage]:
        async with self._pool.connection() as conn:
            result = await conn.execute(
                """
                SELECT role, content FROM (
                    SELECT role, content, created_at
                    FROM chat_messages
                    WHERE thread_id = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                ) sub
                ORDER BY created_at ASC
                """,
                (thread_id, limit),
            )
            rows = await result.fetchall()

        messages: list[BaseMessage] = []
        for row in rows:
            if row["role"] == "human":
                messages.append(HumanMessage(content=row["content"]))
            else:
                messages.append(AIMessage(content=row["content"]))
        return messages

    # ------------------------------------------------------------------
    # Reranked documents
    # ------------------------------------------------------------------

    async def save_reranked_docs(self, thread_id: str, docs: list[Document]) -> None:
        serialized = [
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata,
            }
            for doc in docs
        ]
        docs_json = json.dumps(serialized)
        async with self._pool.connection() as conn:
            await conn.execute(
                """
                INSERT INTO thread_reranked_docs (thread_id, docs_json, updated_at)
                VALUES (%s, %s::jsonb, NOW())
                ON CONFLICT (thread_id)
                DO UPDATE SET docs_json = EXCLUDED.docs_json,
                              updated_at = NOW()
                """,
                (thread_id, docs_json),
            )

    async def get_reranked_docs(self, thread_id: str) -> list[Document]:
        async with self._pool.connection() as conn:
            result = await conn.execute(
                "SELECT docs_json FROM thread_reranked_docs WHERE thread_id = %s",
                (thread_id,),
            )
            row = await result.fetchone()

        if not row:
            return []

        raw = row["docs_json"]
        if isinstance(raw, str):
            raw = json.loads(raw)

        return [
            Document(page_content=item["page_content"], metadata=item["metadata"])
            for item in raw
        ]

    # ------------------------------------------------------------------
    # Feedback
    # ------------------------------------------------------------------

    async def save_feedback(
        self,
        thread_id: str,
        run_id: str,
        score: float,
        comment: str = "",
    ) -> int:
        async with self._pool.connection() as conn:
            result = await conn.execute(
                """
                INSERT INTO api_feedback (thread_id, run_id, score, comment, created_at)
                VALUES (%s, %s, %s, %s, NOW())
                RETURNING id
                """,
                (thread_id, run_id, score, comment),
            )
            row = await result.fetchone()
            return row["id"]

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def delete_thread_data(self, thread_id: str) -> None:
        async with self._pool.connection() as conn:
            await conn.execute(
                "DELETE FROM chat_messages WHERE thread_id = %s", (thread_id,)
            )
            await conn.execute(
                "DELETE FROM thread_reranked_docs WHERE thread_id = %s",
                (thread_id,),
            )
            await conn.execute(
                "DELETE FROM api_feedback WHERE thread_id = %s", (thread_id,)
            )
