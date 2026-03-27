from datetime import datetime, timezone

from psycopg_pool import AsyncConnectionPool

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    file_name TEXT NOT NULL,
    doc_type TEXT NOT NULL,
    file_hash TEXT NOT NULL UNIQUE,
    chunk_count INTEGER NOT NULL,
    uploaded_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""


class MetadataStore:
    def __init__(self, pool: AsyncConnectionPool) -> None:
        self._pool = pool

    async def setup(self) -> None:
        async with self._pool.connection() as conn:
            await conn.execute(CREATE_TABLE_SQL)

    async def insert_document(
        self,
        file_name: str,
        doc_type: str,
        file_hash: str,
        chunk_count: int,
    ) -> int:
        async with self._pool.connection() as conn:
            result = await conn.execute(
                """
                INSERT INTO documents (file_name, doc_type, file_hash, chunk_count, uploaded_at)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    file_name,
                    doc_type,
                    file_hash,
                    chunk_count,
                    datetime.now(timezone.utc),
                ),
            )
            row = await result.fetchone()
            return row["id"]

    async def document_exists(self, file_hash: str) -> bool:
        async with self._pool.connection() as conn:
            result = await conn.execute(
                "SELECT 1 FROM documents WHERE file_hash = %s",
                (file_hash,),
            )
            return await result.fetchone() is not None

    async def get_all_doc_types(self) -> list[str]:
        async with self._pool.connection() as conn:
            result = await conn.execute(
                "SELECT DISTINCT doc_type FROM documents ORDER BY doc_type"
            )
            rows = await result.fetchall()
            return [row["doc_type"] for row in rows]

    async def list_documents(self) -> list[dict]:
        async with self._pool.connection() as conn:
            result = await conn.execute(
                """
                SELECT id, file_name, doc_type, chunk_count, uploaded_at
                FROM documents
                ORDER BY uploaded_at DESC
                """
            )
            return await result.fetchall()
