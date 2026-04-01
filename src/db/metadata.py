from __future__ import annotations

import json
from datetime import datetime, timezone

from langchain_core.documents import Document

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

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    section_title TEXT NOT NULL DEFAULT '',
    summary TEXT NOT NULL DEFAULT '',
    keywords TEXT[] NOT NULL DEFAULT '{}',
    questions TEXT[] NOT NULL DEFAULT '{}',
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    search_vector tsvector
);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id
    ON chunks (document_id);

CREATE INDEX IF NOT EXISTS idx_chunks_search_vector
    ON chunks USING GIN (search_vector);
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

    async def insert_chunks(self, document_id: int, chunks: list[Document]) -> None:
        if not chunks:
            return
        async with self._pool.connection() as conn:
            for idx, chunk in enumerate(chunks):
                chunk_id = str(chunk.metadata.get("chunk_id") or f"{document_id}-{idx}")
                section_title = (
                    chunk.metadata.get("section_title")
                    or chunk.metadata.get("h1")
                    or chunk.metadata.get("h2")
                    or ""
                )
                summary = str(chunk.metadata.get("summary") or "")
                keywords = [
                    str(k) for k in (chunk.metadata.get("keywords") or [])
                ]
                questions = [
                    str(q) for q in (chunk.metadata.get("questions") or [])
                ]
                metadata_json = json.dumps(chunk.metadata)

                await conn.execute(
                    """
                    INSERT INTO chunks (
                        id, document_id, chunk_index, content, section_title,
                        summary, keywords, questions, metadata, search_vector
                    )
                    VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb,
                        to_tsvector('english', %s)
                    )
                    ON CONFLICT (id) DO UPDATE SET
                        document_id = EXCLUDED.document_id,
                        chunk_index = EXCLUDED.chunk_index,
                        content = EXCLUDED.content,
                        section_title = EXCLUDED.section_title,
                        summary = EXCLUDED.summary,
                        keywords = EXCLUDED.keywords,
                        questions = EXCLUDED.questions,
                        metadata = EXCLUDED.metadata,
                        search_vector = EXCLUDED.search_vector
                    """,
                    (
                        chunk_id,
                        document_id,
                        int(chunk.metadata.get("chunk_index", idx)),
                        chunk.page_content,
                        section_title,
                        summary,
                        keywords,
                        questions,
                        metadata_json,
                        chunk.page_content,
                    ),
                )

    async def fts_search(
        self,
        query: str,
        *,
        k: int,
        doc_type_filter: str | None = None,
        source_file_filter: str | None = None,
        section_filter: str | None = None,
    ) -> list[Document]:
        where_clauses = ["search_vector @@ plainto_tsquery('english', %s)"]
        params: list = [query]

        if doc_type_filter:
            where_clauses.append("metadata->>'doc_type' = %s")
            params.append(doc_type_filter)
        if source_file_filter:
            where_clauses.append("metadata->>'source_file' = %s")
            params.append(source_file_filter)
        if section_filter:
            where_clauses.append(
                "(metadata->>'section_title' = %s OR section_title = %s)"
            )
            params.extend([section_filter, section_filter])

        params.append(k)
        where_sql = " AND ".join(where_clauses)

        async with self._pool.connection() as conn:
            result = await conn.execute(
                f"""
                SELECT
                    id,
                    content,
                    metadata,
                    ts_rank_cd(search_vector, plainto_tsquery('english', %s)) AS rank
                FROM chunks
                WHERE {where_sql}
                ORDER BY rank DESC, chunk_index ASC
                LIMIT %s
                """,
                (query, *params),
            )
            rows = await result.fetchall()

        docs: list[Document] = []
        for row in rows:
            metadata = row["metadata"]
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            metadata = dict(metadata or {})
            metadata["chunk_id"] = row["id"]
            metadata["fts_rank"] = float(row["rank"] or 0.0)
            docs.append(Document(page_content=row["content"], metadata=metadata))
        return docs

    async def delete_chunks_by_source_file(self, source_file: str) -> None:
        async with self._pool.connection() as conn:
            await conn.execute(
                "DELETE FROM chunks WHERE metadata->>'source_file' = %s",
                (source_file,),
            )

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

    async def get_document(self, doc_id: int) -> dict | None:
        async with self._pool.connection() as conn:
            result = await conn.execute(
                """
                SELECT id, file_name, doc_type, file_hash, chunk_count, uploaded_at
                FROM documents
                WHERE id = %s
                """,
                (doc_id,),
            )
            return await result.fetchone()

    async def delete_document(self, doc_id: int) -> dict | None:
        async with self._pool.connection() as conn:
            result = await conn.execute(
                """
                DELETE FROM documents
                WHERE id = %s
                RETURNING id, file_name, doc_type, file_hash, chunk_count, uploaded_at
                """,
                (doc_id,),
            )
            return await result.fetchone()

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
