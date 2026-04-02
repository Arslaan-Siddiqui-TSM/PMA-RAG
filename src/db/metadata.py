from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from uuid import uuid4

from langchain_core.documents import Document
from psycopg_pool import AsyncConnectionPool

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS projects (
    id UUID PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    collection_name TEXT,
    vector_cleanup_pending BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    deleted_at TIMESTAMPTZ
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_projects_unique_active_name
    ON projects (name)
    WHERE deleted_at IS NULL;

CREATE INDEX IF NOT EXISTS idx_projects_updated_at
    ON projects (updated_at DESC);

ALTER TABLE IF EXISTS documents
    ADD COLUMN IF NOT EXISTS project_id UUID;

ALTER TABLE IF EXISTS chunks
    ADD COLUMN IF NOT EXISTS project_id UUID;

ALTER TABLE IF EXISTS documents
    DROP CONSTRAINT IF EXISTS documents_file_hash_key;

CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    file_name TEXT NOT NULL,
    doc_type TEXT NOT NULL,
    file_hash TEXT NOT NULL,
    chunk_count INTEGER NOT NULL,
    uploaded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(project_id, file_hash)
);

CREATE INDEX IF NOT EXISTS idx_documents_project_uploaded
    ON documents (project_id, uploaded_at DESC);

CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_project_file_hash
    ON documents (project_id, file_hash);

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
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

CREATE INDEX IF NOT EXISTS idx_chunks_project_id
    ON chunks (project_id);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id
    ON chunks (document_id);

CREATE INDEX IF NOT EXISTS idx_chunks_search_vector
    ON chunks USING GIN (search_vector);

UPDATE chunks c
SET project_id = d.project_id
FROM documents d
WHERE c.document_id = d.id
  AND c.project_id IS NULL
  AND d.project_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS threads (
    thread_id UUID PRIMARY KEY,
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_threads_project_created
    ON threads (project_id, created_at DESC);
"""

_SLUG_INVALID = re.compile(r"[^a-z0-9]+")


def slugify_project_name(name: str) -> str:
    slug = _SLUG_INVALID.sub("-", name.strip().lower()).strip("-")
    return slug or "project"


class MetadataStore:
    def __init__(self, pool: AsyncConnectionPool) -> None:
        self._pool = pool

    async def setup(self) -> None:
        async with self._pool.connection() as conn:
            await conn.execute(CREATE_TABLE_SQL)

    # ------------------------------------------------------------------
    # Project helpers
    # ------------------------------------------------------------------

    async def touch_project(self, project_id: str) -> None:
        async with self._pool.connection() as conn:
            await conn.execute(
                "UPDATE projects SET updated_at = NOW() WHERE id = %s AND deleted_at IS NULL",
                (project_id,),
            )

    async def create_project(self, name: str, description: str = "") -> dict:
        trimmed = name.strip()
        now = datetime.now(timezone.utc)
        pid = str(uuid4())
        col_name = f"{slugify_project_name(trimmed)}__{pid}"
        async with self._pool.connection() as conn:
            result = await conn.execute(
                """
                INSERT INTO projects (id, name, description, collection_name,
                                      vector_cleanup_pending, created_at, updated_at)
                VALUES (%s, %s, %s, %s, FALSE, %s, %s)
                RETURNING id, name, description, collection_name, created_at, updated_at
                """,
                (pid, trimmed, description, col_name, now, now),
            )
            return dict(await result.fetchone())

    async def list_active_projects(self) -> list[dict]:
        async with self._pool.connection() as conn:
            result = await conn.execute(
                """
                SELECT id, name, description, collection_name, created_at, updated_at
                FROM projects
                WHERE deleted_at IS NULL
                ORDER BY updated_at DESC
                """
            )
            return [dict(r) for r in await result.fetchall()]

    async def get_project(
        self, project_id: str, *, include_deleted: bool = False
    ) -> dict | None:
        clause = "id = %s" if include_deleted else "id = %s AND deleted_at IS NULL"
        async with self._pool.connection() as conn:
            result = await conn.execute(
                f"""
                SELECT id, name, description, collection_name,
                       vector_cleanup_pending, created_at, updated_at, deleted_at
                FROM projects WHERE {clause}
                """,
                (project_id,),
            )
            row = await result.fetchone()
            return dict(row) if row else None

    async def set_project_collection_name(
        self, project_id: str, collection_name: str
    ) -> None:
        async with self._pool.connection() as conn:
            await conn.execute(
                "UPDATE projects SET collection_name = %s, updated_at = NOW() WHERE id = %s AND deleted_at IS NULL",
                (collection_name, project_id),
            )

    async def mark_vector_cleanup_pending(
        self, project_id: str, pending: bool = True
    ) -> None:
        async with self._pool.connection() as conn:
            await conn.execute(
                "UPDATE projects SET vector_cleanup_pending = %s, updated_at = NOW() WHERE id = %s",
                (pending, project_id),
            )

    async def soft_delete_project(self, project_id: str) -> dict | None:
        async with self._pool.connection() as conn:
            result = await conn.execute(
                """
                UPDATE projects
                SET deleted_at = NOW(), updated_at = NOW()
                WHERE id = %s AND deleted_at IS NULL
                RETURNING id, name, description, collection_name,
                          vector_cleanup_pending, created_at, updated_at, deleted_at
                """,
                (project_id,),
            )
            row = await result.fetchone()
            return dict(row) if row else None

    async def hard_delete_project_documents(self, project_id: str) -> None:
        async with self._pool.connection() as conn:
            await conn.execute(
                "DELETE FROM documents WHERE project_id = %s", (project_id,)
            )

    # ------------------------------------------------------------------
    # Thread helpers
    # ------------------------------------------------------------------

    async def create_thread(self, thread_id: str, project_id: str) -> None:
        async with self._pool.connection() as conn:
            await conn.execute(
                """
                INSERT INTO threads (thread_id, project_id, created_at, updated_at)
                VALUES (%s, %s, NOW(), NOW())
                ON CONFLICT (thread_id) DO UPDATE SET updated_at = NOW()
                """,
                (thread_id, project_id),
            )
            await conn.execute(
                "UPDATE projects SET updated_at = NOW() WHERE id = %s AND deleted_at IS NULL",
                (project_id,),
            )

    async def get_thread_project_id(self, thread_id: str) -> str | None:
        async with self._pool.connection() as conn:
            result = await conn.execute(
                "SELECT project_id FROM threads WHERE thread_id = %s",
                (thread_id,),
            )
            row = await result.fetchone()
            return str(row["project_id"]) if row else None

    async def list_threads(self, project_id: str) -> list[str]:
        async with self._pool.connection() as conn:
            result = await conn.execute(
                "SELECT thread_id FROM threads WHERE project_id = %s ORDER BY created_at DESC",
                (project_id,),
            )
            return [str(r["thread_id"]) for r in await result.fetchall()]

    async def delete_thread(self, thread_id: str, project_id: str) -> bool:
        async with self._pool.connection() as conn:
            result = await conn.execute(
                "DELETE FROM threads WHERE thread_id = %s AND project_id = %s RETURNING thread_id",
                (thread_id, project_id),
            )
            row = await result.fetchone()
            if row:
                await conn.execute(
                    "UPDATE projects SET updated_at = NOW() WHERE id = %s AND deleted_at IS NULL",
                    (project_id,),
                )
            return row is not None

    async def delete_threads_for_project(self, project_id: str) -> list[str]:
        async with self._pool.connection() as conn:
            result = await conn.execute(
                "DELETE FROM threads WHERE project_id = %s RETURNING thread_id",
                (project_id,),
            )
            return [str(r["thread_id"]) for r in await result.fetchall()]

    # ------------------------------------------------------------------
    # Document CRUD (project-scoped)
    # ------------------------------------------------------------------

    async def insert_document(
        self,
        *,
        project_id: str,
        file_name: str,
        doc_type: str,
        file_hash: str,
        chunk_count: int,
    ) -> int:
        async with self._pool.connection() as conn:
            result = await conn.execute(
                """
                INSERT INTO documents (project_id, file_name, doc_type, file_hash,
                                       chunk_count, uploaded_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    project_id,
                    file_name,
                    doc_type,
                    file_hash,
                    chunk_count,
                    datetime.now(timezone.utc),
                ),
            )
            row = await result.fetchone()
            await conn.execute(
                "UPDATE projects SET updated_at = NOW() WHERE id = %s AND deleted_at IS NULL",
                (project_id,),
            )
            return int(row["id"])

    async def insert_chunks(
        self, project_id: str, document_id: int, chunks: list[Document]
    ) -> None:
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
                keywords = [str(k) for k in (chunk.metadata.get("keywords") or [])]
                questions = [str(q) for q in (chunk.metadata.get("questions") or [])]
                metadata_json = json.dumps(chunk.metadata)

                await conn.execute(
                    """
                    INSERT INTO chunks (
                        id, project_id, document_id, chunk_index, content,
                        section_title, summary, keywords, questions,
                        metadata, search_vector
                    )
                    VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb,
                        to_tsvector('english', %s)
                    )
                    ON CONFLICT (id) DO UPDATE SET
                        project_id = EXCLUDED.project_id,
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
                        project_id,
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

    async def list_chunk_ids_for_document(
        self, project_id: str, doc_id: int
    ) -> list[str]:
        async with self._pool.connection() as conn:
            result = await conn.execute(
                "SELECT id FROM chunks WHERE project_id = %s AND document_id = %s ORDER BY chunk_index",
                (project_id, doc_id),
            )
            return [str(r["id"]) for r in await result.fetchall()]

    async def fts_search(
        self,
        query: str,
        *,
        project_id: str,
        k: int,
        doc_type_filter: str | None = None,
        source_file_filter: str | None = None,
        section_filter: str | None = None,
    ) -> list[Document]:
        where_clauses = [
            "project_id = %s",
            "search_vector @@ plainto_tsquery('english', %s)",
        ]
        params: list[object] = [project_id, query]

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

        where_sql = " AND ".join(where_clauses)
        params.append(k)

        async with self._pool.connection() as conn:
            result = await conn.execute(
                f"""
                SELECT id, content, metadata,
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

    async def delete_chunks_by_source_file(
        self, project_id: str, source_file: str
    ) -> None:
        async with self._pool.connection() as conn:
            await conn.execute(
                "DELETE FROM chunks WHERE project_id = %s AND metadata->>'source_file' = %s",
                (project_id, source_file),
            )

    async def document_exists(self, file_hash: str, project_id: str) -> bool:
        async with self._pool.connection() as conn:
            result = await conn.execute(
                "SELECT 1 FROM documents WHERE project_id = %s AND file_hash = %s",
                (project_id, file_hash),
            )
            return await result.fetchone() is not None

    async def get_all_doc_types(self, project_id: str) -> list[str]:
        async with self._pool.connection() as conn:
            result = await conn.execute(
                "SELECT DISTINCT doc_type FROM documents WHERE project_id = %s ORDER BY doc_type",
                (project_id,),
            )
            return [str(r["doc_type"]) for r in await result.fetchall()]

    async def get_document(self, project_id: str, doc_id: int) -> dict | None:
        async with self._pool.connection() as conn:
            result = await conn.execute(
                """
                SELECT id, file_name, doc_type, file_hash, chunk_count, uploaded_at
                FROM documents WHERE id = %s AND project_id = %s
                """,
                (doc_id, project_id),
            )
            row = await result.fetchone()
            return dict(row) if row else None

    async def delete_document(self, project_id: str, doc_id: int) -> dict | None:
        async with self._pool.connection() as conn:
            result = await conn.execute(
                """
                DELETE FROM documents WHERE id = %s AND project_id = %s
                RETURNING id, file_name, doc_type, file_hash, chunk_count, uploaded_at
                """,
                (doc_id, project_id),
            )
            row = await result.fetchone()
            if row:
                await conn.execute(
                    "UPDATE projects SET updated_at = NOW() WHERE id = %s AND deleted_at IS NULL",
                    (project_id,),
                )
            return dict(row) if row else None

    async def list_documents(
        self, project_id: str, *, doc_type_filter: str | None = None
    ) -> list[dict]:
        clause = "project_id = %s"
        params: list[object] = [project_id]
        if doc_type_filter:
            clause += " AND doc_type = %s"
            params.append(doc_type_filter)
        async with self._pool.connection() as conn:
            result = await conn.execute(
                f"""
                SELECT id, file_name, doc_type, chunk_count, uploaded_at
                FROM documents WHERE {clause}
                ORDER BY uploaded_at DESC
                """,
                tuple(params),
            )
            return [dict(r) for r in await result.fetchall()]
