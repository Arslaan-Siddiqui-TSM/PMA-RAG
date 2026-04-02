from langchain_core.documents import Document
from psycopg_pool import AsyncConnectionPool

from config import settings
from src.db.metadata import MetadataStore


class BM25Index:
    """Postgres FTS-backed lexical retriever (project-scoped)."""

    def __init__(self, pool: AsyncConnectionPool) -> None:
        self._metadata_store = MetadataStore(pool)

    async def add_documents(self, documents: list[Document]) -> None:
        _ = documents

    async def search(
        self,
        query: str,
        *,
        project_id: str,
        doc_type_filter: str | None = None,
        source_file_filter: str | None = None,
        section_filter: str | None = None,
        k: int | None = None,
    ) -> list[Document]:
        return await self._metadata_store.fts_search(
            query,
            project_id=project_id,
            k=k or settings.fts_search_k,
            doc_type_filter=doc_type_filter,
            source_file_filter=source_file_filter,
            section_filter=section_filter,
        )

    async def delete_by_source_file(self, project_id: str, source_file: str) -> None:
        await self._metadata_store.delete_chunks_by_source_file(project_id, source_file)
