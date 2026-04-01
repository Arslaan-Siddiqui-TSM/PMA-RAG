from langchain_core.documents import Document
from psycopg_pool import AsyncConnectionPool

from config import settings
from src.db.metadata import MetadataStore


class BM25Index:
    """Postgres FTS-backed lexical retriever.

    Kept as BM25Index for backward compatibility with existing call sites.
    """

    def __init__(self, pool: AsyncConnectionPool) -> None:
        self._metadata_store = MetadataStore(pool)

    async def add_documents(self, documents: list[Document]) -> None:
        # Chunks are persisted by MetadataStore.insert_chunks in ingestion pipeline.
        _ = documents

    async def search(
        self,
        query: str,
        *,
        doc_type_filter: str | None = None,
        source_file_filter: str | None = None,
        section_filter: str | None = None,
        k: int | None = None,
    ) -> list[Document]:
        return await self._metadata_store.fts_search(
            query,
            k=k or settings.fts_search_k,
            doc_type_filter=doc_type_filter,
            source_file_filter=source_file_filter,
            section_filter=section_filter,
        )

    async def delete_by_source_file(self, source_file: str) -> None:
        await self._metadata_store.delete_chunks_by_source_file(source_file)
