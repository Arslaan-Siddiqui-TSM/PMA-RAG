import asyncio
import logging

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

from config import settings

logger = logging.getLogger(__name__)


class VectorStoreManager:
    def __init__(self) -> None:
        self._embeddings = NVIDIAEmbeddings(
            model=settings.embedding_model, truncate="NONE"
        )
        self._chroma_client = chromadb.CloudClient(
            tenant=settings.chroma_tenant,
            database=settings.chroma_database,
            api_key=settings.chroma_api_key,
        )
        self._stores: dict[str, Chroma] = {}

    def _get_store(self, collection_name: str) -> Chroma:
        if collection_name not in self._stores:
            self._stores[collection_name] = Chroma(
                client=self._chroma_client,
                collection_name=collection_name,
                embedding_function=self._embeddings,
            )
        return self._stores[collection_name]

    def add_documents(
        self,
        collection_name: str,
        documents: list[Document],
        ids: list[str] | None = None,
    ) -> None:
        store = self._get_store(collection_name)
        store.add_documents(documents, ids=ids)

    def as_retriever(
        self,
        collection_name: str,
        filters: dict[str, str] | None = None,
    ) -> VectorStoreRetriever:
        store = self._get_store(collection_name)
        search_kwargs: dict = {"k": settings.vector_search_k}
        if filters:
            search_kwargs["filter"] = dict(filters)
        return store.as_retriever(search_kwargs=search_kwargs)

    async def similarity_search(
        self,
        collection_name: str,
        query: str,
        *,
        k: int | None = None,
        filters: dict[str, str] | None = None,
    ) -> list[Document]:
        store = self._get_store(collection_name)
        return await asyncio.to_thread(
            store.similarity_search,
            query,
            k=k or settings.vector_search_k,
            filter=filters or None,
        )

    def delete_by_ids(self, collection_name: str, ids: list[str]) -> None:
        if not ids:
            return
        collection = self._chroma_client.get_collection(collection_name)
        collection.delete(ids=ids)

    def delete_by_source_file(self, collection_name: str, source_file: str) -> None:
        collection = self._chroma_client.get_collection(collection_name)
        collection.delete(where={"source_file": source_file})

    def delete_collection(self, collection_name: str) -> None:
        try:
            self._chroma_client.delete_collection(collection_name)
        except Exception:
            logger.warning(
                "Failed to delete Chroma collection %s", collection_name, exc_info=True
            )
            raise
        self._stores.pop(collection_name, None)

    def list_collections(self) -> list[str]:
        return [c.name for c in self._chroma_client.list_collections()]

    @property
    def chroma_client(self) -> chromadb.CloudClient:
        return self._chroma_client
