import chromadb
import asyncio
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

from config import settings


class VectorStoreManager:
    def __init__(self) -> None:
        self._embeddings = NVIDIAEmbeddings(model=settings.embedding_model)
        self._chroma_client = chromadb.CloudClient(
            tenant=settings.chroma_tenant,
            database=settings.chroma_database,
            api_key=settings.chroma_api_key,
        )
        self._vectorstore = Chroma(
            client=self._chroma_client,
            collection_name=settings.chroma_collection_name,
            embedding_function=self._embeddings,
        )

    def add_documents(
        self, documents: list[Document], ids: list[str] | None = None
    ) -> None:
        self._vectorstore.add_documents(documents, ids=ids)

    def as_retriever(
        self,
        doc_type_filter: str | None = None,
        filters: dict[str, str] | None = None,
    ) -> VectorStoreRetriever:
        search_kwargs: dict = {"k": settings.vector_search_k}
        merged_filter = dict(filters or {})
        if doc_type_filter and "doc_type" not in merged_filter:
            merged_filter["doc_type"] = doc_type_filter
        if merged_filter:
            search_kwargs["filter"] = merged_filter
        return self._vectorstore.as_retriever(search_kwargs=search_kwargs)

    async def similarity_search(
        self,
        query: str,
        *,
        k: int | None = None,
        filters: dict[str, str] | None = None,
    ) -> list[Document]:
        return await asyncio.to_thread(
            self._vectorstore.similarity_search,
            query,
            k=k or settings.vector_search_k,
            filter=filters or None,
        )

    def delete_by_source_file(self, source_file: str) -> None:
        collection = self._chroma_client.get_collection(
            settings.chroma_collection_name
        )
        collection.delete(where={"source_file": source_file})

    @property
    def vectorstore(self) -> Chroma:
        return self._vectorstore
