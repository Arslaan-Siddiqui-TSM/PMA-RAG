import chromadb
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

    def add_documents(self, documents: list[Document]) -> None:
        self._vectorstore.add_documents(documents)

    def as_retriever(self, doc_type_filter: str | None = None) -> VectorStoreRetriever:
        search_kwargs: dict = {"k": settings.vector_search_k}
        if doc_type_filter:
            search_kwargs["filter"] = {"doc_type": doc_type_filter}
        return self._vectorstore.as_retriever(search_kwargs=search_kwargs)

    @property
    def vectorstore(self) -> Chroma:
        return self._vectorstore
