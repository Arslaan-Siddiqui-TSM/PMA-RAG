import pickle
from pathlib import Path

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from config import settings

BM25_INDEX_DIR = Path("data/bm25_index")
BM25_DOCS_PATH = BM25_INDEX_DIR / "bm25_documents.pkl"


class BM25Index:
    def __init__(self) -> None:
        self._documents: list[Document] = []
        self._load_persisted()

    def _load_persisted(self) -> None:
        if BM25_DOCS_PATH.exists():
            with open(BM25_DOCS_PATH, "rb") as f:
                self._documents = pickle.load(f)

    def _persist(self) -> None:
        BM25_INDEX_DIR.mkdir(parents=True, exist_ok=True)
        with open(BM25_DOCS_PATH, "wb") as f:
            pickle.dump(self._documents, f)

    def add_documents(self, documents: list[Document]) -> None:
        self._documents.extend(documents)
        self._persist()

    def as_retriever(self, doc_type_filter: str | None = None) -> BM25Retriever:
        docs = self._documents
        if doc_type_filter:
            docs = [d for d in docs if d.metadata.get("doc_type") == doc_type_filter]

        if not docs:
            docs = [Document(page_content="empty", metadata={})]

        return BM25Retriever.from_documents(
            docs, k=min(settings.bm25_search_k, len(docs))
        )

    def delete_by_source_file(self, source_file: str) -> None:
        self._documents = [
            d for d in self._documents
            if d.metadata.get("source_file") != source_file
        ]
        self._persist()

    @property
    def document_count(self) -> int:
        return len(self._documents)
