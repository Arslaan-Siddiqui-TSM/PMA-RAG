import hashlib
from pathlib import Path

from src.db.metadata import MetadataStore
from src.ingestion.chunker import chunk_documents
from src.ingestion.loaders import load_document
from src.retrieval.bm25 import BM25Index
from src.retrieval.vectorstore import VectorStoreManager


def compute_file_hash(file_path: str) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()


async def ingest_document(
    file_path: str,
    doc_type: str,
    metadata_store: MetadataStore,
    vectorstore_manager: VectorStoreManager,
    bm25_index: BM25Index,
    original_name: str | None = None,
) -> int:
    """Ingest a single document: load, chunk, embed, store. Returns chunk count."""
    file_hash = compute_file_hash(file_path)

    if await metadata_store.document_exists(file_hash):
        return 0

    docs = load_document(file_path, doc_type=doc_type, original_name=original_name)
    chunks = chunk_documents(docs)

    if not chunks:
        return 0

    vectorstore_manager.add_documents(chunks)
    bm25_index.add_documents(chunks)

    file_name = original_name or Path(file_path).name
    await metadata_store.insert_document(
        file_name=file_name,
        doc_type=doc_type,
        file_hash=file_hash,
        chunk_count=len(chunks),
    )

    return len(chunks)
