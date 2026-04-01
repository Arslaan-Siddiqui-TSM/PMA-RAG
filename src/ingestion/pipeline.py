import hashlib
from pathlib import Path

from src.db.metadata import MetadataStore
from src.ingestion.chunker import chunk_documents
from src.ingestion.enrichment import enrich_chunks
from src.ingestion.loaders import load_document
from src.ingestion.structure import extract_structure
from src.retrieval.bm25 import BM25Index
from src.retrieval.vectorstore import VectorStoreManager
from config import settings


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
    structured_docs = extract_structure(docs)
    chunks = chunk_documents(structured_docs)

    if not chunks:
        return 0

    if settings.enrich_chunks:
        chunks = await enrich_chunks(chunks)

    chunk_ids: list[str] = []
    for idx, chunk in enumerate(chunks):
        chunk_id = f"{file_hash}:{idx}"
        chunk.metadata["chunk_id"] = chunk_id
        chunk_ids.append(chunk_id)

    file_name = original_name or Path(file_path).name
    document_id = await metadata_store.insert_document(
        file_name=file_name,
        doc_type=doc_type,
        file_hash=file_hash,
        chunk_count=len(chunks),
    )

    vectorstore_manager.add_documents(chunks, ids=chunk_ids)
    await metadata_store.insert_chunks(document_id, chunks)
    await bm25_index.add_documents(chunks)

    return len(chunks)
