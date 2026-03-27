from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from config import settings

MARKDOWN_HEADERS = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]


def _build_size_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def _propagate_metadata(original: Document, chunks: list[Document]) -> list[Document]:
    for i, chunk in enumerate(chunks):
        chunk.metadata = {**original.metadata, **chunk.metadata}
        chunk.metadata["chunk_index"] = i
    return chunks


def chunk_markdown_documents(docs: list[Document]) -> list[Document]:
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=MARKDOWN_HEADERS)
    size_splitter = _build_size_splitter()
    all_chunks: list[Document] = []

    for doc in docs:
        header_splits = md_splitter.split_text(doc.page_content)
        sized = size_splitter.split_documents(header_splits)
        all_chunks.extend(_propagate_metadata(doc, sized))

    return all_chunks


def chunk_text_documents(docs: list[Document]) -> list[Document]:
    size_splitter = _build_size_splitter()
    all_chunks: list[Document] = []

    for doc in docs:
        sized = size_splitter.split_documents([doc])
        all_chunks.extend(_propagate_metadata(doc, sized))

    return all_chunks


def chunk_documents(docs: list[Document]) -> list[Document]:
    md_docs = [d for d in docs if d.metadata.get("file_extension") == ".md"]
    other_docs = [d for d in docs if d.metadata.get("file_extension") != ".md"]

    chunks: list[Document] = []
    if md_docs:
        chunks.extend(chunk_markdown_documents(md_docs))
    if other_docs:
        chunks.extend(chunk_text_documents(other_docs))

    return chunks
