"""Structure-aware, token-based document chunker.

Chunks by section/subsection boundaries first, then splits oversized sections
using a token-based ``RecursiveCharacterTextSplitter`` backed by tiktoken.
"""

from __future__ import annotations

import tiktoken
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

_TOKENIZER = tiktoken.get_encoding("cl100k_base")


def _token_length(text: str) -> int:
    return len(_TOKENIZER.encode(text))


def _build_token_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=settings.chunk_size_tokens,
        chunk_overlap=settings.chunk_overlap_tokens,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def _propagate_metadata(original: Document, chunks: list[Document]) -> list[Document]:
    for i, chunk in enumerate(chunks):
        chunk.metadata = {**original.metadata, **chunk.metadata}
        chunk.metadata["chunk_index"] = i
    return chunks


def _group_by_section(docs: list[Document]) -> list[list[Document]]:
    """Group documents that share the same section_title into contiguous runs."""
    if not docs:
        return []
    groups: list[list[Document]] = []
    current_section = docs[0].metadata.get("section_title", "")
    current_group: list[Document] = [docs[0]]

    for doc in docs[1:]:
        section = doc.metadata.get("section_title", "")
        if section == current_section:
            current_group.append(doc)
        else:
            groups.append(current_group)
            current_section = section
            current_group = [doc]
    groups.append(current_group)
    return groups


def chunk_structured_documents(docs: list[Document]) -> list[Document]:
    """Chunk documents respecting section boundaries, then token-split oversized."""
    splitter = _build_token_splitter()
    all_chunks: list[Document] = []

    for section_group in _group_by_section(docs):
        combined_text = "\n\n".join(d.page_content for d in section_group)
        base_meta = {**section_group[0].metadata}

        if _token_length(combined_text) <= settings.chunk_size_tokens:
            chunk = Document(page_content=combined_text, metadata=base_meta)
            all_chunks.append(chunk)
        else:
            sub_chunks = splitter.split_text(combined_text)
            for text in sub_chunks:
                all_chunks.append(Document(page_content=text, metadata={**base_meta}))

    for i, chunk in enumerate(all_chunks):
        chunk.metadata["chunk_index"] = i

    return all_chunks


def chunk_markdown_documents(docs: list[Document]) -> list[Document]:
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=MARKDOWN_HEADERS)
    token_splitter = _build_token_splitter()
    all_chunks: list[Document] = []

    for doc in docs:
        header_splits = md_splitter.split_text(doc.page_content)
        sized = token_splitter.split_documents(header_splits)
        all_chunks.extend(_propagate_metadata(doc, sized))

    return all_chunks


def chunk_documents(docs: list[Document]) -> list[Document]:
    md_docs = [d for d in docs if d.metadata.get("file_extension") == ".md"]
    other_docs = [d for d in docs if d.metadata.get("file_extension") != ".md"]

    chunks: list[Document] = []
    if md_docs:
        chunks.extend(chunk_markdown_documents(md_docs))
    if other_docs:
        chunks.extend(chunk_structured_documents(other_docs))

    return chunks
