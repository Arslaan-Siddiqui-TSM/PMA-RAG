from pathlib import Path

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyMuPDFLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.documents import Document

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".md"}

LOADER_MAP = {
    ".pdf": PyMuPDFLoader,
    ".docx": Docx2txtLoader,
    ".md": UnstructuredMarkdownLoader,
}


def load_document(
    file_path: str,
    doc_type: str | None = None,
    original_name: str | None = None,
) -> list[Document]:
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext not in LOADER_MAP:
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported: {SUPPORTED_EXTENSIONS}"
        )

    loader_cls = LOADER_MAP[ext]
    loader = loader_cls(str(path))
    docs = loader.load()

    for doc in docs:
        doc.metadata["source_file"] = original_name or path.name
        doc.metadata["file_extension"] = ext
        if doc_type:
            doc.metadata["doc_type"] = doc_type

    return docs
