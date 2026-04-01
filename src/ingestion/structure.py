"""Extract and attach section hierarchy metadata to parsed document elements.

After parsing, this module groups elements by detected headings and attaches
``section_title`` and ``parent_section`` to each document's metadata.
"""

from __future__ import annotations

from langchain_core.documents import Document

_HEADING_ELEMENT_TYPES = {"Title", "Header"}


def _is_heading(doc: Document) -> bool:
    el_type = doc.metadata.get("element_type", "")
    if el_type in _HEADING_ELEMENT_TYPES:
        return True
    if doc.metadata.get("section_title") and not doc.metadata.get("parent_section"):
        return False
    return False


def extract_structure(docs: list[Document]) -> list[Document]:
    """Walk through parsed elements and propagate section hierarchy.

    For PDFs parsed by unstructured, ``Title`` elements mark new sections.
    For DOCX, the loader already sets ``section_title`` / ``parent_section``.
    Markdown documents get hierarchy from the chunker's header splitter.

    This function fills in any missing ``section_title`` / ``parent_section``
    fields by tracking the most recent heading elements.
    """
    current_section = ""
    current_parent = ""

    for doc in docs:
        el_type = doc.metadata.get("element_type", "")

        if doc.metadata.get("section_title"):
            current_section = doc.metadata["section_title"]
            current_parent = doc.metadata.get("parent_section", "")
            continue

        if el_type in _HEADING_ELEMENT_TYPES:
            text = doc.page_content.strip()
            if current_section and not current_parent:
                current_parent = current_section
            current_section = text
            doc.metadata["section_title"] = text
            doc.metadata["parent_section"] = current_parent
            continue

        doc.metadata.setdefault("section_title", current_section)
        doc.metadata.setdefault("parent_section", current_parent)

    return docs
