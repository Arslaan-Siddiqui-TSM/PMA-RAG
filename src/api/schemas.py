from datetime import datetime

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

VALID_DOC_TYPES = [
    "PRD",
    "BRD",
    "Technical Spec",
    "Test Plan",
    "Use Case",
    "Functional Spec",
    "Non-Functional Spec",
    "Other",
]


class DocumentOut(BaseModel):
    id: int
    file_name: str
    doc_type: str
    chunk_count: int
    uploaded_at: datetime


class DocumentDetailOut(DocumentOut):
    file_hash: str


class UploadResult(BaseModel):
    file_name: str
    doc_type: str
    chunk_count: int
    status: str = Field(description="'ingested', 'skipped', or 'error'")
    detail: str = ""


class UploadResponse(BaseModel):
    results: list[UploadResult]
    total_chunks: int


class DeleteDocumentResponse(BaseModel):
    id: int
    file_name: str
    deleted: bool = True


class DocTypesResponse(BaseModel):
    doc_types: list[str]


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    thread_id: str | None = Field(
        default=None,
        description="Conversation thread ID. Omit to auto-create a new thread.",
    )
    doc_type_filter: str | None = Field(
        default=None,
        description="Optional document type to filter retrieval.",
    )


class Citation(BaseModel):
    source_file: str
    page: str | int
    section: str
    doc_type: str
    relevance_score: float


class ChatResponse(BaseModel):
    answer: str
    confidence: str
    citations: list[Citation]
    thread_id: str
    run_id: str
    search_documents: bool
    response_style: str


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------

class FeedbackRequest(BaseModel):
    thread_id: str
    run_id: str
    score: float = Field(ge=0.0, le=1.0, description="1.0 = positive, 0.0 = negative")
    comment: str = ""


class FeedbackResponse(BaseModel):
    id: int
    status: str = "ok"


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------

class ConversationOut(BaseModel):
    thread_id: str


class ConversationListResponse(BaseModel):
    conversations: list[ConversationOut]


class ConversationDeleteResponse(BaseModel):
    thread_id: str
    deleted: bool = True
