from datetime import datetime
from uuid import UUID

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
# Projects
# ---------------------------------------------------------------------------


class ProjectCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    description: str = Field(default="", max_length=1000)


class ProjectOut(BaseModel):
    id: UUID
    name: str
    description: str
    created_at: datetime
    updated_at: datetime


class ProjectListResponse(BaseModel):
    projects: list[ProjectOut]


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    project_id: UUID
    thread_id: str | None = Field(
        default=None,
        description="Conversation thread ID. Omit to auto-create a new thread.",
    )

    model_config = {"extra": "forbid"}


class Citation(BaseModel):
    chunk_id: str = ""
    source_file: str
    page: str | int
    section: str
    doc_type: str
    relevance_score: float


class ChatResponse(BaseModel):
    answer: str
    confidence: str
    citations: list[Citation]
    validation_passed: bool = True
    validation_reason: str = ""
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


class ConversationCreateRequest(BaseModel):
    project_id: UUID


class ConversationOut(BaseModel):
    thread_id: str


class ConversationListResponse(BaseModel):
    conversations: list[ConversationOut]


class ConversationMessageOut(BaseModel):
    role: str
    content: str
    created_at: datetime


class ConversationDetailResponse(BaseModel):
    thread_id: str
    messages: list[ConversationMessageOut]
