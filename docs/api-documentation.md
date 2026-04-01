# PMA-RAG API Documentation

## Base Information

- **Base path (versioned routes):** `/api/v1`
- **Health route:** `/health`
- **Content type (JSON APIs):** `application/json`
- **Streaming endpoint:** Server-Sent Events (`text/event-stream`)

---

## 1) Health Check

### Endpoint
- **Method:** `GET`
- **Path:** `/health`
- **Description:** Checks whether the API service is up.

### Params
- None

### Request Schema
- None

### Response Schema
```json
{
  "status": "string"
}
```

### Example Response Payload
```json
{
  "status": "ok"
}
```

---

## 2) Chat Completion

### Endpoint
- **Method:** `POST`
- **Path:** `/api/v1/chat`
- **Description:** Sends a user question to the RAG graph and returns a grounded answer with citations.

### Params
- **Body:** `question` (required), `thread_id` (optional), `doc_type_filter` (optional), `source_file_filter` (optional), `section_filter` (optional)

### Request Schema
```json
{
  "question": "string (min length: 1)",
  "thread_id": "string | null",
  "doc_type_filter": "string | null",
  "source_file_filter": "string | null",
  "section_filter": "string | null"
}
```

### Response Schema
```json
{
  "answer": "string",
  "confidence": "string",
  "citations": [
    {
      "chunk_id": "string",
      "source_file": "string",
      "page": "string | number",
      "section": "string",
      "doc_type": "string",
      "relevance_score": "number"
    }
  ],
  "validation_passed": "boolean",
  "validation_reason": "string",
  "thread_id": "string",
  "run_id": "string",
  "search_documents": "boolean",
  "response_style": "string"
}
```

### Example Request Payload
```json
{
  "question": "Summarize the BRD scope for user onboarding.",
  "thread_id": null,
  "doc_type_filter": "BRD",
  "source_file_filter": null,
  "section_filter": null
}
```

### Example Response Payload
```json
{
  "answer": "The BRD defines onboarding as ...",
  "confidence": "high",
  "citations": [
    {
      "chunk_id": "chunk-018",
      "source_file": "Onboarding-BRD.pdf",
      "page": 7,
      "section": "Scope",
      "doc_type": "BRD",
      "relevance_score": 0.91
    }
  ],
  "validation_passed": true,
  "validation_reason": "",
  "thread_id": "f6e9ef17-56dd-4bd7-a74d-6c2d2b8a4c77",
  "run_id": "11566993-2f0f-47df-9f98-14f8c8f6c3e1",
  "search_documents": true,
  "response_style": "default"
}
```

---

## 3) Streaming Chat

### Endpoint
- **Method:** `POST`
- **Path:** `/api/v1/chat/stream`
- **Description:** Streams answer generation and intermediate metadata as SSE events.

### Params
- **Body:** same as `/api/v1/chat`

### Request Schema
```json
{
  "question": "string (min length: 1)",
  "thread_id": "string | null",
  "doc_type_filter": "string | null",
  "source_file_filter": "string | null",
  "section_filter": "string | null"
}
```

### Response Schema
- **Content-Type:** `text/event-stream`
- **SSE Events emitted:**
  - `thread_id`: `{"thread_id":"string","run_id":"string"}`
  - `intent`: `{"intent":"string","search_documents":"boolean","response_style":"string"}`
  - `confidence`: `{"confidence":"string"}`
  - `token`: `{"token":"string"}`
  - `done`: same payload shape as `ChatResponse`

### Example Request Payload
```json
{
  "question": "What are the acceptance criteria in the test plan?",
  "thread_id": "f6e9ef17-56dd-4bd7-a74d-6c2d2b8a4c77",
  "doc_type_filter": "Test Plan",
  "source_file_filter": null,
  "section_filter": "Acceptance Criteria"
}
```

### Example Final (`done`) Event Payload
```json
{
  "answer": "The acceptance criteria include ...",
  "confidence": "medium",
  "citations": [],
  "validation_passed": true,
  "validation_reason": "",
  "thread_id": "f6e9ef17-56dd-4bd7-a74d-6c2d2b8a4c77",
  "run_id": "f9cb1f49-1770-4f37-b7f5-6401eb90d95a",
  "search_documents": true,
  "response_style": "default"
}
```

---

## 4) Feedback Submission

### Endpoint
- **Method:** `POST`
- **Path:** `/api/v1/feedback`
- **Description:** Stores user feedback for a specific run.

### Params
- **Body:** `thread_id` (required), `run_id` (required), `score` (required, `0.0-1.0`), `comment` (optional)

### Request Schema
```json
{
  "thread_id": "string",
  "run_id": "string",
  "score": "number (0.0 to 1.0)",
  "comment": "string"
}
```

### Response Schema
```json
{
  "id": "number",
  "status": "string"
}
```

### Example Request Payload
```json
{
  "thread_id": "f6e9ef17-56dd-4bd7-a74d-6c2d2b8a4c77",
  "run_id": "11566993-2f0f-47df-9f98-14f8c8f6c3e1",
  "score": 1.0,
  "comment": "Very accurate answer."
}
```

### Example Response Payload
```json
{
  "id": 42,
  "status": "ok"
}
```

---

## 5) Upload Documents

### Endpoint
- **Method:** `POST`
- **Path:** `/api/v1/documents/upload`
- **Description:** Uploads one or more files and ingests them into metadata, vector, and BM25 stores.

### Params
- **Form Data:** `files` (required, list of files), `doc_type` (required)
- **Allowed file extensions:** `.pdf`, `.docx`, `.md`
- **Allowed `doc_type` values:** `PRD`, `BRD`, `Technical Spec`, `Test Plan`, `Use Case`, `Functional Spec`, `Non-Functional Spec`, `Other`

### Request Schema
```json
{
  "files": ["binary file", "binary file", "..."],
  "doc_type": "string"
}
```

### Response Schema
```json
{
  "results": [
    {
      "file_name": "string",
      "doc_type": "string",
      "chunk_count": "number",
      "status": "ingested | skipped | error",
      "detail": "string"
    }
  ],
  "total_chunks": "number"
}
```

### Example Request Payload
```json
{
  "files": ["Onboarding-BRD.pdf", "Payments-PRD.docx"],
  "doc_type": "BRD"
}
```

### Example Response Payload
```json
{
  "results": [
    {
      "file_name": "Onboarding-BRD.pdf",
      "doc_type": "BRD",
      "chunk_count": 18,
      "status": "ingested",
      "detail": ""
    }
  ],
  "total_chunks": 18
}
```

---

## 6) List Documents

### Endpoint
- **Method:** `GET`
- **Path:** `/api/v1/documents`
- **Description:** Returns all ingested document metadata.

### Params
- None

### Request Schema
- None

### Response Schema
```json
[
  {
    "id": "number",
    "file_name": "string",
    "doc_type": "string",
    "chunk_count": "number",
    "uploaded_at": "string (ISO datetime)"
  }
]
```

### Example Response Payload
```json
[
  {
    "id": 7,
    "file_name": "Onboarding-BRD.pdf",
    "doc_type": "BRD",
    "chunk_count": 18,
    "uploaded_at": "2026-03-31T10:22:44.340000Z"
  }
]
```

---

## 7) Get Document By ID

### Endpoint
- **Method:** `GET`
- **Path:** `/api/v1/documents/{doc_id}`
- **Description:** Returns detailed metadata for one document.

### Params
- **Path:** `doc_id` (integer, required)

### Request Schema
- None

### Response Schema
```json
{
  "id": "number",
  "file_name": "string",
  "doc_type": "string",
  "chunk_count": "number",
  "uploaded_at": "string (ISO datetime)",
  "file_hash": "string"
}
```

### Example Response Payload
```json
{
  "id": 7,
  "file_name": "Onboarding-BRD.pdf",
  "doc_type": "BRD",
  "chunk_count": 18,
  "uploaded_at": "2026-03-31T10:22:44.340000Z",
  "file_hash": "7b6a89b95f9e5f2b748f4d6f2f1944e9"
}
```

---

## 8) Delete Document By ID

### Endpoint
- **Method:** `DELETE`
- **Path:** `/api/v1/documents/{doc_id}`
- **Description:** Deletes a document from metadata, vector index, and BM25 index.

### Params
- **Path:** `doc_id` (integer, required)

### Request Schema
- None

### Response Schema
```json
{
  "id": "number",
  "file_name": "string",
  "deleted": "boolean"
}
```

### Example Response Payload
```json
{
  "id": 7,
  "file_name": "Onboarding-BRD.pdf",
  "deleted": true
}
```

---

## 9) Get Available Document Types

### Endpoint
- **Method:** `GET`
- **Path:** `/api/v1/doc-types`
- **Description:** Returns all document types currently available in metadata.

### Params
- None

### Request Schema
- None

### Response Schema
```json
{
  "doc_types": ["string", "..."]
}
```

### Example Response Payload
```json
{
  "doc_types": ["BRD", "PRD", "Test Plan"]
}
```

---

## 10) Create Conversation

### Endpoint
- **Method:** `POST`
- **Path:** `/api/v1/conversations`
- **Description:** Creates a new conversation thread ID.

### Params
- None

### Request Schema
- None

### Response Schema
```json
{
  "thread_id": "string"
}
```

### Example Response Payload
```json
{
  "thread_id": "2da4dd28-12c5-4f4b-a825-e8f2db0522ea"
}
```

---

## 11) List Conversations

### Endpoint
- **Method:** `GET`
- **Path:** `/api/v1/conversations`
- **Description:** Lists distinct conversation threads found in checkpoints.

### Params
- None

### Request Schema
- None

### Response Schema
```json
{
  "conversations": [
    {
      "thread_id": "string"
    }
  ]
}
```

### Example Response Payload
```json
{
  "conversations": [
    {
      "thread_id": "2da4dd28-12c5-4f4b-a825-e8f2db0522ea"
    },
    {
      "thread_id": "f6e9ef17-56dd-4bd7-a74d-6c2d2b8a4c77"
    }
  ]
}
```

---

## 12) Delete Conversation

### Endpoint
- **Method:** `DELETE`
- **Path:** `/api/v1/conversations/{thread_id}`
- **Description:** Deletes all stored checkpoint/chat data for a conversation thread.

### Params
- **Path:** `thread_id` (string UUID, required)

### Request Schema
- None

### Response Schema
```json
{
  "thread_id": "string",
  "deleted": "boolean"
}
```

### Example Response Payload
```json
{
  "thread_id": "2da4dd28-12c5-4f4b-a825-e8f2db0522ea",
  "deleted": true
}
```
