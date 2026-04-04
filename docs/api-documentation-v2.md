# PMA-RAG API Documentation v2 (Project-Scoped)

All endpoints are under the `/api/v1` prefix. Every document, chat, and conversation operation requires a valid `project_id`.

---

## Projects

### POST /api/v1/projects

Create a new project.

**Request body:**

```json
{
  "name": "My Project",
  "description": "Optional description"
}
```

| Field         | Type   | Required | Constraints                                                         |
| ------------- | ------ | -------- | ------------------------------------------------------------------- |
| `name`        | string | yes      | 1–100 chars, trimmed, unique among active projects (case-sensitive) |
| `description` | string | no       | max 1000 chars, defaults to `""`                                    |

**Response (201):**

```json
{
  "id": "uuid",
  "name": "My Project",
  "description": "",
  "created_at": "2026-04-02T12:00:00Z",
  "updated_at": "2026-04-02T12:00:00Z"
}
```

**Errors:**

| Code | Condition                                     |
| ---- | --------------------------------------------- |
| 409  | Project name already exists                   |
| 422  | Validation error (name too long, empty, etc.) |

---

### GET /api/v1/projects

List all active (non-deleted) projects, sorted by `updated_at` descending.

**Response (200):**

```json
{
  "projects": [
    {
      "id": "uuid",
      "name": "My Project",
      "description": "",
      "created_at": "...",
      "updated_at": "..."
    }
  ]
}
```

---

### DELETE /api/v1/projects/{project_id}

Soft-delete a project. Hard-deletes all associated documents, chunks, threads, chat artifacts, LangGraph checkpoints, and the Chroma vector collection.

**Response:** `204 No Content`

**Errors:**

| Code | Condition               |
| ---- | ----------------------- |
| 404  | Project not found       |
| 410  | Project already deleted |
| 422  | Malformed UUID          |

---

## Documents

All document endpoints require `project_id`.

### POST /api/v1/documents/upload

Upload documents to a project.

**Request:** `multipart/form-data`

| Field        | Type          | Required |
| ------------ | ------------- | -------- |
| `files`      | file(s)       | yes      |
| `doc_type`   | string        | yes      |
| `project_id` | UUID (string) | yes      |

**Response (200):**

```json
{
  "results": [
    {
      "file_name": "spec.pdf",
      "doc_type": "PRD",
      "chunk_count": 42,
      "status": "ingested",
      "detail": ""
    }
  ],
  "total_chunks": 42
}
```

---

### GET /api/v1/documents?project_id={uuid}

List documents in a project. Optional `doc_type` query parameter for filtering.

**Query parameters:**

| Param        | Type   | Required |
| ------------ | ------ | -------- |
| `project_id` | UUID   | yes      |
| `doc_type`   | string | no       |

**Response (200):** Array of document objects.

---

### GET /api/v1/documents/{doc_id}?project_id={uuid}

Get document details. Returns `404` if document does not exist in the specified project.

---

### DELETE /api/v1/documents/{doc_id}?project_id={uuid}

Delete a document and its chunks from the project.

**Response (200):**

```json
{
  "id": 1,
  "file_name": "spec.pdf",
  "deleted": true
}
```

---

### GET /api/v1/doc-types?project_id={uuid}

List distinct document types within a project.

---

## Chat

### POST /api/v1/chat

Send a question scoped to a project.

**Request body:**

```json
{
  "question": "What are the key requirements?",
  "project_id": "uuid",
  "thread_id": "uuid (optional)"
}
```

| Field        | Type   | Required | Notes                                                   |
| ------------ | ------ | -------- | ------------------------------------------------------- |
| `question`   | string | yes      | min 1 char                                              |
| `project_id` | UUID   | yes      |                                                         |
| `thread_id`  | UUID   | no       | Omit to auto-create. Unknown IDs generate a new thread. |

**Extra fields are forbidden.** Sending legacy fields (`doc_type_filter`, `source_file_filter`, `section_filter`) returns `422`.

**Response (200):**

```json
{
  "answer": "...",
  "confidence": "High",
  "citations": [...],
  "validation_passed": true,
  "validation_reason": "",
  "thread_id": "uuid",
  "run_id": "uuid",
  "search_documents": true,
  "response_style": "default"
}
```

**Thread binding rules:**

- Threads are permanently bound to one project.
- If `thread_id` is bound to a different project, the API returns `409 Conflict`.
- If `thread_id` is unknown, a new thread UUID is generated and returned.

---

### POST /api/v1/chat/stream

Same request/response contract as `/chat` but returns Server-Sent Events (SSE).

**Events:** `thread_id`, `intent`, `confidence`, `token`, `done`.

---

### POST /api/v1/feedback

Submit feedback for a chat response. Unchanged from v1.

---

## Conversations

### POST /api/v1/conversations

Create a new conversation thread bound to a project.

**Request body:**

```json
{
  "project_id": "uuid"
}
```

**Response (201):**

```json
{
  "thread_id": "uuid"
}
```

---

### GET /api/v1/conversations?project_id={uuid}

List threads for a project, sorted newest first.

**Response (200):**

```json
{
  "conversations": [{ "thread_id": "uuid" }]
}
```

---

### DELETE /api/v1/conversations/{thread_id}?project_id={uuid}

Delete a conversation thread. Requires `project_id` for safety scoping.

**Response:** `204 No Content`

**Errors:**

| Code | Condition                                      |
| ---- | ---------------------------------------------- |
| 404  | Thread not found or belongs to another project |
| 422  | Invalid UUID format                            |

---

## Error Semantics

| Code | Meaning                                                   |
| ---- | --------------------------------------------------------- |
| 404  | Resource not found (also hides cross-project existence)   |
| 409  | Conflict (duplicate name, thread-project mismatch)        |
| 410  | Resource has been deleted (soft-deleted project)          |
| 422  | Validation error (malformed UUID, forbidden fields, etc.) |

---

## Health Check

### GET /health

Returns `{"status": "ok"}`. Unchanged.

---

## Reset Script

A destructive reset script is available at `scripts/reset_all.py`:

```bash
python -m scripts.reset_all
```

This wipes all projects, documents, chunks, threads, chat store tables, LangGraph checkpoints, and all Chroma collections. No confirmation guard.
