"""Integration tests for the project layer API.

Requires a running Postgres instance and valid env vars.
Run with: pytest tests/test_projects_api.py -v
"""

from __future__ import annotations

import uuid

import pytest
from httpx import AsyncClient

from src.api.dependencies import AppComponents

pytestmark = pytest.mark.asyncio(loop_scope="session")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _create_project(
    client: AsyncClient, name: str, description: str = ""
) -> dict:
    resp = await client.post(
        "/api/v1/projects",
        json={"name": name, "description": description},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


# ---------------------------------------------------------------------------
# Project CRUD
# ---------------------------------------------------------------------------


class TestProjectCRUD:
    async def test_create_project(self, client: AsyncClient):
        data = await _create_project(client, f"Test-{uuid.uuid4().hex[:8]}")
        assert "id" in data
        assert "name" in data
        assert "created_at" in data
        assert "updated_at" in data

    async def test_create_duplicate_name_returns_409(self, client: AsyncClient):
        name = f"Dup-{uuid.uuid4().hex[:8]}"
        await _create_project(client, name)
        resp = await client.post("/api/v1/projects", json={"name": name})
        assert resp.status_code == 409

    async def test_create_project_name_trimmed(self, client: AsyncClient):
        name = f"  Trimmed-{uuid.uuid4().hex[:8]}  "
        data = await _create_project(client, name)
        assert data["name"] == name.strip()

    async def test_create_project_name_too_long(self, client: AsyncClient):
        resp = await client.post("/api/v1/projects", json={"name": "x" * 101})
        assert resp.status_code == 422

    async def test_list_projects(self, client: AsyncClient):
        name = f"List-{uuid.uuid4().hex[:8]}"
        await _create_project(client, name)
        resp = await client.get("/api/v1/projects")
        assert resp.status_code == 200
        names = [p["name"] for p in resp.json()["projects"]]
        assert name in names

    async def test_delete_project_returns_204(self, client: AsyncClient):
        data = await _create_project(client, f"Del-{uuid.uuid4().hex[:8]}")
        resp = await client.delete(f"/api/v1/projects/{data['id']}")
        assert resp.status_code == 204

    async def test_delete_already_deleted_returns_410(self, client: AsyncClient):
        data = await _create_project(client, f"Del2-{uuid.uuid4().hex[:8]}")
        await client.delete(f"/api/v1/projects/{data['id']}")
        resp = await client.delete(f"/api/v1/projects/{data['id']}")
        assert resp.status_code == 410

    async def test_delete_nonexistent_returns_404(self, client: AsyncClient):
        fake_id = str(uuid.uuid4())
        resp = await client.delete(f"/api/v1/projects/{fake_id}")
        assert resp.status_code == 404

    async def test_deleted_project_hidden_from_list(self, client: AsyncClient):
        data = await _create_project(client, f"Hidden-{uuid.uuid4().hex[:8]}")
        await client.delete(f"/api/v1/projects/{data['id']}")
        resp = await client.get("/api/v1/projects")
        ids = [p["id"] for p in resp.json()["projects"]]
        assert data["id"] not in ids

    async def test_name_reuse_after_delete(self, client: AsyncClient):
        name = f"Reuse-{uuid.uuid4().hex[:8]}"
        data = await _create_project(client, name)
        await client.delete(f"/api/v1/projects/{data['id']}")
        data2 = await _create_project(client, name)
        assert data2["name"] == name


# ---------------------------------------------------------------------------
# Project-scoped documents
# ---------------------------------------------------------------------------


class TestProjectScopedDocuments:
    async def test_list_documents_requires_project_id(self, client: AsyncClient):
        resp = await client.get("/api/v1/documents")
        assert resp.status_code == 422

    async def test_list_documents_with_valid_project(self, client: AsyncClient):
        data = await _create_project(client, f"DocList-{uuid.uuid4().hex[:8]}")
        resp = await client.get("/api/v1/documents", params={"project_id": data["id"]})
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    async def test_get_document_cross_project_returns_404(self, client: AsyncClient):
        p1 = await _create_project(client, f"Cross1-{uuid.uuid4().hex[:8]}")
        p2 = await _create_project(client, f"Cross2-{uuid.uuid4().hex[:8]}")  # noqa: F841
        resp = await client.get(
            "/api/v1/documents/99999",
            params={"project_id": p1["id"]},
        )
        assert resp.status_code == 404

    async def test_doc_types_requires_project_id(self, client: AsyncClient):
        resp = await client.get("/api/v1/doc-types")
        assert resp.status_code == 422

    async def test_doc_types_with_valid_project(self, client: AsyncClient):
        data = await _create_project(client, f"DocType-{uuid.uuid4().hex[:8]}")
        resp = await client.get("/api/v1/doc-types", params={"project_id": data["id"]})
        assert resp.status_code == 200

    async def test_deleted_project_blocks_doc_list(self, client: AsyncClient):
        data = await _create_project(client, f"Blocked-{uuid.uuid4().hex[:8]}")
        await client.delete(f"/api/v1/projects/{data['id']}")
        resp = await client.get("/api/v1/documents", params={"project_id": data["id"]})
        assert resp.status_code == 410


# ---------------------------------------------------------------------------
# Chat project scoping
# ---------------------------------------------------------------------------


class TestChatProjectScoping:
    async def test_chat_requires_project_id(self, client: AsyncClient):
        resp = await client.post(
            "/api/v1/chat",
            json={"question": "hello"},
        )
        assert resp.status_code == 422

    async def test_chat_rejects_legacy_filters(self, client: AsyncClient):
        data = await _create_project(client, f"ChatFilter-{uuid.uuid4().hex[:8]}")
        resp = await client.post(
            "/api/v1/chat",
            json={
                "question": "hello",
                "project_id": data["id"],
                "doc_type_filter": "PRD",
            },
        )
        assert resp.status_code == 422

    async def test_chat_nonexistent_project_returns_404(self, client: AsyncClient):
        resp = await client.post(
            "/api/v1/chat",
            json={
                "question": "hello",
                "project_id": str(uuid.uuid4()),
            },
        )
        assert resp.status_code == 404

    async def test_chat_deleted_project_returns_410(self, client: AsyncClient):
        data = await _create_project(client, f"ChatDel-{uuid.uuid4().hex[:8]}")
        await client.delete(f"/api/v1/projects/{data['id']}")
        resp = await client.post(
            "/api/v1/chat",
            json={
                "question": "hello",
                "project_id": data["id"],
            },
        )
        assert resp.status_code == 410


# ---------------------------------------------------------------------------
# Thread-project binding
# ---------------------------------------------------------------------------


class TestThreadProjectBinding:
    async def test_create_conversation_requires_project(self, client: AsyncClient):
        resp = await client.post("/api/v1/conversations", json={})
        assert resp.status_code == 422

    async def test_create_conversation_returns_thread_id(self, client: AsyncClient):
        data = await _create_project(client, f"Conv-{uuid.uuid4().hex[:8]}")
        resp = await client.post(
            "/api/v1/conversations",
            json={"project_id": data["id"]},
        )
        assert resp.status_code == 201
        body = resp.json()
        assert "thread_id" in body
        assert body["title"] == "New chat"

    async def test_list_conversations_requires_project(self, client: AsyncClient):
        resp = await client.get("/api/v1/conversations")
        assert resp.status_code == 422

    async def test_list_conversations_scoped(self, client: AsyncClient):
        p1 = await _create_project(client, f"ConvScope1-{uuid.uuid4().hex[:8]}")
        p2 = await _create_project(client, f"ConvScope2-{uuid.uuid4().hex[:8]}")

        r1 = await client.post("/api/v1/conversations", json={"project_id": p1["id"]})
        tid1 = r1.json()["thread_id"]

        r2 = await client.post("/api/v1/conversations", json={"project_id": p2["id"]})
        tid2 = r2.json()["thread_id"]

        resp1 = await client.get(
            "/api/v1/conversations", params={"project_id": p1["id"]}
        )
        data1 = resp1.json()["conversations"]
        tids1 = [c["thread_id"] for c in data1]
        assert tid1 in tids1
        assert tid2 not in tids1
        by_id = {c["thread_id"]: c for c in data1}
        assert by_id[tid1]["title"] == "New chat"

    async def test_delete_conversation_cross_project_returns_404(
        self, client: AsyncClient
    ):
        p1 = await _create_project(client, f"ConvDel1-{uuid.uuid4().hex[:8]}")
        p2 = await _create_project(client, f"ConvDel2-{uuid.uuid4().hex[:8]}")

        r = await client.post("/api/v1/conversations", json={"project_id": p1["id"]})
        tid = r.json()["thread_id"]

        resp = await client.delete(
            f"/api/v1/conversations/{tid}",
            params={"project_id": p2["id"]},
        )
        assert resp.status_code == 404

    async def test_get_conversation_requires_project(self, client: AsyncClient):
        resp = await client.get(f"/api/v1/conversations/{uuid.uuid4()}")
        assert resp.status_code == 422

    async def test_get_conversation_unknown_thread_404(self, client: AsyncClient):
        p = await _create_project(client, f"ConvGet-{uuid.uuid4().hex[:8]}")
        resp = await client.get(
            f"/api/v1/conversations/{uuid.uuid4()}",
            params={"project_id": p["id"]},
        )
        assert resp.status_code == 404

    async def test_get_conversation_cross_project_returns_404(
        self, client: AsyncClient
    ):
        p1 = await _create_project(client, f"ConvGet1-{uuid.uuid4().hex[:8]}")
        p2 = await _create_project(client, f"ConvGet2-{uuid.uuid4().hex[:8]}")
        r = await client.post("/api/v1/conversations", json={"project_id": p1["id"]})
        tid = r.json()["thread_id"]
        resp = await client.get(
            f"/api/v1/conversations/{tid}",
            params={"project_id": p2["id"]},
        )
        assert resp.status_code == 404

    async def test_get_conversation_empty_messages(self, client: AsyncClient):
        p = await _create_project(client, f"ConvEmpty-{uuid.uuid4().hex[:8]}")
        r = await client.post("/api/v1/conversations", json={"project_id": p["id"]})
        tid = r.json()["thread_id"]
        resp = await client.get(
            f"/api/v1/conversations/{tid}",
            params={"project_id": p["id"]},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["thread_id"] == tid
        assert body["title"] == "New chat"
        assert body["messages"] == []

    async def test_get_conversation_descending_order(
        self, client: AsyncClient, components: AppComponents
    ):
        p = await _create_project(client, f"ConvOrder-{uuid.uuid4().hex[:8]}")
        r = await client.post("/api/v1/conversations", json={"project_id": p["id"]})
        tid = r.json()["thread_id"]
        await components.chat_store.append_messages(
            tid, "first question", "first reply"
        )
        await components.chat_store.append_messages(
            tid, "second question", "second reply"
        )
        resp = await client.get(
            f"/api/v1/conversations/{tid}",
            params={"project_id": p["id"]},
        )
        assert resp.status_code == 200
        detail = resp.json()
        msgs = detail["messages"]
        assert len(msgs) == 4
        roles = [m["role"] for m in msgs]
        assert roles == ["ai", "human", "ai", "human"]
        assert "created_at" in msgs[0]
        assert msgs[0]["content"] == "second reply"
        assert msgs[1]["content"] == "second question"
        assert detail["title"] == "first question"

        listed = await client.get(
            "/api/v1/conversations", params={"project_id": p["id"]}
        )
        assert listed.status_code == 200
        conv = next(c for c in listed.json()["conversations"] if c["thread_id"] == tid)
        assert conv["title"] == detail["title"]
