"""Tests for the FastAPI backend."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_storage):
    """Create a test client with a temporary session store."""
    # Patch the module-level session manager before importing
    with patch("api.app.settings") as mock_settings:
        mock_settings.chat_sessions_dir = tmp_storage
        mock_settings.available_providers = ["gemini"]

        # Re-create the session manager with the temp path
        from api.app import app, _session_manager
        from utils.session_manager import SessionManager

        import api.app as api_module
        api_module._session_manager = SessionManager(storage_path=tmp_storage)
        api_module._agents = {}

        yield TestClient(app)


class TestHealthEndpoint:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"


class TestSessionEndpoints:
    def test_create_session(self, client):
        resp = client.post(
            "/sessions",
            json={
                "title": "Test Chat",
                "agent_config": {
                    "framework": "llamaindex",
                    "mode": "api",
                    "llm_provider": "gemini",
                    "llm_model": "gemini-2.5-flash",
                },
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert "session_id" in data
        assert data["title"] == "Test Chat"

    def test_list_sessions(self, client):
        # Create two sessions
        for i in range(2):
            client.post(
                "/sessions",
                json={
                    "title": f"Chat {i}",
                    "agent_config": {
                        "framework": "llamaindex",
                        "mode": "api",
                        "llm_provider": "gemini",
                    },
                },
            )

        resp = client.get("/sessions")
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    def test_get_messages_empty(self, client):
        create_resp = client.post(
            "/sessions",
            json={
                "title": "Empty Chat",
                "agent_config": {
                    "framework": "llamaindex",
                    "mode": "api",
                    "llm_provider": "gemini",
                },
            },
        )
        sid = create_resp.json()["session_id"]

        resp = client.get(f"/sessions/{sid}/messages")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_get_messages_not_found(self, client):
        resp = client.get("/sessions/nonexistent/messages")
        assert resp.status_code == 404

    def test_delete_session(self, client):
        create_resp = client.post(
            "/sessions",
            json={
                "title": "To Delete",
                "agent_config": {
                    "framework": "llamaindex",
                    "mode": "api",
                    "llm_provider": "gemini",
                },
            },
        )
        sid = create_resp.json()["session_id"]

        resp = client.delete(f"/sessions/{sid}")
        assert resp.status_code == 204

        resp = client.get(f"/sessions/{sid}/messages")
        assert resp.status_code == 404

    def test_create_smolagents_local_rejected(self, client):
        resp = client.post(
            "/sessions",
            json={
                "agent_config": {
                    "framework": "smolagents",
                    "mode": "local",
                    "llm_provider": "gemini",
                },
            },
        )
        assert resp.status_code == 422  # Pydantic validation error
