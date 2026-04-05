"""Tests for the session manager."""

import json
from pathlib import Path

import pytest

from utils.session_manager import ChatSession, Message, SessionManager


class TestMessage:
    def test_roundtrip(self):
        msg = Message(role="user", content="hello", metadata={"key": "val"})
        d = msg.to_dict()
        restored = Message.from_dict(d)

        assert restored.role == "user"
        assert restored.content == "hello"
        assert restored.metadata == {"key": "val"}
        assert restored.timestamp == msg.timestamp

    def test_default_timestamp(self):
        msg = Message(role="assistant", content="hi")
        assert msg.timestamp  # should be auto-filled


class TestChatSession:
    def test_create_defaults(self):
        session = ChatSession()
        assert session.session_id
        assert session.title.startswith("Chat ")
        assert session.messages == []
        assert "created_at" in session.metadata

    def test_add_message(self):
        session = ChatSession()
        session.add_message("user", "test prompt")
        assert len(session.messages) == 1
        assert session.messages[0].role == "user"
        assert session.messages[0].content == "test prompt"

    def test_roundtrip(self, sample_agent_config):
        session = ChatSession(title="Test", agent_config=sample_agent_config)
        session.add_message("user", "q1")
        session.add_message("assistant", "a1", metadata={"trace_id": "t1"})

        d = session.to_dict()
        restored = ChatSession.from_dict(d)

        assert restored.session_id == session.session_id
        assert restored.title == "Test"
        assert len(restored.messages) == 2
        assert restored.agent_config == sample_agent_config


class TestSessionManager:
    def test_create_and_load(self, session_manager, sample_agent_config):
        session = session_manager.create_session(
            title="My Chat", agent_config=sample_agent_config
        )

        loaded = session_manager.load_session(session.session_id)
        assert loaded is not None
        assert loaded.title == "My Chat"
        assert loaded.agent_config == sample_agent_config

    def test_save_updates(self, session_manager):
        session = session_manager.create_session(title="Chat A")
        session.add_message("user", "hello")
        session_manager.save_session(session)

        loaded = session_manager.load_session(session.session_id)
        assert len(loaded.messages) == 1

    def test_list_sessions(self, session_manager):
        session_manager.create_session(title="Chat 1")
        session_manager.create_session(title="Chat 2")

        sessions = session_manager.list_sessions()
        assert len(sessions) == 2
        titles = {s["title"] for s in sessions}
        assert titles == {"Chat 1", "Chat 2"}

    def test_list_sessions_limit(self, session_manager):
        for i in range(5):
            session_manager.create_session(title=f"Chat {i}")

        sessions = session_manager.list_sessions(limit=3)
        assert len(sessions) == 3

    def test_delete_session(self, session_manager):
        session = session_manager.create_session(title="To Delete")
        sid = session.session_id

        result = session_manager.delete_session(sid, cleanup_vector_store=False)
        assert result is True

        loaded = session_manager.load_session(sid)
        assert loaded is None

        sessions = session_manager.list_sessions()
        assert all(s["session_id"] != sid for s in sessions)

    def test_load_nonexistent(self, session_manager):
        assert session_manager.load_session("does-not-exist") is None

    def test_persistence_across_instances(self, tmp_storage, sample_agent_config):
        mgr1 = SessionManager(storage_path=tmp_storage)
        session = mgr1.create_session(title="Persist", agent_config=sample_agent_config)
        session.add_message("user", "persisted")
        mgr1.save_session(session)

        mgr2 = SessionManager(storage_path=tmp_storage)
        loaded = mgr2.load_session(session.session_id)
        assert loaded is not None
        assert len(loaded.messages) == 1
        assert loaded.messages[0].content == "persisted"
