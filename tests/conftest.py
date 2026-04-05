"""Shared test fixtures."""

import os
import sys
import tempfile
from pathlib import Path

import pytest

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Set dummy env vars so that pydantic-settings doesn't fail during import
# and so that modules that check for API keys at import time don't crash.
os.environ.setdefault("GOOGLE_API_KEY", "test-google-key")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")


@pytest.fixture()
def tmp_storage(tmp_path):
    """Provide a temporary directory for session storage."""
    return str(tmp_path / "sessions")


@pytest.fixture()
def session_manager(tmp_storage):
    """Provide a SessionManager backed by a temp directory."""
    from utils.session_manager import SessionManager
    return SessionManager(storage_path=tmp_storage)


@pytest.fixture()
def sample_agent_config():
    """Provide a sample agent config dict."""
    return {
        "framework": "llamaindex",
        "mode": "api",
        "llm_provider": "gemini",
        "llm_model": "gemini-2.5-flash",
        "code_execution_enabled": True,
        "media_analysis_enabled": False,
        "img_generation_enabled": False,
        "img_editing_enabled": False,
    }
