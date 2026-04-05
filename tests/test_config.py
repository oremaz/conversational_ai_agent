"""Tests for the centralized configuration."""

import os

import pytest
from pydantic import ValidationError

from config import AgentConfig, AppSettings, Framework, Mode, Provider


class TestAgentConfig:
    def test_defaults(self):
        cfg = AgentConfig()
        assert cfg.framework == "llamaindex"
        assert cfg.mode == "api"
        assert cfg.code_execution_enabled is True

    def test_valid_smolagents_api(self):
        cfg = AgentConfig(framework="smolagents", mode="api", llm_provider="gemini")
        assert cfg.framework == "smolagents"

    def test_smolagents_local_rejected(self):
        with pytest.raises(ValidationError, match="smolagents only supports API mode"):
            AgentConfig(framework="smolagents", mode="local")

    def test_roundtrip_dict(self):
        cfg = AgentConfig(
            framework="llamaindex",
            mode="local",
            llm_model="Qwen/Qwen3.5-35B-A3B",
            model_suite="qwen",
            rag_provider="jina",
        )
        d = cfg.model_dump()
        restored = AgentConfig(**d)
        assert restored == cfg

    def test_enum_values_serialized(self):
        cfg = AgentConfig(framework=Framework.LLAMAINDEX, mode=Mode.API)
        d = cfg.model_dump()
        assert d["framework"] == "llamaindex"
        assert d["mode"] == "api"


class TestAppSettings:
    def test_loads_from_env(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
        monkeypatch.setenv("USE_API_MODE", "true")
        monkeypatch.setenv("LOCAL_MODEL_SUITE", "ministral")

        s = AppSettings()
        assert s.google_api_key == "test-key"
        assert s.use_api_mode is True
        assert s.local_model_suite.value == "ministral"

    def test_loads_gemma4_local_suite(self, monkeypatch):
        monkeypatch.setenv("LOCAL_MODEL_SUITE", "gemma4")
        s = AppSettings()
        assert s.local_model_suite.value == "gemma4"

    def test_has_google_key(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "k")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        s = AppSettings()
        assert s.has_google_key is True
        assert s.has_openai_key is False
        assert "gemini" in s.available_providers
        assert "openai" not in s.available_providers

    def test_api_key_status(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "k")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)

        s = AppSettings()
        status = s.api_key_status()
        assert status["Gemini"] is True
        assert status["OpenAI"] is False
        assert status["Langfuse"] is False

    def test_defaults(self, monkeypatch):
        monkeypatch.delenv("USE_API_MODE", raising=False)
        monkeypatch.delenv("LOCAL_MODEL_SUITE", raising=False)
        monkeypatch.delenv("RAG_PROVIDER", raising=False)

        s = AppSettings()
        assert s.use_api_mode is False
        assert s.local_model_suite.value == "qwen"
        assert s.rag_provider.value == "jina"
