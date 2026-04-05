"""Centralized configuration with Pydantic validation.

All environment variables and application settings are defined here.
Import ``settings`` from this module instead of calling ``os.environ.get()``
directly in application code.
"""

from __future__ import annotations

import os
from enum import Enum
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Framework(str, Enum):
    LLAMAINDEX = "llamaindex"
    SMOLAGENTS = "smolagents"


class Mode(str, Enum):
    API = "api"
    LOCAL = "local"


class Provider(str, Enum):
    GEMINI = "gemini"
    OPENAI = "openai"
    OPENROUTER = "openrouter"


class LocalModelSuite(str, Enum):
    QWEN = "qwen"
    GEMMA4 = "gemma4"
    MINISTRAL = "ministral"
    GPT_OSS = "gpt-oss"
    DEVSTRAL = "devstral"


class RAGProvider(str, Enum):
    JINA = "jina"
    QWEN = "qwen"


# ---------------------------------------------------------------------------
# Typed agent configuration (replaces untyped dicts)
# ---------------------------------------------------------------------------

class AgentConfig(BaseModel):
    """Typed configuration for a chat agent session."""

    framework: Framework = Framework.LLAMAINDEX
    mode: Mode = Mode.API
    llm_provider: Provider = Provider.GEMINI
    llm_model: Optional[str] = None
    model_suite: str = "qwen"
    code_execution_enabled: bool = True
    use_specialized_code_model: bool = False
    use_main_model_for_code_agent: bool = False
    media_analysis_enabled: bool = False
    img_generation_enabled: bool = False
    img_editing_enabled: bool = False
    use_qwen_vl_for_images: bool = True
    qwen_vl_model_id: Optional[str] = None
    rag_provider: str = "jina"

    model_config = {"use_enum_values": True}

    @field_validator("mode")
    @classmethod
    def smolagents_requires_api(cls, v: str, info) -> str:
        data = info.data
        if data.get("framework") == "smolagents" and v != "api":
            raise ValueError("smolagents only supports API mode")
        return v


# ---------------------------------------------------------------------------
# Application settings (loaded from environment / .env file)
# ---------------------------------------------------------------------------

class AppSettings(BaseSettings):
    """Application-wide settings loaded from environment variables."""

    # --- Core API keys ---
    google_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    openrouter_api_key: Optional[str] = None

    # --- Model defaults ---
    use_api_mode: bool = False
    local_model_suite: LocalModelSuite = LocalModelSuite.QWEN
    rag_provider: RAGProvider = RAGProvider.JINA

    # --- HuggingFace ---
    huggingfacehub_api_token: Optional[str] = None

    # --- Langfuse observability ---
    langfuse_secret_key: Optional[str] = None
    langfuse_public_key: Optional[str] = None
    langfuse_host: str = "https://cloud.langfuse.com"

    # --- MCP server credentials ---
    github_personal_access_token: Optional[str] = None
    brave_api_key: Optional[str] = None
    slack_bot_token: Optional[str] = None
    postgres_connection_string: Optional[str] = None
    google_maps_api_key: Optional[str] = None

    # --- OpenAI transcription ---
    openai_transcribe_model: str = "gpt-4o-mini-transcribe"

    # --- Streamlit ---
    chat_sessions_dir: str = ".chat_sessions"
    chroma_db_dir: str = "./chroma_db"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    # -- Convenience helpers ------------------------------------------------

    @property
    def has_google_key(self) -> bool:
        return bool(self.google_api_key)

    @property
    def has_openai_key(self) -> bool:
        return bool(self.openai_api_key)

    @property
    def has_openrouter_key(self) -> bool:
        return bool(self.openrouter_api_key)

    @property
    def has_langfuse(self) -> bool:
        return bool(self.langfuse_secret_key and self.langfuse_public_key)

    @property
    def available_providers(self) -> List[str]:
        providers = []
        if self.has_google_key:
            providers.append("gemini")
        if self.has_openai_key:
            providers.append("openai")
        if self.has_openrouter_key:
            providers.append("openrouter")
        return providers

    def api_key_status(self) -> dict[str, bool]:
        """Return a mapping of service name to key-present boolean."""
        # Use environment variables as fallback so Streamlit in-app edits are reflected immediately.
        google_present = bool(self.google_api_key or os.environ.get("GOOGLE_API_KEY"))
        openai_present = bool(self.openai_api_key or os.environ.get("OPENAI_API_KEY"))
        openrouter_present = bool(self.openrouter_api_key or os.environ.get("OPENROUTER_API_KEY"))
        return {
            "Gemini": google_present,
            "OpenAI": openai_present,
            "OpenRouter": openrouter_present,
            "Langfuse": self.has_langfuse,
            "GitHub MCP": bool(self.github_personal_access_token),
            "Brave Search": bool(self.brave_api_key),
            "Slack": bool(self.slack_bot_token),
        }


# ---------------------------------------------------------------------------
# Singleton instance – import this everywhere
# ---------------------------------------------------------------------------

settings = AppSettings()
