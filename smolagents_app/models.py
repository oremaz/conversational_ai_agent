"""Model initialization for smolagents."""

import logging
import os
from typing import Any, Optional

from smolagents import OpenAIServerModel

logger = logging.getLogger(__name__)

_FORMAT_PROVIDER = None
_FORMAT_MODEL_NAME = None

def get_format_config() -> tuple[str, Optional[str]]:
    """Return provider/model overrides for formatting."""
    return _FORMAT_PROVIDER or "gemini", _FORMAT_MODEL_NAME

def initialize_llm_model(
    provider: str = "gemini",
    model_name: Optional[str] = None,
    temperature: float = 0.7,
    **kwargs,
) -> Any:
    """Initialize an LLM model for smolagents."""
    config = {
        "gemini": {
            "env_var": "GOOGLE_API_KEY",
            "api_base": "https://generativelanguage.googleapis.com/v1beta/openai/",
        },
        "openai": {
            "env_var": "OPENAI_API_KEY",
            "api_base": None,
        },
        "openrouter": {
            "env_var": "OPENROUTER_API_KEY",
            "api_base": "https://openrouter.ai/api/v1",
        },
    }

    if provider not in config:
        raise ValueError(f"Unknown provider: {provider}. Choose from: {', '.join(config.keys())}")

    if not model_name:
        raise ValueError(f"model_name is required for provider '{provider}'")

    api_key = os.environ.get(config[provider]["env_var"])
    if not api_key:
        raise ValueError(f"{config[provider]['env_var']} not found in environment")

    model_kwargs = dict(kwargs)
    api_base = model_kwargs.pop("api_base", config[provider]["api_base"])
    model_kwargs.pop("top_p", None)

    client_kwargs = model_kwargs.pop("client_kwargs", {}) or {}
    client_kwargs.setdefault("timeout", 60)
    client_kwargs.setdefault("max_retries", 2)

    # OpenRouter recommends sending HTTP-Referer / X-Title for analytics.
    # Pass them as default_headers, which the openai SDK forwards on every request.
    if provider == "openrouter":
        default_headers = client_kwargs.setdefault("default_headers", {})
        default_headers.setdefault("HTTP-Referer", model_kwargs.pop("site_url", ""))
        default_headers.setdefault("X-Title", model_kwargs.pop("site_name", ""))
        # Remove keys with empty values to keep headers clean
        client_kwargs["default_headers"] = {
            k: v for k, v in default_headers.items() if v
        } or None

    init_args = {
        "model_id": model_name,
        "api_key": api_key,
        "client_kwargs": client_kwargs,
        "temperature": temperature,
        **model_kwargs,
    }

    if api_base:
        init_args["api_base"] = api_base

    model = OpenAIServerModel(**init_args)

    global _FORMAT_PROVIDER, _FORMAT_MODEL_NAME
    _FORMAT_PROVIDER = provider
    _FORMAT_MODEL_NAME = model_name

    return model
