"""Framework-agnostic model provider interface.

This module defines a thin abstraction layer that model implementations can
use instead of directly coupling to LlamaIndex's ``CustomLLM``.  The concrete
LlamaIndex wrappers can delegate to providers that implement this interface,
making the underlying models usable outside of LlamaIndex (e.g. in smolagents,
evaluation scripts, or tests).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterator, Optional


class ModelProvider(ABC):
    """Minimal interface for a text-generation model.

    Implementations handle model loading, inference, and resource management.
    Framework-specific wrappers (e.g. LlamaIndex CustomLLM) delegate to a
    ``ModelProvider`` so that the same model can be reused across frameworks.
    """

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate a completion for the given prompt."""

    def stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        """Streaming variant.  Default falls back to ``generate``."""
        yield self.generate(prompt, **kwargs)

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the canonical model identifier."""

    @property
    def is_loaded(self) -> bool:
        """Return True if the model is ready for inference."""
        return True

    def unload(self) -> None:
        """Release model resources (GPU memory, handles, etc.)."""


class EmbeddingProvider(ABC):
    """Minimal interface for an embedding model."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return embeddings for a batch of texts."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the canonical model identifier."""

    @property
    @abstractmethod
    def embed_dim(self) -> int:
        """Return the embedding dimensionality."""
