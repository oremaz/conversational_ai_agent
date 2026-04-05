"""OpenRouter embedding wrapper (OpenAI-compatible API)."""

from typing import Any, List, Optional
import logging
import os
import threading

from llama_index.core.embeddings import BaseEmbedding
from pydantic import Field, PrivateAttr

_logger = logging.getLogger(__name__)

class OpenRouterEmbeddings(BaseEmbedding):
    """Lightweight API wrapper for OpenRouter embeddings (OpenAI-compatible endpoint)."""

    model_name: str = Field(default="nvidia/llama-nemotron-embed-vl-1b-v2:free")
    api_key: Optional[str] = Field(default=None)
    base_url: str = Field(default="https://openrouter.ai/api/v1")

    _client = PrivateAttr(default=None)
    _loaded = PrivateAttr(default=False)
    _model_lock = PrivateAttr(default=None)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        try:
            self._model_lock = threading.Lock()
        except Exception:
            self._model_lock = None
        self._loaded = False

    @classmethod
    def class_name(cls) -> str:
        return "openrouter_embeddings"

    def _ensure_client(self):
        if getattr(self, "_loaded", False):
            return
        lock = getattr(self, "_model_lock", None)
        if lock is not None:
            lock.acquire()
        try:
            if getattr(self, "_loaded", False):
                return
            from openai import OpenAI

            key = self.api_key or os.environ.get("OPENROUTER_API_KEY")
            if not key:
                raise ValueError("OPENROUTER_API_KEY not found in environment")
            self._client = OpenAI(
                api_key=key,
                base_url=self.base_url,
            )
            self._loaded = True
            _logger.info("OpenRouter embedding client initialized: %s", self.model_name)
        finally:
            if lock is not None:
                try:
                    lock.release()
                except Exception:
                    pass

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        self._ensure_client()
        result = self._client.embeddings.create(
            model=self.model_name,
            input=texts,
            encoding_format="float",
        )
        return [item.embedding for item in result.data]

    def _get_query_embedding(self, query: str, image_path: Optional[str] = None) -> List[float]:
        return self._embed_texts([query])[0]

    def _get_text_embedding(self, text: str, image_path: Optional[str] = None) -> List[float]:
        return self._embed_texts([text])[0]

    def _get_text_embeddings(
        self,
        texts: List[str],
        image_paths: Optional[List[Optional[str]]] = None,
    ) -> List[List[float]]:
        return self._embed_texts(texts)

    async def _aget_query_embedding(self, query: str, image_path: Optional[str] = None) -> List[float]:
        return self._get_query_embedding(query, image_path)

    async def _aget_text_embedding(self, text: str, image_path: Optional[str] = None) -> List[float]:
        return self._get_text_embedding(text, image_path)
