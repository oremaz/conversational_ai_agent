"""Gemini embedding wrapper using google.genai (natively multimodal)."""

from typing import Any, List, Optional
import logging
import os
import threading

from llama_index.core.embeddings import BaseEmbedding
from pydantic import Field, PrivateAttr

_logger = logging.getLogger(__name__)


class GeminiEmbeddings(BaseEmbedding):
    """Lightweight API wrapper for gemini-embedding-2-preview."""

    model_name: str = Field(default="gemini-embedding-2-preview")
    api_key: Optional[str] = Field(default=None)

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
        return "gemini_embeddings"

    def _ensure_client(self):
        if getattr(self, "_loaded", False):
            return
        lock = getattr(self, "_model_lock", None)
        if lock is not None:
            lock.acquire()
        try:
            if getattr(self, "_loaded", False):
                return
            from google import genai

            key = self.api_key or os.environ.get("GOOGLE_API_KEY")
            self._client = genai.Client(api_key=key) if key else genai.Client()
            self._loaded = True
            _logger.info("Gemini embedding client initialized: %s", self.model_name)
        finally:
            if lock is not None:
                try:
                    lock.release()
                except Exception:
                    pass

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        self._ensure_client()
        result = self._client.models.embed_content(
            model=self.model_name,
            contents=texts,
        )
        return [list(e.values) for e in result.embeddings]

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
