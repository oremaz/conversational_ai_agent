from typing import Any, Dict, List, Optional
import logging
import threading

import numpy as np
import torch
from llama_index.core.embeddings import BaseEmbedding
from pydantic import Field, PrivateAttr
from transformers import AutoModel

_logger = logging.getLogger(__name__)


class Qwen3VLEmbeddings(BaseEmbedding):
    """Qwen3-VL embedder wrapper for local retrieval."""

    model_name: str = Field(default="Qwen/Qwen3-VL-Embedding-2B")
    torch_dtype: Optional[Any] = Field(default=None)
    attn_implementation: Optional[str] = Field(default=None)

    _model = PrivateAttr(default=None)
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
        return "qwen3_vl_embeddings"

    def _ensure_model(self):
        if getattr(self, "_loaded", False):
            return
        lock = getattr(self, "_model_lock", None)
        if lock is not None:
            lock.acquire()
        try:
            if getattr(self, "_loaded", False):
                return
            kwargs: Dict[str, Any] = {"trust_remote_code": True}
            if self.torch_dtype is not None:
                kwargs["torch_dtype"] = self.torch_dtype
            if self.attn_implementation is not None:
                kwargs["attn_implementation"] = self.attn_implementation
            self._model = AutoModel.from_pretrained(self.model_name, **kwargs)
            if hasattr(self._model, "eval"):
                self._model.eval()
            self._loaded = True
        finally:
            if lock is not None:
                try:
                    lock.release()
                except Exception:
                    pass

    def _to_list_vector(self, arr: Any) -> List[float]:
        if isinstance(arr, torch.Tensor):
            arr = arr.detach().cpu().numpy()
        if np is not None and isinstance(arr, np.ndarray):
            if arr.ndim == 2:
                arr = arr[0]
            return [float(x) for x in arr.tolist()]
        if isinstance(arr, list):
            if not arr:
                return []
            first = arr[0]
            if isinstance(first, (list, tuple, torch.Tensor)):
                return self._to_list_vector(first)
            return [float(x) for x in arr]
        try:
            return [float(arr)]
        except Exception:
            return [float(str(arr))]

    def _embed_inputs(self, inputs: List[Dict[str, Any]]) -> List[List[float]]:
        self._ensure_model()
        if hasattr(self._model, "process"):
            embeddings = self._model.process(inputs)
        elif hasattr(self._model, "encode"):
            embeddings = self._model.encode(inputs)
        else:
            raise RuntimeError("Qwen3-VL embedder backend does not expose a process/encode method.")
        results: List[List[float]] = []
        if isinstance(embeddings, list):
            for item in embeddings:
                results.append(self._to_list_vector(item))
        else:
            results.append(self._to_list_vector(embeddings))
        return results

    def _get_query_embedding(self, query: str, image_path: Optional[str] = None) -> List[float]:
        payload = {"text": query}
        if image_path:
            payload["image"] = image_path
        return self._embed_inputs([payload])[0]

    def _get_text_embedding(self, text: str, image_path: Optional[str] = None) -> List[float]:
        payload = {"text": text}
        if image_path:
            payload["image"] = image_path
        return self._embed_inputs([payload])[0]

    def _get_text_embeddings(
        self,
        texts: List[str],
        image_paths: Optional[List[Optional[str]]] = None,
    ) -> List[List[float]]:
        image_paths = image_paths or [None] * len(texts)
        inputs = []
        for text, image_path in zip(texts, image_paths):
            payload = {"text": text}
            if image_path:
                payload["image"] = image_path
            inputs.append(payload)
        return self._embed_inputs(inputs)

    async def _aget_query_embedding(self, query: str, image_path: Optional[str] = None) -> List[float]:
        return self._get_query_embedding(query, image_path)

    async def _aget_text_embedding(self, text: str, image_path: Optional[str] = None) -> List[float]:
        return self._get_text_embedding(text, image_path)
