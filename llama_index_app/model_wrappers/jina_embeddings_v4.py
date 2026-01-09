from typing import Any, List, Optional
import logging
import os
import threading

import numpy as np
from PIL import Image
import torch
from transformers import AutoModel, AutoProcessor, BitsAndBytesConfig
from llama_index.core.embeddings import BaseEmbedding
from pydantic import Field, PrivateAttr

_logger = logging.getLogger(__name__)


class JinaEmbeddingsV4(BaseEmbedding):
    """Memory-constrained wrapper for jinaai/jina-embeddings-v4 (GPU only)."""

    model_name: str = Field(default="jinaai/jina-embeddings-v4")

    max_length_query: int = Field(default=512)
    max_length_passage: int = Field(default=768)
    batch_size: int = Field(default=1)
    truncate_dim: Optional[int] = Field(default=None)
    offload_folder: str = Field(default="./offload_jina_v4")

    _model = PrivateAttr(default=None)
    _processor = PrivateAttr(default=None)
    _loaded = PrivateAttr(default=False)
    _model_lock = PrivateAttr(default=None)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        try:
            self._model_lock = threading.Lock()
        except Exception:
            self._model_lock = None
        self._loaded = False

    def _ensure_model(self):
        if getattr(self, "_loaded", False):
            return
        lock = getattr(self, "_model_lock", None)
        if lock is not None:
            lock.acquire()
        try:
            if getattr(self, "_loaded", False):
                return

            if not torch.cuda.is_available():
                raise RuntimeError("Jina embeddings require CUDA; CPU mode is not supported.")

            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
            os.makedirs(self.offload_folder, exist_ok=True)

            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )

            self._model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                device_map="cuda:0",
                offload_folder=self.offload_folder,
                torch_dtype=torch.float16,
                quantization_config=bnb_cfg,
            ).eval()

            try:
                self._processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
            except Exception:
                self._processor = None

            self._loaded = True
        finally:
            if lock is not None:
                try:
                    lock.release()
                except Exception:
                    pass

    @classmethod
    def class_name(cls) -> str:
        return "jina_v4_memory_guarded"

    def _embed(
        self,
        texts: List[str],
        image_paths: Optional[List[Optional[str]]] = None,
        task: str = "retrieval",
        prompt_name: Optional[str] = "passage",
        return_multivector: bool = False,
        truncate_dim: Optional[int] = None,
        max_length: Optional[int] = None,
        max_pixels: Optional[int] = None,
        batch_size: Optional[int] = None,
        is_query: bool = False,
    ) -> List[List[float]]:
        self._ensure_model()

        model = self._model

        eff_batch = 1 if batch_size is None else max(1, int(batch_size))
        default_max_len = self.max_length_query if is_query else self.max_length_passage
        eff_max_len = int(max_length or default_max_len)
        eff_truncate_dim = truncate_dim or self.truncate_dim

        image_paths = image_paths or [None] * len(texts)
        if len(image_paths) == 1 and len(texts) > 1:
            image_paths = image_paths * len(texts)

        def _to_list_vector(arr):
            if isinstance(arr, torch.Tensor):
                arr = arr.detach().cpu().numpy()
            if np is not None and isinstance(arr, np.ndarray):
                if arr.ndim == 3:
                    arr = arr.mean(axis=1)
                if arr.ndim == 2:
                    arr = arr[0]
                return [float(x) for x in arr.tolist()]
            if isinstance(arr, list):
                if not arr:
                    return []
                first = arr[0]
                if isinstance(first, (list, tuple, torch.Tensor)):
                    return _to_list_vector(first)
                return [float(x) for x in arr]
            try:
                return [float(arr)]
            except Exception:
                return [float(str(arr))]

        results: List[List[float]] = []

        with torch.inference_mode():
            for text, img_path in zip(texts, image_paths):
                base_kwargs = dict(
                    task=task,
                    return_multivector=return_multivector,
                    truncate_dim=eff_truncate_dim,
                    max_length=eff_max_len,
                    batch_size=eff_batch,
                )
                image_kwargs = dict(base_kwargs)
                if max_pixels is not None:
                    image_kwargs["max_pixels"] = max_pixels

                if img_path is not None and text:
                    img = Image.open(img_path).convert("RGB")
                    emb = model.encode(text=text, image=img, **image_kwargs)
                    results.append(_to_list_vector(emb))
                elif img_path is not None:
                    img = Image.open(img_path).convert("RGB")
                    emb = model.encode_image(images=[img], **image_kwargs)
                    results.append(_to_list_vector(emb))
                else:
                    ekw = {"texts": [text], **base_kwargs}
                    if prompt_name:
                        ekw["prompt_name"] = prompt_name
                    emb = model.encode_text(**ekw)
                    results.append(_to_list_vector(emb))

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()

        return results

    def _get_query_embedding(self, query: str, image_path: Optional[str] = None) -> List[float]:
        return self._embed(
            [query],
            [image_path] if image_path else None,
            task="retrieval",
            prompt_name=None if image_path is not None else "query",
            is_query=True,
        )[0]

    def _get_text_embedding(self, text: str, image_path: Optional[str] = None) -> List[float]:
        return self._embed(
            [text],
            [image_path] if image_path else None,
            task="retrieval",
            prompt_name=None if image_path is not None else "passage",
            is_query=False,
        )[0]

    def _get_text_embeddings(
        self,
        texts: List[str],
        image_paths: Optional[List[Optional[str]]] = None,
    ) -> List[List[float]]:
        return self._embed(
            texts,
            image_paths,
            task="retrieval",
            prompt_name="passage",
            is_query=False,
        )

    async def _aget_query_embedding(self, query: str, image_path: Optional[str] = None) -> List[float]:
        return self._get_query_embedding(query, image_path)

    async def _aget_text_embedding(self, text: str, image_path: Optional[str] = None) -> List[float]:
        return self._get_text_embedding(text, image_path)
