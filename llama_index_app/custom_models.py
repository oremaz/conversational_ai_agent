from typing import Optional
import os
import threading
import logging

import torch

from .model_wrappers.jina_embeddings_v4 import JinaEmbeddingsV4
from .model_wrappers.jina_multimodal_reranker import JinaMultimodalReranker
from .model_wrappers.qwen3_vl_embeddings import Qwen3VLEmbeddings
from .model_wrappers.qwen3_vl_reranker import Qwen3VLReranker
from .model_wrappers.devstral_llm import DevstralLLM
from .model_wrappers.qwen35_multimodal import Qwen35MultiModal
from .model_wrappers.ministral_multimodal import MinistralMultiModal
from .model_wrappers.gpt_oss_llm import GPTOSSLLM
from .model_wrappers.gemini_multimodal import GeminiMultimodalLLM
from .model_wrappers.openai_multimodal import OpenAIMultimodalLLM
from .model_wrappers.openrouter_multimodal import OpenRouterMultimodalLLM
from .model_wrappers.qwen3_omni_multimodal import Qwen3OmniMultiModal
from .model_wrappers.qwen_image_generator import QwenImageGenerator
from .model_wrappers.qwen_image_editor import QwenImageEditor
from .model_wrappers.utils import (
    DIFFUSERS_AVAILABLE,
    register_image_cache,
    register_rag_caches,
)

# Lock to prevent races when creating cached instances concurrently
_CACHE_LOCK = threading.Lock()
_logger = logging.getLogger(__name__)

# Module-level caches to avoid creating multiple heavyweight model instances
_EMBEDDER_CACHE = {}
_RERANKER_CACHE = {}
_LLM_CACHE = {}
_MINISTRAL_CACHE = {}
_API_CLIENT_CACHE = {}

register_rag_caches(_EMBEDDER_CACHE, _RERANKER_CACHE)
register_image_cache(_API_CLIENT_CACHE)


def get_or_create_jina_embedder(model_name: Optional[str] = None, device: Optional[str] = None):
    """Return a cached JinaEmbeddingsV4 instance or create one."""
    key = (model_name or "jinaai/jina-embeddings-v4", device or "auto")
    inst = _EMBEDDER_CACHE.get(key)
    if inst is not None:
        return inst

    with _CACHE_LOCK:
        inst = _EMBEDDER_CACHE.get(key)
        if inst is not None:
            return inst

        _logger.info("Creating Jina embedder for key=%s", key)
        try:
            before_alloc = None
            try:
                if torch.cuda.is_available():
                    before_alloc = torch.cuda.memory_allocated()
            except Exception:
                before_alloc = None

            inst = JinaEmbeddingsV4(model_name=key[0])

            after_alloc = None
            try:
                if torch.cuda.is_available():
                    after_alloc = torch.cuda.memory_allocated()
            except Exception:
                after_alloc = None

            _logger.info("Jina embedder created for key=%s (mem_before=%s, mem_after=%s)", key, before_alloc, after_alloc)
        except Exception:
            _logger.exception("Failed to create Jina embedder for key=%s", key)
            raise

        _EMBEDDER_CACHE[key] = inst
        return inst


def get_or_create_jina_reranker(model_name: Optional[str] = None, top_n: int = 5, device: str = "auto"):
    """Return a cached JinaMultimodalReranker instance or create one."""
    key = (model_name or "jinaai/jina-reranker-m0", top_n, device)
    inst = _RERANKER_CACHE.get(key)
    if inst is not None:
        return inst

    with _CACHE_LOCK:
        inst = _RERANKER_CACHE.get(key)
        if inst is not None:
            return inst

        _logger.info("Creating Jina reranker for key=%s", key)
        try:
            before_alloc = None
            if torch.cuda.is_available():
                try:
                    before_alloc = torch.cuda.memory_allocated()
                except Exception:
                    pass

            inst = JinaMultimodalReranker(model_name=key[0], top_n=key[1], device=key[2])

            after_alloc = None
            if torch.cuda.is_available():
                try:
                    after_alloc = torch.cuda.memory_allocated()
                except Exception:
                    pass

            _logger.info(
                "Jina reranker created for key=%s (mem_before=%s, mem_after=%s)",
                key,
                before_alloc,
                after_alloc
            )
        except Exception:
            _logger.exception("Failed to create Jina reranker for key=%s", key)
            raise

        _RERANKER_CACHE[key] = inst
        return inst


def get_or_create_qwen_embedder(model_name: Optional[str] = None):
    """Return cached Qwen3-VL embedder or create one."""
    key = (model_name or "Qwen/Qwen3-VL-Embedding-2B",)
    inst = _EMBEDDER_CACHE.get(key)
    if inst is not None:
        return inst

    with _CACHE_LOCK:
        inst = _EMBEDDER_CACHE.get(key)
        if inst is not None:
            return inst

        _logger.info("Creating Qwen3-VL embedder for key=%s", key)
        try:
            inst = Qwen3VLEmbeddings(model_name=key[0])
        except Exception:
            _logger.exception("Failed to create Qwen3-VL embedder for key=%s", key)
            raise

        _EMBEDDER_CACHE[key] = inst
        return inst


def get_or_create_qwen_reranker(model_name: Optional[str] = None, top_n: int = 5):
    """Return cached Qwen3-VL reranker or create one."""
    key = (model_name or "Qwen/Qwen3-VL-Reranker-2B", top_n)
    inst = _RERANKER_CACHE.get(key)
    if inst is not None:
        return inst

    with _CACHE_LOCK:
        inst = _RERANKER_CACHE.get(key)
        if inst is not None:
            return inst

        _logger.info("Creating Qwen3-VL reranker for key=%s", key)
        try:
            inst = Qwen3VLReranker(model_name=key[0], top_n=key[1])
        except Exception:
            _logger.exception("Failed to create Qwen3-VL reranker for key=%s", key)
            raise

        _RERANKER_CACHE[key] = inst
        return inst


def get_or_create_qwen35_llm(model_name: Optional[str] = None, device: Optional[str] = None):
    """Return cached Qwen35MultiModal or create one."""
    key = (model_name or "Qwen/Qwen3.5-35B-A3B", device or "auto")
    inst = _LLM_CACHE.get(key)
    if inst is not None:
        return inst

    with _CACHE_LOCK:
        inst = _LLM_CACHE.get(key)
        if inst is not None:
            return inst

        _logger.info("Creating Qwen35MultiModal for key=%s", key)
        try:
            before_alloc = None
            try:
                if torch.cuda.is_available():
                    before_alloc = torch.cuda.memory_allocated()
            except Exception:
                before_alloc = None

            inst = Qwen35MultiModal(model_id=key[0], device_map=key[1])

            after_alloc = None
            try:
                if torch.cuda.is_available():
                    after_alloc = torch.cuda.memory_allocated()
            except Exception:
                after_alloc = None

            _logger.info(
                "Qwen35MultiModal created for key=%s (mem_before=%s, mem_after=%s)",
                key,
                before_alloc,
                after_alloc,
            )
        except Exception:
            _logger.exception("Failed to create Qwen35MultiModal for key=%s", key)
            raise

        _LLM_CACHE[key] = inst
        return inst


def get_or_create_ministral_llm(model_name: Optional[str] = None, device: Optional[str] = None):
    """Return cached MinistralMultiModal or create one."""
    key = (model_name or "mistralai/Ministral-3-8B-Instruct-2512", device or "auto")
    inst = _MINISTRAL_CACHE.get(key)
    if inst is not None:
        return inst

    with _CACHE_LOCK:
        inst = _MINISTRAL_CACHE.get(key)
        if inst is not None:
            return inst

        _logger.info("Creating MinistralMultiModal for key=%s", key)
        try:
            before_alloc = None
            if torch.cuda.is_available():
                try:
                    before_alloc = torch.cuda.memory_allocated()
                except Exception:
                    pass

            inst = MinistralMultiModal(model_id=key[0], device_map=key[1])

            after_alloc = None
            if torch.cuda.is_available():
                try:
                    after_alloc = torch.cuda.memory_allocated()
                except Exception:
                    pass

            _logger.info(
                "MinistralMultiModal created (mem_before=%s, mem_after=%s)",
                before_alloc, after_alloc
            )
        except Exception:
            _logger.exception("Failed to create MinistralMultiModal for key=%s", key)
            raise

        _MINISTRAL_CACHE[key] = inst
        return inst


def get_or_create_devstral_llm(model_name: Optional[str] = None, device: Optional[str] = None):
    """Return cached Devstral LLM configured for agentic software engineering tasks."""
    key = (model_name or "mistralai/Devstral-Small-2-24B-Instruct-2512", device or "auto")
    inst = _LLM_CACHE.get(key)
    if inst is not None:
        return inst

    with _CACHE_LOCK:
        inst = _LLM_CACHE.get(key)
        if inst is not None:
            return inst

        _logger.info("Creating Devstral LLM for key=%s", key)
        try:
            before_alloc = None
            if torch.cuda.is_available():
                try:
                    before_alloc = torch.cuda.memory_allocated()
                except Exception:
                    pass

            inst = DevstralLLM(model_id=key[0], device_map=key[1])

            after_alloc = None
            if torch.cuda.is_available():
                try:
                    after_alloc = torch.cuda.memory_allocated()
                except Exception:
                    pass

            _logger.info(
                "DevstralLLM created for key=%s (mem_before=%s, mem_after=%s)",
                key,
                before_alloc,
                after_alloc,
            )
        except Exception:
            _logger.exception("Failed to create DevstralLLM for key=%s", key)
            raise

        _LLM_CACHE[key] = inst
        return inst


def get_or_create_gpt_oss_llm(model_name: Optional[str] = None, device: Optional[str] = None):
    """Return cached GPT-OSS LLM wrapper."""
    key = (model_name or "openai/gpt-oss-20b", device or "auto")
    inst = _LLM_CACHE.get(key)
    if inst is not None:
        return inst

    with _CACHE_LOCK:
        inst = _LLM_CACHE.get(key)
        if inst is not None:
            return inst

        _logger.info("Creating GPT-OSS LLM for key=%s", key)
        try:
            before_alloc = None
            if torch.cuda.is_available():
                try:
                    before_alloc = torch.cuda.memory_allocated()
                except Exception:
                    pass

            inst = GPTOSSLLM(model_id=key[0], device_map=key[1])

            after_alloc = None
            if torch.cuda.is_available():
                try:
                    after_alloc = torch.cuda.memory_allocated()
                except Exception:
                    pass

            _logger.info(
                "GPT-OSS LLM created for key=%s (mem_before=%s, mem_after=%s)",
                key,
                before_alloc,
                after_alloc,
            )
        except Exception:
            _logger.exception("Failed to create GPT-OSS LLM for key=%s", key)
            raise

        _LLM_CACHE[key] = inst
        return inst


def get_or_create_qwen3_omni_llm(model_name: Optional[str] = None, device: Optional[str] = None):
    """Return cached Qwen3OmniMultiModal or create one."""
    key = (model_name or "Qwen/Qwen3-Omni-30B-A3B-Instruct", device or "auto")
    inst = _LLM_CACHE.get(key)
    if inst is not None:
        return inst

    with _CACHE_LOCK:
        inst = _LLM_CACHE.get(key)
        if inst is not None:
            return inst

        _logger.info("Creating Qwen3OmniMultiModal for key=%s", key)
        inst = Qwen3OmniMultiModal(model_id=key[0], device_map=key[1])
        _LLM_CACHE[key] = inst
        return inst


def get_or_create_gemini_llm(
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    session_id: Optional[str] = None,
):
    """Return cached GeminiMultimodalLLM or create one."""
    key = (model_name or "gemini-3-pro-preview", api_key or os.environ.get("GOOGLE_API_KEY"), session_id)
    inst = _API_CLIENT_CACHE.get(key)
    if inst is not None:
        return inst

    with _CACHE_LOCK:
        inst = _API_CLIENT_CACHE.get(key)
        if inst is not None:
            return inst

        _logger.info("Creating GeminiMultimodalLLM for key=%s", key[0])
        inst = GeminiMultimodalLLM(model_id=key[0], api_key=key[1])
        _API_CLIENT_CACHE[key] = inst
        return inst


def get_or_create_openai_llm(
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    session_id: Optional[str] = None,
):
    """Return cached OpenAIMultimodalLLM or create one."""
    key = (model_name or "gpt-4o", api_key or os.environ.get("OPENAI_API_KEY"), session_id)
    inst = _API_CLIENT_CACHE.get(key)
    if inst is not None:
        return inst

    with _CACHE_LOCK:
        inst = _API_CLIENT_CACHE.get(key)
        if inst is not None:
            return inst

        _logger.info("Creating OpenAIMultimodalLLM for key=%s", key[0])
        inst = OpenAIMultimodalLLM(model_id=key[0], api_key=key[1])
        _API_CLIENT_CACHE[key] = inst
        return inst


def get_or_create_gemini_embedder(model_name: Optional[str] = None):
    """Return cached GeminiEmbeddings or create one."""
    from .model_wrappers.gemini_embeddings import GeminiEmbeddings

    key = ("gemini_embed", model_name or "gemini-embedding-2-preview")
    inst = _API_CLIENT_CACHE.get(key)
    if inst is not None:
        return inst

    with _CACHE_LOCK:
        inst = _API_CLIENT_CACHE.get(key)
        if inst is not None:
            return inst

        _logger.info("Creating Gemini embedder for key=%s", key)
        inst = GeminiEmbeddings(model_name=key[1])
        _API_CLIENT_CACHE[key] = inst
        return inst


def get_or_create_openai_embedder(model_name: Optional[str] = None):
    """Return cached OpenAIEmbeddings or create one."""
    from .model_wrappers.openai_embeddings import OpenAIEmbeddings

    key = ("openai_embed", model_name or "text-embedding-3-small")
    inst = _API_CLIENT_CACHE.get(key)
    if inst is not None:
        return inst

    with _CACHE_LOCK:
        inst = _API_CLIENT_CACHE.get(key)
        if inst is not None:
            return inst

        _logger.info("Creating OpenAI embedder for key=%s", key)
        inst = OpenAIEmbeddings(model_name=key[1])
        _API_CLIENT_CACHE[key] = inst
        return inst


def get_or_create_openrouter_llm(
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    session_id: Optional[str] = None,
):
    """Return cached OpenRouterMultimodalLLM or create one."""
    key = (model_name or "openai/gpt-5-mini", api_key or os.environ.get("OPENROUTER_API_KEY"), session_id)
    inst = _API_CLIENT_CACHE.get(key)
    if inst is not None:
        return inst

    with _CACHE_LOCK:
        inst = _API_CLIENT_CACHE.get(key)
        if inst is not None:
            return inst

        _logger.info("Creating OpenRouterMultimodalLLM for key=%s", key[0])
        inst = OpenRouterMultimodalLLM(model_id=key[0], api_key=key[1])
        _API_CLIENT_CACHE[key] = inst
        return inst


def get_or_create_openrouter_embedder(model_name: Optional[str] = None):
    """Return cached OpenRouterEmbeddings or create one."""
    from .model_wrappers.openrouter_embeddings import OpenRouterEmbeddings

    key = ("openrouter_embed", model_name or "nvidia/llama-nemotron-embed-vl-1b-v2:free")
    inst = _API_CLIENT_CACHE.get(key)
    if inst is not None:
        return inst

    with _CACHE_LOCK:
        inst = _API_CLIENT_CACHE.get(key)
        if inst is not None:
            return inst

        _logger.info("Creating OpenRouter embedder for key=%s", key)
        inst = OpenRouterEmbeddings(model_name=key[1])
        _API_CLIENT_CACHE[key] = inst
        return inst


def get_or_create_image_generator(model_name: Optional[str] = None):
    """Return cached QwenImageGenerator or create one."""
    if not DIFFUSERS_AVAILABLE:
        raise RuntimeError("Diffusers library not available. Cannot create image generator.")

    key = model_name or "Qwen/Qwen-Image-2512"
    inst = _API_CLIENT_CACHE.get(("img_gen", key))
    if inst is not None:
        return inst

    with _CACHE_LOCK:
        cache_key = ("img_gen", key)
        inst = _API_CLIENT_CACHE.get(cache_key)
        if inst is not None:
            return inst

        _logger.info("Creating QwenImageGenerator for model=%s", key)
        inst = QwenImageGenerator(model_name=key)
        _API_CLIENT_CACHE[cache_key] = inst
        return inst


def get_or_create_image_editor(model_name: Optional[str] = None):
    """Return cached QwenImageEditor or create one."""
    if not DIFFUSERS_AVAILABLE:
        raise RuntimeError("Diffusers library not available. Cannot create image editor.")

    key = model_name or "Qwen/Qwen-Image-Edit-2511"
    inst = _API_CLIENT_CACHE.get(("img_edit", key))
    if inst is not None:
        return inst

    with _CACHE_LOCK:
        cache_key = ("img_edit", key)
        inst = _API_CLIENT_CACHE.get(cache_key)
        if inst is not None:
            return inst

        _logger.info("Creating QwenImageEditor for model=%s", key)
        inst = QwenImageEditor(model_name=key)
        _API_CLIENT_CACHE[cache_key] = inst
        return inst
