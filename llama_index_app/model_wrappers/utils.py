from typing import Optional, List
import logging
import threading

import torch

try:
    from diffusers import DiffusionPipeline, QwenImageEditPlusPipeline
    import soundfile as sf
    DIFFUSERS_AVAILABLE = True
except (ImportError, AttributeError) as e:
    DIFFUSERS_AVAILABLE = False
    DiffusionPipeline, QwenImageEditPlusPipeline, sf = None, None, None
    logging.getLogger(__name__).warning("Diffusers not available: %s", e)

_logger = logging.getLogger(__name__)
_ACTIVE_LLM_LOCK = threading.Lock()
_ACTIVE_LLM = None
_RAG_CACHES = {"embedder": None, "reranker": None}
_IMAGE_CACHE = None


def log_cuda_memory(label: str) -> None:
    """Log a small CUDA memory snapshot for debugging."""
    if not torch.cuda.is_available():
        return
    try:
        device_count = torch.cuda.device_count()
    except Exception:
        device_count = 1
    total_alloc = 0
    total_reserved = 0
    parts = []
    for idx in range(device_count):
        alloc = torch.cuda.memory_allocated(idx)
        reserved = torch.cuda.memory_reserved(idx)
        total_alloc += alloc
        total_reserved += reserved
        parts.append(f"cuda:{idx}={int(alloc / (1024 * 1024))}MB/{int(reserved / (1024 * 1024))}MB")
    if parts:
        _logger.debug(
            "%s cuda_mem alloc=%sMB reserved=%sMB devices=%s",
            label,
            int(total_alloc / (1024 * 1024)),
            int(total_reserved / (1024 * 1024)),
            " ".join(parts),
        )


def register_rag_caches(embedder_cache: dict, reranker_cache: dict) -> None:
    """Register caches used by offload_rag_models."""
    _RAG_CACHES["embedder"] = embedder_cache
    _RAG_CACHES["reranker"] = reranker_cache


def register_image_cache(api_cache: dict) -> None:
    """Register cache used by offload_image_models."""
    global _IMAGE_CACHE
    _IMAGE_CACHE = api_cache


def truncate_on_stop(text: str, stop: Optional[List[str]]) -> str:
    if not stop:
        return text
    idxs = [text.find(s) for s in stop if s and s in text]
    if not idxs:
        return text
    cut = min(i for i in idxs if i >= 0)
    return text[:cut]


def unload_model_from_gpu(model):
    """Move a model to CPU and clear GPU cache."""
    try:
        if hasattr(model, "_model") and model._model is not None:
            _logger.info("Unloading model from GPU to CPU")
            model._model.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        elif hasattr(model, "_pipe") and getattr(model._pipe, "model", None) is not None:
            _logger.info("Unloading pipeline model from GPU to CPU")
            model._pipe.model.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        return True
    except Exception as e:
        _logger.warning("Failed to unload model: %s", e)
        return False


def unload_diffusion_pipeline(pipeline) -> bool:
    """Move a diffusers pipeline to CPU and clear GPU cache."""
    try:
        if pipeline is not None and hasattr(pipeline, "to"):
            _logger.info("Unloading diffusion pipeline from GPU to CPU")
            pipeline.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        return True
    except Exception as e:
        _logger.warning("Failed to unload diffusion pipeline: %s", e)
        return False


def _offload_cached_models(cache: dict, label: str) -> None:
    """Best-effort offload of cached models to free VRAM."""
    for key, inst in list(cache.items()):
        if inst is None:
            continue
        try:
            if hasattr(inst, "_model") and inst._model is not None:
                unload_model_from_gpu(inst)
                inst._model = None
            if hasattr(inst, "_processor"):
                inst._processor = None
            if hasattr(inst, "_tokenizer"):
                inst._tokenizer = None
            if hasattr(inst, "_loaded"):
                inst._loaded = False
            _logger.info("Offloaded %s model for key=%s", label, key)
        except Exception as exc:
            _logger.warning("Failed to offload %s model for key=%s: %s", label, key, exc)


def offload_rag_models() -> None:
    """Offload embedding and reranker models to free VRAM."""
    embedder_cache = _RAG_CACHES.get("embedder")
    reranker_cache = _RAG_CACHES.get("reranker")
    if embedder_cache is None or reranker_cache is None:
        _logger.debug("RAG caches not registered; skipping offload.")
        return
    log_cuda_memory("before_rag_offload")
    _offload_cached_models(embedder_cache, "embedder")
    _offload_cached_models(reranker_cache, "reranker")
    log_cuda_memory("after_rag_offload")


def offload_image_models() -> None:
    """Offload image generation/editing pipelines to free VRAM."""
    if _IMAGE_CACHE is None:
        _logger.debug("Image cache not registered; skipping offload.")
        return
    log_cuda_memory("before_image_offload")
    for key, inst in list(_IMAGE_CACHE.items()):
        if not isinstance(key, tuple) or not key:
            continue
        if key[0] not in ("img_gen", "img_edit"):
            continue
        try:
            pipeline = getattr(inst, "pipeline", None)
            if pipeline is not None:
                unload_diffusion_pipeline(pipeline)
                inst.pipeline = None
            _logger.info("Offloaded image pipeline for key=%s", key)
        except Exception as exc:
            _logger.warning("Failed to offload image pipeline for key=%s: %s", key, exc)
    log_cuda_memory("after_image_offload")


def reload_model_to_gpu(model):
    """Move a model back to GPU if available."""
    try:
        if hasattr(model, "_model") and model._model is not None:
            _logger.info("Reloading model to GPU")
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            model._model.to(device)
        elif hasattr(model, "_pipe") and getattr(model._pipe, "model", None) is not None:
            _logger.info("Reloading pipeline model to GPU")
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            model._pipe.model.to(device)
        return True
    except Exception as e:
        _logger.warning("Failed to reload model: %s", e)
        return False


def _reset_llm_state(llm) -> None:
    """Best-effort reset of an LLM wrapper so it can be reloaded later."""
    unload_model_from_gpu(llm)
    if hasattr(llm, "_model"):
        llm._model = None
    if hasattr(llm, "_pipe"):
        llm._pipe = None
    if hasattr(llm, "_processor"):
        llm._processor = None
    if hasattr(llm, "_tokenizer"):
        llm._tokenizer = None
    if hasattr(llm, "_hf_loaded"):
        llm._hf_loaded = False


def prepare_llm_for_inference(current_llm) -> None:
    """Offload the previously used LLM before loading a new one."""
    global _ACTIVE_LLM
    with _ACTIVE_LLM_LOCK:
        prev_llm = _ACTIVE_LLM
        if prev_llm is current_llm:
            return
        _ACTIVE_LLM = current_llm
    if prev_llm is not None:
        log_cuda_memory("before_llm_offload")
        _logger.info("Offloading previous LLM before switching agents")
        _reset_llm_state(prev_llm)
        log_cuda_memory("after_llm_offload")
