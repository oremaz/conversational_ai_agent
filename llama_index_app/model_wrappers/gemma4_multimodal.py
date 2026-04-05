"""Gemma 4 multimodal LLM wrapper (AutoModelForMultimodalLM, natively multimodal)."""

from typing import Optional, Any, Dict, Sequence, List
import os
import threading
import logging

import torch
from pydantic import Field, PrivateAttr
from llama_index.core.llms import CustomLLM, CompletionResponse, CompletionResponseGen, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core.schema import ImageDocument
from transformers import (
    AutoProcessor,
    AutoModelForMultimodalLM,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)

from .utils import prepare_llm_for_inference, truncate_on_stop

_logger = logging.getLogger(__name__)

# E4B is a 4B MoE (active params) that runs fast on a single GPU and supports Text+Image+Audio.
# Swap to "google/gemma-4-26B-A4B-it" (256K ctx, Text+Image, fast MoE)
#         "google/gemma-4-31B-it"      (256K ctx, Text+Image, dense)
_DEFAULT_GEMMA4 = "google/gemma-4-E4B-it"

# Context windows per variant
_CTX_WINDOWS = {
    "E2B": 131072,
    "E4B": 131072,
    "26B": 262144,
    "31B": 262144,
}


def _ctx_for_model(model_id: str) -> int:
    for key, ctx in _CTX_WINDOWS.items():
        if key in model_id:
            return ctx
    return 131072


class Gemma4MultiModal(CustomLLM):
    """CustomLLM wrapper for Gemma 4 natively multimodal models (Text / Image / Audio).

    All three instruction-tuned variants are supported:
      - google/gemma-4-E4B-it   (128K ctx, Text+Image+Audio, ~4B active params MoE)
      - google/gemma-4-26B-A4B-it (256K ctx, Text+Image, ~4B active params MoE)
      - google/gemma-4-31B-it   (256K ctx, Text+Image, 31B dense)
    """

    model_id: str = Field(default=_DEFAULT_GEMMA4)
    max_new_tokens: int = Field(default=4096)
    # Google-recommended defaults for Gemma 4
    temperature: float = Field(default=1.0)
    top_p: float = Field(default=0.95)
    top_k: int = Field(default=64)
    device_map: str = Field(default="auto")
    use_flash_attn2: bool = Field(default=False)
    max_input_tokens: int = Field(default=8192)
    # Set True to expose model thinking in complete() via processor.parse_response()
    enable_thinking: bool = Field(default=False)

    _processor: Any = PrivateAttr(default=None)
    _model: Any = PrivateAttr(default=None)
    _hf_loaded: bool = PrivateAttr(default=False)
    _hf_lock: Any = PrivateAttr(default=None)

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        try:
            self._hf_lock = threading.Lock()
        except Exception:
            self._hf_lock = None
        self._hf_loaded = False

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            model_name=self.model_id,
            context_window=_ctx_for_model(self.model_id),
            num_output=self.max_new_tokens,
            is_chat_model=True,
            is_function_calling_model=True,
        )

    @property
    def model_name(self) -> str:
        return self.model_id

    def _init_hf(self) -> None:
        try:
            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128,expandable_segments:True")
        except Exception:
            pass

        self._processor = AutoProcessor.from_pretrained(self.model_id)

        compute_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        quant_cfg = None
        try:
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            )
        except Exception as exc:
            _logger.warning("BitsAndBytes unavailable (%s); falling back to bfloat16 for Gemma 4.", exc)

        model_kwargs: Dict[str, Any] = {
            "device_map": self.device_map,
            "low_cpu_mem_usage": True,
            "torch_dtype": compute_dtype,
        }
        if quant_cfg is not None:
            model_kwargs["quantization_config"] = quant_cfg
        if self.use_flash_attn2:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        # AutoModelForMultimodalLM enables image + audio modalities in addition to text.
        self._model = AutoModelForMultimodalLM.from_pretrained(self.model_id, **model_kwargs)
        self._model.eval()
        _logger.info("Gemma 4 loaded: %s (thinking=%s)", self.model_id, self.enable_thinking)

    def _ensure_hf(self) -> None:
        if getattr(self, "_hf_loaded", False):
            return
        lock = getattr(self, "_hf_lock", None)
        if lock is not None:
            lock.acquire()
        try:
            if getattr(self, "_hf_loaded", False):
                return
            self._init_hf()
            self._hf_loaded = True
        finally:
            if lock is not None:
                try:
                    lock.release()
                except Exception:
                    pass

    def _build_messages(
        self,
        prompt: str,
        image_documents: Optional[Sequence[ImageDocument]] = None,
    ) -> List[Dict[str, Any]]:
        """Build the chat messages list.

        Images are placed *before* the text, as recommended by Google for best performance.
        Each image is referenced via its local file path using the ``url`` key; transformers'
        Gemma 4 processor will open the file automatically.
        """
        content: List[Dict[str, Any]] = []

        for img_doc in image_documents or []:
            if img_doc.image_path:
                content.append({"type": "image", "url": img_doc.image_path})

        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    def _prepare_inputs(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Tokenize messages via apply_chat_template (tokenize=True, return_dict=True)."""
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )

        try:
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        except Exception:
            pass

        # Truncate to max_input_tokens if needed
        if "input_ids" in inputs:
            input_ids = inputs["input_ids"]
            if input_ids.shape[-1] > self.max_input_tokens:
                _logger.warning(
                    "Truncating Gemma 4 input from %d to %d tokens",
                    input_ids.shape[-1], self.max_input_tokens,
                )
                inputs["input_ids"] = input_ids[:, -self.max_input_tokens:]
                if "attention_mask" in inputs:
                    inputs["attention_mask"] = inputs["attention_mask"][:, -self.max_input_tokens:]

        return inputs

    def _parse_output(self, raw: str) -> str:
        """Decode model output via processor.parse_response() and return the response text."""
        try:
            parsed = self._processor.parse_response(raw)
            # parse_response returns {"thinking": ..., "text": ...} when thinking is enabled,
            # or just a string when disabled.
            if isinstance(parsed, dict):
                return (parsed.get("text") or parsed.get("thinking") or "").strip()
            return str(parsed).strip()
        except Exception:
            return raw.strip()

    @llm_completion_callback()
    def complete(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        prepare_llm_for_inference(self)
        self._ensure_hf()
        image_documents: Optional[Sequence[ImageDocument]] = kwargs.pop("image_documents", None)
        messages = self._build_messages(prompt, image_documents)
        inputs = self._prepare_inputs(messages)
        input_len = inputs["input_ids"].shape[-1]

        gen_kwargs: Dict[str, Any] = dict(
            max_new_tokens=min(int(kwargs.get("max_new_tokens", self.max_new_tokens)), self.max_new_tokens),
            do_sample=(kwargs.get("temperature", self.temperature) or 0.0) > 0.0,
            temperature=kwargs.get("temperature", self.temperature),
            top_p=kwargs.get("top_p", self.top_p),
            top_k=self.top_k,
            use_cache=False,
        )
        reserved = set(gen_kwargs.keys())
        merged_inputs = {k: v for k, v in inputs.items() if k not in reserved}

        with torch.inference_mode():
            output_ids = self._model.generate(**merged_inputs, **gen_kwargs)

        # Decode new tokens only; keep special tokens for parse_response
        raw = self._processor.decode(output_ids[0][input_len:], skip_special_tokens=False)
        text = self._parse_output(raw)

        if stop:
            text = truncate_on_stop(text, stop)
        return CompletionResponse(text=text)

    @llm_completion_callback()
    def stream_complete(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> CompletionResponseGen:
        prepare_llm_for_inference(self)
        self._ensure_hf()
        image_documents: Optional[Sequence[ImageDocument]] = kwargs.pop("image_documents", None)
        messages = self._build_messages(prompt, image_documents)
        inputs = self._prepare_inputs(messages)

        gen_kwargs: Dict[str, Any] = dict(
            max_new_tokens=min(int(kwargs.get("max_new_tokens", self.max_new_tokens)), self.max_new_tokens),
            do_sample=(kwargs.get("temperature", self.temperature) or 0.0) > 0.0,
            temperature=kwargs.get("temperature", self.temperature),
            top_p=kwargs.get("top_p", self.top_p),
            top_k=self.top_k,
            use_cache=False,
        )

        # skip_special_tokens=True for streaming to avoid emitting raw thinking tokens
        streamer = TextIteratorStreamer(
            self._processor.tokenizer,
            skip_special_tokens=True,
            skip_prompt=True,
            clean_up_tokenization_spaces=False,
        )

        reserved = set(gen_kwargs.keys())
        merged_inputs = {k: v for k, v in inputs.items() if k not in reserved}

        thread = threading.Thread(
            target=self._model.generate,
            kwargs={**merged_inputs, **gen_kwargs, "streamer": streamer},
            daemon=True,
        )
        thread.start()

        text_accum = ""
        for delta in streamer:
            text_accum += delta
            if stop and any(s and s in text_accum for s in stop):
                trimmed = truncate_on_stop(text_accum, stop)
                yield CompletionResponse(text=trimmed, delta="")
                break
            yield CompletionResponse(text=text_accum, delta=delta)
