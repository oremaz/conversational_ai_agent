from typing import Optional, Any, Dict
import os
import threading
import logging

import torch
from pydantic import Field, PrivateAttr
from llama_index.core.llms import CustomLLM, CompletionResponse, CompletionResponseGen, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback
from transformers import Mistral3ForConditionalGeneration, TextIteratorStreamer

try:
    from transformers import MistralCommonBackend
except Exception:
    MistralCommonBackend = None

from .utils import prepare_llm_for_inference, truncate_on_stop

_logger = logging.getLogger(__name__)


class MinistralMultiModal(CustomLLM):
    """CustomLLM wrapper for Mistral Ministral-3 series in native FP8."""

    model_id: str = Field(default="mistralai/Ministral-3-8B-Instruct-2512")
    max_new_tokens: int = Field(default=2048)
    temperature: float = Field(default=0.7)
    top_p: float = Field(default=0.95)
    top_k: int = Field(default=50)
    device_map: str = Field(default="auto")
    use_flash_attn2: bool = Field(default=False)
    max_input_tokens: int = Field(default=8192)

    _tokenizer: Any = PrivateAttr(default=None)
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
            context_window=32768,
            num_output=self.max_new_tokens,
            is_chat_model=True,
            is_function_calling_model=False,
        )

    @property
    def model_name(self) -> str:
        return self.model_id

    def _init_hf(self) -> None:
        try:
            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128,expandable_segments:True")
        except Exception:
            pass

        if MistralCommonBackend is None:
            raise RuntimeError("MistralCommonBackend not available; Mistral models cannot be initialized.")

        self._tokenizer = MistralCommonBackend.from_pretrained(self.model_id)

        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        model_kwargs: Dict[str, Any] = {
            "device_map": self.device_map,
            "low_cpu_mem_usage": True,
            "torch_dtype": "auto",
            "trust_remote_code": True,
        }

        try:
            offload_folder = os.path.abspath("./offload_ministral")
            os.makedirs(offload_folder, exist_ok=True)
            model_kwargs["offload_folder"] = offload_folder
        except Exception:
            pass

        if self.use_flash_attn2:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        self._model = Mistral3ForConditionalGeneration.from_pretrained(self.model_id, **model_kwargs)
        self._model.eval()

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

    def _truncate_input(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.shape[-1] > self.max_input_tokens:
            _logger.warning(
                "Truncating Ministral input from %d to %d tokens",
                input_ids.shape[-1], self.max_input_tokens
            )
            return input_ids[:, -self.max_input_tokens:]
        return input_ids

    @llm_completion_callback()
    def complete(self, prompt: str, stop: Optional[list[str]] = None, **kwargs: Any) -> CompletionResponse:
        prepare_llm_for_inference(self)
        self._ensure_hf()

        messages = [{"role": "user", "content": prompt}]
        inputs = self._tokenizer.apply_chat_template(
            conversation=messages,
            return_tensors="pt",
            return_dict=True,
        )

        try:
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        except Exception:
            pass

        if "input_ids" in inputs:
            inputs["input_ids"] = self._truncate_input(inputs["input_ids"])
        if "attention_mask" in inputs:
            inputs["attention_mask"] = inputs["attention_mask"][:, -self.max_input_tokens:]

        gen_kwargs = dict(
            max_new_tokens=min(int(kwargs.get("max_new_tokens", self.max_new_tokens)), self.max_new_tokens),
            do_sample=(kwargs.get("temperature", self.temperature) or 0.0) > 0.0,
            temperature=kwargs.get("temperature", self.temperature) or 0.7,
            top_p=kwargs.get("top_p", self.top_p),
            top_k=self.top_k,
            pad_token_id=self._tokenizer.pad_token_id,
            eos_token_id=self._tokenizer.eos_token_id,
        )

        with torch.inference_mode():
            output_ids = self._model.generate(**inputs, **gen_kwargs)

        new_tokens = output_ids[0][len(inputs["input_ids"][0]):]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)

        if stop:
            text = truncate_on_stop(text, stop)

        return CompletionResponse(text=text.strip())

    @llm_completion_callback()
    def stream_complete(self, prompt: str, stop: Optional[list[str]] = None, **kwargs: Any) -> CompletionResponseGen:
        prepare_llm_for_inference(self)
        self._ensure_hf()

        messages = [{"role": "user", "content": prompt}]
        inputs = self._tokenizer.apply_chat_template(
            conversation=messages,
            return_tensors="pt",
            return_dict=True,
        )

        try:
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        except Exception:
            pass

        if "input_ids" in inputs:
            inputs["input_ids"] = self._truncate_input(inputs["input_ids"])
        if "attention_mask" in inputs:
            inputs["attention_mask"] = inputs["attention_mask"][:, -self.max_input_tokens:]

        gen_kwargs = dict(
            max_new_tokens=min(int(kwargs.get("max_new_tokens", self.max_new_tokens)), self.max_new_tokens),
            do_sample=(kwargs.get("temperature", self.temperature) or 0.0) > 0.0,
            temperature=kwargs.get("temperature", self.temperature) or 0.7,
            top_p=kwargs.get("top_p", self.top_p),
            top_k=self.top_k,
            pad_token_id=self._tokenizer.pad_token_id,
            eos_token_id=self._tokenizer.eos_token_id,
        )

        streamer = TextIteratorStreamer(
            self._tokenizer.tokenizer if hasattr(self._tokenizer, "tokenizer") else self._tokenizer,
            skip_special_tokens=True,
            skip_prompt=True,
        )

        thread = threading.Thread(
            target=self._model.generate,
            kwargs={**inputs, **gen_kwargs, "streamer": streamer},
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
