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
    AutoModelForCausalLM,
    AutoProcessor,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)
from PIL import Image

from .utils import prepare_llm_for_inference, truncate_on_stop

_logger = logging.getLogger(__name__)

_DEFAULT_QWEN35 = "Qwen/Qwen3.5-35B-A3B"


class Qwen35MultiModal(CustomLLM):
    """CustomLLM wrapper for Qwen3.5 natively multimodal models with optional 4-bit quantization."""

    model_id: str = Field(default=_DEFAULT_QWEN35)
    max_new_tokens: int = Field(default=4096)
    temperature: float = Field(default=0.2)
    top_p: float = Field(default=0.95)
    device_map: str = Field(default="auto")
    min_pixels: Optional[int] = Field(default=None)
    max_pixels: Optional[int] = Field(default=None)
    use_flash_attn2: bool = Field(default=False)
    max_input_tokens: int = Field(default=4096)

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
            context_window=262144,
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

        processor_kwargs: Dict[str, Any] = {}
        if self.min_pixels is not None:
            processor_kwargs["min_pixels"] = self.min_pixels
        if self.max_pixels is not None:
            processor_kwargs["max_pixels"] = self.max_pixels
        self._processor = AutoProcessor.from_pretrained(self.model_id, **processor_kwargs)

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
            _logger.warning("BitsAndBytes unavailable (%s); falling back to fp16 load for Qwen3.5.", exc)

        model_kwargs: Dict[str, Any] = {
            "device_map": self.device_map,
            "low_cpu_mem_usage": True,
            "torch_dtype": compute_dtype,
            "trust_remote_code": True,
        }
        if quant_cfg is not None:
            model_kwargs["quantization_config"] = quant_cfg
        if self.use_flash_attn2:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            **model_kwargs,
        )
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

    def _build_user_messages(
        self,
        prompt: str,
        image_documents: Optional[Sequence[ImageDocument]] = None,
    ) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for image_document in image_documents or []:
            if image_document.image_path:
                content.append({"type": "image", "image": image_document.image_path})
        return [{"role": "user", "content": content}]

    def _prepare_inputs_from_messages(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        images = []
        for message in messages:
            for item in message.get("content", []):
                if item.get("type") == "image" and item.get("image"):
                    images.append(Image.open(item["image"]).convert("RGB"))

        if images:
            inputs = self._processor(text=[text], images=images, return_tensors="pt")
        else:
            inputs = self._processor(text=[text], return_tensors="pt")

        try:
            for key, value in inputs.items():
                inputs[key] = value.to(self._model.device)
        except Exception:
            pass

        if "input_ids" in inputs:
            input_ids = inputs["input_ids"]
            if input_ids.shape[-1] > self.max_input_tokens:
                _logger.warning(
                    "Truncating Qwen3.5 input from %d to %d tokens",
                    input_ids.shape[-1], self.max_input_tokens
                )
                inputs["input_ids"] = input_ids[:, -self.max_input_tokens:]
        if "attention_mask" in inputs:
            inputs["attention_mask"] = inputs["attention_mask"][:, -self.max_input_tokens:]

        if len(inputs.get("input_ids", [])) > 1:
            _logger.warning("Trimmed input batch dimension to 1 for Qwen3.5 inference.")
            for key in inputs:
                inputs[key] = inputs[key][:1]

        return inputs

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
        messages = self._build_user_messages(prompt, image_documents)
        inputs = self._prepare_inputs_from_messages(messages)
        gen_kwargs = dict(
            max_new_tokens=min(int(kwargs.get("max_new_tokens", self.max_new_tokens)), self.max_new_tokens),
            do_sample=(kwargs.get("temperature", self.temperature) or 0.0) > 0.0,
            temperature=kwargs.get("temperature", self.temperature),
            top_p=kwargs.get("top_p", self.top_p),
            use_cache=False,
        )
        reserved = set(gen_kwargs.keys())
        merged_inputs = {k: v for k, v in inputs.items() if k not in reserved}
        with torch.inference_mode():
            output_ids = self._model.generate(
                **merged_inputs,
                **gen_kwargs,
            )
        output_ids = output_ids[0][len(inputs["input_ids"][0]):]
        text = self._processor.tokenizer.decode(output_ids, skip_special_tokens=True)
        if stop:
            text = truncate_on_stop(text, stop)
        return CompletionResponse(text=text.strip())

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
        messages = self._build_user_messages(prompt, image_documents)
        inputs = self._prepare_inputs_from_messages(messages)
        gen_kwargs = dict(
            max_new_tokens=min(int(kwargs.get("max_new_tokens", self.max_new_tokens)), 64),
            do_sample=(kwargs.get("temperature", self.temperature) or 0.0) > 0.0,
            temperature=kwargs.get("temperature", self.temperature),
            top_p=kwargs.get("top_p", self.top_p),
            use_cache=False,
        )
        streamer = TextIteratorStreamer(
            self._processor.tokenizer,
            skip_special_tokens=True,
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
