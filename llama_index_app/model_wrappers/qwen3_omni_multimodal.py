from typing import Optional, Any, Dict
import os
import threading

import torch
from pydantic import Field, PrivateAttr
from llama_index.core.llms import CustomLLM, CompletionResponse, CompletionResponseGen, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core.schema import ImageDocument
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

from .utils import prepare_llm_for_inference


class Qwen3OmniMultiModal(CustomLLM):
    """Qwen3-Omni model wrapper for audio, video, and text processing."""

    model_id: str = Field(default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    max_new_tokens: int = Field(default=4096)
    temperature: float = Field(default=0.2)
    top_p: float = Field(default=0.95)
    device_map: str = Field(default="auto")
    use_flash_attn2: bool = Field(default=False)
    use_audio_in_video: bool = Field(default=True)
    speaker: str = Field(default="Ethan")

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

        model_kwargs: Dict[str, Any] = {
            "device_map": self.device_map,
            "low_cpu_mem_usage": True,
            "torch_dtype": "auto",
            "trust_remote_code": True,
        }
        if self.use_flash_attn2:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        self._model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            self.model_id,
            **model_kwargs
        )
        self._model.eval()

        self._processor = Qwen3OmniMoeProcessor.from_pretrained(self.model_id)

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

    @llm_completion_callback()
    def complete(self, prompt: str, image_documents: Optional[list[ImageDocument]] = None, **kwargs: Any) -> CompletionResponse:
        prepare_llm_for_inference(self)
        self._ensure_hf()

        try:
            from qwen_omni_utils import process_mm_input
        except Exception:
            return CompletionResponse(text="Qwen Omni utils not available. Please install qwen_omni_utils.")

        messages = [{"role": "user", "content": prompt}]

        if image_documents:
            for img in image_documents:
                if img.image_path:
                    messages.append({
                        "role": "user",
                        "content": {
                            "type": "image",
                            "image": img.image_path,
                        },
                    })

        inputs = process_mm_input(
            messages,
            self._processor,
            self.speaker,
            use_audio_in_video=self.use_audio_in_video,
        )

        gen_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=(self.temperature or 0.0) > 0.0,
            temperature=self.temperature,
            top_p=self.top_p,
            use_cache=False,
        )

        with torch.inference_mode():
            output = self._model.generate(**inputs, **gen_kwargs)

        output_ids = output[0][len(inputs["input_ids"][0]):]
        text = self._processor.decode(output_ids, skip_special_tokens=True)

        return CompletionResponse(text=text.strip())

    @llm_completion_callback()
    def stream_complete(self, prompt: str, image_documents: Optional[list[ImageDocument]] = None, **kwargs: Any) -> CompletionResponseGen:
        resp = self.complete(prompt, image_documents=image_documents, **kwargs)
        yield CompletionResponse(text=resp.text, delta=resp.text)
