from typing import Optional, Any
import threading

from pydantic import Field, PrivateAttr
from llama_index.core.llms import CustomLLM, CompletionResponse, CompletionResponseGen, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback
from transformers import pipeline

from .utils import prepare_llm_for_inference, truncate_on_stop


class GPTOSSLLM(CustomLLM):
    """GPT-OSS text model wrapper using the HF pipeline API."""

    model_id: str = Field(default="openai/gpt-oss-20b")
    max_new_tokens: int = Field(default=1024)
    temperature: float = Field(default=0.7)
    top_p: float = Field(default=0.95)
    device_map: str = Field(default="auto")
    max_input_tokens: int = Field(default=32768)

    _pipe: Any = PrivateAttr(default=None)
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

    def _init_pipeline(self) -> None:
        self._pipe = pipeline(
            "text-generation",
            model=self.model_id,
            torch_dtype="auto",
            device_map=self.device_map,
        )

    def _ensure_pipeline(self) -> None:
        if getattr(self, "_hf_loaded", False):
            return
        lock = getattr(self, "_hf_lock", None)
        if lock is not None:
            lock.acquire()
        try:
            if getattr(self, "_hf_loaded", False):
                return
            self._init_pipeline()
            self._hf_loaded = True
        finally:
            if lock is not None:
                try:
                    lock.release()
                except Exception:
                    pass

    def _extract_generated_text(self, outputs: Any) -> str:
        if isinstance(outputs, list) and outputs:
            payload = outputs[0]
        else:
            payload = outputs

        if isinstance(payload, dict):
            generated = payload.get("generated_text")
        else:
            generated = payload

        if isinstance(generated, list):
            last = generated[-1] if generated else ""
            if isinstance(last, dict):
                return (last.get("content") or last.get("text") or "").strip()
            return str(last).strip()
        if isinstance(generated, dict):
            return (generated.get("content") or generated.get("text") or "").strip()
        if isinstance(generated, str):
            return generated.strip()
        return str(generated).strip()

    @llm_completion_callback()
    def complete(self, prompt: str, stop: Optional[list[str]] = None, **kwargs: Any) -> CompletionResponse:
        prepare_llm_for_inference(self)
        self._ensure_pipeline()

        messages = [{"role": "user", "content": prompt}]
        outputs = self._pipe(
            messages,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )

        content = self._extract_generated_text(outputs)
        if stop:
            content = truncate_on_stop(content, stop)
        return CompletionResponse(text=content.strip())

    @llm_completion_callback()
    def stream_complete(self, prompt: str, stop: Optional[list[str]] = None, **kwargs: Any) -> CompletionResponseGen:
        resp = self.complete(prompt, stop=stop, **kwargs)
        yield CompletionResponse(text=resp.text, delta=resp.text)
