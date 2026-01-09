from typing import Optional, Any
import threading

from pydantic import Field, PrivateAttr
from llama_index.core.llms import CustomLLM, CompletionResponse, CompletionResponseGen, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

from .utils import prepare_llm_for_inference, truncate_on_stop


class Qwen3TextLLM(CustomLLM):
    """Qwen3 text-only model wrapper without vision capabilities."""

    model_id: str = Field(default="Qwen/Qwen3-4B-Instruct-2507-FP8")
    max_new_tokens: int = Field(default=16384)
    temperature: float = Field(default=0.6)
    top_p: float = Field(default=0.95)
    device_map: str = Field(default="auto")
    max_input_tokens: int = Field(default=32768)

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
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True
        )

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype="auto",
            device_map=self.device_map,
            trust_remote_code=True
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

    @llm_completion_callback()
    def complete(self, prompt: str, stop: Optional[list[str]] = None, **kwargs: Any) -> CompletionResponse:
        prepare_llm_for_inference(self)
        self._ensure_hf()

        messages = [{"role": "user", "content": prompt}]
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        model_inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)

        generated_ids = self._model.generate(
            **model_inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )

        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        content = self._tokenizer.decode(output_ids, skip_special_tokens=True)

        if stop:
            content = truncate_on_stop(content, stop)

        return CompletionResponse(text=content.strip())

    @llm_completion_callback()
    def stream_complete(self, prompt: str, stop: Optional[list[str]] = None, **kwargs: Any) -> CompletionResponseGen:
        prepare_llm_for_inference(self)
        self._ensure_hf()

        messages = [{"role": "user", "content": prompt}]
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        model_inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)

        streamer = TextIteratorStreamer(
            self._tokenizer,
            skip_special_tokens=True,
            skip_prompt=True,
        )

        thread = threading.Thread(
            target=self._model.generate,
            kwargs={
                **model_inputs,
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "streamer": streamer
            },
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
