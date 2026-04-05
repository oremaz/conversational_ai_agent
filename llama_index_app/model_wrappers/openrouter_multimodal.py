"""OpenRouter multimodal LLM wrapper (OpenAI-compatible chat completions API)."""

from typing import Optional, Any, Dict, List
import os
import base64
import mimetypes

from pydantic import Field, PrivateAttr
from llama_index.core.llms import CustomLLM, CompletionResponse, CompletionResponseGen, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core.schema import ImageDocument
from openai import OpenAI


class OpenRouterMultimodalLLM(CustomLLM):
    """OpenRouter API client with multimodal support (OpenAI-compatible chat completions)."""

    model_id: str = Field(default="qwen/qwen3.6-plus:free")
    max_new_tokens: int = Field(default=4096)
    temperature: float = Field(default=0.7)
    api_key: Optional[str] = Field(default=None)
    base_url: str = Field(default="https://openrouter.ai/api/v1")
    site_url: Optional[str] = Field(default=None)
    site_name: Optional[str] = Field(default=None)

    _client: Any = PrivateAttr(default=None)

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)

        api_key = self.api_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment")

        extra_headers: Dict[str, str] = {}
        if self.site_url:
            extra_headers["HTTP-Referer"] = self.site_url
        if self.site_name:
            extra_headers["X-Title"] = self.site_name

        self._client = OpenAI(
            api_key=api_key,
            base_url=self.base_url,
            default_headers=extra_headers if extra_headers else None,
        )

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            model_name=self.model_id,
            context_window=128000,
            num_output=self.max_new_tokens,
            is_chat_model=True,
            is_function_calling_model=True,
        )

    @property
    def model_name(self) -> str:
        return self.model_id

    def _prepare_messages(
        self,
        prompt: str,
        image_documents: Optional[List[ImageDocument]] = None,
    ) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]

        if image_documents:
            for img_doc in image_documents:
                file_path = img_doc.image_path
                mime_type, _ = mimetypes.guess_type(file_path)
                mime_type = mime_type or "image/jpeg"

                with open(file_path, "rb") as f:
                    image_data = base64.b64encode(f.read()).decode("utf-8")

                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{image_data}",
                    },
                })

        return [{"role": "user", "content": content}]

    def _extract_text(self, response: Any) -> str:
        try:
            return response.choices[0].message.content or ""
        except (AttributeError, IndexError):
            return ""

    def _extract_stream_delta(self, event: Any) -> str:
        """Extract incremental text from OpenAI/OpenRouter stream event variants."""
        # Some SDK versions wrap chat chunks as ChunkEvent(chunk=...)
        chunk = getattr(event, "chunk", event)

        try:
            if getattr(chunk, "choices", None):
                delta = chunk.choices[0].delta
                return getattr(delta, "content", "") or ""
        except Exception:
            pass

        # Responses-style event fallback.
        event_type = getattr(chunk, "type", None)
        if event_type in {"response.output_text.delta", "output_text.delta"}:
            return getattr(chunk, "delta", "") or ""

        return ""

    @llm_completion_callback()
    def complete(
        self,
        prompt: str,
        image_documents: Optional[List[ImageDocument]] = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        messages = self._prepare_messages(prompt, image_documents)
        response = self._client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        text = self._extract_text(response)
        return CompletionResponse(text=text)

    @llm_completion_callback()
    def stream_complete(
        self,
        prompt: str,
        image_documents: Optional[List[ImageDocument]] = None,
        **kwargs: Any,
    ) -> CompletionResponseGen:
        messages = self._prepare_messages(prompt, image_documents)
        accumulated = ""
        with self._client.chat.completions.stream(
            model=self.model_id,
            messages=messages,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
        ) as stream:
            for chunk in stream:
                delta = self._extract_stream_delta(chunk)
                accumulated += delta
                yield CompletionResponse(text=accumulated, delta=delta)
