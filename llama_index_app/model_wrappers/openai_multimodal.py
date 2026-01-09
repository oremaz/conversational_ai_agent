from typing import Optional, Any, Dict, List
import os
import base64
import mimetypes

from pydantic import Field, PrivateAttr
from llama_index.core.llms import CustomLLM, CompletionResponse, CompletionResponseGen, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core.schema import ImageDocument
from openai import OpenAI


class OpenAIMultimodalLLM(CustomLLM):
    """OpenAI API client with multimodal support."""

    model_id: str = Field(default="gpt-4o")
    max_new_tokens: int = Field(default=4096)
    temperature: float = Field(default=0.7)
    api_key: Optional[str] = Field(default=None)

    _client: Any = PrivateAttr(default=None)
    _conversation_id: Optional[str] = PrivateAttr(default=None)

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)

        api_key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")

        self._client = OpenAI(api_key=api_key)
        self._conversation_id = None

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

    def _ensure_conversation_id(self) -> Optional[str]:
        if self._conversation_id:
            return self._conversation_id

        try:
            convo = self._client.conversations.create()
        except Exception:
            return None

        convo_id = getattr(convo, "id", None)
        if convo_id:
            self._conversation_id = convo_id
        return self._conversation_id

    def _prepare_input(self, prompt: str, image_documents: Optional[List[ImageDocument]] = None) -> List[Dict]:
        content = [{"type": "input_text", "text": prompt}]

        if image_documents:
            for img_doc in image_documents:
                file_path = img_doc.image_path

                with open(file_path, "rb") as f:
                    image_data = base64.b64encode(f.read()).decode("utf-8")

                mime_type, _ = mimetypes.guess_type(file_path)
                mime_type = mime_type or "image/jpeg"

                content.append({
                    "type": "input_image",
                    "image_url": f"data:{mime_type};base64,{image_data}",
                })

        return [{"role": "user", "content": content}]

    def _extract_response_text(self, response: Any) -> str:
        text = getattr(response, "output_text", None)
        if text:
            return text.strip()

        output = getattr(response, "output", None) or getattr(response, "outputs", None) or []
        for item in output:
            content = getattr(item, "content", None) if not isinstance(item, dict) else item.get("content")
            if not content:
                continue
            for part in content:
                part_type = part.get("type") if isinstance(part, dict) else getattr(part, "type", None)
                if part_type in ("output_text", "text"):
                    text = part.get("text") if isinstance(part, dict) else getattr(part, "text", None)
                    if text:
                        return text.strip()
        return ""

    @llm_completion_callback()
    def complete(self, prompt: str, image_documents: Optional[List[ImageDocument]] = None, **kwargs: Any) -> CompletionResponse:
        payload = self._prepare_input(prompt, image_documents)
        convo_id = self._ensure_conversation_id()

        response_kwargs = {
            "model": self.model_id,
            "input": payload,
            "max_output_tokens": self.max_new_tokens,
            "temperature": self.temperature,
        }
        if convo_id:
            response_kwargs["conversation"] = convo_id

        response = self._client.responses.create(**response_kwargs)

        text = self._extract_response_text(response)
        return CompletionResponse(text=text)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, image_documents: Optional[List[ImageDocument]] = None, **kwargs: Any) -> CompletionResponseGen:
        payload = self._prepare_input(prompt, image_documents)
        convo_id = self._ensure_conversation_id()

        response_kwargs = {
            "model": self.model_id,
            "input": payload,
            "max_output_tokens": self.max_new_tokens,
            "temperature": self.temperature,
        }
        if convo_id:
            response_kwargs["conversation"] = convo_id

        response = self._client.responses.create(**response_kwargs)

        text = self._extract_response_text(response)
        yield CompletionResponse(text=text, delta=text)
