from typing import Optional, Any, Dict, List
import os
import base64
import mimetypes

from pydantic import Field, PrivateAttr
from llama_index.core.llms import CustomLLM, CompletionResponse, CompletionResponseGen, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core.schema import ImageDocument
from google import genai


class GeminiMultimodalLLM(CustomLLM):
    """Gemini API client with multimodal support."""

    model_id: str = Field(default="gemini-3-pro-preview")
    max_new_tokens: int = Field(default=16000)
    temperature: float = Field(default=0.6)
    top_p: float = Field(default=0.95)
    top_k: int = Field(default=20)
    api_key: Optional[str] = Field(default=None)

    _client: Any = PrivateAttr(default=None)
    _previous_interaction_id: Optional[str] = PrivateAttr(default=None)

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)

        api_key = self.api_key or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment")

        self._client = genai.Client(api_key=api_key, http_options={"api_version": "v1alpha"})
        self._previous_interaction_id = None

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            model_name=self.model_id,
            context_window=1000000,
            num_output=self.max_new_tokens,
            is_chat_model=True,
            is_function_calling_model=True,
        )

    @property
    def model_name(self) -> str:
        return self.model_id

    def _infer_interaction_type(self, mime_type: str) -> str:
        if mime_type.startswith("image/"):
            return "image"
        if mime_type.startswith("audio/"):
            return "audio"
        if mime_type.startswith("video/"):
            return "video"
        if mime_type == "application/pdf":
            return "document"
        return "document"

    def _prepare_interaction_input(
        self,
        prompt: str,
        image_documents: Optional[List[ImageDocument]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        inputs: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]

        if image_documents:
            for img_doc in image_documents:
                file_path = img_doc.image_path
                file_size = os.path.getsize(file_path)
                mime_type, _ = mimetypes.guess_type(file_path)
                mime_type = mime_type or "application/octet-stream"
                input_type = self._infer_interaction_type(mime_type)

                if file_size > 20 * 1024 * 1024:
                    uploaded_file = self._client.files.upload(file=file_path)
                    inputs.append({
                        "type": input_type,
                        "uri": uploaded_file.uri,
                    })
                else:
                    with open(file_path, "rb") as f:
                        encoded = base64.b64encode(f.read()).decode("utf-8")
                    inputs.append({
                        "type": input_type,
                        "data": encoded,
                        "mime_type": mime_type,
                    })

        return inputs

    def _extract_text(self, interaction: Any) -> str:
        outputs = getattr(interaction, "outputs", None) or []
        for output in reversed(outputs):
            text = getattr(output, "text", None)
            if text:
                return text.strip()
            if isinstance(output, dict):
                text = output.get("text")
                if text:
                    return text.strip()
        return ""

    @llm_completion_callback()
    def complete(self, prompt: str, image_documents: Optional[List[ImageDocument]] = None, **kwargs: Any) -> CompletionResponse:
        inputs = self._prepare_interaction_input(prompt, image_documents, **kwargs)

        generation_config = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_output_tokens": self.max_new_tokens,
        }

        interaction = self._client.interactions.create(
            model=self.model_id,
            input=inputs,
            previous_interaction_id=self._previous_interaction_id,
            generation_config=generation_config,
        )

        self._previous_interaction_id = getattr(interaction, "id", None)
        text = self._extract_text(interaction)
        return CompletionResponse(text=text)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, image_documents: Optional[List[ImageDocument]] = None, **kwargs: Any) -> CompletionResponseGen:
        inputs = self._prepare_interaction_input(prompt, image_documents, **kwargs)

        generation_config = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_output_tokens": self.max_new_tokens,
        }

        interaction = self._client.interactions.create(
            model=self.model_id,
            input=inputs,
            previous_interaction_id=self._previous_interaction_id,
            generation_config=generation_config,
        )

        self._previous_interaction_id = getattr(interaction, "id", None)
        text = self._extract_text(interaction)
        yield CompletionResponse(text=text, delta=text)
