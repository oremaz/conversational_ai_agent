"""Tool definitions for the smolagents runner."""

import base64
import mimetypes
import os
import re
from typing import Any, Dict, Optional

import requests
from markdownify import markdownify
from requests.exceptions import RequestException
from smolagents import Tool, tool

from google import genai
from google.genai import types
from openai import OpenAI

from .prompts import (
    FINAL_ANSWER_TOOL_DESCRIPTION,
    MULTIMODAL_TOOL_DESCRIPTION,
    MULTIMODAL_TASK_PROMPTS,
)

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class FinalAnswerTool(Tool):
    name = "final_answer"
    description = FINAL_ANSWER_TOOL_DESCRIPTION
    inputs = {
        "answer": {
            "type": "any",
            "description": "The final answer to the problem",
        },
        "original_question": {
            "type": "string",
            "description": "The original question to determine answer format",
            "nullable": True,
        },
    }
    output_type = "any"

    def forward(self, answer: Any, original_question: str = "") -> Any:
        return answer


@tool
def get_youtube_transcript(youtube_url: str) -> str:
    """Fetch a YouTube transcript given a URL or video id.

    Args:
        youtube_url: YouTube URL or 11-character video id to fetch transcripts for.
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound

        video_id = None
        if "youtube" in youtube_url or "youtu.be" in youtube_url:
            match = re.search(r"(?:v=|youtu.be/)([\w-]{11})", youtube_url)
            if match:
                video_id = match.group(1)
        else:
            if re.fullmatch(r"[\w-]{11}", youtube_url.strip()):
                video_id = youtube_url.strip()

        if not video_id:
            return "Could not extract a valid YouTube video ID."

        ytt_api = YouTubeTranscriptApi()
        fetched_transcript = ytt_api.fetch(video_id)
        transcript_text = " ".join([snippet.text for snippet in fetched_transcript])
        return transcript_text
    except ImportError:
        return "Install 'youtube-transcript-api' to use this tool: pip install youtube-transcript-api."
    except TranscriptsDisabled:
        return "Transcripts are disabled for this video."
    except NoTranscriptFound:
        return "No transcript found for this video."
    except Exception as exc:
        return f"Error fetching transcript: {str(exc)}"


@tool
def visit_webpage(url: str) -> str:
    """Fetch a webpage and return its content as Markdown.

    Args:
        url: Webpage URL to fetch and convert to Markdown.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')
        content = soup.get_text(separator="\n", strip=True)
        markdown_content = markdownify(content)
        markdown_content = re.sub(r'\n+', '\n', markdown_content)
        markdown_content = re.sub(r'\s+', ' ', markdown_content)
        markdown_content = markdown_content.strip()
        return markdown_content

    except RequestException as exc:
        return f"Error fetching the webpage: {str(exc)}"
    except Exception as exc:
        return f"An unexpected error occurred: {str(exc)}"


class UnifiedMultimodalTool(Tool):
    name = "multimodal_processor"
    description = MULTIMODAL_TOOL_DESCRIPTION
    inputs = {
        "file_path": {
            "type": "string",
            "description": "Path to the media file (audio, video, or image) to process",
        },
        "task": {
            "type": "string",
            "description": "Processing task: analyze, transcribe, extract, caption, summarize, or search",
            "nullable": True,
        },
        "modality": {
            "type": "string",
            "description": "Force specific modality: auto, audio, video, or image",
            "nullable": True,
        },
        "additional_context": {
            "type": "string",
            "description": "Extra instructions for processing",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, provider: str, model_name: str, api_key: str):
        super().__init__()
        self.provider = provider
        self.model_name = model_name
        self.transcription_model = self._select_transcription_model()

        if provider == "gemini":
            self.client = genai.Client(api_key=api_key, http_options={'api_version': 'v1alpha'})
        elif provider == "openai":
            self.client = OpenAI(api_key=api_key)
        elif provider == "openrouter":
            # OpenRouter exposes an OpenAI-compatible chat completions API.
            # Audio transcription is not available on OpenRouter; images are handled
            # via standard vision messages (base64 data URLs).
            self.client = OpenAI(
                api_key=api_key,
                base_url=_OPENROUTER_BASE_URL,
            )
        else:
            raise ValueError(f"Unsupported provider for multimodal tool: {provider}")
        mimetypes.init()

    def forward(
        self,
        file_path: str,
        task: str = "analyze",
        modality: str = "auto",
        additional_context: str = "",
    ) -> str:
        """Process multimedia files with a unified interface."""
        try:
            if modality == "auto":
                modality = self._detect_modality(file_path)

            if modality in ['pdf', 'unsupported']:
                return "Error: PDF files are not supported by this tool. Please use Docling for PDF processing."

            prompt = self._generate_prompt(task, modality, additional_context)

            if self.provider == "openai":
                if modality == "image":
                    return self._process_openai_image(file_path, prompt)
                if modality in {"audio", "video"}:
                    return self._process_openai_audio(file_path)
                return f"Unsupported modality for OpenAI: {modality}"

            if self.provider == "openrouter":
                if modality == "image":
                    return self._process_openrouter_image(file_path, prompt)
                # OpenRouter has no transcription endpoint; fall back to a helpful message.
                if modality in {"audio", "video"}:
                    return (
                        "Audio/video transcription is not supported via OpenRouter. "
                        "Use the get_youtube_transcript tool for YouTube videos, or switch to "
                        "provider='openai' or provider='gemini' for direct audio/video processing."
                    )
                return f"Unsupported modality for OpenRouter: {modality}"

            # Gemini path
            file_size = os.path.getsize(file_path)
            use_files_api = file_size > 20 * 1024 * 1024

            if use_files_api:
                return self._process_with_files_api(file_path, prompt, modality)
            return self._process_inline(file_path, prompt, modality)

        except Exception as exc:
            return f"Error processing {modality} file: {str(exc)}"

    def _detect_modality(self, file_path: str) -> str:
        mime_type, _ = mimetypes.guess_type(file_path)

        if mime_type:
            if mime_type.startswith('audio/'):
                return 'audio'
            if mime_type.startswith('video/'):
                return 'video'
            if mime_type.startswith('image/'):
                return 'image'
            if mime_type == 'application/pdf':
                return 'unsupported'

        ext = file_path.lower().split('.')[-1]
        audio_exts = {'mp3', 'wav', 'm4a', 'ogg', 'flac', 'aac', 'wma'}
        video_exts = {'mp4', 'avi', 'mov', 'mkv', 'webm', 'flv', 'wmv', '3gp'}
        image_exts = {'jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp', 'tiff', 'svg'}

        if ext in audio_exts:
            return 'audio'
        if ext in video_exts:
            return 'video'
        if ext in image_exts:
            return 'image'
        if ext == 'pdf':
            return 'unsupported'
        return 'unknown'

    def _get_mime_type(self, file_path: str) -> str:
        mime_type, _ = mimetypes.guess_type(file_path)

        if mime_type:
            return mime_type

        ext = file_path.lower().split('.')[-1]
        fallback_mappings = {
            'm4a': 'audio/mp4',
            'mkv': 'video/x-matroska',
            'webm': 'video/webm',
            'flac': 'audio/flac',
            'ogg': 'audio/ogg',
            'webp': 'image/webp',
            'pdf': 'application/pdf',
        }
        return fallback_mappings.get(ext, 'application/octet-stream')

    def _generate_prompt(self, task: str, modality: str, context: str) -> str:
        base_prompt = MULTIMODAL_TASK_PROMPTS.get(task, MULTIMODAL_TASK_PROMPTS['analyze']).get(
            modality,
            'Analyze this media file.',
        )
        if context:
            base_prompt += f" Additional context: {context}"
        return base_prompt

    def _get_media_resolution(self, modality: str) -> Optional[Dict[str, str]]:
        if modality == 'image':
            return {"level": "media_resolution_high"}
        if modality == 'video':
            return {"level": "media_resolution_low"}
        return None

    # ------------------------------------------------------------------ Gemini

    def _process_with_files_api(self, file_path: str, prompt: str, modality: str) -> str:
        uploaded_file = self.client.files.upload(file=file_path)

        resolution = self._get_media_resolution(modality)

        if resolution:
            contents = [
                types.Content(
                    parts=[
                        types.Part(text=prompt),
                        types.Part(
                            file_data=types.FileData(
                                file_uri=uploaded_file.uri,
                                mime_type=uploaded_file.mime_type,
                            ),
                            media_resolution=resolution,
                        ),
                    ]
                )
            ]
        else:
            contents = [prompt, uploaded_file]

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
        )
        return response.text

    def _process_inline(self, file_path: str, prompt: str, modality: str) -> str:
        with open(file_path, "rb") as handle:
            file_data = handle.read()

        mime_type = self._get_mime_type(file_path)
        resolution = self._get_media_resolution(modality)

        if resolution:
            contents = [
                types.Content(
                    parts=[
                        types.Part(text=prompt),
                        types.Part(
                            inline_data=types.Blob(
                                data=file_data,
                                mime_type=mime_type,
                            ),
                            media_resolution=resolution,
                        ),
                    ]
                )
            ]
        else:
            contents = [
                prompt,
                types.Part.from_bytes(
                    data=file_data,
                    mime_type=mime_type,
                ),
            ]

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
        )
        return response.text

    # ------------------------------------------------------------------ OpenAI

    def _select_transcription_model(self) -> str:
        env_model = os.environ.get("OPENAI_TRANSCRIBE_MODEL")
        if env_model:
            return env_model
        return "gpt-4o-mini-transcribe"

    def _process_openai_image(self, file_path: str, prompt: str) -> str:
        with open(file_path, "rb") as handle:
            result = self.client.files.create(
                file=handle,
                purpose="vision",
            )
        image_part = {"type": "input_image", "file_id": result.id}
        response = self.client.responses.create(
            model=self.model_name,
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    image_part,
                ],
            }],
        )
        return (response.output_text or "").strip()

    def _process_openai_audio(self, file_path: str) -> str:
        ext = os.path.splitext(file_path)[1].lower()
        allowed_exts = {".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm"}
        if ext not in allowed_exts:
            return "Unsupported file type for OpenAI transcription."

        file_size = os.path.getsize(file_path)
        if file_size > 25 * 1024 * 1024:
            return "OpenAI transcription file size limit is 25 MB."

        with open(file_path, "rb") as handle:
            transcription = self.client.audio.transcriptions.create(
                model=self.transcription_model,
                file=handle,
            )

        transcript_text = getattr(transcription, "text", None)
        if not transcript_text:
            transcript_text = str(transcription)

        return transcript_text.strip()

    # --------------------------------------------------------------- OpenRouter

    def _process_openrouter_image(self, file_path: str, prompt: str) -> str:
        """Send an image to an OpenRouter vision model via base64 data URL."""
        mime_type = self._get_mime_type(file_path)
        with open(file_path, "rb") as handle:
            image_b64 = base64.b64encode(handle.read()).decode("utf-8")

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{image_b64}"},
                    },
                ],
            }],
        )
        return (response.choices[0].message.content or "").strip()

    # ------------------------------------------------------------------ Utility

    def get_file_info(self, file_path: str) -> Dict[str, str]:
        mime_type, encoding = mimetypes.guess_type(file_path)
        modality = self._detect_modality(file_path)
        file_size = os.path.getsize(file_path)

        return {
            'file_path': file_path,
            'mime_type': mime_type or 'unknown',
            'encoding': encoding or 'none',
            'modality': modality,
            'file_size': f"{file_size:,} bytes",
            'processing_method': 'Files API' if file_size > 20 * 1024 * 1024 else 'Inline',
        }
