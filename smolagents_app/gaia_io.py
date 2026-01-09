"""Helpers for GAIA task files and response parsing."""

import logging
import mimetypes
import os
import re
from typing import Optional

import requests

from .prompts import GAIA_MEDIA_CONTEXT_TEMPLATE, GAIA_DOCUMENT_CONTEXT_TEMPLATE

logger = logging.getLogger(__name__)


def gaia_file_to_context(file_path: str) -> str:
    """Convert a GAIA task file into prompt context."""
    if not file_path:
        return ""

    mime_type, _ = mimetypes.guess_type(file_path)
    mime_type = mime_type or ""
    ext = os.path.splitext(file_path)[1].lower()
    filename = os.path.basename(file_path)

    if mime_type.startswith("image") or ext in {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}:
        return GAIA_MEDIA_CONTEXT_TEMPLATE.format(file_path=file_path, modality="image")

    if mime_type.startswith("video") or ext in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
        return GAIA_MEDIA_CONTEXT_TEMPLATE.format(file_path=file_path, modality="video")

    if mime_type.startswith("audio") or ext in {".mp3", ".wav", ".m4a"}:
        return GAIA_MEDIA_CONTEXT_TEMPLATE.format(file_path=file_path, modality="audio")

    try:
        from llama_index.readers.docling import DoclingReader

        docling_reader = DoclingReader(
            export_type=DoclingReader.ExportType.MARKDOWN,
        )
        documents = docling_reader.load_data(file_path)
        content = "\n\n".join([doc.text for doc in documents if doc.text])
        if content:
            return GAIA_DOCUMENT_CONTEXT_TEMPLATE.format(filename=filename, content=content)
    except Exception as exc:
        logger.exception("DoclingReader failed for %s: %s", filename, exc)

    return ""


def download_gaia_file(task_id: str, api_url: str = "https://agents-course-unit4-scoring.hf.space") -> Optional[str]:
    """Download a GAIA task file and return its local path."""
    try:
        response = requests.get(f"{api_url}/files/{task_id}", timeout=30)
        response.raise_for_status()

        content_disp = response.headers.get("content-disposition", "")
        match = re.search(r'filename="(.+)"', content_disp)
        if match:
            filename = match.group(1)
        else:
            raise ValueError("Filename not found in response headers")

        with open(filename, 'wb') as handle:
            handle.write(response.content)

        logger.info("Downloaded file saved as %s", filename)
        return os.path.abspath(filename)

    except Exception as exc:
        logger.exception("Failed to download file for task %s: %s", task_id, exc)
        return None


def extract_final_answer(response: str) -> str:
    """Extract the final answer from an agent response."""
    if "FINAL ANSWER:" in response:
        return response.split("FINAL ANSWER:")[-1].strip()

    lines = [line.strip() for line in response.split("\n") if line.strip()]
    return lines[-1] if lines else response
