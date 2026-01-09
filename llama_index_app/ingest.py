"""Ingestion utilities for prompt media processing."""

import json
import logging
import os
from typing import List

import weave
from llama_index.core import Document
from llama_index.core.schema import ImageDocument
from llama_index.readers.docling import DoclingReader
from llama_index.readers.file import PandasCSVReader, PandasExcelReader

from . import models
from .prompts import IMAGE_DESCRIPTION_PROMPT, MEDIA_DESCRIPTION_PROMPT

logger = logging.getLogger(__name__)


@weave.op
def read_and_parse_content(input_path: str) -> List[Document]:
    """Parse a file into LlamaIndex Documents.

    This handles Docling for PDFs and Office files, specialized readers for
    CSV/Excel/JSON/text, and multimodal LLM analysis for images and media.
    """
    if not os.path.exists(input_path):
        return [Document(text=f"Error: File not found at {input_path}")]

    file_extension = os.path.splitext(input_path)[1].lower()

    if file_extension == '.pdf':
        try:
            logger.info("Using DoclingReader for PDF file: %s", input_path)
            reader = DoclingReader(export_type=DoclingReader.ExportType.MARKDOWN)
            documents = reader.load_data(input_path)

            for doc in documents:
                doc.metadata["source"] = input_path
                doc.metadata["loader"] = "docling"
                doc.metadata["file_type"] = "pdf"

            logger.info("DoclingReader extracted %d documents from PDF", len(documents))
            return documents

        except Exception as exc:
            logger.exception("DoclingReader failed for PDF %s: %s", input_path, exc)
            return [Document(text=f"Error loading PDF with DoclingReader: {exc}")]

    if file_extension in ['.docx', '.doc', '.pptx', '.ppt', '.html', '.htm']:
        try:
            logger.info("Using DoclingReader for %s file", file_extension)
            reader = DoclingReader(export_type=DoclingReader.ExportType.MARKDOWN)
            documents = reader.load_data(input_path)

            for doc in documents:
                doc.metadata["source"] = input_path
                doc.metadata["loader"] = "docling"
                doc.metadata["file_type"] = file_extension[1:]

            logger.info("DoclingReader extracted %d documents", len(documents))
            return documents

        except Exception as exc:
            logger.exception("DoclingReader failed for %s: %s", input_path, exc)
            return [Document(text=f"Error loading file with DoclingReader: {exc}")]

    readers_map = {
        '.csv': PandasCSVReader(),
        '.xlsx': PandasExcelReader(),
    }

    if file_extension in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
        try:
            if not models.USE_API_MODE and models.img_analysis_llm is None:
                return [Document(text="Multimodal model not available for image processing")]
            image_llm = models.img_analysis_llm or models.IMAGE_CAPTION_LLM
            if image_llm is None:
                return [Document(text="Multimodal model not available for image processing")]

            logger.info("Using multimodal LLM to describe image: %s", input_path)

            img_doc = ImageDocument(image_path=input_path, metadata={"source": input_path, "type": "input_image"})
            resp = image_llm.complete(IMAGE_DESCRIPTION_PROMPT, image_documents=[img_doc])
            description = getattr(resp, "text", str(resp)).strip()

            return [Document(
                text=description,
                metadata={
                    "source": input_path,
                    "type": "image_description",
                    "path": input_path,
                },
            )]
        except Exception as exc:
            logger.exception("Multimodal image processing failed: %s", exc)
            return [Document(text=f"Error processing image with LLM: {exc}")]

    if file_extension in ['.mp3', '.mp4', '.wav', '.m4a']:
        try:
            media_llm = models.media_analysis_llm
            if models.USE_API_MODE and models.LOCAL_MODEL_SUITE == "gemini":
                media_llm = models.IMAGE_CAPTION_LLM
            elif models.USE_API_MODE and models.LOCAL_MODEL_SUITE == "openai":
                return [Document(text="OpenAI does not support audio/video inputs in this app.")]
            if media_llm is None:
                return [Document(text="Multimodal model not available for audio/video processing")]

            logger.info("Using multimodal LLM to process audio/video: %s", input_path)

            is_video = file_extension in ['.mp4']
            prompt = MEDIA_DESCRIPTION_PROMPT.format(modality="video" if is_video else "audio")

            media_doc = ImageDocument(image_path=input_path, metadata={"source": input_path, "type": "input_media"})
            resp = media_llm.complete(prompt, image_documents=[media_doc])
            description = getattr(resp, "text", str(resp)).strip()

            return [Document(
                text=description,
                metadata={
                    "source": input_path,
                    "type": "audio_video_description",
                    "path": input_path,
                },
            )]
        except Exception as exc:
            logger.exception("Multimodal audio/video processing failed: %s", exc)
            return [Document(text=f"Error processing audio/video with LLM: {exc}")]

    documents: List[Document] = []

    if file_extension == ".json":
        try:
            with open(input_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            content = json.dumps(data, ensure_ascii=True, indent=2, sort_keys=True)
            documents = [Document(text=content, metadata={"source": input_path})]
        except Exception as exc:
            return [Document(text=f"Error loading JSON: {exc}")]
    elif file_extension in readers_map:
        loader = readers_map[file_extension]
        try:
            documents = loader.load_data(input_path)
        except Exception as exc:
            return [Document(text=f"Error loading file with reader: {exc}")]
    else:
        try:
            with open(input_path, 'r', encoding='utf-8') as handle:
                content = handle.read()
            documents = [Document(text=content, metadata={"source": input_path})]
        except Exception as exc:
            return [Document(text=f"Error reading file as plain text: {exc}")]

    for doc in documents:
        doc.metadata["source"] = input_path

    return documents
