"""Formatting helpers for smolagents responses."""

import json
import logging
import os

from pydantic import BaseModel, Field

from google import genai
from openai import OpenAI

from .models import get_format_config
from .prompts import FORMAT_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)

_GENAI_CLIENT = None
_OPENAI_CLIENT = None
_OPENROUTER_CLIENT = None

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class ExtractedAnswer(BaseModel):
    """Structured format for extracted answer."""
    reasoning: str = Field(description="Brief reasoning process")
    final_answer: str = Field(description="The exact final answer extracted from the response")


def llm_reformat(response: str, question: str) -> str:
    """Extract the final answer from a response using an LLM with structured outputs."""
    format_prompt = FORMAT_PROMPT_TEMPLATE.format(question=question, response=response)

    try:
        provider, model_name = get_format_config()
        model_name = model_name or "gemini-3-pro-preview"

        if provider == "openai":
            global _OPENAI_CLIENT
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                return response
            if _OPENAI_CLIENT is None:
                _OPENAI_CLIENT = OpenAI(api_key=api_key)

            completion = _OPENAI_CLIENT.beta.chat.completions.parse(
                model=model_name,
                messages=[{"role": "user", "content": format_prompt}],
                response_format=ExtractedAnswer,
            )
            if completion.choices[0].message.parsed:
                return completion.choices[0].message.parsed.final_answer
            return response

        if provider == "openrouter":
            global _OPENROUTER_CLIENT
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                return response
            if _OPENROUTER_CLIENT is None:
                _OPENROUTER_CLIENT = OpenAI(
                    api_key=api_key,
                    base_url=_OPENROUTER_BASE_URL,
                )

            # OpenRouter supports structured outputs via the same OpenAI-compatible endpoint.
            completion = _OPENROUTER_CLIENT.beta.chat.completions.parse(
                model=model_name,
                messages=[{"role": "user", "content": format_prompt}],
                response_format=ExtractedAnswer,
            )
            if completion.choices[0].message.parsed:
                return completion.choices[0].message.parsed.final_answer
            return response

        # Gemini path (default)
        global _GENAI_CLIENT
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            return response
        if _GENAI_CLIENT is None:
            _GENAI_CLIENT = genai.Client(api_key=api_key, http_options={'api_version': 'v1alpha'})

        formatting_response = _GENAI_CLIENT.models.generate_content(
            model=model_name,
            contents=[format_prompt],
            config={
                "response_mime_type": "application/json",
                "response_json_schema": ExtractedAnswer.model_json_schema(),
            },
        )
        parsed = json.loads(formatting_response.text)
        return parsed.get("final_answer", response)

    except Exception as exc:
        logger.warning("LLM reformatting failed: %s", exc)
        return response
