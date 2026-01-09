"""Response formatting helpers for the LlamaIndex agent."""

import logging

from llama_index.core.program import LLMTextCompletionProgram
from pydantic import BaseModel, Field

from . import models
from .prompts import FORMAT_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)


class StructuredResponse(BaseModel):
    """Structured answer format with validation."""
    reasoning: str = Field(
        description="Step-by-step reasoning process used to arrive at the answer",
    )
    final_answer: str = Field(
        description="Exact answer in concise format",
    )
    confidence: float = Field(
        description="Confidence score between 0 and 1",
        ge=0.0,
        le=1.0,
        default=0.8,
    )


def llm_reformat(response: str, question: str) -> str:
    """Extract a concise final answer from an LLM response."""
    try:
        program = LLMTextCompletionProgram.from_defaults(
            output_cls=StructuredResponse,
            prompt_template_str=FORMAT_PROMPT_TEMPLATE,
            llm=models.proj_llm,
            verbose=True,
        )

        response_obj = program(
            query_str=question,
            context=response,
        )

        logger.info("Structured response - Confidence: %.2f", response_obj.confidence)
        logger.info("Reasoning: %s...", response_obj.reasoning[:100])

        return response_obj.final_answer

    except Exception as exc:
        logger.exception("Pydantic parsing failed: %s", exc)
        return response.strip()
