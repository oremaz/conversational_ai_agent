"""smolagents runner for tool-heavy tasks."""

import logging
import os
import sys
import warnings
from typing import Any, Dict, List, Optional

from smolagents import CodeAgent

# Disable weave instrumentation for this module to avoid serialization issues with CodeAgent
os.environ["WEAVE_DISABLED"] = "true"

from .gaia_io import download_gaia_file, extract_final_answer, gaia_file_to_context
from .models import initialize_llm_model
from .observability import setup_langfuse_observability
from .prompts import GAIA_SYSTEM_PROMPT, GAIA_CONTEXT_QUESTION_TEMPLATE
from .tools import FinalAnswerTool, UnifiedMultimodalTool, visit_webpage
from .web_search import WebSearchTool
from .utils.mcp_connectors import load_multiple_mcp_servers

logger = logging.getLogger(__name__)

# Send logs to stdout so they show up in notebooks/terminals.
root_logger = logging.getLogger()
if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    handler.setLevel(logging.INFO)
    root_logger.addHandler(handler)
root_logger.setLevel(logging.INFO)
logger.setLevel(logging.INFO)

# Map each provider to the settings attribute that holds its API key.
_PROVIDER_API_KEY_ATTR = {
    "gemini": "google_api_key",
    "openai": "openai_api_key",
    "openrouter": "openrouter_api_key",
}

class GAIAAgent:
    """GAIA agent using smolagents with Langfuse observability."""

    def __init__(
        self,
        user_id: str = None,
        session_id: str = None,
        provider: str = "gemini",
        model_name: Optional[str] = None,
        mcp_servers: Optional[List[str]] = None,
    ):
        self.tracer = setup_langfuse_observability()

        self.model = initialize_llm_model(provider=provider, model_name=model_name, temperature=0.0)

        self.user_id = user_id or "gaia-user"
        self.session_id = session_id or "gaia-session"

        from config import settings as _app_settings

        key_attr = _PROVIDER_API_KEY_ATTR.get(provider, "openai_api_key")
        api_key = getattr(_app_settings, key_attr, None)

        if api_key:
            self.multimodal_tool = UnifiedMultimodalTool(
                provider=provider,
                model_name=model_name or self.model.model_id,
                api_key=api_key,
            )
        else:
            self.multimodal_tool = None
            logger.warning("Multimodal tool disabled; missing API key for provider %s.", provider)

        self.system_prompt = GAIA_SYSTEM_PROMPT

        self.mcp_tools = []
        if mcp_servers:
            mcp_collections = load_multiple_mcp_servers(mcp_servers, trust_remote_code=True)
            for collection in mcp_collections:
                self.mcp_tools.extend(collection.tools)
            logger.info("Loaded %s MCP tools from %s servers", len(self.mcp_tools), len(mcp_collections))

        self.agent = None
        self._create_agent()

        self.langfuse = None
        self.last_trace_id = None
        try:
            from langfuse import Langfuse
            self.langfuse = Langfuse()
        except Exception as exc:
            logger.warning("Langfuse client unavailable: %s", exc)

    def _create_agent(self):
        base_tools = [
            WebSearchTool(),
            visit_webpage,
            FinalAnswerTool(),
        ]

        if self.multimodal_tool is not None:
            base_tools.append(self.multimodal_tool)

        all_tools = base_tools + self.mcp_tools

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            weave_logger = logging.getLogger("weave")
            original_level = weave_logger.level
            weave_logger.setLevel(logging.ERROR)

            try:
                self.agent = CodeAgent(
                    tools=all_tools,
                    model=self.model,
                    additional_authorized_imports=[
                        "math", "statistics", "itertools", "datetime",
                        "random", "re", "json", "csv", "os", "sys",
                        "collections", "functools", "pathlib",
                        "numpy", "pandas", "matplotlib", "seaborn",
                        "scipy", "sklearn", "requests", "bs4",
                        "PIL", "yaml", "tqdm",
                    ],
                    max_steps=6,
                )
            finally:
                weave_logger.setLevel(original_level)

    def _log_run_trace(self, prompt: str, response: str) -> Optional[str]:
        if not self.langfuse:
            return None

        metadata = {
            "provider": getattr(self.model, "provider", None),
            "model_id": getattr(self.model, "model_id", None),
            "user_id": self.user_id,
            "session_id": self.session_id,
        }

        try:
            with self.langfuse.start_as_current_observation(
                as_type="span",
                name="chat",
            ) as observation:
                observation.update(
                    input=prompt,
                    output=response,
                    metadata=metadata,
                )

                trace_id = observation.trace_id

            self.langfuse.flush()
            return trace_id
        except Exception as exc:
            logger.exception("Langfuse trace creation failed: %s", exc)
            return None

    def run(self, query: str, max_steps: Optional[int] = None, reset_documents: bool = False) -> tuple[str, Optional[str]]:
        """Run the agent on a user query."""
        try:
            full_query = query

            if max_steps:
                original_max_steps = self.agent.max_steps
                self.agent.max_steps = max_steps

            response = self.agent.run(full_query)

            if max_steps:
                self.agent.max_steps = original_max_steps

            response_text = str(response)
            trace_id = self._log_run_trace(full_query, response_text)
            self.last_trace_id = trace_id
            return response_text, trace_id

        except Exception as exc:
            error_msg = f"Error during agent execution: {exc}"
            logger.exception("Error during agent execution: %s", exc)
            import traceback
            traceback.print_exc()
            self.last_trace_id = None
            return error_msg, None

    def solve_gaia_question(self, question_dict: Dict[str, Any]) -> str:
        """Solve a GAIA benchmark question (legacy method for evaluation)."""
        question_text = question_dict.get("Question", "")
        task_id = question_dict.get("task_id")

        if task_id:
            file_path = download_gaia_file(task_id)
            if file_path:
                context = gaia_file_to_context(file_path)
                if context:
                    question_text = GAIA_CONTEXT_QUESTION_TEMPLATE.format(
                        context=context,
                        question=question_text,
                    )

        response, _ = self.run(question_text)
        final_answer = extract_final_answer(response)

        return final_answer

    def add_user_feedback(self, trace_id: str, feedback_score: int, comment: str = None):
        """Add user feedback to a specific trace."""
        if not self.langfuse or not trace_id:
            logger.warning("Langfuse client not available or trace_id is None")
            return

        try:
            self.langfuse.score(
                trace_id=trace_id,
                name="user-feedback",
                value=feedback_score,
                comment=comment,
            )
            logger.info("User feedback added: %s/5", feedback_score)
        except Exception as exc:
            logger.exception("Error adding user feedback: %s", exc)
