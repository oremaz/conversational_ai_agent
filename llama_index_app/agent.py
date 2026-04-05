"""LlamaIndex multi-agent conversational runtime."""

import asyncio
import logging
import os
import sys
from typing import List, Optional

import weave
from llama_index.core.agent.workflow import ReActAgent, AgentStream
from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import Context
from llama_index.core.agent.workflow import AgentWorkflow

from . import models
from .formatting import llm_reformat
from .model_wrappers.utils import offload_image_models, offload_rag_models
from .prompts import (
    EXTERNAL_KNOWLEDGE_SYSTEM_PROMPT,
    CODE_EXECUTION_SYSTEM_PROMPT,
    IMAGE_ANALYSIS_SYSTEM_PROMPT,
    MEDIA_ANALYSIS_SYSTEM_PROMPT,
    IMG_GENERATION_SYSTEM_PROMPT_GEMINI,
    IMG_GENERATION_SYSTEM_PROMPT_OPENAI,
    IMG_GENERATION_SYSTEM_PROMPT_QWEN,
    IMG_EDITING_SYSTEM_PROMPT_GEMINI,
    IMG_EDITING_SYSTEM_PROMPT_OPENAI,
    IMG_EDITING_SYSTEM_PROMPT_QWEN,
    build_context_prompt,
)
from .tools import execute_python_code, make_enhanced_web_search_tool
from .utils.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ConversationalAgent:
    def __init__(
        self,
        use_api_mode: Optional[bool] = None,
        model_suite: str = "qwen",
        local_model_id: Optional[str] = None,
        session_id: Optional[str] = None,
        media_analysis_enabled: bool = False,
        code_execution_enabled: bool = True,
        use_specialized_code_model: bool = False,
        img_generation_enabled: bool = False,
        img_editing_enabled: bool = False,
        use_qwen_vl_for_images: bool = True,
        use_main_model_for_code_agent: bool = False,
        qwen_vl_model_id: Optional[str] = None,
        rag_provider: str = "jina",
    ):
        logger.info("Initializing Conversational Agent...")

        if use_api_mode is None:
            use_api_mode = models.USE_API_MODE
        if not model_suite:
            model_suite = models.LOCAL_MODEL_SUITE

        self.use_api_mode = bool(use_api_mode)
        self.session_id = session_id

        models.configure_models(
            use_api_mode=use_api_mode,
            model_suite=model_suite,
            local_model_id=local_model_id,
            session_id=self.session_id,
            use_qwen_vl_for_images=use_qwen_vl_for_images,
            use_main_model_for_code_agent=use_main_model_for_code_agent,
            qwen_vl_model_id=qwen_vl_model_id,
            media_analysis_enabled=media_analysis_enabled,
            img_generation_enabled=img_generation_enabled,
            img_editing_enabled=img_editing_enabled,
            rag_provider=rag_provider,
        )

        hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        if not hf_token:
            logger.warning("Warning: HUGGINGFACEHUB_API_TOKEN not found, some features may not work")

        if self.use_api_mode:
            self.vector_store_manager = None
        else:
            self.vector_store_manager = VectorStoreManager(
                conversations_dir="./chroma_db/conversations",
                embedder_provider=rag_provider,
            )

        self.web_tool = make_enhanced_web_search_tool()
        self.web_function_tool = FunctionTool.from_defaults(
            fn=self.web_tool,
            name="enhanced_web_search",
        )

        self.exec_function_tool = FunctionTool.from_defaults(
            fn=execute_python_code,
            name="execute_python_code",
        )

        agents_list = []

        self.external_knowledge_agent = ReActAgent(
            tools=[self.web_function_tool],
            llm=models.proj_llm,
            max_steps=8,
            system_prompt=EXTERNAL_KNOWLEDGE_SYSTEM_PROMPT,
            verbose=True,
            name="external_knowledge_agent",
            description="Handles external knowledge retrieval, web searches, and document processing",
        )
        agents_list.append(self.external_knowledge_agent)

        use_api = use_api_mode

        if code_execution_enabled:
            if use_api:
                if model_suite == "openai" and use_specialized_code_model:
                    from .custom_models import get_or_create_openai_llm
                    code_exec_llm = get_or_create_openai_llm(
                        model_name="gpt-5.2-codex",
                        session_id=self.session_id,
                    )
                    logger.info("Using OpenAI gpt-5.2-codex for code execution")
                else:
                    code_exec_llm = models.proj_llm
                    logger.info("Using main model (%s) for code execution", model_suite)
            else:
                code_exec_llm = models.code_llm

            self.code_agent = ReActAgent(
                tools=[self.exec_function_tool],
                llm=code_exec_llm,
                max_steps=6,
                system_prompt=CODE_EXECUTION_SYSTEM_PROMPT,
                verbose=True,
                name="code_execution_agent",
                description="Executes Python code, performs calculations, data analysis, and mathematical operations",
            )
            agents_list.append(self.code_agent)
            logger.info("Code execution agent activated")

        if not use_api and models.img_analysis_llm is not None:
            self.img_analysis_agent = ReActAgent(
                tools=[],
                llm=models.img_analysis_llm,
                max_steps=4,
                system_prompt=IMAGE_ANALYSIS_SYSTEM_PROMPT,
                verbose=True,
                name="img_analysis_agent",
                description="Analyzes visual content and images using Qwen-VL",
            )
            agents_list.append(self.img_analysis_agent)
            logger.info("Image analysis agent activated (Qwen-VL)")

        if not use_api and media_analysis_enabled and models.media_analysis_llm is not None:
            self.med_analysis_agent = ReActAgent(
                tools=[],
                llm=models.media_analysis_llm,
                max_steps=4,
                system_prompt=MEDIA_ANALYSIS_SYSTEM_PROMPT,
                verbose=True,
                name="med_analysis_agent",
                description="Analyzes audio and video content using Qwen-Omni",
            )
            agents_list.append(self.med_analysis_agent)
            logger.info("Media analysis agent activated (Qwen-Omni)")

        if img_generation_enabled:
            if use_api:
                if model_suite == "gemini":
                    from .custom_models import get_or_create_gemini_llm
                    img_gen_llm = get_or_create_gemini_llm(model_name="gemini-3-pro-image-preview")
                    self.img_generation_agent = ReActAgent(
                        tools=[],
                        llm=img_gen_llm,
                        max_steps=3,
                        system_prompt=IMG_GENERATION_SYSTEM_PROMPT_GEMINI,
                        verbose=True,
                        name="img_generation_agent",
                        description="Generates images from text descriptions using Gemini",
                    )
                    agents_list.append(self.img_generation_agent)
                    logger.info("Image generation agent activated (Gemini gemini-3-pro-image-preview)")
                elif model_suite == "openai":
                    self.img_generation_agent = ReActAgent(
                        tools=[],
                        llm=models.proj_llm,
                        max_steps=3,
                        system_prompt=IMG_GENERATION_SYSTEM_PROMPT_OPENAI,
                        verbose=True,
                        name="img_generation_agent",
                        description="Generates images from text descriptions using OpenAI",
                    )
                    self.img_generation_agent._use_image_tool = True
                    agents_list.append(self.img_generation_agent)
                    logger.info("Image generation agent activated (OpenAI with image_generation tool)")
                else:
                    logger.warning("Image generation not supported for this provider, skipping")
            else:
                if models.img_gen_model is not None:
                    self.img_generation_agent = ReActAgent(
                        tools=[],
                        llm=models.proj_llm,
                        max_steps=3,
                        system_prompt=IMG_GENERATION_SYSTEM_PROMPT_QWEN,
                        verbose=True,
                        name="img_generation_agent",
                        description="Generates images from text descriptions using Qwen",
                    )
                    self.img_generation_agent._generator = models.img_gen_model
                    agents_list.append(self.img_generation_agent)
                    logger.info("Image generation agent activated (Qwen)")

        if img_editing_enabled:
            if use_api:
                if model_suite == "gemini":
                    from .custom_models import get_or_create_gemini_llm
                    img_edit_llm = get_or_create_gemini_llm(model_name="gemini-3-pro-image-preview")
                    self.img_editing_agent = ReActAgent(
                        tools=[],
                        llm=img_edit_llm,
                        max_steps=3,
                        system_prompt=IMG_EDITING_SYSTEM_PROMPT_GEMINI,
                        verbose=True,
                        name="img_editing_agent",
                        description="Edits and modifies existing images using Gemini",
                    )
                    agents_list.append(self.img_editing_agent)
                    logger.info("Image editing agent activated (Gemini gemini-3-pro-image-preview)")
                elif model_suite == "openai":
                    self.img_editing_agent = ReActAgent(
                        tools=[],
                        llm=models.proj_llm,
                        max_steps=3,
                        system_prompt=IMG_EDITING_SYSTEM_PROMPT_OPENAI,
                        verbose=True,
                        name="img_editing_agent",
                        description="Edits and modifies existing images using OpenAI",
                    )
                    self.img_editing_agent._use_image_tool = True
                    agents_list.append(self.img_editing_agent)
                    logger.info("Image editing agent activated (OpenAI)")
                else:
                    logger.warning("Image editing not supported for this provider, skipping")
            else:
                if models.img_edit_model is not None:
                    self.img_editing_agent = ReActAgent(
                        tools=[],
                        llm=models.proj_llm,
                        max_steps=3,
                        system_prompt=IMG_EDITING_SYSTEM_PROMPT_QWEN,
                        verbose=True,
                        name="img_editing_agent",
                        description="Edits and modifies existing images using Qwen",
                    )
                    self.img_editing_agent._editor = models.img_edit_model
                    agents_list.append(self.img_editing_agent)
                    logger.info("Image editing agent activated (Qwen)")

        self.coordinator = AgentWorkflow(
            agents=agents_list,
            root_agent="external_knowledge_agent",
        )

        logger.info("Coordinator initialized with %d agents", len(agents_list))

    @weave.op
    def _record_steps(self, deltas: List[str]) -> int:
        """Record streamed deltas as a single Weave op."""
        return len(deltas)

    @weave.op
    def run(self, query: str, max_steps: Optional[int] = None) -> str:
        """Run the agent on a user query."""
        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
                response = loop.run_until_complete(self.process_query(query))
            else:
                response = asyncio.run(self.process_query(query))

            return str(response)
        except Exception as exc:
            error_msg = f"Error during agent execution: {exc}"
            logger.exception(error_msg)
            return error_msg

    async def process_query(self, query: str) -> str:
        """Process a query with knowledge base integration."""
        if not self.use_api_mode:
            try:
                offload_rag_models()
            except Exception as exc:
                logger.debug("RAG offload skipped: %s", exc)

        context_prompt = build_context_prompt(query)

        try:
            ctx = Context(self.coordinator)
            logger.info("=== AGENT REASONING STEPS ===")
            if self.vector_store_manager:
                stats = self.vector_store_manager.get_stats()
                logger.info("Cached sources available: %s", stats.get("library_sources", 0))

            handler = self.coordinator.run(ctx=ctx, user_msg=context_prompt)
            full_response = ""
            step_deltas: List[str] = []

            current_agent_name: Optional[str] = None
            image_agents = {"img_generation_agent", "img_editing_agent"}

            async for event in handler.stream_events():
                event_agent = getattr(event, "current_agent_name", None)
                if event_agent and event_agent != current_agent_name:
                    if (
                        not self.use_api_mode
                        and current_agent_name in image_agents
                        and event_agent not in image_agents
                    ):
                        try:
                            offload_image_models()
                        except Exception as exc:
                            logger.debug("Image offload skipped: %s", exc)
                    current_agent_name = event_agent

                if isinstance(event, AgentStream):
                    sys.stdout.write(event.delta)
                    sys.stdout.flush()
                    full_response += event.delta
                    step_deltas.append(event.delta)

            final_response = await handler
            logger.info("\n=== END REASONING ===")

            final_answer = llm_reformat(str(final_response), query)
            self._record_steps(step_deltas)

            return final_answer
        except Exception as exc:
            error_msg = f"Error processing question: {str(exc)}"
            logger.exception(error_msg)
            return error_msg

    def get_knowledge_base_stats(self):
        """Return statistics about the current knowledge base."""
        if self.vector_store_manager:
            return self.vector_store_manager.get_stats()
        return {}
