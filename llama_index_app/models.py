"""Model initialization and configuration for the LlamaIndex agent."""

import logging
import os
import sys
import warnings
from typing import Dict, Any, Optional

import weave
from llama_index.core import Settings

# Suppress a noisy third-party warning that does not affect runtime behavior.
warnings.filterwarnings(
    "ignore",
    message=r".*The 'validate_default' attribute with value True was provided to the `Field\(\)` function.*",
)
warnings.filterwarnings(
    "ignore",
    message=r".*Accessing `__path__` from `\.models\.zoedepth\.image_processing_zoedepth`.*",
)

from .custom_models import (
    get_or_create_jina_reranker,
    get_or_create_jina_embedder,
    get_or_create_qwen35_llm,
    get_or_create_gemma4_llm,
    get_or_create_devstral_llm,
    get_or_create_qwen_embedder,
    get_or_create_gpt_oss_llm,
    get_or_create_ministral_llm,
    get_or_create_gemini_llm,
    get_or_create_openai_llm,
    get_or_create_openrouter_llm,
    get_or_create_qwen3_omni_llm,
    get_or_create_gemini_embedder,
    get_or_create_openai_embedder,
    get_or_create_openrouter_embedder,
)

# Weave tracing is opt-in to avoid interactive wandb login prompts in local/dev runs.
_WEAVE_ENABLED = os.environ.get("WEAVE_ENABLED", "false").lower() in {"1", "true", "yes", "on"}
if _WEAVE_ENABLED:
    try:
        weave.init("conversational-ai-agent")
    except Exception:
        # Keep application startup non-blocking if tracing cannot initialize.
        logging.getLogger(__name__).exception("Weave initialization failed; continuing without tracing")

# Setup logging - force INFO logs to stdout so they are visible in notebooks/terminals
root_logger = logging.getLogger()
# Remove any existing StreamHandlers (they may be configured to stderr or filtered)
for handler in list(root_logger.handlers):
    if isinstance(handler, logging.StreamHandler):
        root_logger.removeHandler(handler)

# Create a StreamHandler that writes to stdout and set it to INFO
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
stdout_handler.setLevel(logging.INFO)
root_logger.addHandler(stdout_handler)
root_logger.setLevel(logging.INFO)
# Ensure basicConfig is set for libraries that check it
logging.basicConfig(level=logging.INFO)

# Make llama_index and related libraries more verbose for debugging (they will propagate to root handler)
logging.getLogger("llama_index").setLevel(logging.DEBUG)
logging.getLogger("llama_index.core.agent").setLevel(logging.DEBUG)
logging.getLogger("llama_index.llms").setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Use centralized config for defaults, but keep module-level mutables
# for backward compatibility with configure_models()
from config import settings as _app_settings

USE_API_MODE = _app_settings.use_api_mode
LOCAL_MODEL_SUITE = _app_settings.local_model_suite.value
RAG_PROVIDER = _app_settings.rag_provider.value

# Lazy-initialized globals (configured in configure_models)
embed_model = None
proj_llm = None
code_llm = None
IMAGE_CAPTION_LLM = None
img_analysis_llm = None
media_analysis_llm = None
img_gen_model = None
img_edit_model = None

_ACTIVE_MODEL_CONFIG: Dict[str, Any] = {}


@weave.op
def initialize_models(
    use_api_mode: bool = False,
    model_suite: str = "qwen",
    local_model_id: Optional[str] = None,
    api_model_name: Optional[str] = None,
    session_id: Optional[str] = None,
    use_qwen_vl_for_images: bool = True,
    use_main_model_for_code_agent: bool = False,
    qwen_vl_model_id: Optional[str] = None,
    media_analysis_enabled: bool = False,
    img_generation_enabled: bool = False,
    img_editing_enabled: bool = False,
    rag_provider: str = "jina",
):
    """Initialize models for API or local mode."""
    if use_api_mode:
        # API Mode - Using native multimodal API clients
        try:
            logger.info("Initializing models in API mode...")

            provider_order = []
            if model_suite in ("gemini", "openai", "openrouter"):
                provider_order.append(model_suite)
            for provider in ("gemini", "openai", "openrouter"):
                if provider not in provider_order:
                    provider_order.append(provider)

            for provider in provider_order:
                if provider == "gemini":
                    google_api_key = os.environ.get("GOOGLE_API_KEY")
                    if not google_api_key:
                        continue
                    logger.info("Using Gemini API with model: %s", api_model_name or "default")
                    proj_llm_instance = get_or_create_gemini_llm(
                        model_name=api_model_name,
                        session_id=session_id,
                    )
                    code_llm_instance = proj_llm_instance
                    embed_model_instance = get_or_create_gemini_embedder()
                    return embed_model_instance, proj_llm_instance, code_llm_instance

                if provider == "openai":
                    openai_api_key = os.environ.get("OPENAI_API_KEY")
                    if not openai_api_key:
                        continue
                    logger.info("Using OpenAI API with model: %s", api_model_name or "default")
                    proj_llm_instance = get_or_create_openai_llm(
                        model_name=api_model_name,
                        session_id=session_id,
                    )
                    code_llm_instance = proj_llm_instance
                    embed_model_instance = get_or_create_openai_embedder()
                    return embed_model_instance, proj_llm_instance, code_llm_instance

                if provider == "openrouter":
                    openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
                    if not openrouter_api_key:
                        continue
                    logger.info("Using OpenRouter API with model: %s", api_model_name or "default")
                    proj_llm_instance = get_or_create_openrouter_llm(
                        model_name=api_model_name,
                        session_id=session_id,
                    )
                    code_llm_instance = proj_llm_instance
                    embed_model_instance = get_or_create_openrouter_embedder()
                    return embed_model_instance, proj_llm_instance, code_llm_instance

            # No API keys found, fall back to local mode
            logger.warning("No API keys found. Falling back to local mode...")
            return initialize_models(
                use_api_mode=False,
                model_suite=model_suite,
                local_model_id=local_model_id,
                use_qwen_vl_for_images=use_qwen_vl_for_images,
                use_main_model_for_code_agent=use_main_model_for_code_agent,
                qwen_vl_model_id=qwen_vl_model_id,
                media_analysis_enabled=media_analysis_enabled,
                img_generation_enabled=img_generation_enabled,
                img_editing_enabled=img_editing_enabled,
                rag_provider=rag_provider,
            )

        except Exception as exc:
            logger.exception("Error initializing API mode: %s", exc)
            logger.info("Falling back to local mode...")
            return initialize_models(
                use_api_mode=False,
                model_suite=model_suite,
                local_model_id=local_model_id,
                use_qwen_vl_for_images=use_qwen_vl_for_images,
                use_main_model_for_code_agent=use_main_model_for_code_agent,
                qwen_vl_model_id=qwen_vl_model_id,
                media_analysis_enabled=media_analysis_enabled,
                img_generation_enabled=img_generation_enabled,
                img_editing_enabled=img_editing_enabled,
                rag_provider=rag_provider,
            )
    logger.info("Initializing models in local mode...")
    try:
        if rag_provider == "qwen":
            embed_model_instance = get_or_create_qwen_embedder()
        else:
            embed_model_instance = get_or_create_jina_embedder()

        img_analysis_llm_instance = None
        media_analysis_llm_instance = None
        img_gen_model_instance = None
        img_edit_model_instance = None

        if model_suite == "gemma4":
            model_id = local_model_id or "google/gemma-4-E4B-it"
            logger.info("Initializing Gemma 4 suite: %s", model_id)
            proj_llm_instance = get_or_create_gemma4_llm(model_name=model_id, device="auto")
            if use_main_model_for_code_agent:
                code_llm_instance = proj_llm_instance
                logger.info("Using Gemma 4 main model for code execution")
            else:
                code_llm_instance = get_or_create_devstral_llm()

            if media_analysis_enabled:
                logger.info("Initializing Qwen3-Omni-30B for media analysis")
                media_analysis_llm_instance = get_or_create_qwen3_omni_llm(
                    model_name="Qwen/Qwen3-Omni-30B-A3B-Instruct",
                    device="auto",
                )

            if img_generation_enabled or img_editing_enabled:
                try:
                    from .custom_models import DIFFUSERS_AVAILABLE
                    if DIFFUSERS_AVAILABLE:
                        from .custom_models import get_or_create_image_generator, get_or_create_image_editor
                        if img_generation_enabled:
                            img_gen_model_instance = get_or_create_image_generator()
                        if img_editing_enabled:
                            img_edit_model_instance = get_or_create_image_editor()
                except Exception as exc:
                    logger.warning("Image generation/editing initialization failed: %s", exc)
        elif model_suite == "ministral":
            model_id = local_model_id or "mistralai/Ministral-3-8B-Instruct-2512"
            logger.info("Initializing Ministral suite: %s", model_id)
            proj_llm_instance = get_or_create_ministral_llm(model_name=model_id, device="auto")
            if use_main_model_for_code_agent:
                code_llm_instance = proj_llm_instance
                logger.info("Using Ministral main model for code execution")
            else:
                code_llm_instance = get_or_create_devstral_llm()
            if media_analysis_enabled:
                logger.info("Initializing Qwen3-Omni-30B for media analysis")
                media_analysis_llm_instance = get_or_create_qwen3_omni_llm(
                    model_name="Qwen/Qwen3-Omni-30B-A3B-Instruct",
                    device="auto",
                )
            if img_generation_enabled or img_editing_enabled:
                try:
                    from .custom_models import DIFFUSERS_AVAILABLE
                    if DIFFUSERS_AVAILABLE:
                        from .custom_models import get_or_create_image_generator, get_or_create_image_editor
                        if img_generation_enabled:
                            img_gen_model_instance = get_or_create_image_generator()
                        if img_editing_enabled:
                            img_edit_model_instance = get_or_create_image_editor()
                except Exception as exc:
                    logger.warning("Image generation/editing initialization failed: %s", exc)
        elif model_suite == "gpt-oss":
            model_id = local_model_id or "openai/gpt-oss-20b"
            logger.info("Initializing GPT-OSS model: %s", model_id)
            proj_llm_instance = get_or_create_gpt_oss_llm(model_name=model_id, device="auto")

            if use_qwen_vl_for_images:
                vl_model = qwen_vl_model_id or "Qwen/Qwen3.5-4B"
                logger.info("Initializing Qwen3.5 for image analysis: %s", vl_model)
                img_analysis_llm_instance = get_or_create_qwen35_llm(
                    model_name=vl_model,
                    device="auto",
                )

            if media_analysis_enabled:
                logger.info("Initializing Qwen3-Omni-30B for media analysis")
                media_analysis_llm_instance = get_or_create_qwen3_omni_llm(
                    model_name="Qwen/Qwen3-Omni-30B-A3B-Instruct",
                    device="auto",
                )

            if img_generation_enabled or img_editing_enabled:
                try:
                    from .custom_models import DIFFUSERS_AVAILABLE
                    if DIFFUSERS_AVAILABLE:
                        from .custom_models import get_or_create_image_generator, get_or_create_image_editor
                        if img_generation_enabled:
                            logger.info("Initializing Qwen image generation model")
                            img_gen_model_instance = get_or_create_image_generator()
                        if img_editing_enabled:
                            logger.info("Initializing Qwen image editing model")
                            img_edit_model_instance = get_or_create_image_editor()
                except Exception as exc:
                    logger.warning("Image generation/editing initialization failed: %s", exc)

            if use_main_model_for_code_agent:
                code_llm_instance = proj_llm_instance
                logger.info("Using GPT-OSS for code execution")
            else:
                code_llm_instance = get_or_create_devstral_llm()
        else:
            model_id = local_model_id or "Qwen/Qwen3.5-35B-A3B"
            logger.info("Initializing Qwen3.5 model: %s", model_id)
            proj_llm_instance = get_or_create_qwen35_llm(model_name=model_id, device="auto")

            if media_analysis_enabled:
                logger.info("Initializing Qwen3-Omni-30B for media analysis")
                media_analysis_llm_instance = get_or_create_qwen3_omni_llm(
                    model_name="Qwen/Qwen3-Omni-30B-A3B-Instruct",
                    device="auto",
                )

            if img_generation_enabled or img_editing_enabled:
                try:
                    from .custom_models import DIFFUSERS_AVAILABLE
                    if DIFFUSERS_AVAILABLE:
                        from .custom_models import get_or_create_image_generator, get_or_create_image_editor
                        if img_generation_enabled:
                            logger.info("Initializing Qwen image generation model")
                            img_gen_model_instance = get_or_create_image_generator()
                        if img_editing_enabled:
                            logger.info("Initializing Qwen image editing model")
                            img_edit_model_instance = get_or_create_image_editor()
                except Exception as exc:
                    logger.warning("Image generation/editing initialization failed: %s", exc)

            code_llm_instance = get_or_create_devstral_llm()

        return (
            embed_model_instance,
            proj_llm_instance,
            code_llm_instance,
            img_analysis_llm_instance,
            media_analysis_llm_instance,
            img_gen_model_instance,
            img_edit_model_instance,
        )
    except Exception as exc:
        logger.exception("Error initializing models: %s", exc)
        raise


@weave.op
def configure_models(
    use_api_mode: bool,
    model_suite: str = "qwen",
    local_model_id: Optional[str] = None,
    session_id: Optional[str] = None,
    use_qwen_vl_for_images: bool = True,
    use_main_model_for_code_agent: bool = False,
    qwen_vl_model_id: Optional[str] = None,
    media_analysis_enabled: bool = False,
    img_generation_enabled: bool = False,
    img_editing_enabled: bool = False,
    rag_provider: str = "jina",
):
    """Configure global model settings for the current agent session."""
    global USE_API_MODE, LOCAL_MODEL_SUITE, embed_model, proj_llm, code_llm, IMAGE_CAPTION_LLM, _ACTIVE_MODEL_CONFIG
    global img_analysis_llm, media_analysis_llm, img_gen_model, img_edit_model
    global RAG_PROVIDER

    resolved_use_api = bool(use_api_mode)

    resolved_model_id = local_model_id
    config = {
        "use_api_mode": resolved_use_api,
        "model_suite": model_suite,
        "local_model_id": resolved_model_id,
        "session_id": session_id if resolved_use_api else None,
        "use_qwen_vl_for_images": use_qwen_vl_for_images if not resolved_use_api else None,
        "use_main_model_for_code_agent": use_main_model_for_code_agent if not resolved_use_api else None,
        "qwen_vl_model_id": qwen_vl_model_id if not resolved_use_api else None,
        "media_analysis_enabled": media_analysis_enabled if not resolved_use_api else None,
        "img_generation_enabled": img_generation_enabled if not resolved_use_api else None,
        "img_editing_enabled": img_editing_enabled if not resolved_use_api else None,
        "rag_provider": rag_provider if not resolved_use_api else None,
    }

    if _ACTIVE_MODEL_CONFIG == config:
        return

    USE_API_MODE = resolved_use_api
    LOCAL_MODEL_SUITE = model_suite
    RAG_PROVIDER = rag_provider

    api_model_name = resolved_model_id if use_api_mode else None

    if use_api_mode:
        embed_model, proj_llm, code_llm = initialize_models(
            use_api_mode=USE_API_MODE,
            model_suite=LOCAL_MODEL_SUITE,
            local_model_id=None,
            api_model_name=api_model_name,
            session_id=session_id,
            rag_provider=rag_provider,
        )
        img_analysis_llm = None
        media_analysis_llm = None
        img_gen_model = None
        img_edit_model = None
    else:
        result = initialize_models(
            use_api_mode=USE_API_MODE,
            model_suite=LOCAL_MODEL_SUITE,
            local_model_id=local_model_id,
            api_model_name=None,
            session_id=session_id,
            use_qwen_vl_for_images=use_qwen_vl_for_images,
            use_main_model_for_code_agent=use_main_model_for_code_agent,
            qwen_vl_model_id=qwen_vl_model_id,
            media_analysis_enabled=media_analysis_enabled,
            img_generation_enabled=img_generation_enabled,
            img_editing_enabled=img_editing_enabled,
            rag_provider=rag_provider,
        )
        embed_model, proj_llm, code_llm, img_analysis_llm, media_analysis_llm, img_gen_model, img_edit_model = result

    Settings.llm = proj_llm
    Settings.embed_model = embed_model
    IMAGE_CAPTION_LLM = proj_llm
    _ACTIVE_MODEL_CONFIG = config

    logger.info(
        "Model configuration set: use_api_mode=%s model_suite=%s model_id=%s",
        USE_API_MODE,
        LOCAL_MODEL_SUITE,
        resolved_model_id,
    )
