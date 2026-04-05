"""Session management and sidebar UI components."""

import logging
import os
from typing import Any, Dict

import streamlit as st

from config import AgentConfig, settings
from llama_index_app.agent import ConversationalAgent
from llama_index_app.utils.vector_store import VectorStoreManager
from smolagents_app.agent import GAIAAgent
from utils.session_manager import ChatSession

logger = logging.getLogger(__name__)


def _has_any_api_key() -> bool:
    """Return whether any supported API key is present in runtime env."""
    return any(
        bool(os.environ.get(name, "").strip())
        for name in ("GOOGLE_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY")
    )


# ============================================================================
# Agent Initialization
# ============================================================================


def initialize_agent_for_session(session: ChatSession):
    """Initialize the agent for a session configuration."""

    agent_config = session.agent_config

    with st.spinner("Initializing agent..."):
        try:
            if agent_config["framework"] == "llamaindex":
                use_api_mode = agent_config["mode"] == "api"
                provider = agent_config.get("llm_provider", "gemini")
                local_model_id = agent_config.get("llm_model")

                agent = ConversationalAgent(
                    use_api_mode=use_api_mode,
                    model_suite=agent_config.get("model_suite", provider if use_api_mode else "qwen"),
                    local_model_id=local_model_id,
                    session_id=session.session_id,
                    media_analysis_enabled=agent_config.get("media_analysis_enabled", False),
                    code_execution_enabled=agent_config.get("code_execution_enabled", True),
                    use_specialized_code_model=agent_config.get("use_specialized_code_model", False),
                    img_generation_enabled=agent_config.get("img_generation_enabled", False),
                    img_editing_enabled=agent_config.get("img_editing_enabled", False),
                    use_qwen_vl_for_images=agent_config.get("use_qwen_vl_for_images", True),
                    use_main_model_for_code_agent=agent_config.get("use_main_model_for_code_agent", False),
                    qwen_vl_model_id=agent_config.get("qwen_vl_model_id"),
                    rag_provider=agent_config.get("rag_provider", "jina"),
                )

                st.session_state.agent = agent

                if agent_config["mode"] == "local":
                    rag_provider = agent_config.get("rag_provider", "jina")
                else:
                    rag_provider = agent_config.get("llm_provider", "gemini")

                if (
                    st.session_state.vector_store_manager is None
                    or st.session_state.get("vector_store_embedder_provider") != rag_provider
                ):
                    st.session_state.vector_store_manager = VectorStoreManager(
                        embedder_provider=rag_provider
                    )
                    st.session_state.vector_store_embedder_provider = rag_provider

                st.session_state.conversation_index = None

                if agent_config["mode"] == "local":
                    st.success(f"LlamaIndex initialized with {rag_provider} RAG")
                else:
                    st.success(f"LlamaIndex initialized (API mode + {rag_provider} embeddings)")

            elif agent_config["framework"] == "smolagents":
                if agent_config["mode"] != "api":
                    st.error("smolagents only supports API mode")
                    return

                agent = GAIAAgent(
                    user_id="streamlit_user",
                    session_id=session.session_id,
                    provider=agent_config["llm_provider"],
                    model_name=agent_config["llm_model"],
                    mcp_servers=st.session_state.mcp_servers,
                )

                st.session_state.agent = agent
                st.session_state.vector_store_manager = None
                st.session_state.conversation_index = None

                st.success("smolagents initialized (API mode)")

        except Exception as e:
            st.error(f"Initialization failed: {e}")
            logger.exception("Agent initialization error")


# ============================================================================
# Session Management
# ============================================================================


def render_session_management():
    """Render the session management sidebar."""

    st.sidebar.header("Chat Sessions")

    if st.sidebar.button("New Chat", type="primary"):
        st.session_state.show_new_chat_config = True

    if st.session_state.show_new_chat_config:
        show_new_chat_config()

    sessions = st.session_state.session_manager.list_sessions(limit=20)

    if sessions:
        st.sidebar.subheader("Recent Chats")

        for sess in sessions:
            col1, col2 = st.sidebar.columns([4, 1])

            with col1:
                if st.button(
                    f"{sess['title'][:30]}",
                    key=f"load_{sess['session_id']}",
                    use_container_width=True,
                ):
                    load_session(sess["session_id"])

            with col2:
                if st.button(
                    "\U0001f5d1\ufe0f",
                    key=f"del_{sess['session_id']}",
                    help="Delete chat",
                ):
                    delete_session(sess["session_id"])


def show_new_chat_config():
    """Show the new chat configuration panel."""

    with st.sidebar.expander("Configure New Chat", expanded=True):
        framework = st.selectbox(
            "Framework",
            ["llamaindex", "smolagents"],
            key="new_chat_framework",
            help="LlamaIndex: Multimodal RAG | smolagents: Tool-calling",
        )

        mode = st.radio(
            "Mode",
            ["API", "Local"],
            key="new_chat_mode",
            help="API: Cloud providers | Local: On-device models",
        )

        use_api = mode == "API"

        if framework == "smolagents" and not use_api:
            st.error("smolagents only supports API mode")
            return

        if use_api:
            if not _has_any_api_key():
                st.error(
                    "API mode requires one environment variable: GOOGLE_API_KEY, OPENAI_API_KEY, or OPENROUTER_API_KEY"
                )
                return
            agent_config = _build_api_config(framework)
        else:
            agent_config = _build_local_config(framework)

        if agent_config is None:
            return

        # MCP Servers (smolagents only)
        if framework == "smolagents":
            _render_mcp_selection()

        if st.button("Create Chat", key="create_chat_btn", type="primary"):
            create_chat_with_config(agent_config)


def _build_api_config(framework: str) -> Dict[str, Any]:
    """Build agent config for API mode."""

    provider_options = ["gemini", "openai"]
    if framework == "llamaindex":
        provider_options.append("openrouter")

    provider = st.selectbox("Provider", provider_options, key="new_chat_provider")

    if provider == "gemini":
        models = ["gemini-3-pro-preview", "gemini-3-flash-preview", "gemini-2.5-flash-lite", "gemini-2.5-flash"]
        model = st.selectbox("Model", models, key="new_chat_model")
    elif provider == "openai":
        models = ["gpt-5.2", "gpt-5.2-pro", "gpt-5-mini", "gpt-5-nano"]
        model = st.selectbox("Model", models, key="new_chat_model")
    else:
        model = st.text_input(
            "OpenRouter Model Path",
            value="openai/gpt-5-mini",
            key="new_chat_openrouter_model",
            help="Default model path for OpenRouter. You can change this per chat.",
        )

    provider_to_env = {
        "gemini": "GOOGLE_API_KEY",
        "openai": "OPENAI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }
    required_key = provider_to_env[provider]
    if not os.environ.get(required_key, "").strip():
        st.error(f"{required_key} is required for provider '{provider}'. Set it in your environment.")
        return None

    use_specialized_code_model = False
    use_image_tools = False

    st.markdown("**Specialized Agents**")

    if provider == "openai":
        use_specialized_code_model = st.checkbox(
            "Specialized Model for Code Agent (gpt-5.2-codex)",
            value=True,
            key="new_chat_code_agent",
            help="Use specialized gpt-5.2-codex model for code execution. If disabled, uses the selected model above.",
        )

    use_image_tools = st.checkbox(
        "Image Tools (Generation + Editing)",
        value=False,
        key="new_chat_image_tools_api",
        help=f"Enable image generation and editing with {model}",
    )

    return {
        "framework": framework,
        "mode": "api",
        "llm_provider": provider,
        "llm_model": model,
        "code_execution_enabled": True,
        "use_specialized_code_model": use_specialized_code_model,
        "img_generation_enabled": use_image_tools,
        "img_editing_enabled": use_image_tools,
        "media_analysis_enabled": False,
    }


def _build_local_config(framework: str) -> Dict[str, Any]:
    """Build agent config for local mode."""

    suite = st.selectbox(
        "Model Suite",
        ["qwen", "ministral", "gpt-oss"],
        key="new_chat_suite",
        help="Qwen: Multimodal models | Ministral: Fast lightweight models | GPT-OSS: Local OpenAI OSS model",
    )

    if suite == "qwen":
        models = ["Qwen/Qwen3.5-35B-A3B", "Qwen/Qwen3.5-9B"]
    elif suite == "gpt-oss":
        models = ["openai/gpt-oss-20b"]
    else:
        models = [
            "mistralai/Ministral-3-3B-Instruct-2512",
            "mistralai/Ministral-3-8B-Instruct-2512",
            "mistralai/Ministral-3-14B-Instruct-2512",
        ]

    model = st.selectbox("Main Model", models, key="new_chat_model", help="Main LLM for general tasks")

    st.markdown("**Specialized Agents**")

    code_label = "Code Agent (Devstral 24B)"
    code_help = "Enable Devstral-Small-2-24B for software engineering tasks"
    if suite == "gpt-oss":
        code_label = "Code Agent"
        code_help = "Enable a code execution agent (GPT-OSS or Devstral)"
    elif suite == "qwen" and model == "Qwen/Qwen3.5-35B-A3B":
        code_label = "Code Agent"
        code_help = "Enable a code execution agent (Qwen3.5 35B or Devstral)"

    use_code_llm = st.checkbox(code_label, value=True, key="new_chat_code_llm", help=code_help)

    use_main_model_for_code_agent = False
    if use_code_llm:
        if suite == "gpt-oss":
            code_model_options = ["Default (Devstral 24B)", "openai/gpt-oss-20b"]
            selected_code_model = st.selectbox(
                "Code Agent Model",
                code_model_options,
                key="new_chat_gpt_oss_code_model",
                help="Pick which model the code agent should use",
            )
            use_main_model_for_code_agent = selected_code_model == "openai/gpt-oss-20b"
        elif suite == "qwen" and model == "Qwen/Qwen3.5-35B-A3B":
            code_model_options = ["Default (Devstral 24B)", "Qwen/Qwen3.5-35B-A3B"]
            selected_code_model = st.selectbox(
                "Code Agent Model",
                code_model_options,
                key="new_chat_qwen_35b_code_model",
                help="Pick which model the code agent should use",
            )
            use_main_model_for_code_agent = selected_code_model == "Qwen/Qwen3.5-35B-A3B"

    qwen_vl_model_id = None
    use_qwen_vl_for_images = False
    if suite == "gpt-oss":
        vl_options = ["None", "Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B"]
        selected_vl = st.selectbox(
            "Image Analysis Model (Qwen3.5)",
            vl_options,
            key="new_chat_gpt_oss_vl_model",
            help="Pick a Qwen3.5 model for image analysis, or select None to disable",
        )
        if selected_vl != "None":
            qwen_vl_model_id = selected_vl
            use_qwen_vl_for_images = True
        else:
            use_qwen_vl_for_images = st.checkbox(
                "Image Analysis (Qwen3.5)",
                value=True,
                key="new_chat_gpt_oss_image_agent",
                help="Enable image analysis using the default Qwen3.5 model",
            )

    use_media_analysis = st.checkbox(
        "Media Analysis (Qwen-Omni)",
        value=False,
        key="new_chat_media_analysis",
        help="Enable Qwen-Omni for audio/video transcription and analysis",
    )

    use_image_gen = st.checkbox(
        "Image Generation (Qwen-Image-2512)",
        value=False,
        key="new_chat_image_gen",
        help="Enable text-to-image generation with Qwen",
    )

    use_image_edit = st.checkbox(
        "Image Editing (Qwen-Image-Edit-2511)",
        value=False,
        key="new_chat_image_edit",
        help="Enable image editing with Qwen",
    )

    st.markdown("**Retrieval (RAG)**")
    rag_provider = st.selectbox(
        "Embeddings + Reranker",
        ["jina", "qwen"],
        key="new_chat_rag_provider",
        help="Qwen requires Qwen3-VL embedding/reranker models to be available",
    )

    return {
        "framework": framework,
        "mode": "local",
        "model_suite": suite,
        "llm_model": model,
        "media_analysis_enabled": use_media_analysis,
        "code_execution_enabled": use_code_llm,
        "img_generation_enabled": use_image_gen,
        "img_editing_enabled": use_image_edit,
        "use_qwen_vl_for_images": use_qwen_vl_for_images,
        "use_main_model_for_code_agent": use_main_model_for_code_agent,
        "qwen_vl_model_id": qwen_vl_model_id,
        "rag_provider": rag_provider,
    }


def _render_mcp_selection():
    """Render MCP server selection UI."""

    from smolagents_app.utils.mcp_connectors import check_mcp_server_requirements, get_available_mcp_servers

    st.subheader("MCP Servers")
    available_servers = get_available_mcp_servers()

    selected_servers = st.multiselect(
        "Enable MCP Tools",
        available_servers,
        key="new_chat_mcp",
        help="Model Context Protocol servers",
    )

    for server in selected_servers:
        met, missing = check_mcp_server_requirements(server)
        if not met:
            st.warning(f"{server}: Missing {', '.join(missing)}")

    st.session_state.mcp_servers = selected_servers


def create_chat_with_config(agent_config: Dict[str, Any]):
    """Create a new chat session with a specific configuration."""

    framework = agent_config["framework"]
    model = agent_config.get("llm_model", "")
    title = f"{framework.capitalize()} - {model.split('/')[-1][:20]}"

    new_session = st.session_state.session_manager.create_session(
        title=title, agent_config=agent_config
    )

    st.session_state.current_session = new_session
    st.session_state.show_new_chat_config = False

    initialize_agent_for_session(new_session)

    st.success("Chat created")
    st.rerun()


def load_session(session_id: str):
    """Load an existing session and initialize its agent."""

    session = st.session_state.session_manager.load_session(session_id)

    if session:
        st.session_state.current_session = session
        initialize_agent_for_session(session)
        st.rerun()
    else:
        st.error("Failed to load session")


def delete_session(session_id: str):
    """Delete a session and its associated data."""

    if st.session_state.session_manager.delete_session(session_id, cleanup_vector_store=True):
        if (
            st.session_state.current_session
            and st.session_state.current_session.session_id == session_id
        ):
            st.session_state.current_session = None
            st.session_state.agent = None

        st.success("Chat deleted")
        st.rerun()
    else:
        st.error("Failed to delete chat")


# ============================================================================
# Settings
# ============================================================================


def render_settings():
    """Render the settings sidebar."""

    st.sidebar.header("Settings")

    st.sidebar.caption(
        "API keys are read from environment variables only (not editable in the app)."
    )

    with st.sidebar.expander("API Keys Status"):
        for name, present in settings.api_key_status().items():
            if present:
                st.success(name)
            else:
                st.error(name)

    st.sidebar.subheader("Data Management")

    if st.sidebar.button("Export All Chats"):
        _export_all_sessions()

    if st.sidebar.button("Clear All Data"):
        if st.sidebar.checkbox("I understand this is irreversible"):
            _clear_all_data()


def _export_all_sessions():
    """Export all chat sessions to JSON."""

    import json
    from datetime import datetime

    sessions = st.session_state.session_manager.list_sessions()

    if not sessions:
        st.sidebar.warning("No sessions to export")
        return

    export_data = {
        "export_date": datetime.now().isoformat(),
        "total_sessions": len(sessions),
        "sessions": [],
    }

    for sess_info in sessions:
        session = st.session_state.session_manager.load_session(sess_info["session_id"])
        if session:
            export_data["sessions"].append(session.to_dict())

    export_json = json.dumps(export_data, indent=2, ensure_ascii=False)

    st.sidebar.download_button(
        label="Download JSON",
        data=export_json,
        file_name=f"ai_agent_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
    )


def _clear_all_data():
    """Clear all application data and reset state."""

    import shutil
    from pathlib import Path

    try:
        if Path(".chat_sessions").exists():
            shutil.rmtree(".chat_sessions")

        if Path("./chroma_db").exists():
            shutil.rmtree("./chroma_db")

        for key in list(st.session_state.keys()):
            del st.session_state[key]

        st.sidebar.success("All data cleared")
        st.rerun()

    except Exception as e:
        st.sidebar.error(f"Clear failed: {e}")
