"""Chat interface UI components."""

import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

from ui.feedback import render_feedback_section
from ui.vector_store import _format_library_label
from utils.session_manager import ChatSession

logger = logging.getLogger(__name__)


# ============================================================================
# Chat Interface
# ============================================================================


def render_chat_interface():
    """Render the main chat interface."""

    if not st.session_state.current_session:
        st.markdown('<div class="main-header">AI Agent</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sub-header">Create a new chat to get started</div>',
            unsafe_allow_html=True,
        )
        return

    session = st.session_state.current_session
    agent_config = session.agent_config

    st.title(f"{session.title}")

    config_info = (
        f"Framework: {agent_config['framework'].capitalize()} | "
        f"Mode: {agent_config['mode'].upper()} | "
        f"Model: {agent_config.get('llm_model', 'N/A').split('/')[-1]}"
    )
    st.caption(config_info)

    st.divider()

    # Display chat history
    for msg in session.messages:
        with st.chat_message(msg.role):
            st.markdown(msg.content)

            if msg.metadata.get("type") == "document_upload":
                storage = msg.metadata.get("storage", "unknown")
                if storage == "vector_store":
                    st.caption("Stored in vector store (searchable via RAG)")
                elif storage == "memory":
                    st.caption("Stored in chat memory (full content)")

    render_feedback_section()

    # Document uploads for smolagents
    if agent_config["framework"] == "smolagents":
        _render_smolagents_uploads(agent_config)

    # Prompt media for LlamaIndex
    if agent_config["framework"] == "llamaindex":
        _render_llamaindex_media(agent_config)

    prompt = st.chat_input("Ask me anything...", key="chat_input")

    pending_prompt_media = session.metadata.get("pending_prompt_media", [])

    if prompt and pending_prompt_media:
        handle_user_message_with_media(prompt, [])
    elif prompt:
        handle_user_message(prompt)


def _render_smolagents_uploads(agent_config: Dict[str, Any]):
    """Render document/media upload UI for smolagents."""

    from ui.documents import process_chat_document

    with st.sidebar.expander("Document Uploads", expanded=False):
        st.markdown("**Chat files (Docling)**")
        doc_file = st.file_uploader(
            "Add to chat memory",
            type=[
                "pdf", "docx", "doc", "pptx", "ppt", "html", "htm",
                "csv", "xlsx", "xls", "txt", "md", "json",
            ],
            key="smol_doc_persist",
            help="Processed with Docling and prepended before chat history",
        )
        if doc_file and st.sidebar.button("Process and save", key="smol_doc_persist_btn"):
            process_chat_document(doc_file)

        st.markdown("**Prompt media (multimodal)**")
        prompt_media = st.file_uploader(
            "Attach media to next prompt",
            type=["png", "jpg", "jpeg", "gif", "webp", "mp3", "wav", "m4a", "mp4", "webm"],
            accept_multiple_files=True,
            key="smol_prompt_media",
            help="Attached to the next prompt and processed via the multimodal tool",
        )
        if prompt_media and st.sidebar.button("Attach media to prompt", key="smol_prompt_media_btn"):
            add_prompt_media_files(prompt_media)


def _render_llamaindex_media(agent_config: Dict[str, Any]):
    """Render prompt media upload UI for LlamaIndex."""

    label = (
        "Prompt Media (LlamaIndex Local)"
        if agent_config["mode"] == "local"
        else "Prompt Media (LlamaIndex API)"
    )
    allow_audio_video = agent_config["mode"] == "api" or agent_config.get(
        "media_analysis_enabled", False
    )
    media_types = ["png", "jpg", "jpeg", "gif", "webp"]
    if allow_audio_video:
        media_types.extend(["mp3", "wav", "m4a", "mp4"])
    media_help = "Adds media to the next prompt without indexing in the vector store"
    if agent_config["mode"] == "local" and not allow_audio_video:
        media_help = "Image-only unless Media Analysis (Audio/Video) is enabled"
    media_label = (
        "Attach images/audio/video to next prompt" if allow_audio_video else "Attach images to next prompt"
    )
    with st.sidebar.expander(label, expanded=False):
        prompt_media = st.file_uploader(
            media_label,
            type=media_types,
            accept_multiple_files=True,
            key="llama_prompt_media",
            help=media_help,
        )
        if prompt_media and st.sidebar.button("Attach media to prompt", key="llama_prompt_media_btn"):
            add_prompt_media_files(prompt_media)


# ============================================================================
# Message Handling
# ============================================================================


def handle_user_message(prompt: str):
    """Handle a user message and generate a response."""

    session = st.session_state.current_session

    session.add_message("user", prompt)
    st.session_state.session_manager.save_session(session)

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response, trace_id = generate_response(prompt)
        st.markdown(response)

    metadata = {"trace_id": trace_id} if trace_id else None
    session.add_message("assistant", response, metadata=metadata)
    if trace_id:
        session.metadata["last_trace_id"] = trace_id
    st.session_state.session_manager.save_session(session)

    st.rerun()


def handle_user_message_with_media(prompt: str, media_files: List):
    """Handle a user message with attached media files."""

    session = st.session_state.current_session
    agent_config = session.agent_config
    prompt_media = _consume_prompt_media(session)

    import tempfile

    media_paths = []
    media_info = []

    try:
        for info in prompt_media:
            path = info.get("path")
            if not path:
                continue
            media_paths.append(path)
            media_info.append({
                "filename": info.get("filename") or Path(path).name,
                "path": path,
                "type": info.get("type") or _infer_media_type(Path(path).suffix.lower()),
            })

        for media_file in media_files:
            ext = Path(media_file.name).suffix.lower()
            media_type = _infer_media_type(ext)

            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
                tmp_file.write(media_file.getvalue())
                abs_path = os.path.abspath(tmp_file.name)
                media_paths.append(abs_path)
                media_info.append({
                    "filename": media_file.name,
                    "path": abs_path,
                    "type": media_type,
                })

        if agent_config["framework"] == "smolagents":
            media_summary = ", ".join(
                [f"{m['path']} ({m['type']})" for m in media_info if m["type"] != "document"]
            )
        else:
            media_summary = ", ".join(
                [f"{m['filename']} ({m['type']})" for m in media_info]
            )
        message_content = f"{prompt}\n\n[Attached: {media_summary}]"

        session.add_message(
            "user",
            message_content,
            metadata={
                "type": "multimodal_message",
                "media_files": media_info,
                "media_paths": media_paths,
            },
        )
        st.session_state.session_manager.save_session(session)

        with st.chat_message("user"):
            st.markdown(prompt)
            for info in media_info:
                if info["type"] == "image":
                    st.image(info["path"], caption=info["filename"], width=300)
                elif info["type"] == "audio":
                    st.audio(info["path"])
                elif info["type"] == "video":
                    st.video(info["path"])

        non_pdf_paths = [p for p in media_paths if Path(p).suffix.lower() != ".pdf"]

        with st.chat_message("assistant"):
            with st.spinner("Processing multimodal content..."):
                if non_pdf_paths:
                    response, trace_id = generate_multimodal_response(prompt, non_pdf_paths)
                else:
                    response, trace_id = generate_response(prompt)
            st.markdown(response)

        metadata = {"trace_id": trace_id} if trace_id else None
        session.add_message("assistant", response, metadata=metadata)
        if trace_id:
            session.metadata["last_trace_id"] = trace_id
        st.session_state.session_manager.save_session(session)

        st.rerun()

    except Exception as e:
        st.error(f"Error processing media files: {e}")
        logger.exception("Multimodal message processing error")
    finally:
        for path in media_paths:
            try:
                os.unlink(path)
            except Exception:
                pass
        _cleanup_prompt_media(prompt_media)


# ============================================================================
# Response Generation
# ============================================================================


def _parse_agent_response(result) -> tuple[str, Optional[str]]:
    """Normalize agent outputs to (response_text, trace_id)."""
    if isinstance(result, tuple) and len(result) == 2:
        response_text, trace_id = result
        return str(response_text), trace_id
    if isinstance(result, dict):
        response_text = result.get("response") or result.get("text") or ""
        trace_id = result.get("trace_id")
        return str(response_text), trace_id
    return str(result), None


def _build_prompt(
    prompt: str,
    session: ChatSession,
    agent_config: Dict[str, Any],
    max_history: int = 8,
) -> str:
    """Build a prompt with documents, vector store metadata, and history."""
    parts = []

    def _retrieval_block() -> Optional[str]:
        if agent_config.get("framework") != "llamaindex":
            return None
        if st.session_state.vector_store_manager is None:
            return None
        session_id = session.session_id
        if not session_id:
            return None

        try:
            conv_store = st.session_state.vector_store_manager.get_or_create_conversation_store(session_id)
            collection = getattr(conv_store, "chroma_collection", None)
            if collection is None or collection.count() == 0:
                return None

            from llama_index.core import VectorStoreIndex
            from llama_index.core import StorageContext

            storage_context = StorageContext.from_defaults(vector_store=conv_store)
            conv_index = VectorStoreIndex.from_vector_store(
                conv_store,
                storage_context=storage_context,
                embed_model=st.session_state.vector_store_manager.embed_model,
            )
            query_engine = conv_index.as_query_engine(similarity_top_k=4)
            retrieved = query_engine.query(prompt)
            retrieved_text = str(retrieved).strip()
            if not retrieved_text:
                return None
            return "Retrieved context from long-term memory:\n" + retrieved_text
        except Exception as exc:
            logger.debug("Vector retrieval skipped: %s", exc)
            return None

    def _document_block() -> Optional[str]:
        docs = []
        for msg in session.messages:
            meta = msg.metadata or {}
            if meta.get("type") == "document_upload" and meta.get("storage") == "memory":
                filename = meta.get("filename", "document")
                file_type = meta.get("file_type", "unknown")
                content = meta.get("full_content") or ""
                if not content:
                    continue
                docs.append(f"[{filename} | {file_type}]\n{content}")
        if not docs:
            return None
        return "Documents uploaded in this conversation:\n" + "\n\n".join(docs)

    def _vector_store_block() -> Optional[str]:
        sources = []
        for msg in session.messages:
            meta = msg.metadata or {}
            if meta.get("type") == "document_upload" and meta.get("storage") == "vector_store":
                filename = meta.get("filename", "document")
                file_type = meta.get("file_type", "unknown")
                sources.append(f"- {filename} ({file_type})")

        linked = session.metadata.get("linked_sources", [])
        if linked and st.session_state.vector_store_manager:
            lib = st.session_state.vector_store_manager.library_index.get("sources", {})
            for source_id in linked:
                entry = lib.get(source_id, {})
                sources.append(f"- {_format_library_label(entry)}")

        if not sources:
            return None

        seen = set()
        uniq = []
        for item in sources:
            if item in seen:
                continue
            seen.add(item)
            uniq.append(item)
        return "Vector store sources linked to this chat:\n" + "\n".join(uniq)

    def _history_block() -> Optional[str]:
        history = []
        for msg in session.messages:
            meta = msg.metadata or {}
            if meta.get("type") == "document_upload":
                continue
            if msg.role not in ("user", "assistant"):
                continue
            history.append(msg)

        if not history:
            return None

        recent = history[-max_history:]
        lines = []
        for msg in recent:
            role = "User" if msg.role == "user" else "Assistant"
            lines.append(f"{role}: {msg.content}")
        return "Conversation history:\n" + "\n".join(lines)

    retrieval_block = _retrieval_block()
    if retrieval_block:
        parts.append(retrieval_block)

    if agent_config["framework"] == "llamaindex":
        vector_block = _vector_store_block()
        if vector_block:
            parts.append(vector_block)
        history_block = _history_block()
        if history_block:
            parts.append(history_block)
    elif agent_config["framework"] == "smolagents":
        doc_block = _document_block()
        if doc_block:
            parts.append(doc_block)
        history_block = _history_block()
        if history_block:
            parts.append(history_block)
    else:
        doc_block = _document_block()
        if doc_block:
            parts.append(doc_block)

    parts.append(f"User question:\n{prompt}")
    return "\n\n".join(parts)


def generate_response(prompt: str) -> tuple[str, Optional[str]]:
    """Generate a response using the configured agent."""

    session = st.session_state.current_session
    agent_config = session.agent_config

    full_prompt = _build_prompt(prompt, session, agent_config)

    try:
        agent = st.session_state.agent

        if not agent:
            return "Agent not initialized. Please create a new chat.", None

        response = agent.run(full_prompt)
        return _parse_agent_response(response)

    except Exception as e:
        error_msg = f"Error generating response: {e}"
        logger.exception("Response generation error")
        return error_msg, None


def generate_multimodal_response(
    prompt: str,
    media_paths: List[str],
) -> tuple[str, Optional[str]]:
    """Generate a response with multimodal content."""

    session = st.session_state.current_session
    agent_config = session.agent_config

    try:
        agent = st.session_state.agent

        if not agent:
            return "Agent not initialized. Please create a new chat.", None

        if agent_config["framework"] == "llamaindex":
            from llama_index_app.ingest import read_and_parse_content

            media_descriptions = []
            for path in media_paths:
                try:
                    docs = read_and_parse_content(path)
                    if docs:
                        description = docs[0].text
                        media_descriptions.append(f"File {path}:\n{description}")
                except Exception as e:
                    logger.warning(f"Failed to process media file {path}: {e}")
                    media_descriptions.append(f"File {path}: [Processing failed]")

            base_prompt = _build_prompt(prompt, session, agent_config)
            multimodal_context = "\n\n".join(media_descriptions + [base_prompt])

            response = agent.run(multimodal_context)
            return _parse_agent_response(response)

        elif agent_config["framework"] == "smolagents":
            media_lines = []
            for path in media_paths:
                if Path(path).suffix.lower() == ".pdf":
                    continue
                ext = Path(path).suffix.lower()
                if ext in {".png", ".jpg", ".jpeg", ".gif", ".webp"}:
                    media_type = "image"
                elif ext in {".mp3", ".wav", ".m4a"}:
                    media_type = "audio"
                elif ext in {".mp4", ".avi", ".mov", ".webm"}:
                    media_type = "video"
                else:
                    media_type = "file"
                media_lines.append(f"- {path} ({media_type})")

            attachments = "\n".join(media_lines)
            base_prompt = _build_prompt(prompt, session, agent_config)
            full_context = (
                "Attached media files:\n"
                f"{attachments}\n\n"
                "Use the multimodal_processor tool on each file_path before answering.\n\n"
                f"{base_prompt}"
            )
            response = agent.run(full_context)
            return _parse_agent_response(response)

        else:
            return "Multimodal processing not supported for this agent configuration.", None

    except Exception as e:
        error_msg = f"Error generating multimodal response: {e}"
        logger.exception("Multimodal response generation error")
        return error_msg, None


# ============================================================================
# Prompt Media Helpers
# ============================================================================


def _infer_media_type(extension: str) -> str:
    ext = extension.lower()
    if not ext.startswith("."):
        ext = f".{ext}"
    if ext in [".png", ".jpg", ".jpeg", ".gif", ".webp"]:
        return "image"
    if ext in [".mp3", ".wav", ".m4a"]:
        return "audio"
    if ext in [".mp4", ".webm"]:
        return "video"
    return "file"


def add_prompt_media_files(uploaded_files):
    """Store media files to attach to the next prompt without indexing."""
    session = st.session_state.current_session
    if not session:
        st.sidebar.warning("Create or select a chat first.")
        return

    if not uploaded_files:
        st.sidebar.warning("Select at least one media file to attach.")
        return

    if not isinstance(uploaded_files, list):
        uploaded_files = [uploaded_files]

    agent_config = session.agent_config or {}
    restrict_audio_video = (
        agent_config.get("framework") == "llamaindex"
        and agent_config.get("mode") == "local"
        and not agent_config.get("media_analysis_enabled", False)
    )

    upload_dir = Path(".prompt_media") / session.session_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    pending = session.metadata.get("pending_prompt_media", [])
    added = 0
    skipped = []

    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.getvalue()
        safe_name = Path(uploaded_file.name).name
        ext = Path(safe_name).suffix.lower()
        media_type = _infer_media_type(ext)
        if restrict_audio_video and media_type in ("audio", "video"):
            skipped.append(safe_name)
            continue
        unique_name = f"{uuid.uuid4()}_{safe_name}"
        file_path = upload_dir / unique_name

        with open(file_path, "wb") as handle:
            handle.write(file_bytes)

        pending.append({
            "filename": safe_name,
            "path": os.path.abspath(file_path),
            "extension": ext,
            "type": media_type,
            "size_bytes": len(file_bytes),
        })
        added += 1

    session.metadata["pending_prompt_media"] = pending
    st.session_state.session_manager.save_session(session)
    if added:
        st.sidebar.success(f"Attached {added} media file(s) to the next prompt.")
    if skipped:
        st.sidebar.warning(
            "Skipped audio/video files (enable Media Analysis to attach): "
            + ", ".join(skipped)
        )


def _consume_prompt_media(session: ChatSession) -> List[Dict[str, Any]]:
    """Return and clear prompt media for the next user message."""
    if not session:
        return []
    return session.metadata.pop("pending_prompt_media", [])


def _cleanup_prompt_media(prompt_media: List[Dict[str, Any]]) -> None:
    """Remove temporary prompt media attachments from disk."""
    for info in prompt_media or []:
        path = info.get("path")
        if not path:
            continue
        try:
            Path(path).unlink(missing_ok=True)
        except Exception as exc:
            logger.warning("Failed to cleanup prompt media %s: %s", path, exc)
