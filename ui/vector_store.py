"""Vector store management UI components for LlamaIndex chats."""

import hashlib
import logging
from typing import Any, Dict, List, Optional

import streamlit as st

from utils.session_manager import ChatSession

logger = logging.getLogger(__name__)


def render_vector_store_section():
    """Render the vector store management section."""

    if not st.session_state.current_session:
        st.sidebar.info("Create or select a chat first")
        return

    agent_config = st.session_state.current_session.agent_config

    if agent_config.get("framework") != "llamaindex":
        st.sidebar.info("Vector stores available only with LlamaIndex")
        return

    session = st.session_state.current_session
    st.sidebar.header("Conversation Knowledge Base")
    st.sidebar.caption("Sources are cached and can be reused across chats")

    # Upload to conversation store
    st.sidebar.subheader("Upload Documents")

    uploaded_file = st.sidebar.file_uploader(
        "Add to this chat",
        type=["pdf", "docx", "txt", "csv", "xlsx", "json", "md", "pptx"],
        help="Documents are cached and linked to this chat",
        key="conversation_upload",
    )

    if uploaded_file:
        if st.sidebar.button("Upload to Chat", key="upload_conversation_btn"):
            upload_to_conversation_store(uploaded_file)

    # Add URL
    st.sidebar.subheader("Add Web Content")

    url_input = st.sidebar.text_input(
        "URL",
        placeholder="https://example.com",
        key="conversation_url",
    )

    if url_input and st.sidebar.button("Fetch and Add to Chat", key="fetch_conversation_btn"):
        fetch_url_to_conversation(url_input)

    # Library sources
    if st.session_state.vector_store_manager:
        _render_library_section(session)


def _render_library_section(session: ChatSession):
    """Render the library source linking section."""

    sources = st.session_state.vector_store_manager.list_library_sources()
    linked_sources = set(session.metadata.get("linked_sources", []))

    st.sidebar.subheader("Library")

    if sources:
        source_lookup = {entry["source_id"]: entry for entry in sources}
        available_sources = [
            entry["source_id"]
            for entry in sources
            if entry["source_id"] not in linked_sources
        ]

        if available_sources:
            selected_sources = st.sidebar.multiselect(
                "Add cached sources",
                options=available_sources,
                format_func=lambda sid: _format_library_label(source_lookup.get(sid, {})),
                key="library_selection",
            )

            if selected_sources and st.sidebar.button("Link Selected to Chat", key="link_library_btn"):
                add_library_sources_to_conversation(selected_sources)
        else:
            st.sidebar.caption("All cached sources are already linked.")
    else:
        st.sidebar.caption("No cached documents yet.")

    if linked_sources:
        st.sidebar.subheader("Linked to This Chat")
        for source_id in linked_sources:
            entry = st.session_state.vector_store_manager.library_index.get("sources", {}).get(
                source_id, {}
            )
            st.sidebar.caption(f"- {_format_library_label(entry)}")

    stats = st.session_state.vector_store_manager.get_stats()
    chat_doc_count = st.session_state.vector_store_manager.get_conversation_document_count(
        session.session_id
    )
    st.sidebar.subheader("Store Statistics")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Chat Docs", chat_doc_count)
    with col2:
        st.metric("Cached Sources", stats.get("library_sources", 0))


def _format_library_label(entry: Dict[str, Any]) -> str:
    """Format a library entry for UI display."""
    label = entry.get("label") or entry.get("source_key") or "Unknown"
    source_type = entry.get("source_type", "source")
    return f"{label} ({source_type})"


def _get_linked_sources(session: ChatSession) -> set:
    """Return linked source ids for the session."""
    return set(session.metadata.get("linked_sources", []))


def _link_source_to_session(session: ChatSession, source_id: str) -> bool:
    """Link a source to the session if not already linked."""
    linked_sources = _get_linked_sources(session)
    if source_id in linked_sources:
        return False
    linked_sources.add(source_id)
    session.metadata["linked_sources"] = sorted(linked_sources)
    return True


def upload_to_conversation_store(uploaded_file):
    """Upload a file to the conversation vector store with caching."""

    with st.spinner(f"Processing {uploaded_file.name}..."):
        try:
            if st.session_state.vector_store_manager is None:
                st.sidebar.error("Vector store not initialized (local LlamaIndex only)")
                return

            file_bytes = uploaded_file.getvalue()
            file_hash = hashlib.sha256(file_bytes).hexdigest()
            source_id = st.session_state.vector_store_manager.get_library_source_id(
                "file", file_hash
            )

            if source_id:
                linked = add_library_sources_to_conversation([source_id], rerun=False)
                if linked:
                    st.sidebar.info("File already cached. Linked to this chat.")
                else:
                    st.sidebar.info("File already linked to this chat.")
                st.rerun()
                return

            docs, file_type = st.session_state.doc_processor.process_uploaded_file(uploaded_file)

            if not docs:
                st.error("No content extracted")
                return

            source_id, _ = st.session_state.vector_store_manager.register_library_source(
                "file",
                file_hash,
                uploaded_file.name,
                docs,
                {"file_type": file_type},
            )

            add_library_sources_to_conversation(
                [source_id], rerun=False, announce_label=uploaded_file.name
            )
            st.sidebar.success(f"Added {len(docs)} chunks ({file_type}) to this chat")
            st.rerun()

        except Exception as e:
            st.sidebar.error(f"Upload failed: {e}")
            logger.exception("Conversation upload error")


def fetch_url_to_conversation(url: str):
    """Fetch a URL and add it to the conversation vector store with caching."""

    with st.spinner(f"Fetching {url}..."):
        try:
            if st.session_state.vector_store_manager is None:
                st.sidebar.error("Vector store not initialized (local LlamaIndex only)")
                return

            normalized_url = url.strip()
            source_id = st.session_state.vector_store_manager.get_library_source_id(
                "url", normalized_url
            )

            if source_id:
                linked = add_library_sources_to_conversation([source_id], rerun=False)
                if linked:
                    st.sidebar.info("URL already cached. Linked to this chat.")
                else:
                    st.sidebar.info("URL already linked to this chat.")
                st.rerun()
                return

            docs, url_type = st.session_state.doc_processor.process_url(normalized_url)

            if not docs:
                st.error("No content extracted")
                return

            source_id, _ = st.session_state.vector_store_manager.register_library_source(
                "url",
                normalized_url,
                normalized_url,
                docs,
                {"url_type": url_type},
            )

            add_library_sources_to_conversation(
                [source_id], rerun=False, announce_label=normalized_url
            )
            st.sidebar.success(f"Added {len(docs)} chunks ({url_type}) to this chat")
            st.rerun()

        except Exception as e:
            st.sidebar.error(f"Fetch failed: {e}")
            logger.exception("URL fetch error")


def add_library_sources_to_conversation(
    source_ids: List[str],
    rerun: bool = True,
    announce_label: Optional[str] = None,
) -> bool:
    """Add cached library sources to the current conversation."""
    session = st.session_state.current_session
    if not session or not st.session_state.vector_store_manager:
        return False

    linked_any = False
    for source_id in source_ids:
        if not _link_source_to_session(session, source_id):
            continue

        conv_index = st.session_state.vector_store_manager.add_library_source_to_conversation(
            source_id,
            session.session_id,
            st.session_state.conversation_index,
        )
        st.session_state.conversation_index = conv_index

        entry = st.session_state.vector_store_manager.library_index.get("sources", {}).get(
            source_id, {}
        )
        label = announce_label or entry.get("label") or source_id

        session.add_message(
            "user",
            f"Linked: {label}\n\nCached content added to conversation vector store",
            metadata={
                "type": "document_upload",
                "source_id": source_id,
                "storage": "vector_store",
            },
        )
        linked_any = True

    st.session_state.session_manager.save_session(session)
    if rerun:
        st.rerun()
    return linked_any
