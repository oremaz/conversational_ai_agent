"""Document upload handling for in-chat documents."""

import logging
from pathlib import Path

import streamlit as st

logger = logging.getLogger(__name__)


def process_chat_document(uploaded_file):
    """Process a document uploaded within a chat."""

    session = st.session_state.current_session
    agent_config = session.agent_config

    with st.spinner(f"Processing {uploaded_file.name}..."):
        try:
            docs, file_type = st.session_state.doc_processor.process_uploaded_file(uploaded_file)

            if not docs:
                st.error("Failed to process document")
                return

            full_content = "\n\n".join([doc.text for doc in docs])

            if agent_config["framework"] == "llamaindex":
                if st.session_state.vector_store_manager is None:
                    st.error("Vector store not initialized")
                    return

                conv_index = st.session_state.vector_store_manager.add_documents_to_conversation(
                    docs,
                    session.session_id,
                    st.session_state.conversation_index,
                )
                st.session_state.conversation_index = conv_index

                session.add_message(
                    "user",
                    f"Uploaded: {uploaded_file.name}\n\nFile indexed in conversation vector store",
                    metadata={
                        "type": "document_upload",
                        "filename": uploaded_file.name,
                        "file_type": file_type,
                        "storage": "vector_store",
                    },
                )

                provider = st.session_state.get("vector_store_embedder_provider", "active")
                st.success(f"{uploaded_file.name} indexed with {provider} embeddings")

            else:
                session.add_message(
                    "user",
                    f"Uploaded: {uploaded_file.name}\n\nFull content loaded ({len(full_content):,} characters)",
                    metadata={
                        "type": "document_upload",
                        "filename": uploaded_file.name,
                        "full_content": full_content,
                        "file_type": file_type,
                        "storage": "memory",
                    },
                )

                st.success(f"{uploaded_file.name} loaded into chat memory")

            st.session_state.session_manager.save_session(session)
            st.rerun()

        except Exception as e:
            st.error(f"Processing failed: {e}")
            logger.exception("Chat document processing error")
