"""Session state initialization for the Streamlit UI."""

import streamlit as st

from utils.session_manager import SessionManager
from llama_index_app.utils.document_processor import DocumentProcessor


def initialize_session_state():
    """Initialize Streamlit session state variables."""

    if "session_manager" not in st.session_state:
        st.session_state.session_manager = SessionManager()

    if "current_session" not in st.session_state:
        st.session_state.current_session = None

    if "vector_store_manager" not in st.session_state:
        st.session_state.vector_store_manager = None

    if "conversation_index" not in st.session_state:
        st.session_state.conversation_index = None

    if "doc_processor" not in st.session_state:
        st.session_state.doc_processor = DocumentProcessor()

    if "agent" not in st.session_state:
        st.session_state.agent = None

    if "mcp_servers" not in st.session_state:
        st.session_state.mcp_servers = []

    if "show_new_chat_config" not in st.session_state:
        st.session_state.show_new_chat_config = False
