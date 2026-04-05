"""Streamlit UI for the Conversational AI Agent."""

import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Page configuration
st.set_page_config(
    page_title="AI Agent",
    page_icon="\U0001f916",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""",
    unsafe_allow_html=True,
)

from ui.state import initialize_session_state
from ui.sidebar import render_session_management, render_settings
from ui.vector_store import render_vector_store_section
from ui.chat import render_chat_interface


def main():
    """Main application entry point."""

    initialize_session_state()

    # Sidebar
    render_session_management()
    st.sidebar.divider()
    render_vector_store_section()
    st.sidebar.divider()
    render_settings()

    # Main area
    render_chat_interface()

    # Footer
    st.sidebar.divider()
    st.sidebar.caption("Conversational AI Agent v2.0")
    st.sidebar.caption("Powered by LlamaIndex and smolagents")


if __name__ == "__main__":
    main()
