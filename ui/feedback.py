"""Feedback UI components."""

import logging

import streamlit as st

logger = logging.getLogger(__name__)


def submit_feedback(trace_id: str, score: int, comment: str) -> bool:
    """Send feedback to Langfuse using the agent hook."""
    agent = st.session_state.agent
    if not agent or not hasattr(agent, "add_user_feedback"):
        return False

    try:
        agent.add_user_feedback(trace_id=trace_id, feedback_score=score, comment=comment or None)
        return True
    except Exception as e:
        logger.exception("Agent feedback submission failed: %s", e)
        return False


def render_feedback_section():
    """Render feedback controls for the latest assistant message."""
    session = st.session_state.current_session
    if not session:
        return

    latest_msg = None
    for msg in reversed(session.messages):
        trace_id = msg.metadata.get("trace_id") if msg.metadata else None
        if msg.role == "assistant" and trace_id:
            latest_msg = msg
            break

    if not latest_msg:
        return

    trace_id = latest_msg.metadata.get("trace_id")
    st.subheader("Rate the last response")
    feedback_choice = st.radio(
        "Feedback",
        options=["Helpful", "Not helpful"],
        horizontal=True,
        key=f"feedback_choice_{trace_id}",
    )
    comment = st.text_input(
        "Comment (optional)",
        key=f"feedback_comment_{trace_id}",
    )
    if st.button("Submit feedback", key=f"feedback_submit_{trace_id}"):
        score = 1 if feedback_choice == "Helpful" else 0
        if submit_feedback(trace_id, score, comment):
            st.success("Thanks for the feedback.")
        else:
            st.warning("Feedback is unavailable. Check Langfuse credentials.")
