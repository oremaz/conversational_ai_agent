"""FastAPI application for the Conversational AI Agent.

Run with:
    uvicorn api.app:app --reload --port 8000
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException

from config import settings
from utils.session_manager import SessionManager

from .schemas import (
    ChatRequest,
    ChatResponse,
    CreateSessionRequest,
    CreateSessionResponse,
    HealthResponse,
    MessageOut,
    SessionInfo,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Conversational AI Agent API",
    version="2.0.0",
    description="REST API for the multi-framework conversational AI agent.",
)

# ---------------------------------------------------------------------------
# Shared state (process-scoped singletons)
# ---------------------------------------------------------------------------

_session_manager = SessionManager(storage_path=settings.chat_sessions_dir)

# agent_id -> agent instance
_agents: Dict[str, object] = {}


def _get_or_create_agent(session_id: str):
    """Return an initialized agent for the given session, creating if needed."""
    if session_id in _agents:
        return _agents[session_id]

    session = _session_manager.load_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    agent_config = session.agent_config

    if agent_config.get("framework") == "llamaindex":
        from llama_index_app.agent import ConversationalAgent

        use_api_mode = agent_config.get("mode") == "api"
        provider = agent_config.get("llm_provider", "gemini")

        agent = ConversationalAgent(
            use_api_mode=use_api_mode,
            model_suite=agent_config.get("model_suite", provider if use_api_mode else "qwen"),
            local_model_id=agent_config.get("llm_model"),
            session_id=session_id,
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

    elif agent_config.get("framework") == "smolagents":
        from smolagents_app.agent import GAIAAgent

        if agent_config.get("mode") != "api":
            raise HTTPException(status_code=400, detail="smolagents only supports API mode")

        agent = GAIAAgent(
            user_id="api_user",
            session_id=session_id,
            provider=agent_config["llm_provider"],
            model_name=agent_config["llm_model"],
        )
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown framework: {agent_config.get('framework')}",
        )

    _agents[session_id] = agent
    return agent


def _parse_agent_response(result) -> tuple[str, Optional[str]]:
    """Normalize agent outputs to (response_text, trace_id)."""
    if isinstance(result, tuple) and len(result) == 2:
        return str(result[0]), result[1]
    if isinstance(result, dict):
        return str(result.get("response") or result.get("text") or ""), result.get("trace_id")
    return str(result), None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        available_providers=settings.available_providers,
    )


@app.get("/sessions", response_model=List[SessionInfo])
async def list_sessions(limit: int = 20):
    """List recent chat sessions."""
    raw = _session_manager.list_sessions(limit=limit)
    return [SessionInfo(session_id=s["session_id"], **{k: v for k, v in s.items() if k != "session_id"}) for s in raw]


@app.post("/sessions", response_model=CreateSessionResponse, status_code=201)
async def create_session(req: CreateSessionRequest):
    """Create a new chat session."""
    config_dict = req.agent_config.model_dump()
    framework = config_dict.get("framework", "")
    model = config_dict.get("llm_model", "")
    title = req.title or f"{framework.capitalize()} - {(model or '').split('/')[-1][:20]}"

    session = _session_manager.create_session(title=title, agent_config=config_dict)

    return CreateSessionResponse(
        session_id=session.session_id,
        title=session.title,
        agent_config=config_dict,
    )


@app.get("/sessions/{session_id}/messages", response_model=List[MessageOut])
async def get_messages(session_id: str):
    """Get all messages for a session."""
    session = _session_manager.load_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    return [
        MessageOut(
            role=m.role,
            content=m.content,
            timestamp=m.timestamp,
            metadata=m.metadata,
        )
        for m in session.messages
    ]


@app.post("/sessions/{session_id}/chat", response_model=ChatResponse)
async def chat(session_id: str, req: ChatRequest):
    """Send a message and get an agent response."""
    session = _session_manager.load_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    agent = _get_or_create_agent(session_id)

    # Save user message
    session.add_message("user", req.prompt)
    _session_manager.save_session(session)

    # Generate response
    try:
        result = agent.run(req.prompt)
        response_text, trace_id = _parse_agent_response(result)
    except Exception as exc:
        logger.exception("Agent error for session %s", session_id)
        raise HTTPException(status_code=500, detail=str(exc))

    # Save assistant message
    metadata = {"trace_id": trace_id} if trace_id else None
    session.add_message("assistant", response_text, metadata=metadata)
    _session_manager.save_session(session)

    return ChatResponse(response=response_text, trace_id=trace_id)


@app.delete("/sessions/{session_id}", status_code=204)
async def delete_session(session_id: str):
    """Delete a chat session."""
    if session_id in _agents:
        del _agents[session_id]

    if not _session_manager.delete_session(session_id, cleanup_vector_store=True):
        raise HTTPException(status_code=404, detail="Session not found")
