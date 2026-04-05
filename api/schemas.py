"""Request/response schemas for the API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from config import AgentConfig


class CreateSessionRequest(BaseModel):
    """Request body for creating a new chat session."""
    title: Optional[str] = None
    agent_config: AgentConfig


class CreateSessionResponse(BaseModel):
    """Response for a newly created session."""
    session_id: str
    title: str
    agent_config: Dict[str, Any]


class SessionInfo(BaseModel):
    """Summary info for a chat session."""
    session_id: str
    title: str
    created_at: str = ""
    updated_at: str = ""
    message_count: int = 0
    agent_config: Dict[str, Any] = {}


class MessageOut(BaseModel):
    """A single chat message."""
    role: str
    content: str
    timestamp: str
    metadata: Dict[str, Any] = {}


class ChatRequest(BaseModel):
    """Request body for sending a chat message."""
    prompt: str
    media_paths: List[str] = Field(default_factory=list)


class ChatResponse(BaseModel):
    """Response from the agent."""
    response: str
    trace_id: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "ok"
    available_providers: List[str] = []
