"""
session/store.py — In-memory conversation session store.

Holds per-session state for the /chat/converse multi-turn endpoint.
Swap this module for a Redis-backed implementation later without touching
any endpoint code — just change the three functions below.
"""
import uuid
from typing import Optional

from ai.models.schemas import ConversationState

# ---------------------------------------------------------------------------
# In-memory store (resets on server restart)
# ---------------------------------------------------------------------------
_sessions: dict[str, ConversationState] = {}


def get_or_create(
    session_id: Optional[str],
    user_contraindications: Optional[list[str]] = None,
) -> tuple[str, ConversationState]:
    """
    Return (session_id, state).
    Creates a fresh ConversationState if session_id is None or unknown.
    """
    sid = session_id or str(uuid.uuid4())
    if sid not in _sessions:
        _sessions[sid] = ConversationState(
            session_id=sid,
            user_contraindications=user_contraindications or [],
        )
    return sid, _sessions[sid]


def save(session_id: str, state: ConversationState) -> None:
    """Persist updated session state."""
    _sessions[session_id] = state


def delete(session_id: str) -> None:
    """Remove a completed or expired session."""
    _sessions.pop(session_id, None)


def get(session_id: str) -> Optional[ConversationState]:
    """Look up a session by ID, returns None if not found."""
    return _sessions.get(session_id)
