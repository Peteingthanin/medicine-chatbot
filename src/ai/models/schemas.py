"""
models/schemas.py — All Pydantic request/response schemas for the chatbot API
"""
from typing import Any, Optional
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Single-turn chat endpoints (/chat/vector, /chat/hybrid, /eval)
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    query:                  str
    session_id:             Optional[str] = None
    user_contraindications: Optional[list[str]] = []   # for hybrid pipeline
    top_k:                  Optional[int] = 8
    model:                  str = "deepseek"  # "local" | "deepseek" | "minimax" | "cloud"


class SourceChunk(BaseModel):
    pill_name:  str
    section:    str
    subsection: Optional[str]
    detail:     list[str]
    score:      float


class ChatResponse(BaseModel):
    answer:       str
    pipeline:     str
    sources:      list[SourceChunk]
    retrieval_ms: float


class EvalResponse(BaseModel):
    query:  str
    phase1: ChatResponse
    phase2: ChatResponse


# ---------------------------------------------------------------------------
# Conversational multi-turn endpoint (/chat/converse)
# ---------------------------------------------------------------------------

class ConversationState(BaseModel):
    session_id:             str
    turn:                   int = 0
    history:                list[dict] = []   # [{role, content}]
    symptoms:               list[str]  = []   # accumulated across turns
    current_candidates:     list[str]  = []
    chunks:                 list[Any]  = []   # cached from latest retrieval
    user_contraindications: list[str]  = []


class ConverseRequest(BaseModel):
    query:                  str
    session_id:             Optional[str] = None
    model:                  str = "deepseek"
    user_contraindications: Optional[list[str]] = []


class ConverseResponse(BaseModel):
    session_id:   str
    answer:       str
    is_final:     bool
    turn:         int
    candidates:   list[str]
    sources:      list[SourceChunk]
    retrieval_ms: float
