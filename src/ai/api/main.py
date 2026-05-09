"""
api/main.py — FastAPI application factory and startup lifespan
"""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from llama_cpp import Llama
from qdrant_client import QdrantClient
from openai import AsyncOpenAI

from ai.config.llm import (
    EMBED_MODEL_PATH, EMBED_N_CTX, EMBED_N_GPU_LAYERS,
    CHAT_MODEL_PATH,  CHAT_N_CTX,  CHAT_N_GPU_LAYERS,
    USE_LOCAL_CHAT_MODEL,
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL,
    OPENROUTER_API_KEY, OPENROUTER_BASE_URL,
)
from ai.config.db import QDRANT_PATH
from ai.retrieval import VectorRetriever, HybridRetriever
from ai.api.router import router


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def load_embed_model() -> Llama:
    print("[startup] Loading embedding model...")
    return Llama(
        model_path=EMBED_MODEL_PATH,
        embedding=True,
        n_ctx=EMBED_N_CTX,
        n_gpu_layers=EMBED_N_GPU_LAYERS,
        verbose=False,
    )


def load_chat_model() -> Llama:
    print("[startup] Loading chat model...")
    return Llama(
        model_path=CHAT_MODEL_PATH,
        n_ctx=CHAT_N_CTX,
        n_gpu_layers=CHAT_N_GPU_LAYERS,
        chat_format="chatml",
        verbose=False,
    )


# ---------------------------------------------------------------------------
# App lifespan — load all models and clients once at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.embed_model = load_embed_model()

    if USE_LOCAL_CHAT_MODEL:
        app.state.chat_model = load_chat_model()
    else:
        app.state.chat_model = None
        print("[startup] Skipping local chat model (USE_LOCAL_CHAT_MODEL=False)")

    # DeepSeek client (primary cloud model)
    if DEEPSEEK_API_KEY:
        app.state.deepseek_client = AsyncOpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
        )
        print("[startup] DeepSeek client initialized.")
    else:
        app.state.deepseek_client = None

    # OpenRouter client (fallback cloud model)
    if OPENROUTER_API_KEY:
        app.state.openai_client = AsyncOpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
        )
        print("[startup] OpenRouter client initialized.")
    else:
        app.state.openai_client = None

    app.state.qdrant = QdrantClient(path=QDRANT_PATH)

    app.state.vector_retriever = VectorRetriever(
        app.state.embed_model, app.state.qdrant
    )
    app.state.hybrid_retriever = HybridRetriever(
        app.state.embed_model, app.state.chat_model, app.state.qdrant,
        app.state.deepseek_client, app.state.openai_client,
    )
    print("[startup] All models loaded. API ready.")
    yield
    print("[shutdown] Shutting down.")


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Medication RAG Chatbot API",
    description="Thai medication information chatbot with vector and hybrid RAG pipelines",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

# Serve frontend at root
_FRONTEND_DIR = Path(__file__).parent.parent.parent / "frontend"
_VUE_DIST = _FRONTEND_DIR / "dist"
_FRONTEND_HTML = _FRONTEND_DIR / "index.html"

if _VUE_DIST.exists() and (_VUE_DIST / "index.html").exists():
    _SERVE_DIR = _VUE_DIST
    _SERVE_HTML = _VUE_DIST / "index.html"
else:
    _SERVE_DIR = _FRONTEND_DIR
    _SERVE_HTML = _FRONTEND_HTML

@app.get("/")
async def serve_frontend():
    """Serve the frontend index.html at root URL."""
    return FileResponse(_SERVE_HTML)

if _SERVE_DIR == _VUE_DIST:
    app.mount("/assets", StaticFiles(directory=_SERVE_DIR / "assets"), name="assets")

