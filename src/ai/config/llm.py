"""
config/llm.py — LLM model paths, API keys, and generation settings
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Local model paths (relative to project root, mounted into Docker at /app)
# ---------------------------------------------------------------------------
EMBED_MODEL_PATH = "llm/Qwen3-Embedding-4B-Q6_K.gguf"
CHAT_MODEL_PATH  = "llm/Qwen3.5-4B-Q6_K.gguf"
# Larger 9B model — used only by offline pipeline scripts (03_extract_graph.py)
GRAPH_MODEL_PATH = "llm/Qwen3.5-9B-UD-Q6_K_XL.gguf"

# ---------------------------------------------------------------------------
# Embedding model settings
# ---------------------------------------------------------------------------
EMBED_N_CTX        = 8192   # context window
EMBED_N_GPU_LAYERS = -1     # -1 = offload all layers to GPU
EMBED_DIMENSIONS   = 2560   # Qwen3-Embedding-4B output dimensions

# Instruction prefix applied to QUERY embeddings only (not to documents)
EMBED_QUERY_PREFIX = (
    "Instruct: Given a medical query in Thai, retrieve relevant drug "
    "information that answers the question\nQuery: "
)

# ---------------------------------------------------------------------------
# Local chat model settings
# ---------------------------------------------------------------------------
# Set False to skip loading the local model into RAM (saves ~8 GB)
USE_LOCAL_CHAT_MODEL = os.getenv("USE_LOCAL_CHAT_MODEL", "false").lower() == "true"

CHAT_N_CTX        = 32768
CHAT_N_GPU_LAYERS = -1
CHAT_MAX_TOKENS   = 8192
CHAT_TEMPERATURE  = 0.5

# ---------------------------------------------------------------------------
# DeepSeek (primary cloud model)
# ---------------------------------------------------------------------------
DEEPSEEK_API_KEY  = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL_ID = "deepseek-v4-flash"

# DeepSeek used by offline graph extraction pipeline (may share key)
DEEPSEEK_GRAPH_API_KEY  = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_GRAPH_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_GRAPH_MODEL_ID = "deepseek-v4-flash"

# ---------------------------------------------------------------------------
# OpenRouter / MiniMax (fallback cloud model)
# ---------------------------------------------------------------------------
OPENROUTER_API_KEY  = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL_ID = "minimax/minimax-m2.5:free"

# Default cloud model (used when model == "cloud" for backwards compat)
DEFAULT_CLOUD_MODEL = "deepseek"
