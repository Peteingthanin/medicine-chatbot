"""
config/app.py — Application behavioural settings and data paths
"""
# ---------------------------------------------------------------------------
# LLM output post-processing
# ---------------------------------------------------------------------------
# Set True to strip <think>...</think> chain-of-thought blocks from responses.
# Recommended True for API use (Qwen3 thinking mode).
STRIP_THINKING = True

# ---------------------------------------------------------------------------
# Conversational chatbot (POST /chat/converse)
# ---------------------------------------------------------------------------
CONVERSE_MAX_CANDIDATES = 10    # graph top-k for conversational mode
CONVERSE_TOP_K_PER_DRUG = 3     # vector chunks retrieved per candidate drug
CONVERSE_STOP_THRESHOLD = 3     # stop asking when ≤ N candidates remain
CONVERSE_MAX_TURNS      = 3     # hard cap on clarifying questions per session
CONVERSE_SCORE_GAP      = 0.20  # stop if top-1 chunk beats top-2 by this margin

# ---------------------------------------------------------------------------
# Offline pipeline data paths (used by pipeline scripts, not the server)
# ---------------------------------------------------------------------------
CHUNKS_DIR = r".\extracted_data\chunks"
