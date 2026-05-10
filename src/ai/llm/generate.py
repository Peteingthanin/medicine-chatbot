"""
llm/generate.py — Shared LLM generation helpers used across endpoints.

Provides:
  strip_thinking(text) -> str
  generate_answer(...)  -> str   single-turn generation with context + query
  call_model(...)       -> str   low-level model call with pre-built messages
"""
import re
from typing import Optional

from ai.config.llm import DEEPSEEK_MODEL_ID, OPENROUTER_MODEL_ID
from ai.config.app import STRIP_THINKING

# Generation defaults (previously in config/llm.py as CHAT_* constants)
_MAX_TOKENS  = 8192
_TEMPERATURE = 0.5

# ---------------------------------------------------------------------------
# System prompt (Thai medication assistant)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
คุณคือผู้ช่วยด้านยาและสุขภาพ ตอบคำถามเป็นภาษาไทยเท่านั้น
ตอบโดยอ้างอิงจากข้อมูลยาที่ให้ไว้เท่านั้น
หากไม่พบข้อมูลในเอกสาร ให้แจ้งว่าไม่ทราบและแนะนำให้ปรึกษาแพทย์หรือเภสัชกร
ห้ามแต่งเติมข้อมูลที่ไม่มีในเอกสาร
เมื่อมีหลายตัวยาที่เหมาะสม ให้นำเสนอตัวเลือก 2-3 อันดับแรก พร้อมเหตุผล
"""


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def strip_thinking(text: str) -> str:
    """Remove <think>...</think> chain-of-thought blocks from model output."""
    # Complete blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Unclosed blocks (model hit max_tokens mid-thought)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
    # Header-style reasoning blocks
    text = re.sub(
        r"^(?:Thinking Process|Reasoning):.*?\n\n", "", text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return text.strip()


# ---------------------------------------------------------------------------
# Core generation helpers
# ---------------------------------------------------------------------------

async def generate_answer(
    chat_model,
    deepseek_client,
    openai_client,
    context: str,
    query: str,
    model_choice: str,
    phase_name: str = "LLM",
) -> str:
    """
    Build a single-turn medication Q&A prompt and generate a response.
    Routes to the correct backend based on model_choice.
    """
    user_msg = f"ข้อมูลยา:\n\n{context}\n\nคำถาม: {query}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_msg},
    ]
    return await call_model(
        chat_model, deepseek_client, openai_client,
        messages, model_choice, phase_name,
    )


async def call_model(
    chat_model,
    deepseek_client,
    openai_client,
    messages: list[dict],
    model_choice: str,
    phase_name: str,
) -> str:
    """
    Low-level model dispatch. Accepts a fully-formed messages list and routes
    to deepseek / openrouter / local llama_cpp based on model_choice.
    """
    print(f"[{phase_name}] Calling {model_choice} model...")

    # --- Cloud routing ---
    if model_choice in ("deepseek", "cloud") and deepseek_client is not None:
        resp = await deepseek_client.chat.completions.create(
            model=DEEPSEEK_MODEL_ID,
            messages=messages,
            max_tokens=_MAX_TOKENS,
            temperature=_TEMPERATURE,
        )
        answer = resp.choices[0].message.content

    elif model_choice == "minimax" and openai_client is not None:
        resp = await openai_client.chat.completions.create(
            model=OPENROUTER_MODEL_ID,
            messages=messages,
            max_tokens=_MAX_TOKENS,
            temperature=_TEMPERATURE,
        )
        answer = resp.choices[0].message.content

    # --- Local llama_cpp ---
    elif chat_model is not None:
        resp = chat_model.create_chat_completion(
            messages=messages,
            max_tokens=_MAX_TOKENS,
            temperature=_TEMPERATURE,
        )
        answer = resp["choices"][0]["message"]["content"]

    else:
        return "Error: No model configured."

    # --- Post-processing ---
    if STRIP_THINKING:
        raw = answer
        print(f"[{phase_name}-Debug] Raw output: {repr(raw[:120])}...")
        cleaned = strip_thinking(raw)
        # Safety net: if cleaning wiped everything, return raw
        if not cleaned and raw.strip():
            return raw.strip()
        return cleaned

    return answer
