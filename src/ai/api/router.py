"""
api/router.py — FastAPI route handlers for the RAG chatbot
"""
import asyncio
from fastapi import APIRouter, HTTPException, Request

from ai.config.db  import QDRANT_COLLECTION
from ai.config.llm import EMBED_MODEL_PATH, CHAT_MODEL_PATH
from ai.config.app import (
    CONVERSE_MAX_CANDIDATES, CONVERSE_TOP_K_PER_DRUG,
    CONVERSE_STOP_THRESHOLD, CONVERSE_MAX_TURNS, CONVERSE_SCORE_GAP,
)
from ai.models.schemas import (
    ChatRequest, ChatResponse, SourceChunk, EvalResponse,
    ConversationState, ConverseRequest, ConverseResponse,
)
from ai.llm.generate   import generate_answer, call_model, SYSTEM_PROMPT
from ai.utils.text     import chunks_to_context, to_source_chunks
from ai.session        import store as session_store

router = APIRouter()


# ---------------------------------------------------------------------------
# Single-turn endpoints
# ---------------------------------------------------------------------------

@router.post("/chat/vector", response_model=ChatResponse, tags=["Phase 1"])
async def chat_vector(req: ChatRequest, request: Request):
    """Phase 1 — pure vector retrieval."""
    print(f"[Phase 1] Starting vector retrieval for query: '{req.query}'")
    retriever = request.app.state.vector_retriever
    result    = retriever.retrieve(query=req.query, top_k=req.top_k)

    if not result.chunks:
        print("[Phase 1] ❌ No chunks found.")
        raise HTTPException(status_code=404, detail="No relevant information found.")

    print(f"[Phase 1] ✅ Found {len(result.chunks)} chunks. Preparing context...")
    context = chunks_to_context(result.chunks)
    answer  = await generate_answer(
        request.app.state.chat_model,
        getattr(request.app.state, "deepseek_client", None),
        getattr(request.app.state, "openai_client", None),
        context, req.query, req.model, "Phase 1",
    )
    print(f"[Phase 1] ✅ Finished in {result.retrieval_ms}ms")

    return ChatResponse(
        answer=answer, pipeline="vector",
        sources=to_source_chunks(result.chunks),
        retrieval_ms=result.retrieval_ms,
    )


@router.post("/chat/hybrid", response_model=ChatResponse, tags=["Phase 2"])
async def chat_hybrid(req: ChatRequest, request: Request):
    """Phase 2 — graph candidate finding + vector detail retrieval."""
    print(f"[Phase 2] Starting hybrid retrieval for query: '{req.query}'")
    retriever = request.app.state.hybrid_retriever
    result    = await retriever.retrieve(
        query=req.query,
        user_contraindications=req.user_contraindications or [],
        model_choice=req.model,
    )

    if not result.chunks:
        print("[Phase 2] ❌ No chunks found.")
        raise HTTPException(status_code=404, detail="No relevant information found.")

    candidate_info = ""
    if result.candidate_drugs:
        candidate_info = ("ยาที่อาจเหมาะสม (จากการวิเคราะห์กราฟ): "
                          + ", ".join(result.candidate_drugs) + "\n\n")
    if result.filtered_out:
        candidate_info += ("ยาที่ถูกคัดออก (ข้อห้ามใช้): "
                           + ", ".join(result.filtered_out) + "\n\n")

    print(f"[Phase 2] ✅ Found {len(result.chunks)} chunks across {len(result.candidate_drugs)} drugs. Preparing context...")
    context = candidate_info + chunks_to_context(result.chunks)
    answer  = await generate_answer(
        request.app.state.chat_model,
        getattr(request.app.state, "deepseek_client", None),
        getattr(request.app.state, "openai_client", None),
        context, req.query, req.model, "Phase 2",
    )
    print(f"[Phase 2] ✅ Finished in {result.retrieval_ms}ms")

    return ChatResponse(
        answer=answer, pipeline="hybrid",
        sources=to_source_chunks(result.chunks),
        retrieval_ms=result.retrieval_ms,
    )


@router.post("/eval", response_model=EvalResponse, tags=["Evaluation"])
async def eval_both(req: ChatRequest, request: Request):
    """Run both pipelines on the same query for side-by-side comparison."""
    print(f"\n========================================================")
    print(f"🚀 NEW EVALUATION REQUEST")
    print(f"Query: '{req.query}'")
    print(f"Model: {req.model}")
    print(f"========================================================\n")
    if req.model == "cloud":
        phase1, phase2 = await asyncio.gather(
            chat_vector(req, request),
            chat_hybrid(req, request),
        )
    else:
        phase1 = await chat_vector(req, request)
        phase2 = await chat_hybrid(req, request)
    return EvalResponse(query=req.query, phase1=phase1, phase2=phase2)


# ---------------------------------------------------------------------------
# Multi-turn conversational endpoint
# ---------------------------------------------------------------------------

@router.post("/chat/converse", response_model=ConverseResponse, tags=["Conversational"])
async def chat_converse(req: ConverseRequest, request: Request):
    """
    Phase 3 — Conversational clarification loop.
    Turn 0: returns candidate drug list + clarifying question.
    Turn 1+: narrows candidates, asks another question or gives final answer.
    """
    retriever  = request.app.state.hybrid_retriever
    chat_model = request.app.state.chat_model
    deepseek_cl = getattr(request.app.state, "deepseek_client", None)
    openai_cl  = getattr(request.app.state, "openai_client", None)

    # --- Load or create session ---
    session_id, session = session_store.get_or_create(
        req.session_id, req.user_contraindications
    )
    session.history.append({"role": "user", "content": req.query})

    # --- State-Aware Extraction ---
    extracted_data = await retriever._extract_intent_and_entities(req.query, req.model)
    new_symptoms   = extracted_data["symptoms"]
    intent         = extracted_data["intent"]
    retrieval_ms   = 0.0

    if session.turn == 0 or new_symptoms:
        if new_symptoms:
            session.symptoms.extend(new_symptoms)
            session.symptoms = list(dict.fromkeys(session.symptoms))
            print(f"[Converse] Turn {session.turn} — Updated known symptoms: {session.symptoms}")

        extracted_data["symptoms"] = session.symptoms
        combined_query = " ".join(m["content"] for m in session.history if m["role"] == "user")

        result = await retriever.retrieve(
            query=combined_query,
            user_contraindications=session.user_contraindications,
            top_k_per_drug=CONVERSE_TOP_K_PER_DRUG,
            max_candidates=CONVERSE_MAX_CANDIDATES,
            model_choice=req.model,
            pre_extracted_data=extracted_data,
        )
        session.current_candidates = result.candidate_drugs
        session.chunks = result.chunks
        retrieval_ms   = result.retrieval_ms
        intent         = result.intent

    candidates = session.current_candidates
    chunks     = session.chunks

    # --- Stopping criteria ---
    score_gap_reached = False
    if len(chunks) >= 2:
        sorted_scores = sorted([c.score for c in chunks], reverse=True)
        score_gap_reached = (sorted_scores[0] - sorted_scores[1]) >= CONVERSE_SCORE_GAP

    if intent in ["drug_inquiry", "general_knowledge"]:
        should_stop = True
    else:
        should_stop = (
            len(candidates) <= CONVERSE_STOP_THRESHOLD
            or session.turn >= CONVERSE_MAX_TURNS
            or score_gap_reached
        )

    if not chunks:
        raise HTTPException(status_code=404, detail="No relevant information found.")

    context = chunks_to_context(chunks)

    if should_stop:
        # --- Final answer ---
        print(f"[Converse] Turn {session.turn} — Giving final answer. Candidates: {candidates}")
        if intent == "symptom_search":
            final_system = (
                SYSTEM_PROMPT
                + "\nตอบแบบครบถ้วน แนะนำยาที่เหมาะสมที่สุด 1-3 อันดับแรกพร้อมเหตุผล\n"
                "Think briefly and provide the final recommendation."
            )
            user_action = "กรุณาแนะนำยาที่เหมาะสมสำหรับอาการของผู้ใช้"
        else:
            final_system = (
                SYSTEM_PROMPT
                + "\nตอบคำถามของผู้ใช้โดยอ้างอิงจากข้อมูลยาที่ให้ไว้ ตอบให้ตรงคำถามและเข้าใจง่าย\n"
                "Think briefly and provide a helpful answer."
            )
            user_action = "กรุณาตอบคำถามของผู้ใช้โดยอ้างอิงจากข้อมูลยาที่ค้นพบ"

        conversation_context = "\n".join(
            f"{'ผู้ใช้' if m['role'] == 'user' else 'ผู้ช่วย'}: {m['content']}"
            for m in session.history
        )
        user_msg = (
            f"บทสนทนา:\n{conversation_context}\n\n"
            f"ข้อมูลยา:\n{context}\n\n"
            f"{user_action}"
        )
        messages = [
            {"role": "system", "content": final_system},
            {"role": "user",   "content": user_msg},
        ]
        answer = await call_model(chat_model, deepseek_cl, openai_cl, messages, req.model, "Converse-Final")

        session_store.delete(session_id)

        return ConverseResponse(
            session_id=session_id, answer=answer, is_final=True,
            turn=session.turn, candidates=candidates,
            sources=to_source_chunks(chunks), retrieval_ms=retrieval_ms,
        )

    else:
        # --- Clarifying question ---
        print(f"[Converse] Turn {session.turn} — Asking question. Candidates: {candidates}")

        top_shown = candidates[:5]
        candidate_lines = "\n".join(f"  • {p}" for p in top_shown)
        if len(candidates) > 5:
            candidate_lines += f"\n  และยาอื่นๆ อีก {len(candidates)-5} ชนิด"

        question_system = (
            "คุณคือผู้ช่วยด้านยาและสุขภาพ ตอบเป็นภาษาไทย\n"
            "หน้าที่ของคุณคือถามคำถามสั้นๆ 1 ข้อ เพื่อระบุยาที่เหมาะสมที่สุด\n"
            "ห้ามแนะนำยาในขั้นตอนนี้ ให้ถามคำถามเพื่อสอบถามข้อมูลเพิ่มเติมเท่านั้น\n"
            "Think briefly and get straight to the question.\n\n"
            "--- ตัวอย่าง (EXAMPLE) ---\n"
            "User: ผมมีอาการปวดหลัง ยาที่เป็นไปได้: Ibuprofen, Paracetamol\n"
            "Assistant: <think>Ibuprofen มีผลต่อกระเพาะ ต้องถามประวัติโรคกระเพาะ</think>\nคุณมีประวัติเป็นโรคกระเพาะอาหารหรือโรคไตหรือไม่ครับ?\n"
            "--- จบตัวอย่าง (END OF EXAMPLE) ---"
        )
        conversation_context = "\n".join(
            f"{'ผู้ใช้' if m['role'] == 'user' else 'ผู้ช่วย'}: {m['content']}"
            for m in session.history
        )
        user_msg = (
            f"บทสนทนา:\n{conversation_context}\n\n"
            f"ยาที่อาจเหมาะสม: {', '.join(top_shown)}\n\n"
            f"ถามคำถามสั้นๆ 1 ข้อ เพื่อระบุยาที่เหมาะสมที่สุด"
        )
        messages = [
            {"role": "system", "content": question_system},
            {"role": "user",   "content": user_msg},
        ]
        question = await call_model(chat_model, deepseek_cl, openai_cl, messages, req.model, "Converse-Q")

        answer = (
            f"ยาที่อาจเหมาะสมกับอาการของคุณ ได้แก่:\n{candidate_lines}\n\n{question}"
            if session.turn == 0 else question
        )

        session.history.append({"role": "assistant", "content": answer})
        session.turn += 1
        session.current_candidates = candidates
        session_store.save(session_id, session)

        return ConverseResponse(
            session_id=session_id, answer=answer, is_final=False,
            turn=session.turn, candidates=candidates,
            sources=to_source_chunks(chunks), retrieval_ms=retrieval_ms,
        )


# ---------------------------------------------------------------------------
# Utility endpoints
# ---------------------------------------------------------------------------

@router.get("/drugs", tags=["Utils"])
async def list_drugs(request: Request):
    """List all drugs currently in the vector store."""
    drugs = request.app.state.vector_retriever.get_all_drugs()
    return {"drugs": drugs, "count": len(drugs)}


@router.get("/health", tags=["Utils"])
async def health(request: Request):
    """Check that models and databases are ready."""
    try:
        info = request.app.state.qdrant.get_collection(QDRANT_COLLECTION)
        vector_count = info.points_count
    except Exception:
        vector_count = -1

    neo4j_nodes = 0
    neo4j_rels  = 0
    try:
        driver = request.app.state.hybrid_retriever.neo4j_driver
        if driver:
            with driver.session() as s:
                neo4j_nodes = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
                neo4j_rels  = s.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
    except Exception:
        pass

    return {
        "status":       "ok",
        "embed_model":  EMBED_MODEL_PATH,
        "chat_model":   CHAT_MODEL_PATH,
        "vector_store": {"collection": QDRANT_COLLECTION, "points": vector_count},
        "graph_store":  {"backend": "neo4j", "nodes": neo4j_nodes, "edges": neo4j_rels},
    }
